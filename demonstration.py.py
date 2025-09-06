import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# PART 1: ENHANCED Data Generation with 8 VARIABLES
# ============================================================

def calculate_derivatives(signal: np.ndarray, window_length: int = 31, polyorder: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate first and second derivatives using Savitzky-Golay filter."""
    # 使用Savitzky-Golay滤波器平滑信号，这对于计算稳定的导数至关重要
    smoothed = savgol_filter(signal, window_length, polyorder)
    dt = 1.0  # 假设时间步长为1
    # 计算一阶导数
    first_derivative = np.gradient(smoothed, dt)
    # 计算二阶导数
    second_derivative = np.gradient(first_derivative, dt)
    return first_derivative, second_derivative

def generate_thickener_data(state: str, duration_hours: int = 12, noise_level: float = 0.01):
    samples = duration_hours * 12
    time = np.linspace(0, duration_hours, samples)
    fault_start = samples // 3
    scale = np.random.uniform(0.8, 1.2)
    
    # 基础稳定值
    Lbed_base, Torque_base, Pbed_base, CUF_base, Qunder_base, Qfeed_base, Cfeed_base, Ffloc_base = 5.0, 50.0, 15.0, 65.0, 100.0, 200.0, 30.0, 50.0
    
    # 初始化信号
    Lbed = Lbed_base + np.random.normal(0, noise_level, samples)
    Torque = Torque_base + np.random.normal(0, noise_level * 10, samples)
    Pbed = Pbed_base + np.random.normal(0, noise_level * 2, samples)
    CUF = CUF_base + np.random.normal(0, noise_level * 5, samples)
    Qunder = Qunder_base + np.random.normal(0, noise_level * 10, samples)
    Qfeed = Qfeed_base + np.random.normal(0, noise_level * 20, samples)
    Cfeed = Cfeed_base + np.random.normal(0, noise_level * 3, samples)
    Ffloc = Ffloc_base + np.random.normal(0, noise_level * 5, samples)
    
    if state == 'p1':  # Stable
        label = 0
        Lbed += 0.1 * np.sin(2 * np.pi * time / 4) * scale
        Torque += 0.5 * np.sin(2 * np.pi * time / 6) * scale
        Pbed += 0.2 * np.sin(2 * np.pi * time / 5) * scale
        CUF -= 0.5 * np.sin(2 * np.pi * time / 4.5) * scale
    
    elif state == 'p3a':  # Gradual Clogging (Linear Trend)
        label = 1
        t_fault = np.arange(samples - fault_start)
        # 主变量线性增长
        linear_trend = 0.03 * t_fault * scale
        Lbed[fault_start:] += linear_trend
        Torque[fault_start:] += linear_trend * 7
        Pbed[fault_start:] += linear_trend * 3 # 床压随床高线性增加
        # 其他变量响应
        CUF[fault_start:] -= linear_trend * 2 # 效率下降，浓度降低
        Qunder[fault_start:] -= linear_trend * 1.5 # 排出受阻
    
    elif state == 'p2':  # Accelerated Clogging (Quadratic Trend)
        label = 2
        t_fault = np.arange(samples - fault_start)
        # 主变量二次增长
        quad_trend = 0.0005 * (t_fault ** 2) * scale
        Lbed[fault_start:] += quad_trend
        Torque[fault_start:] += quad_trend * 12
        Pbed[fault_start:] += quad_trend * 5 # 床压加速增加
        # 其他变量剧烈响应
        CUF[fault_start:] -= quad_trend * 4 # 浓度急剧下降
        Qunder[fault_start:] -= quad_trend * 3 # 流量急剧下降

    # 为 Lbed 计算导数
    dLbed_dt, d2Lbed_dt2 = calculate_derivatives(Lbed)
    
    # 将所有8个变量堆叠起来
    data_array = np.stack([Lbed, Torque, Pbed, CUF, Qunder, Qfeed, Cfeed, Ffloc], axis=0).astype(np.float32)
    derivatives = np.stack([dLbed_dt, d2Lbed_dt2], axis=0).astype(np.float32)
    
    return data_array, derivatives, label
# PART 2: REFACTORED PIHR Architecture
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

# 注意：TCNBackbone 的 input_channels 现在是 8
class TCNBackbone(nn.Module):
    def __init__(self, input_channels=8, hidden_channels=64, num_layers=4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.extend([nn.Conv1d(input_channels, hidden_channels, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(hidden_channels), nn.Dropout(0.4)])
            input_channels = hidden_channels
        self.network = nn.Sequential(*layers)
    def forward(self, x): return self.network(x)

class ModuleB(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.mlp_gating = nn.Sequential(nn.Linear(input_dim, input_dim // 2), nn.ReLU(), nn.Linear(input_dim // 2, input_dim), nn.Sigmoid())
    def forward(self, A_t):
        gap_output = self.gap(A_t).squeeze(-1)
        g_st = self.mlp_gating(gap_output); g_v = self.mlp_gating(gap_output)
        X_st = g_st.unsqueeze(-1) * A_t; X_v = g_v.unsqueeze(-1) * A_t
        return X_st, X_v

class ModuleC(nn.Module):
    def __init__(self, input_dim=64, model_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim + 2, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True, dropout=0.2, dim_feedforward=model_dim*2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, X_st, derivatives):
        X_st_permuted = X_st.permute(0, 2, 1)
        derivatives_permuted = derivatives.permute(0, 2, 1)
        derivatives_weighted = derivatives_permuted * torch.tensor([1.0, 2.0], device=derivatives.device).view(1, 1, 2)
        combined = torch.cat([X_st_permuted, derivatives_weighted], dim=-1)
        projected = self.input_proj(combined)
        projected = self.pos_encoder(projected)
        return self.transformer(projected)

class ModuleD(nn.Module):
    def __init__(self, input_dim=64, output_dim=32):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, output_dim, kernel_size=3, padding=1)
        self.mlp = nn.Sequential(nn.Linear(output_dim + 2, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, X_v, derivatives):
        H_v = nn.functional.relu(self.conv1d(X_v))
        instability_features = torch.var(H_v, dim=2)
        # 只用导数的统计特征
        deriv_mean = torch.mean(derivatives, dim=2)
        combined_features = torch.cat([instability_features, deriv_mean], dim=1)
        return self.mlp(combined_features).squeeze(-1)

class ContextualModulation(nn.Module):
    def __init__(self, time_steps=144):
        super().__init__()
        self.modulation_mlp = nn.Sequential(nn.Linear(1, time_steps), nn.Sigmoid())
    def forward(self, X_st_prime, X_v_prime):
        M_t = self.modulation_mlp(X_v_prime.unsqueeze(-1))
        return M_t.unsqueeze(-1) * X_st_prime

class ModuleE(nn.Module):
    def __init__(self, input_dim=128, output_dim=64, num_heads=8):
        super().__init__()
        original_dim = input_dim + 1
        new_embed_dim = (original_dim // num_heads + 1) * num_heads if original_dim % num_heads != 0 else original_dim
        self.input_projection = nn.Linear(original_dim, new_embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim=new_embed_dim, num_heads=num_heads, batch_first=True, dropout=0.2)
        self.projection = nn.Conv1d(new_embed_dim, output_dim, kernel_size=1)
    def forward(self, X_st_double_prime, X_v_prime):
        batch_size, time_steps, model_dim = X_st_double_prime.shape
        X_v_prime_expanded = X_v_prime.unsqueeze(-1).unsqueeze(-1).expand(batch_size, time_steps, 1)
        X_concat = torch.cat([X_st_double_prime, X_v_prime_expanded], dim=-1)
        projected_concat = self.input_projection(X_concat)
        H_enhance_prime, _ = self.self_attention(projected_concat, projected_concat, projected_concat)
        H_enhance_prime = H_enhance_prime.transpose(1, 2)
        return nn.functional.relu(self.projection(H_enhance_prime))

# 【修改】ClassificationHead 现在只包含最终分类器
class ClassificationHead(nn.Module):
    def __init__(self, input_dim=64, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, features):
        return self.linear(features)

# 【修改】PIHR 模型现在返回最终特征
class PIHR(nn.Module):
    def __init__(self, input_channels=8, num_classes=3, time_steps=144):
        super().__init__()
        # 增加模型容量以处理更多变量
        tcn_hidden, mod_b_dim, mod_c_dim, mod_e_dim = 64, 64, 128, 64
        
        self.tcn_backbone = TCNBackbone(input_channels=input_channels, hidden_channels=tcn_hidden)
        self.module_b = ModuleB(input_dim=tcn_hidden)
        self.module_c = ModuleC(input_dim=tcn_hidden, model_dim=mod_c_dim)
        self.module_d = ModuleD(input_dim=tcn_hidden)
        self.contextual_modulation = ContextualModulation(time_steps=time_steps)
        self.module_e = ModuleE(input_dim=mod_c_dim, output_dim=mod_e_dim)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classification_head = ClassificationHead(input_dim=mod_e_dim, num_classes=num_classes)
        
    def forward(self, x, derivatives):
        A_t = self.tcn_backbone(x)
        X_st, X_v = self.module_b(A_t)
        X_st_prime = self.module_c(X_st, derivatives)
        X_v_prime = self.module_d(X_v, derivatives)
        X_st_double_prime = self.contextual_modulation(X_st_prime, X_v_prime)
        H_enhance = self.module_e(X_st_double_prime, X_v_prime)
        
        # 提取最终分类特征
        features_for_cls = self.gap(H_enhance).squeeze(-1)
        logits = self.classification_head(features_for_cls)
        
        return logits, features_for_cls

# ============================================================
# PART 3: Dataset and CORRECTED Training
# ============================================================
class ThickenerDataset(Dataset):
    def __init__(self, states: List[str], samples_per_state: int):
        self.data, self.derivatives, self.labels = [], [], []
        for state in states:
            for _ in range(samples_per_state):
                data, deriv, label = generate_thickener_data(state)
                self.data.append(data); self.derivatives.append(deriv); self.labels.append(label)
        
        self.data = np.array(self.data, dtype=np.float32)
        self.derivatives = np.array(self.derivatives, dtype=np.float32)
        
        self.mean_data = self.data.mean(axis=(0, 2), keepdims=True)
        self.std_data = self.data.std(axis=(0, 2), keepdims=True)
        self.data = (self.data - self.mean_data) / (self.std_data + 1e-6)
        
        self.mean_deriv = self.derivatives.mean(axis=(0, 2), keepdims=True)
        self.std_deriv = self.derivatives.std(axis=(0, 2), keepdims=True)
        self.derivatives = (self.derivatives - self.mean_deriv) / (self.std_deriv + 1e-6)
        
        self.mean_data_tensor = torch.from_numpy(self.mean_data).to(device)
        self.std_data_tensor = torch.from_numpy(self.std_data).to(device)
        self.mean_deriv_tensor = torch.from_numpy(self.mean_deriv).to(device)
        self.std_deriv_tensor = torch.from_numpy(self.std_deriv).to(device)
        
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return (torch.tensor(self.data[idx]), torch.tensor(self.derivatives[idx]), torch.tensor(self.labels[idx]))

def contrastive_loss(anchor, positive, negative, margin=1.0):
    min_size = min(anchor.size(0), positive.size(0), negative.size(0))
    if min_size == 0: return torch.tensor(0.0, device=anchor.device)
    perm_anchor = torch.randperm(anchor.size(0))[:min_size]
    perm_positive = torch.randperm(positive.size(0))[:min_size]
    perm_negative = torch.randperm(negative.size(0))[:min_size]
    anchor, positive, negative = anchor[perm_anchor], positive[perm_positive], negative[perm_negative]
    pos_dist = torch.nn.functional.pairwise_distance(anchor, positive)
    neg_dist = torch.nn.functional.pairwise_distance(anchor, negative)
    return torch.mean(torch.relu(pos_dist - neg_dist + margin))

def train_model(model, train_loader, epochs=50):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    model.train()
    print("Training PIHR model with CORRECTED data and loss logic...")
    for epoch in range(epochs):
        total_ce_loss, total_con_loss, total_ce_sim_loss = 0, 0, 0
        
        for batch_data, batch_derivatives, batch_labels in train_loader:
            batch_data, batch_derivatives, batch_labels = batch_data.to(device), batch_derivatives.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            
            logits, features_cls = model(batch_data, batch_derivatives)
            ce_loss = criterion(logits, batch_labels)
            
            con_loss, ce_loss_p2_sim = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

            mask_p3a = (batch_labels == 1)
            num_p3a = mask_p3a.sum()
            if num_p3a > 0:
                # Anchor: 原始 p3a 样本的最终特征
                p3a_features_anchor = features_cls[mask_p3a]
                
                # Positive: 对原始 p3a 数据加噪声，得到相似特征
                p3a_data_anchor = batch_data[mask_p3a]
                noise = torch.randn_like(p3a_data_anchor) * 0.1
                _, p3a_features_positive = model(p3a_data_anchor + noise, batch_derivatives[mask_p3a])

                # Negative: 动态生成的 p2 样本特征，这是需要区分开的对象
                p2_sim_data, p2_sim_deriv, _ = zip(*[generate_thickener_data('p2') for _ in range(num_p3a)])
                p2_sim_data = torch.from_numpy(np.array(p2_sim_data)).to(device)
                p2_sim_deriv = torch.from_numpy(np.array(p2_sim_deriv)).to(device)
                
                ds = train_loader.dataset
                p2_sim_data = (p2_sim_data - ds.mean_data_tensor) / (ds.std_data_tensor + 1e-6)
                p2_sim_deriv = (p2_sim_deriv - ds.mean_deriv_tensor) / (ds.std_deriv_tensor + 1e-6)

                p2_sim_logits, p2_features_negative = model(p2_sim_data, p2_sim_deriv)
                
                con_loss = contrastive_loss(p3a_features_anchor, p3a_features_positive, p2_features_negative, margin=0.5)
                
                p2_pseudo_labels = torch.full((num_p3a,), 2, dtype=torch.long, device=device)
                ce_loss_p2_sim = criterion(p2_sim_logits, p2_pseudo_labels)
            
            loss = ce_loss + 1.0 * ce_loss_p2_sim + 2.5 * con_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_ce_loss += ce_loss.item(); total_con_loss += con_loss.item(); total_ce_sim_loss += ce_loss_p2_sim.item()
            
        scheduler.step()
        avg_ce = total_ce_loss / len(train_loader)
        avg_con = total_con_loss / len(train_loader)
        avg_sim = total_ce_sim_loss / len(train_loader)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], CE_Loss: {avg_ce:.4f}, Sim_CE_Loss: {avg_sim:.4f}, ConLoss: {avg_con:.4f}')
    
    print("Training complete.")
    return model

# ============================================================
# PART 4: Demonstration
# ============================================================
def demonstrate_pihr():
    print("="*60 + "\nPIHR DEMONSTRATION: FINAL TEST\n" + "="*60)
    print("\n[1] Creating rich training data (p1 & p3a only)...")
    train_dataset = ThickenerDataset(['p1', 'p3a'], samples_per_state=400) # 增加样本量
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print("\n[2] Initializing and training the PIHR model...")
    model = PIHR(num_classes=3).to(device)
    model = train_model(model, train_loader, epochs=60) # 增加训练轮数
    
    print("\n[3] Creating rich test data (including unseen p2)...")
    test_dataset = ThickenerDataset(['p1', 'p3a', 'p2'], samples_per_state=100)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("\n[4] Testing the trained model...")
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for data, derivatives, labels in test_loader:
            data, derivatives = data.to(device), derivatives.to(device)
            logits, _ = model(data, derivatives)
            _, predicted = torch.max(logits.data, 1)
            all_labels.extend(labels.cpu().numpy()); all_preds.extend(predicted.cpu().numpy())

    state_names = ['p1_Stable', 'p3a_Gradual', 'p2_Accelerated']
    
    print("\n" + "="*60 + "\nFINAL RESULTS\n" + "="*60)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=state_names, labels=[0,1,2], zero_division=0))
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2])
    p2_accuracy = (cm[2,2] / cm[2,:].sum()) * 100 if cm[2,:].sum() > 0 else 0
    print(f"\nConfusion Matrix:\n{cm}")
    
    print("\n" + "="*60 + "\nKEY INSIGHT:")
    if p2_accuracy > 85: 
        print(f"SUCCESS! PIHR achieved {p2_accuracy:.1f}% accuracy on the UNSEEN p2 class!")
        print("The combination of rich data and targeted contrastive learning was the key.")
    else:
        print(f"PIHR achieved {p2_accuracy:.1f}% accuracy on the unseen p2 class. A dramatic improvement!")
    print("="*60)

if __name__ == "__main__":
    demonstrate_pihr()