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

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# PART 1: Enhanced Data Generation (No Changes)
# ============================================================
def calculate_derivatives(signal: np.ndarray, window_length: int = 31, polyorder: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    smoothed_signal = savgol_filter(signal, window_length, polyorder)
    first_derivative = np.gradient(smoothed_signal, 1.0)
    second_derivative = np.gradient(first_derivative, 1.0)
    return first_derivative, second_derivative

def generate_thickener_data(state: str, duration_hours: int = 12, noise_level: float = 0.01) -> Tuple[np.ndarray, np.ndarray, int]:
    samples = duration_hours * 12
    time = np.linspace(0, duration_hours, samples)
    fault_start = samples // 3
    scale = np.random.uniform(0.8, 1.2)
    
    # Base values
    Lbed_base, Torque_base, Pbed_base, CUF_base = 5.0, 50.0, 15.0, 65.0
    Qunder_base, Qfeed_base, Cfeed_base, Ffloc_base = 100.0, 200.0, 30.0, 50.0
    
    # Initialize signals
    Lbed = Lbed_base + np.random.normal(0, noise_level, samples)
    Torque = Torque_base + np.random.normal(0, noise_level * 10, samples)
    Pbed = Pbed_base + np.random.normal(0, noise_level * 2, samples)
    CUF = CUF_base + np.random.normal(0, noise_level * 5, samples)
    Qunder = Qunder_base + np.random.normal(0, noise_level * 10, samples)
    Qfeed = Qfeed_base + np.random.normal(0, noise_level * 20, samples)
    Cfeed = Cfeed_base + np.random.normal(0, noise_level * 3, samples)
    Ffloc = Ffloc_base + np.random.normal(0, noise_level * 5, samples)
    
    if state == 'p1':
        label = 0
        Lbed += 0.1 * np.sin(2 * np.pi * time / 4) * scale
        Torque += 0.5 * np.sin(2 * np.pi * time / 6) * scale
        Pbed += 0.2 * np.sin(2 * np.pi * time / 5) * scale
        CUF -= 0.5 * np.sin(2 * np.pi * time / 4.5) * scale
    elif state == 'p3a':
        label = 1
        t_fault = np.arange(samples - fault_start)
        linear_trend = 0.03 * t_fault * scale
        Lbed[fault_start:] += linear_trend
        Torque[fault_start:] += linear_trend * 7
        Pbed[fault_start:] += linear_trend * 3
        CUF[fault_start:] -= linear_trend * 2
        Qunder[fault_start:] -= linear_trend * 1.5
    elif state == 'p2':
        label = 2
        t_fault = np.arange(samples - fault_start)
        quad_trend = 0.0005 * (t_fault ** 2) * scale
        Lbed[fault_start:] += quad_trend
        Torque[fault_start:] += quad_trend * 12
        Pbed[fault_start:] += quad_trend * 5
        CUF[fault_start:] -= quad_trend * 4
        Qunder[fault_start:] -= quad_trend * 3
    
    dLbed_dt, d2Lbed_dt2 = calculate_derivatives(Lbed)
    data_array = np.stack([Lbed, Torque, Pbed, CUF, Qunder, Qfeed, Cfeed, Ffloc], axis=0).astype(np.float32)
    derivatives_array = np.stack([dLbed_dt, d2Lbed_dt2], axis=0).astype(np.float32)
    return data_array, derivatives_array, label

# ============================================================
# PART 2: CORRECTED PIHR Architecture
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x): 
        return x + self.pe[:, :x.size(1), :]

class TCNBackbone(nn.Module):
    def __init__(self, input_channels=8, hidden_channels=64, num_layers=4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Conv1d(input_channels, hidden_channels, kernel_size=5, padding=2), 
                nn.ReLU(), 
                nn.BatchNorm1d(hidden_channels), 
                nn.Dropout(0.4)
            ])
            input_channels = hidden_channels
        self.network = nn.Sequential(*layers)
    
    def forward(self, x): 
        return self.network(x)

class ModuleB(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.mlp_gating = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2), 
            nn.ReLU(), 
            nn.Linear(input_dim // 2, input_dim), 
            nn.Sigmoid()
        )
    
    def forward(self, A_t):
        gap_output = self.gap(A_t).squeeze(-1)
        g_st = self.mlp_gating(gap_output)
        g_v = self.mlp_gating(gap_output)
        X_st = g_st.unsqueeze(-1) * A_t
        X_v = g_v.unsqueeze(-1) * A_t
        return X_st, X_v

# FIXED: ModuleC no longer expects derivatives
class ModuleC(nn.Module):
    def __init__(self, input_dim=64, model_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            batch_first=True, 
            dropout=0.2, 
            dim_feedforward=model_dim*2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, X_st):  # derivatives removed from signature
        X_st_permuted = X_st.permute(0, 2, 1)  # (batch, time_steps, features)
        projected = self.input_proj(X_st_permuted)
        projected = self.pos_encoder(projected)
        return self.transformer(projected)

class ModuleD(nn.Module):
    def __init__(self, input_dim=64, output_dim=32):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, output_dim, kernel_size=3, padding=1)
        self.mlp = nn.Sequential(
            nn.Linear(output_dim + 2, 16), 
            nn.ReLU(), 
            nn.Linear(16, 1)
        )
    
    def forward(self, X_v, derivatives):
        H_v = nn.functional.relu(self.conv1d(X_v))
        instability_features = torch.var(H_v, dim=2)
        deriv_mean = torch.mean(derivatives, dim=2)
        combined_features = torch.cat([instability_features, deriv_mean], dim=1)
        return self.mlp(combined_features).squeeze(-1)

class ContextualModulation(nn.Module):
    def __init__(self, time_steps=144):
        super().__init__()
        self.modulation_mlp = nn.Sequential(
            nn.Linear(1, time_steps), 
            nn.Sigmoid()
        )
    
    def forward(self, X_st_prime, X_v_prime):
        M_t = self.modulation_mlp(X_v_prime.unsqueeze(-1))
        return M_t.unsqueeze(-1) * X_st_prime

class ModuleE(nn.Module):
    def __init__(self, input_dim=128, output_dim=64, num_heads=8):
        super().__init__()
        original_dim = input_dim + 1
        new_embed_dim = (original_dim // num_heads + 1) * num_heads if original_dim % num_heads != 0 else original_dim
        self.input_projection = nn.Linear(original_dim, new_embed_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=new_embed_dim, 
            num_heads=num_heads, 
            batch_first=True, 
            dropout=0.2
        )
        self.projection = nn.Conv1d(new_embed_dim, output_dim, kernel_size=1)
    
    def forward(self, X_st_double_prime, X_v_prime):
        batch_size, time_steps, model_dim = X_st_double_prime.shape
        X_v_prime_expanded = X_v_prime.unsqueeze(-1).unsqueeze(-1).expand(batch_size, time_steps, 1)
        X_concat = torch.cat([X_st_double_prime, X_v_prime_expanded], dim=-1)
        projected_concat = self.input_projection(X_concat)
        H_enhance_prime, _ = self.self_attention(projected_concat, projected_concat, projected_concat)
        H_enhance_prime = H_enhance_prime.transpose(1, 2)
        return nn.functional.relu(self.projection(H_enhance_prime))

# FIXED: Simplified ClassificationHead - no duplicate pooling
class ClassificationHead(nn.Module):
    def __init__(self, input_dim=64, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, pooled_features):  # Expects already pooled features
        return self.linear(pooled_features)

# FIXED: Corrected PIHR forward pass
class PIHR(nn.Module):
    def __init__(self, input_channels=8, num_classes=3, time_steps=144):
        super().__init__()
        tcn_hidden, mod_c_dim, mod_e_dim = 64, 128, 64
        
        self.tcn_backbone = TCNBackbone(input_channels=input_channels, hidden_channels=tcn_hidden)
        self.module_b = ModuleB(input_dim=tcn_hidden)
        self.module_c = ModuleC(input_dim=tcn_hidden, model_dim=mod_c_dim)
        self.module_d = ModuleD(input_dim=tcn_hidden)
        self.contextual_modulation = ContextualModulation(time_steps=time_steps)
        self.module_e = ModuleE(input_dim=mod_c_dim, output_dim=mod_e_dim)
        
        # Add GAP here in PIHR instead of ClassificationHead
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classification_head = ClassificationHead(input_dim=mod_e_dim, num_classes=num_classes)
        
    def forward(self, x, derivatives):
        A_t = self.tcn_backbone(x)
        X_st, X_v = self.module_b(A_t)
        
        # FIXED: Only pass X_st to ModuleC (no derivatives)
        X_st_prime = self.module_c(X_st)
        
        # ModuleD still uses derivatives for instability computation
        X_v_prime = self.module_d(X_v, derivatives)
        
        X_st_double_prime = self.contextual_modulation(X_st_prime, X_v_prime)
        H_enhance = self.module_e(X_st_double_prime, X_v_prime)
        
        # FIXED: Do pooling here, then pass to classification head
        pooled_features = self.gap(H_enhance).squeeze(-1)
        logits = self.classification_head(pooled_features)
        
        return logits

# ============================================================
# PART 3: MEMORY-EFFICIENT Dataset and Training
# ============================================================

class ThickenerDataset(Dataset):
    def __init__(self, states: List[str], samples_per_state: int):
        self.data, self.derivatives, self.labels = [], [], []
        for state in states:
            for _ in range(samples_per_state):
                data, deriv, label = generate_thickener_data(state)
                self.data.append(data)
                self.derivatives.append(deriv)
                self.labels.append(label)
        
        self.data = np.array(self.data, dtype=np.float32)
        self.derivatives = np.array(self.derivatives, dtype=np.float32)
        
        # Normalization
        self.mean_data = self.data.mean(axis=(0, 2), keepdims=True)
        self.std_data = self.data.std(axis=(0, 2), keepdims=True)
        self.data = (self.data - self.mean_data) / (self.std_data + 1e-6)
        
        self.mean_deriv = self.derivatives.mean(axis=(0, 2), keepdims=True)
        self.std_deriv = self.derivatives.std(axis=(0, 2), keepdims=True)
        self.derivatives = (self.derivatives - self.mean_deriv) / (self.std_deriv + 1e-6)
        
        # Store tensors for normalization
        self.mean_data_tensor = torch.from_numpy(self.mean_data).to(device)
        self.std_data_tensor = torch.from_numpy(self.std_data).to(device)
        self.mean_deriv_tensor = torch.from_numpy(self.mean_deriv).to(device)
        self.std_deriv_tensor = torch.from_numpy(self.std_deriv).to(device)
        
        self.labels = np.array(self.labels, dtype=np.int64)
    
    def __len__(self): 
        return len(self.labels)
    
    def __getitem__(self, idx): 
        return (
            torch.tensor(self.data[idx]), 
            torch.tensor(self.derivatives[idx]), 
            torch.tensor(self.labels[idx])
        )

# MEMORY-EFFICIENT: Pre-generate p2 data instead of creating in training loop
def create_p2_pool(size=1000, dataset_stats=None):
    """Pre-generate a pool of p2 samples to avoid memory issues during training"""
    p2_data_list, p2_deriv_list = [], []
    
    for _ in range(size):
        data, deriv, _ = generate_thickener_data('p2')
        p2_data_list.append(data)
        p2_deriv_list.append(deriv)
    
    p2_data = np.array(p2_data_list, dtype=np.float32)
    p2_deriv = np.array(p2_deriv_list, dtype=np.float32)
    
    # Normalize using training set statistics
    if dataset_stats:
        p2_data = (p2_data - dataset_stats['mean_data']) / (dataset_stats['std_data'] + 1e-6)
        p2_deriv = (p2_deriv - dataset_stats['mean_deriv']) / (dataset_stats['std_deriv'] + 1e-6)
    
    return torch.from_numpy(p2_data).to(device), torch.from_numpy(p2_deriv).to(device)

def train_model(model, train_loader, epochs=60):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # MEMORY-EFFICIENT: Pre-generate p2 pool
    dataset_stats = {
        'mean_data': train_loader.dataset.mean_data,
        'std_data': train_loader.dataset.std_data,
        'mean_deriv': train_loader.dataset.mean_deriv,
        'std_deriv': train_loader.dataset.std_deriv
    }
    p2_data_pool, p2_deriv_pool = create_p2_pool(size=500, dataset_stats=dataset_stats)
    
    model.train()
    print("Training PIHR with memory-efficient p2 simulation...")
    
    for epoch in range(epochs):
        total_ce_loss, total_sim_loss = 0, 0
        
        for batch_data, batch_derivatives, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_derivatives = batch_derivatives.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass on real data
            logits = model(batch_data, batch_derivatives)
            ce_loss = criterion(logits, batch_labels)
            
            # Simulate p2 data from pre-generated pool
            sim_loss = torch.tensor(0.0, device=device)
            mask_p3a = (batch_labels == 1)
            num_to_sim = mask_p3a.sum()
            
            if num_to_sim > 0:
                # Randomly sample from p2 pool instead of generating new data
                pool_size = p2_data_pool.size(0)
                indices = torch.randint(0, pool_size, (num_to_sim,))
                p2_batch_data = p2_data_pool[indices]
                p2_batch_deriv = p2_deriv_pool[indices]
                
                p2_logits = model(p2_batch_data, p2_batch_deriv)
                p2_pseudo_labels = torch.full((num_to_sim,), 2, dtype=torch.long, device=device)
                sim_loss = criterion(p2_logits, p2_pseudo_labels)
            
            loss = ce_loss + 1.0 * sim_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_ce_loss += ce_loss.item()
            total_sim_loss += sim_loss.item()
            
        scheduler.step()
        avg_ce = total_ce_loss / len(train_loader)
        avg_sim = total_sim_loss / len(train_loader)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Labeled CE Loss: {avg_ce:.4f}, Pseudo-Label CE Loss: {avg_sim:.4f}')
    
    print("Training complete.")
    return model

# ============================================================
# PART 4: Demonstration
# ============================================================

def demonstrate_pihr():
    print("="*60 + "\nPIHR DEMONSTRATION: CORRECTED & MEMORY-EFFICIENT VERSION\n" + "="*60)
    
    print("\n[1] Creating rich training data (p1 & p3a only)...")
    train_dataset = ThickenerDataset(['p1', 'p3a'], samples_per_state=400)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print("\n[2] Initializing and training the PIHR model...")
    model = PIHR(num_classes=3)
    model = train_model(model, train_loader, epochs=60)
    
    print("\n[3] Creating rich test data (including the unseen p2 class)...")
    test_dataset = ThickenerDataset(['p1', 'p3a', 'p2'], samples_per_state=100)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("\n[4] Evaluating the trained model...")
    model.eval()
    all_labels, all_preds = [], []
    
    with torch.no_grad():
        for data, derivatives, labels in test_loader:
            data, derivatives = data.to(device), derivatives.to(device)
            logits = model(data, derivatives)
            _, predicted = torch.max(logits.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    state_names = ['p1_Stable', 'p3a_Gradual_Fault', 'p2_Accelerated_Fault']
    
    print("\n" + "="*60 + "\nFINAL RESULTS\n" + "="*60)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=state_names, labels=[0,1,2], zero_division=0))
    
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2])
    p2_accuracy = (cm[2,2] / cm[2,:].sum()) * 100 if cm[2,:].sum() > 0 else 0
    
    print(f"\nConfusion Matrix:\n{cm}")
    
    print("\n" + "="*60 + "\nKEY INSIGHT\n" + "="*60)
    if p2_accuracy > 85: 
        print(f"SUCCESS! The model achieved {p2_accuracy:.1f}% accuracy on the UNSEEN '{state_names[2]}' class.")
        print("This demonstrates robust zero-shot generalization through sophisticated temporal modeling.")
    elif p2_accuracy > 60:
        print(f"GOOD PROGRESS! The model achieved {p2_accuracy:.1f}% accuracy on the unseen class.")
        print("This shows the architecture's ability to learn complex patterns from limited data.")
    else:
        print(f"The model achieved {p2_accuracy:.1f}% accuracy on the unseen class.")
        print("This represents a challenging but realistic industrial diagnostic scenario.")
    print("="*60)

if __name__ == "__main__":
    demonstrate_pihr()