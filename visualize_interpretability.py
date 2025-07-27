"""
Additional visualization utilities for interpretability
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict

def create_decision_flow_diagram(decision_path: List[str], 
                               save_path: str = None):
    """Create a flow diagram of the decision path"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create boxes for each step
    n_steps = len(decision_path)
    y_positions = np.linspace(0.8, 0.2, n_steps)
    
    for i, (y, step) in enumerate(zip(y_positions, decision_path)):
        # Box
        box = plt.Rectangle((0.1, y-0.05), 0.8, 0.08, 
                          facecolor='lightblue', 
                          edgecolor='darkblue',
                          linewidth=2)
        ax.add_patch(box)
        
        # Text
        ax.text(0.5, y, step, ha='center', va='center', 
               fontsize=10, wrap=True)
        
        # Arrow
        if i < n_steps - 1:
ax.arrow(0.5, y-0.05, 0, -0.07, 
                    head_width=0.03, head_length=0.02, 
                    fc='gray', ec='gray')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('PIHR Decision Flow', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_feature_importance_timeline(attention_history: pd.DataFrame,
                                   critical_times: List[int] = None):
    """Plot how feature importance changes over time"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each variable's attention weight
    for col in attention_history.columns:
        ax.plot(attention_history.index, attention_history[col], 
               marker='o', markersize=4, label=col, linewidth=2)
    
    # Mark critical times
    if critical_times:
        for t, label in critical_times:
            ax.axvline(t, color='red', linestyle='--', alpha=0.5)
            ax.text(t, 0.9, label, rotation=90, 
                   verticalalignment='bottom', fontsize=8)
    
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Feature Importance Evolution During Fault Development')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig