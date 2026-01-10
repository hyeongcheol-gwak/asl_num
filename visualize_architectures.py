import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def setup_plot(title, figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    plt.title(title, fontsize=20, fontweight='bold', pad=20, color='#2c3e50')
    return fig, ax

def draw_box(ax, x, y, w, h, text, color='#3498db', text_color='white', fontsize=12):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                  linewidth=2, edgecolor='#2c3e50', facecolor=color, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, color=text_color, fontsize=fontsize, 
            ha='center', va='center', fontweight='bold')

def draw_arrow(ax, x, y, dx, dy):
    ax.arrow(x, y, dx, dy, head_width=1.5, head_length=2, fc='#34495e', ec='#34495e', 
             length_includes_head=True)

def visualize_mlp():
    fig, ax = setup_plot("MLP Architecture (Baseline)")
    
    # Input
    draw_box(ax, 30, 90, 40, 6, "Input [Landmarks + Angles + Dists]\n[81 - D]", color='#95a5a6')
    draw_arrow(ax, 50, 90, 0, -4)
    
    # Layers
    layers = [
        "Linear [81 -> 512] + BN + SiLU",
        "Linear [512 -> 256] + BN + SiLU",
        "Linear [256 -> 128] + BN + SiLU",
        "Linear [128 -> 64] + BN + SiLU",
        "Linear [64 -> 10] [Output]"
    ]
    
    curr_y = 78
    for i, label in enumerate(layers):
        color = '#3498db' if i < len(layers)-1 else '#e74c3c'
        draw_box(ax, 30, curr_y, 40, 6, label, color=color)
        if i < len(layers) - 1:
            draw_arrow(ax, 50, curr_y, 0, -6)
            curr_y -= 12
            
    plt.tight_layout()
    plt.savefig('arch_mlp.png', dpi=300, bbox_inches='tight')
    print("Saved arch_mlp.png")

def visualize_hybrid():
    fig, ax = setup_plot("Hybrid Model Architecture\n(Transformer + Geometric MLP)")
    
    # Input splitting
    draw_box(ax, 15, 90, 25, 6, "Landmark Coordinates\n[(21, 3) - D]", color='#2ecc71')
    draw_box(ax, 60, 90, 25, 6, "Geometric Features\n[20 - D]", color='#f1c40f', text_color='#2c3e50')
    
    # Transformer Stream
    draw_arrow(ax, 27.5, 90, 0, -4)
    draw_box(ax, 15, 80, 25, 6, "Input Projection [64 - D]", color='#2ecc71')
    draw_arrow(ax, 27.5, 80, 0, -4)
    draw_box(ax, 15, 70, 25, 6, "Positional Embedding", color='#2ecc71')
    draw_arrow(ax, 27.5, 70, 0, -4)
    draw_box(ax, 15, 55, 25, 10, "Transformer Encoder\n[3 Layers, 4 Heads]", color='#2ecc71')
    draw_arrow(ax, 27.5, 55, 0, -4)
    draw_box(ax, 15, 45, 25, 6, "CLS Token Feature", color='#2ecc71')
    
    # MLP Stream
    draw_arrow(ax, 72.5, 90, 0, -15)
    draw_box(ax, 60, 65, 25, 10, "Geometric MLP\n[Linear + BN + GELU]", color='#f1c40f', text_color='#2c3e50')
    draw_arrow(ax, 72.5, 65, 0, -14)
    draw_box(ax, 59, 45, 27, 6, "Geometric Feature [32 - D]", color='#f1c40f', text_color='#2c3e50')
    
    # Fusion
    draw_arrow(ax, 27.5, 45, 15, -10)
    draw_arrow(ax, 72.5, 45, -15, -10)
    draw_box(ax, 33, 25, 34, 10, "Fusion Layer\n[Concat + Linear + BN + GELU]", color='#3498db')
    draw_arrow(ax, 50, 25, 0, -5)
    draw_box(ax, 40, 14, 20, 6, "Output [10 - D]", color='#e74c3c')
    
    plt.tight_layout()
    plt.savefig('arch_hybrid.png', dpi=300, bbox_inches='tight')
    print("Saved arch_hybrid.png")

def visualize_ensemble():
    fig, ax = setup_plot("Ensemble Stacking Architecture")
    
    # Base Models
    models = ["ResNet1D", "Transformer", "GCN", "XGBoost", "LightGBM", "CatBoost"]
    x_pos = [5, 20, 35, 50, 65, 80]
    
    for x, name in zip(x_pos, models):
        color = '#1abc9c' if 'GBM' not in name and 'Boost' not in name else '#f39c12'
        draw_box(ax, x, 70, 14, 8, name, color=color, fontsize=10)
        draw_arrow(ax, x+7, 70, 50-(x+7), -20)
        
    # Meta Model
    draw_box(ax, 35, 40, 30, 10, "Stacking Layer\n(Meta-Classifier / Mean)", color='#3498db')
    draw_arrow(ax, 50, 40, 0, -10)
    draw_box(ax, 40, 24, 20, 6, "Final Output", color='#e74c3c')
    
    # Legend/Note
    ax.text(50, 10, "* Uses 5-Fold Cross Validation for each model", ha='center', fontsize=10, style='italic', color='#7f8c8d')
    
    plt.tight_layout()
    plt.savefig('arch_ensemble.png', dpi=300, bbox_inches='tight')
    print("Saved arch_ensemble.png")

if __name__ == "__main__":
    visualize_mlp()
    visualize_hybrid()
    visualize_ensemble()
