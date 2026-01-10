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

def visualize_resnet1d():
    fig, ax = setup_plot("ResNet1D Architecture")
    
    # Two-stream input
    draw_box(ax, 15, 90, 25, 6, "Landmarks [21, 3]", color='#95a5a6')
    draw_box(ax, 60, 90, 25, 6, "Geo Features [20]", color='#f1c40f', text_color='#2c3e50')
    
    # Left stream: Conv layers
    draw_arrow(ax, 27.5, 90, 0, -6)
    draw_box(ax, 12, 78, 31, 6, "Conv1d [3->64] + BN + GELU", color='#1abc9c', fontsize=10)
    draw_arrow(ax, 27.5, 78, 0, -6)
    draw_box(ax, 12, 66, 31, 6, "Conv1d [64->128] + BN + GELU", color='#1abc9c', fontsize=10)
    draw_arrow(ax, 27.5, 66, 0, -6)
    draw_box(ax, 12, 54, 31, 6, "Conv1d [128->256] + BN + GELU", color='#1abc9c', fontsize=10)
    draw_arrow(ax, 27.5, 54, 0, -6)
    draw_box(ax, 17, 42, 21, 6, "Global Mean Pool", color='#16a085', fontsize=10)
    
    # Right stream: Geo FC
    draw_arrow(ax, 72.5, 90, 0, -42)
    draw_box(ax, 62, 42, 21, 6, "FC [20->64]", color='#f39c12', text_color='#2c3e50', fontsize=10)
    
    # Fusion
    draw_arrow(ax, 27.5, 42, 15, -10)
    draw_arrow(ax, 72.5, 42, -15, -10)
    draw_box(ax, 32, 28, 36, 6, "Concat + Linear Head [10]", color='#3498db', fontsize=10)
    draw_arrow(ax, 50, 28, 0, -6)
    draw_box(ax, 40, 16, 20, 6, "Output [10]", color='#e74c3c')
    
    plt.tight_layout()
    plt.savefig('arch_resnet1d.png', dpi=300, bbox_inches='tight')
    print("Saved arch_resnet1d.png")

def visualize_transformer_ultimate():
    fig, ax = setup_plot("Ultimate Transformer Architecture")
    
    # Two-stream input
    draw_box(ax, 15, 90, 25, 6, "Landmarks [21, 3]", color='#95a5a6')
    draw_box(ax, 60, 90, 25, 6, "Geo Features [20]", color='#f1c40f', text_color='#2c3e50')
    
    # Left stream: Transformer
    draw_arrow(ax, 27.5, 90, 0, -6)
    draw_box(ax, 12, 78, 31, 6, "Linear Projection [3->128]", color='#9b59b6', fontsize=10)
    draw_arrow(ax, 27.5, 78, 0, -6)
    draw_box(ax, 12, 66, 31, 6, "Positional Embedding", color='#8e44ad', fontsize=10)
    draw_arrow(ax, 27.5, 66, 0, -6)
    draw_box(ax, 10, 52, 35, 10, "Transformer Encoder\n[8 Heads, 4 Layers]", color='#8e44ad', fontsize=10)
    draw_arrow(ax, 27.5, 52, 0, -6)
    draw_box(ax, 17, 40, 21, 6, "Global Mean Pool", color='#9b59b6', fontsize=10)
    
    # Right stream: Geo FC
    draw_arrow(ax, 72.5, 90, 0, -44)
    draw_box(ax, 62, 40, 21, 6, "FC [20->64]", color='#f39c12', text_color='#2c3e50', fontsize=10)
    
    # Fusion
    draw_arrow(ax, 27.5, 40, 15, -10)
    draw_arrow(ax, 72.5, 40, -15, -10)
    draw_box(ax, 32, 26, 36, 6, "Concat + Linear Head [10]", color='#3498db', fontsize=10)
    draw_arrow(ax, 50, 26, 0, -6)
    draw_box(ax, 40, 14, 20, 6, "Output [10]", color='#e74c3c')
    
    plt.tight_layout()
    plt.savefig('arch_transformer_ultimate.png', dpi=300, bbox_inches='tight')
    print("Saved arch_transformer_ultimate.png")

def visualize_gcn():
    fig, ax = setup_plot("GCN Architecture")
    
    # Input with adjacency hint
    draw_box(ax, 30, 90, 40, 6, "Input Nodes [21, 3]", color='#95a5a6')
    ax.text(77, 87, "With Adjacency\nMatrix", ha='left', fontsize=9, color='#7f8c8d', style='italic')
    
    # GCN Layers
    draw_arrow(ax, 50, 90, 0, -6)
    draw_box(ax, 25, 78, 50, 7, "GCN Layer 1: [3->64]\nAdj @ X @ W + GELU", color='#2ecc71', fontsize=10)
    draw_arrow(ax, 50, 78, 0, -6)
    draw_box(ax, 25, 65, 50, 7, "GCN Layer 2: [64->128]\nAdj @ X @ W + GELU", color='#2ecc71', fontsize=10)
    draw_arrow(ax, 50, 65, 0, -6)
    draw_box(ax, 25, 52, 50, 7, "GCN Layer 3: [128->256]\nAdj @ X @ W + GELU", color='#2ecc71', fontsize=10)
    draw_arrow(ax, 50, 52, 0, -6)
    draw_box(ax, 35, 40, 30, 6, "Global Mean Pool", color='#27ae60')
    draw_arrow(ax, 50, 40, 0, -6)
    draw_box(ax, 35, 28, 30, 6, "Linear Head [256->10]", color='#3498db')
    draw_arrow(ax, 50, 28, 0, -6)
    draw_box(ax, 40, 16, 20, 6, "Output [10]", color='#e74c3c')
    
    plt.tight_layout()
    plt.savefig('arch_gcn.png', dpi=300, bbox_inches='tight')
    print("Saved arch_gcn.png")

def visualize_boosting(model_name, description, filename):
    fig, ax = setup_plot(f"{model_name} Architecture")
    
    # Input
    draw_box(ax, 25, 90, 50, 6, "Input Features [Coords + Geo]", color='#95a5a6')
    draw_arrow(ax, 50, 90, 0, -8)
    
    # Boosting concept
    ax.text(50, 77, f"Gradient Boosting Framework ({description})", ha='center', 
            fontsize=11, fontweight='bold', color='#2c3e50')
    
    # Sequential trees
    tree_y = 65
    for i in range(3):
        x = 20 + i * 20
        # Triangle for tree
        triangle = patches.Polygon(
            [[x, tree_y], [x-7, tree_y-12], [x+7, tree_y-12]], 
            closed=True, facecolor='#f39c12', alpha=0.8, edgecolor='#2c3e50', linewidth=2
        )
        ax.add_patch(triangle)
        ax.text(x, tree_y-16, f"Tree {i+1}", ha='center', fontsize=9, fontweight='bold')
        
        if i < 2:
            # Arrow to next tree
            ax.annotate('', xy=(x+13, tree_y-6), xytext=(x+7, tree_y-6),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='#34495e'))
    
    # Ellipsis for more trees
    ax.text(67, tree_y-6, "···", fontsize=24, ha='center', color='#7f8c8d')
    
    # Weighted sum
    ax.text(50, 40, "Weighted Sum of Tree Predictions", ha='center', 
            fontsize=10, style='italic', color='#7f8c8d')
    
    # Convergence arrows
    for x in [20, 40, 60]:
        draw_arrow(ax, x, 50, 50-x, -15)
    
    draw_box(ax, 35, 25, 30, 6, "Class Probabilities", color='#3498db')
    draw_arrow(ax, 50, 25, 0, -6)
    draw_box(ax, 40, 13, 20, 6, "Output [10]", color='#e74c3c')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")

if __name__ == "__main__":
    visualize_mlp()
    visualize_hybrid()
    visualize_ensemble()
    
    # Individual ensemble base models
    visualize_resnet1d()
    visualize_transformer_ultimate()
    visualize_gcn()
    visualize_boosting("XGBoost", "Symmetric Tree Growth", 'arch_xgboost.png')
    visualize_boosting("LightGBM", "Leaf-wise Growth", 'arch_lightgbm.png')
    visualize_boosting("CatBoost", "Ordered Boosting", 'arch_catboost.png')
