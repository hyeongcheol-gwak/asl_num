import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import pickle
import warnings

warnings.filterwarnings("ignore")

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
SEED = 76


# Set seed
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


seed_everything(SEED)

# ==========================================
# 1. Model Definitions
# ==========================================


# 1.1 MLP
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.model(x)


# 1.2 HybridHandModel
class HybridHandModel(nn.Module):
    def __init__(
        self, num_classes=10, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1
    ):
        super().__init__()
        self.input_projection = nn.Linear(3, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 22, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.geo_mlp = nn.Sequential(
            nn.Linear(20, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model + 32, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_coords, x_geo):
        batch_size = x_coords.size(0)
        x_emb = self.input_projection(x_coords)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat((cls_tokens, x_emb), dim=1)
        # Handle pos embedding size matching
        seq_len = x_seq.size(1)
        x_seq = x_seq + self.pos_embedding[:, :seq_len, :]
        t_out = self.transformer(x_seq)
        t_feature = t_out[:, 0, :]
        g_feature = self.geo_mlp(x_geo)
        return self.fusion_layer(torch.cat((t_feature, g_feature), dim=1))


# 1.3 ResNet1D
class ResNet1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, 3, 2, 1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, 3, 2, 1),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )
        self.geo_fc = nn.Sequential(nn.Linear(20, 64), nn.GELU())
        self.head = nn.Linear(256 + 64, num_classes)

    def forward(self, x, geo):
        f = self.conv(x.permute(0, 2, 1)).mean(dim=2)
        g = self.geo_fc(geo)
        return self.head(torch.cat([f, g], dim=1))


# 1.4 UltimateTransformer
class UltimateTransformerModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.emb = nn.Linear(3, 128)
        self.pos = nn.Parameter(torch.randn(1, 21, 128) * 0.02)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(128, 8, 256, 0.2, "gelu", batch_first=True), 4
        )
        self.geo_fc = nn.Sequential(nn.Linear(20, 64), nn.GELU())
        self.head = nn.Linear(128 + 64, num_classes)

    def forward(self, x, geo):
        x = self.enc(self.emb(x) + self.pos).mean(dim=1)
        g = self.geo_fc(geo)
        return self.head(torch.cat([x, g], dim=1))


# 1.5 GCNModel
def get_adj():
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ]
    A = np.eye(21, dtype=np.float32)
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    return torch.tensor(
        np.diag(np.power(np.sum(A, axis=0), -0.5)) @ A @ np.diag(np.power(np.sum(A, axis=0), -0.5)),
        dtype=torch.float32,
    ).to(DEVICE)


ADJ_MATRIX = get_adj()


class GCNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.gc1 = nn.Linear(3, 64)
        self.gc2 = nn.Linear(64, 128)
        self.gc3 = nn.Linear(128, 256)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x, geo):
        adj = ADJ_MATRIX
        x = F.gelu(torch.matmul(adj, self.gc1(x)))
        x = F.gelu(torch.matmul(adj, self.gc2(x)))
        x = F.gelu(torch.matmul(adj, self.gc3(x))).mean(dim=1)
        return self.head(x)


# ==========================================
# 2. Preprocessing Functions
# ==========================================
def extract_features_mlp(landmarks):
    # landmarks: (N, 21, 3)
    N = landmarks.shape[0]
    lm_flat = landmarks.reshape(N, -1)  # (N, 63)

    # Calculate angles
    fingers = [
        [1, 2, 3],
        [2, 3, 4],
        [0, 5, 6],
        [5, 6, 7],
        [6, 7, 8],
        [0, 9, 10],
        [9, 10, 11],
        [10, 11, 12],
        [0, 13, 14],
        [13, 14, 15],
        [14, 15, 16],
        [0, 17, 18],
        [17, 18, 19],
        [18, 19, 20],
    ]
    angles_list = []
    for f in fingers:
        v1 = landmarks[:, f[0]] - landmarks[:, f[1]]
        v2 = landmarks[:, f[2]] - landmarks[:, f[1]]

        norm1 = np.linalg.norm(v1, axis=1)
        norm2 = np.linalg.norm(v2, axis=1)

        dot = np.sum(v1 * v2, axis=1)
        angle = np.degrees(np.arccos(np.clip(dot / (norm1 * norm2 + 1e-8), -1.0, 1.0))) / 180.0
        angles_list.append(angle[:, np.newaxis])

    angles = np.concatenate(angles_list, axis=1)  # (N, 14)

    # Distances
    dists_list = []
    for t in [8, 12, 16, 20]:
        d = np.linalg.norm(landmarks[:, 4] - landmarks[:, t], axis=1)
        dists_list.append(d[:, np.newaxis])
    dists = np.concatenate(dists_list, axis=1)  # (N, 4)

    return np.concatenate([lm_flat, angles, dists], axis=1)


def compute_geometric_features(landmarks):
    # landmarks: (N, 21, 3)
    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ]
    vecs = (
        landmarks[:, [c[1] for c in connections], :] - landmarks[:, [c[0] for c in connections], :]
    )
    norms = np.linalg.norm(vecs, axis=2) + 1e-8

    finger_indices = []
    for f in range(5):
        finger_indices.extend([(f * 4, f * 4 + 1), (f * 4 + 1, f * 4 + 2), (f * 4 + 2, f * 4 + 3)])

    v1 = vecs[:, [f[0] for f in finger_indices], :]
    v2 = vecs[:, [f[1] for f in finger_indices], :]

    dot = np.sum(v1 * v2, axis=2)
    norm_mul = norms[:, [f[0] for f in finger_indices]] * norms[:, [f[1] for f in finger_indices]]
    angles = np.arccos(np.clip(dot / norm_mul, -1.0, 1.0))

    dists = np.linalg.norm(landmarks[:, [4, 8, 12, 16, 20], :] - landmarks[:, 0, None, :], axis=2)
    return np.concatenate([angles, dists], axis=1)


# ==========================================
# 3. Ensemble Helper
# ==========================================
class EnsemblePredictor:
    def __init__(self, models_dir="models_final", num_classes=10):
        self.models_dir = models_dir
        self.num_classes = num_classes
        self.dl_models = []
        self.ml_models = []
        self.meta_model = None
        self.load_models()

    def load_models(self):
        print("Loading Ensemble Models...")
        for name, cls in {
            "res": ResNet1D,
            "trans": UltimateTransformerModel,
            "gcn": GCNModel,
        }.items():
            for fold in range(5):
                path = os.path.join(self.models_dir, f"{name}_fold{fold}.pth")
                if os.path.exists(path):
                    model = cls(self.num_classes).to(DEVICE)
                    model.load_state_dict(torch.load(path, map_location=DEVICE))
                    model.eval()
                    self.dl_models.append((name, fold, model))

        try:
            import xgboost, lightgbm, catboost

            ml_names = ["xgb", "lgbm", "cat"]
            for name in ml_names:
                for fold in range(5):
                    path = os.path.join(self.models_dir, f"{name}_fold{fold}.pkl")
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            self.ml_models.append((name, fold, pickle.load(f)))
        except ImportError:
            print("Warning: ML libraries not found.")

        meta_path = os.path.join(self.models_dir, "meta_model.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                self.meta_model = pickle.load(f)

    def predict(self, coords, geo):
        # coords: Tensor (B, 21, 3)
        # geo: Tensor (B, 20)
        batch_size = coords.size(0)

        preds = {}
        with torch.no_grad():
            for name, _, model in self.dl_models:
                p = F.softmax(model(coords, geo), dim=1).cpu().numpy()
                preds.setdefault(name, []).append(p)

        ml_in = np.concatenate(
            [coords.cpu().numpy().reshape(batch_size, -1), geo.cpu().numpy()], axis=1
        )
        for name, _, model in self.ml_models:
            # ML models usually don't support batch predict nicely if passed purely as object but here they should
            # Sklearn style predict_proba
            preds.setdefault(name, []).append(model.predict_proba(ml_in))

        final_feats = []
        # Average across folds for each model type
        for name in ["res", "trans", "gcn", "xgb", "lgbm", "cat"]:
            if name in preds and preds[name]:
                # Mean across folds (list of (B, Class))
                # Stack folds: (Folds, B, Class) -> Mean(0) -> (B, Class)
                final_feats.append(np.mean(np.array(preds[name]), axis=0))
            else:
                final_feats.append(np.zeros((batch_size, self.num_classes)))

        stacking_input = np.concatenate(final_feats, axis=1)  # (B, 6*Class)

        if self.meta_model:
            return self.meta_model.predict(
                stacking_input
            )  # Returns labels directly if predict, predict_proba for probs
        else:
            return np.argmax(np.mean(final_feats, axis=0), axis=1)


# ==========================================
# 4. Main Evaluation Logic
# ==========================================
def evaluate_model(model, X_test, y_test, model_type="mlp", batch_size=256):
    model.eval() if hasattr(model, "eval") else None

    preds = []

    # Simple Batch Loop
    num_samples = len(X_test)
    for i in range(0, num_samples, batch_size):
        X_batch = X_test[i : i + batch_size]

        # Preprocess
        # Normalize
        wrist = X_batch[:, 0]
        X_batch_rel = X_batch - wrist[:, None]
        max_val = np.max(np.linalg.norm(X_batch_rel, axis=2), axis=1, keepdims=True)
        # Fix broadcasting: (B, 1) -> (B, 1, 1)
        max_val = max_val[..., np.newaxis]
        X_norm = X_batch_rel / (max_val + 1e-8)

        if model_type == "mlp":
            inp = extract_features_mlp(X_norm)
            inp_tensor = torch.FloatTensor(inp).to(DEVICE)
            with torch.no_grad():
                out = model(inp_tensor)
                p = out.argmax(dim=1).cpu().numpy()
                preds.extend(p)

        elif model_type == "hybrid":
            geo = compute_geometric_features(X_norm)
            x_t = torch.FloatTensor(X_norm).to(DEVICE)
            g_t = torch.FloatTensor(geo).to(DEVICE)
            with torch.no_grad():
                out = model(x_t, g_t)
                p = out.argmax(dim=1).cpu().numpy()
                preds.extend(p)

        elif model_type == "ensemble":
            # Ensemble class handles its own batching/predict logic if we wrote it well,
            # but our predict() takes tensors. Ideally we used the predict method which calls models.
            # But the predict method I modified above returns LABELS (argmax).
            # The EnsemblePredictor I copied from test_realtime.py uses batch processing?
            # No, test_realtime.py processes 1 frame.
            # I updated predict() above to handle batches.
            geo = compute_geometric_features(X_norm)
            x_t = torch.FloatTensor(X_norm).to(DEVICE)
            g_t = torch.FloatTensor(geo).to(DEVICE)

            p = model.predict(x_t, g_t)  # Returns numpy array of labels
            preds.extend(p)

    return np.mean(np.array(preds) == y_test)


def main():
    print("Loading Data...")
    df = pd.read_csv("../data/dataset.csv")
    X = df.iloc[:, 1:].values.astype(np.float32).reshape(-1, 21, 3)
    y = df.iloc[:, 0].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    NUM_CLASSES = len(le.classes_)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.1, stratify=y_enc, random_state=SEED
    )
    print(f"Test Set Size: {len(X_test)}")

    results = {}

    # 1. MLP
    if os.path.exists("../train/mlp.pth"):
        print("Evaluating MLP...")
        # Check input dim for MLP. 63 + 14 + 4 = 81
        mlp_model = MLP(81, NUM_CLASSES).to(DEVICE)
        mlp_model.load_state_dict(torch.load("../train/mlp.pth", map_location=DEVICE))
        acc = evaluate_model(mlp_model, X_test, y_test, "mlp")
        results["MLP"] = acc
        print(f"MLP Acc: {acc*100:.2f}%")

    # 2. Hybrid (Transformer)
    if os.path.exists("../train/transformer.pth"):
        print("Evaluating Hybrid...")
        hybrid_model = HybridHandModel(NUM_CLASSES).to(DEVICE)
        hybrid_model.load_state_dict(torch.load("../train/transformer.pth", map_location=DEVICE))
        acc = evaluate_model(hybrid_model, X_test, y_test, "hybrid")
        results["Hybrid"] = acc
        print(f"Hybrid Acc: {acc*100:.2f}%")

    # Plotting
    print("\nGenerating Graph...")
    if not results:
        print("No models evaluated.")
        return

    plt.figure(figsize=(10, 6))

    # Rename keys for display if needed
    display_names = {
        "MLP": "MLP (Baseline)",
        "Hybrid": "Hybrid (Transformer + Geometric MLP)",
    }

    names = [display_names.get(k, k) for k in results.keys()]
    values = [v * 100 for v in results.values()]
    colors = ["#bdc3c7", "#3498db"]
    # Match colors to keys if possible, but safe to just list them.
    # If fewer models, slice colors.
    colors = colors[: len(results)]

    bars = plt.bar(names, values, color=colors, alpha=0.9, width=0.6)

    min_val = min(values)
    plt.ylim(max(0, min_val - 5), 100.5)

    plt.ylabel("Validation Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title("Model Validation Accuracy", fontsize=16, fontweight="bold", pad=20)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    output_path = "validation_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Graph saved to {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
