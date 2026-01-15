import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import copy
import os
import warnings

# 경고 무시
warnings.filterwarnings("ignore")

# 설정
LABEL_ENCODER_PATH = "../train/label_encoder.pkl"
THRESHOLD = 0.0
MODELS_DIR = "models_final"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 모델 정의 (All Classes)
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
        # Fix: Ensure pos_embedding size matches x_seq (handle different sequence lengths if needed, though here fixed)
        x_seq = x_seq + self.pos_embedding[:, : x_seq.size(1), :]
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
# 2. 전처리 함수
# ==========================================
def compute_geometric_features(landmarks):
    if landmarks.ndim == 2:
        landmarks = landmarks[np.newaxis, ...]
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


def extract_features_mlp(landmarks):
    lm_flat = landmarks.reshape(-1, 3)
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
    angles = [
        np.degrees(
            np.arccos(
                np.clip(
                    np.dot(lm_flat[f[0]] - lm_flat[f[1]], lm_flat[f[2]] - lm_flat[f[1]])
                    / (
                        np.linalg.norm(lm_flat[f[0]] - lm_flat[f[1]])
                        * np.linalg.norm(lm_flat[f[2]] - lm_flat[f[1]])
                        + 1e-8
                    ),
                    -1.0,
                    1.0,
                )
            )
        )
        / 180.0
        for f in fingers
    ]
    dists = [np.linalg.norm(lm_flat[4] - lm_flat[t]) for t in [8, 12, 16, 20]]
    return np.concatenate([lm_flat.flatten(), angles, dists])


def preprocess_input(landmarks, model_type):
    landmarks = np.array(landmarks)
    wrist = landmarks[0]
    relative = landmarks - wrist
    max_val = np.max(np.linalg.norm(relative, axis=1))
    normalized = relative / max_val if max_val > 0 else relative

    if model_type == "mlp":
        return torch.FloatTensor(extract_features_mlp(normalized)).unsqueeze(0).to(DEVICE)
    elif model_type in ["hybrid", "ultimate_transformer", "resnet1d", "gcn", "ensemble"]:
        return (
            torch.FloatTensor(normalized).unsqueeze(0).to(DEVICE),
            torch.FloatTensor(compute_geometric_features(normalized)).to(DEVICE),
        )
    return None


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
        preds = {}
        with torch.no_grad():
            for name, _, model in self.dl_models:
                p = F.softmax(model(coords, geo), dim=1).cpu().numpy()
                preds.setdefault(name, []).append(p)

        ml_in = np.concatenate([coords.cpu().numpy().reshape(1, -1), geo.cpu().numpy()], axis=1)
        for name, _, model in self.ml_models:
            preds.setdefault(name, []).append(model.predict_proba(ml_in))

        final_feats = []
        for name in ["res", "trans", "gcn", "xgb", "lgbm", "cat"]:
            if name in preds and preds[name]:
                final_feats.append(np.mean(preds[name], axis=0))
            else:
                final_feats.append(np.zeros((1, self.num_classes)))

        stacking_input = np.concatenate(final_feats, axis=1)
        return torch.FloatTensor(
            self.meta_model.predict_proba(stacking_input)
            if self.meta_model
            else np.mean(final_feats, axis=0)
        ).to(DEVICE)


# ==========================================
# 4. Main Menu & Loop
# ==========================================
def detect_model_type_from_path(path):
    try:
        state = torch.load(path, map_location="cpu")
        keys = list(state.keys())
        keys_str = " ".join(keys)
        if "gc1.weight" in keys:
            return "gcn"
        if "conv.0.weight" in keys_str:
            return "resnet1d"
        if "cls_token" in keys and "fusion_layer.0.weight" in keys:
            return "hybrid"
        if "enc.layers" in keys_str and "emb.weight" in keys:
            return "ultimate_transformer"
        if "transformer" in keys_str or "pos_embedding" in keys:
            return "transformer"
        return "mlp"
    except:
        return "mlp"


def main():
    print("\n=== ASL Real-time Recognition ===")
    print("1) Load Single Model (.pth)")
    print("2) Full Ensemble - Requires 'models_final/'")

    choice = input("Select (1/2, default 2): ").strip()
    if not choice:
        choice = "2"  # Default to Ensemble as user requested "God Mode" implicitly

    ensemble_predictor = None
    model = None
    model_type = "mlp"

    try:
        with open(LABEL_ENCODER_PATH, "rb") as f:
            le = pickle.load(f)
        num_classes = len(le.classes_)
    except Exception as e:
        print(f"Failed to load label encoder: {e}")
        return

    if choice == "2":
        model_type = "ensemble"
        ensemble_predictor = EnsemblePredictor(MODELS_DIR, num_classes)
        print("God Mode Loaded.")
    else:
        path = input("Enter .pth path (default: ../train/transformer.pth): ").strip()
        if not path:
            path = "../train/transformer.pth"
        if not os.path.exists(path):
            print("File not found.")
            return
        model_type = detect_model_type_from_path(path)
        print(f"Detected Type: {model_type}")

        if model_type == "mlp":
            model = MLP(81, num_classes).to(DEVICE)
        elif model_type == "hybrid":
            model = HybridHandModel(num_classes).to(DEVICE)
        elif model_type == "resnet1d":
            model = ResNet1D(num_classes).to(DEVICE)
        elif model_type == "ultimate_transformer":
            model = UltimateTransformerModel(num_classes).to(DEVICE)
        elif model_type == "gcn":
            model = GCNModel(num_classes).to(DEVICE)
        else:  # Generic Transformer
            model = HybridHandModel(num_classes).to(DEVICE)  # Assuming default

        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    cap = cv2.VideoCapture(0)  # Standard Webcam
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)  # Try external

    print("Starting Webcam... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        txt = "Waiting..."
        color = (200, 200, 200)

        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                lm = [[l.x, l.y, l.z] for l in hl.landmark]

                try:
                    inp = preprocess_input(lm, model_type)
                    if model_type == "ensemble":
                        out = ensemble_predictor.predict(inp[0], inp[1])
                    elif model_type == "mlp":
                        out = model(inp)
                    else:
                        out = model(inp[0], inp[1])

                    probs = F.softmax(out, dim=1)
                    max_p, idx = torch.max(probs, 1)
                    conf = max_p.item()

                    if conf > THRESHOLD:
                        label = le.inverse_transform([idx.item()])[0]
                        txt = f"{label + 1} ({conf*100:.0f}%)"
                        color = (0, 255, 0)
                    else:
                        txt = f"Unknown ({conf*100:.0f}%)"
                        color = (0, 0, 255)

                except Exception as e:
                    print(e)

        cv2.putText(frame, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Realtime ASL God Mode", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
