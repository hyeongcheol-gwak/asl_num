import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import warnings

# 경고 무시
warnings.filterwarnings("ignore")

# 설정
LABEL_ENCODER_PATH = "../train/label_encoder.pkl"
MODEL_PATH = "../train/transformer.pth"  # 기본 모델 경로
THRESHOLD = 0.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 모델 정의
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
        # Fix: Ensure pos_embedding size matches x_seq
        x_seq = x_seq + self.pos_embedding[:, : x_seq.size(1), :]
        t_out = self.transformer(x_seq)
        t_feature = t_out[:, 0, :]
        g_feature = self.geo_mlp(x_geo)
        return self.fusion_layer(torch.cat((t_feature, g_feature), dim=1))


# ==========================================
# 2. 전처리 함수
# ==========================================
def compute_geometric_features(landmarks):
    if landmarks.ndim == 2:
        landmarks = landmarks[np.newaxis, ...]
    connections = [
        (0, 1),(1, 2),(2, 3),(3, 4),(0, 5),(5, 6),(6, 7),(7, 8),
        (0, 9),(9, 10),(10, 11),(11, 12),(0, 13),(13, 14),(14, 15),(15, 16),
        (0, 17),(17, 18),(18, 19),(19, 20),
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
        [1, 2, 3], [2, 3, 4], [0, 5, 6], [5, 6, 7], [6, 7, 8],
        [0, 9, 10], [9, 10, 11], [10, 11, 12], [0, 13, 14],
        [13, 14, 15], [14, 15, 16], [0, 17, 18], [17, 18, 19], [18, 19, 20],
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
    elif model_type == "hybrid":
        return (
            torch.FloatTensor(normalized).unsqueeze(0).to(DEVICE),
            torch.FloatTensor(compute_geometric_features(normalized)).to(DEVICE),
        )
    return None


# ==========================================
# 3. Main
# ==========================================
def detect_model_type_from_path(path):
    try:
        state = torch.load(path, map_location="cpu")
        keys = list(state.keys())
        keys_str = " ".join(keys)
        if "cls_token" in keys and "fusion_layer.0.weight" in keys:
            return "hybrid"
        if "transformer" in keys_str or "pos_embedding" in keys:
            return "hybrid"
        return "mlp"
    except:
        return "mlp"


def main():
    print("\n=== ASL Real-time Recognition ===")
    
    # 모델 경로 입력
    path = input(f"Enter .pth path (default: {MODEL_PATH}): ").strip()
    if not path:
        path = MODEL_PATH
        
    if not os.path.exists(path):
        print("File not found.")
        return
        
    # Load label encoder
    try:
        with open(LABEL_ENCODER_PATH, "rb") as f:
            le = pickle.load(f)
        num_classes = len(le.classes_)
    except Exception as e:
        print(f"Failed to load label encoder: {e}")
        return

    # Load model
    model_type = detect_model_type_from_path(path)
    print(f"Detected Type: {model_type}")

    if model_type == "mlp":
        model = MLP(81, num_classes).to(DEVICE)
    elif model_type == "hybrid":
        model = HybridHandModel(num_classes).to(DEVICE)
    else:
        print(f"Unsupported model type: {model_type}")
        return

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully!")

    # Mediapipe setup
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
                    if model_type == "mlp":
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
        cv2.imshow("Realtime ASL Recognition", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
