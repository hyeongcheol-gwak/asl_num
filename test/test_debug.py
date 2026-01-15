import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from collections import deque
import os
import warnings

# 경고 무시
warnings.filterwarnings("ignore")

# 설정
INPUT_VIDEO_PATH = "test_video.mp4"
OUTPUT_TXT_PATH = "debug_log.txt"
LABEL_ENCODER_PATH = "../train/label_encoder.pkl"
MODEL_PATH = "../train/mlp.pth"  # 사용할 모델 경로
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
        x_seq = x_seq + self.pos_embedding[:, : x_seq.size(1), :]
        t_out = self.transformer(x_seq)
        t_feature = t_out[:, 0, :]
        g_feature = self.geo_mlp(x_geo)
        combined = torch.cat((t_feature, g_feature), dim=1)
        return self.fusion_layer(combined)


# ==========================================
# 2. 특징 추출 및 전처리 함수
# ==========================================


def compute_geometric_features(landmarks):
    # landmarks: (N, 21, 3) or (21, 3)
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
        base = f * 4
        finger_indices.extend([(base, base + 1), (base + 1, base + 2), (base + 2, base + 3)])

    v1 = vecs[:, [f[0] for f in finger_indices], :]
    v2 = vecs[:, [f[1] for f in finger_indices], :]

    dot = np.sum(v1 * v2, axis=2)
    norm_mul = norms[:, [f[0] for f in finger_indices]] * norms[:, [f[1] for f in finger_indices]]
    angles = np.arccos(np.clip(dot / norm_mul, -1.0, 1.0))

    tips = [4, 8, 12, 16, 20]
    wrist = landmarks[:, 0, :]
    dists = np.linalg.norm(landmarks[:, tips, :] - wrist[:, None, :], axis=2)

    return np.concatenate([angles, dists], axis=1)  # (N, 20)


def extract_features_mlp(landmarks):
    # MLP용 81차원 특징 (normalized coords 63 + angles 14 + dists 4)
    lm_flat = landmarks.reshape(-1, 3)

    # 1. Angles (14)
    fingers = [
        [1, 2, 3], [2, 3, 4], [0, 5, 6], [5, 6, 7], [6, 7, 8],
        [0, 9, 10], [9, 10, 11], [10, 11, 12], [0, 13, 14],
        [13, 14, 15], [14, 15, 16], [0, 17, 18], [17, 18, 19], [18, 19, 20],
    ]
    angles = []
    for f in fingers:
        v1 = lm_flat[f[0]] - lm_flat[f[1]]
        v2 = lm_flat[f[2]] - lm_flat[f[1]]
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        angle = np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0))) / 180.0
        angles.append(angle)

    # 2. Dists (4)
    thumb_tip = lm_flat[4]
    dists = []
    for t in [8, 12, 16, 20]:
        dists.append(np.linalg.norm(thumb_tip - lm_flat[t]))

    return np.concatenate([lm_flat.flatten(), angles, dists])


def preprocess_input(landmarks, model_type):
    landmarks = np.array(landmarks)
    wrist = landmarks[0]

    # Common Normalization
    relative = landmarks - wrist
    max_val = np.max(np.linalg.norm(relative, axis=1))
    if max_val > 0:
        normalized = relative / max_val
    else:
        normalized = relative

    if model_type == "mlp":
        # MLP: 1D array (63 + features)
        features = extract_features_mlp(normalized)
        return torch.FloatTensor(features).unsqueeze(0).to(DEVICE)

    elif model_type == "hybrid":
        # Hybrid Model: (1, 21, 3) coords AND (1, 20) geo
        geo = compute_geometric_features(normalized)  # (1, 20)
        return (
            torch.FloatTensor(normalized).unsqueeze(0).to(DEVICE),
            torch.FloatTensor(geo).to(DEVICE),
        )

    return None


# ==========================================
# 3. Main Logic
# ==========================================


def detect_model_type_from_path(path):
    # Determine model type from file content
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
    print("=== ASL Recognition Test (Video - Debug Mode) ===")

    # Label Encoder Load
    try:
        with open(LABEL_ENCODER_PATH, "rb") as f:
            le = pickle.load(f)
        num_classes = len(le.classes_)
    except Exception as e:
        print(f"Error loading label encoder: {e}")
        return

    # Model Loading
    print(f"Loading Model: '{MODEL_PATH}'")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return

    model_type = detect_model_type_from_path(MODEL_PATH)
    print(f"Detected Model Type: {model_type}")

    if model_type == "mlp":
        model = MLP(81, num_classes).to(DEVICE)
    elif model_type == "hybrid":
        model = HybridHandModel(num_classes).to(DEVICE)
    else:
        print(f"Unsupported model type: {model_type}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully!")

    # Video Processing
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video: {INPUT_VIDEO_PATH}")
        return

    prediction_buffer = deque(maxlen=20)
    output_triggered = False
    last_output_gesture = None

    with open(OUTPUT_TXT_PATH, "w") as f:
        f.write("=== Debug Log ===\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            pred_label = "None"
            confidence = 0.0

            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    lm_list = [[lm.x, lm.y, lm.z] for lm in hl.landmark]

                    try:
                        inp = preprocess_input(lm_list, model_type)

                        if model_type == "mlp":
                            outputs = model(inp)
                        else:  # hybrid
                            outputs = model(inp[0], inp[1])

                        probs = F.softmax(outputs, dim=1)
                        max_prob, idx = torch.max(probs, 1)
                        confidence = max_prob.item()

                        if confidence > THRESHOLD:
                            pred_label = le.inverse_transform([idx.item()])[0]
                            prediction_buffer.append(pred_label + 1)  # Class 0 -> 1
                        else:
                            prediction_buffer.append("None")

                    except Exception as e:
                        print(e)
                        f.write(f"Error: {e}\n")
            else:
                prediction_buffer.append("None")

            # Stability Check
            if len(prediction_buffer) == 20:
                most_common = max(set(prediction_buffer), key=prediction_buffer.count)
                cnt = prediction_buffer.count(most_common)
                
                f.write(f"Buffer: {list(prediction_buffer)} | Most: {most_common} ({cnt}/20) | Conf: {confidence:.3f}\n")
                
                if cnt >= 15 and most_common != "None":
                    if most_common != last_output_gesture:
                        print(f"Recognized: {most_common}")
                        f.write(f">>> RECOGNIZED: {most_common}\n")
                        f.flush()
                        last_output_gesture = most_common
                        output_triggered = True

                elif most_common == "None":
                    output_triggered = False

            cv2.imshow("Debug Test", frame)
            if cv2.waitKey(1) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Debug log saved to: {OUTPUT_TXT_PATH}")


if __name__ == "__main__":
    main()
