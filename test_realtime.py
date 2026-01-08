import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import copy

LABEL_ENCODER_PATH = "label_encoder.pkl"
THRESHOLD = 0.85


def choose_model_path():
    print("\n=== 사용할 모델 선택 ===")
    print("1) mlp.pth        (MLP 기반 모델)")
    print("2) transformer.pth (Hybrid Transformer + MLP 모델)")
    print("3) 직접 경로 입력 (.pth 파일)")

    choice = input("번호를 입력하세요 (기본: 2): ").strip()

    if choice == "1":
        return "mlp.pth"
    elif choice == "3":
        path = input("사용할 .pth 파일 경로를 입력하세요: ").strip()
        return path if path else "transformer.pth"
    else:
        return "transformer.pth"


MODEL_PATH = choose_model_path()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim=3,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        num_classes=10,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Embedding(21, d_model)

        encoder_layer = TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
        self.transformer = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

        self.norm = nn.LayerNorm(d_model)
        self.dropout_final = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        x = self.input_projection(x)

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb

        for layer in self.transformer:
            x = layer(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.dropout_final(x)
        x = self.classifier(x)

        return x


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
        output = self.fusion_layer(combined)

        return output


def detect_model_type(model_path):
    if "transformer" in model_path.lower():
        try:
            state_dict = torch.load(model_path, map_location="cpu")
            if "geo_mlp" in state_dict.keys() or "fusion_layer" in state_dict.keys():
                return "hybrid"
            return "transformer"
        except:
            return "transformer"
    elif "mlp" in model_path.lower():
        return "mlp"
    else:
        try:
            state_dict = torch.load(model_path, map_location="cpu")
            if "geo_mlp" in state_dict.keys() or "fusion_layer" in state_dict.keys():
                return "hybrid"
            for key in state_dict.keys():
                if "transformer" in key or "pos_embedding" in key:
                    return "transformer"
            return "mlp"
        except:
            return "mlp"


try:
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    num_classes = len(le.classes_)
    model_type = detect_model_type(MODEL_PATH)
    print(f"사용할 모델 유형: {model_type}")

    if model_type == "hybrid":
        model = HybridHandModel(
            num_classes=num_classes, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1
        ).to(device)
        print(f"Hybrid 모델 로드: {MODEL_PATH}")
    elif model_type == "transformer":
        model = TransformerModel(
            input_dim=3,
            d_model=64,
            nhead=4,
            num_layers=3,
            dim_feedforward=128,
            num_classes=num_classes,
            dropout=0.1,
        ).to(device)
        print(f"Transformer 모델 로드: {MODEL_PATH}")
    else:
        model = MLP(81, num_classes).to(device)
        print(f"MLP 모델 로드: {MODEL_PATH}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"모델 로드 성공: {MODEL_PATH}")
except FileNotFoundError:
    print("오류: 모델 파일이나 라벨 인코더를 찾을 수 없습니다.")
    exit()
except Exception as e:
    print(f"모델 로드 오류: {e}")
    exit()


def get_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta)) / 180.0


def extract_features(landmarks):
    features = []
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
    for f in fingers:
        v1 = landmarks[f[0]] - landmarks[f[1]]
        v2 = landmarks[f[2]] - landmarks[f[1]]
        features.append(get_angle(v1, v2))

    thumb_tip = landmarks[4]
    for tip_idx in [8, 12, 16, 20]:
        features.append(np.linalg.norm(thumb_tip - landmarks[tip_idx]))

    return np.array(features)


def compute_geometric_features(landmarks):
    connections = np.array([
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16],
        [0, 17], [17, 18], [18, 19], [19, 20],
    ])

    vectors = landmarks[connections[:, 1]] - landmarks[connections[:, 0]]
    norms = np.linalg.norm(vectors, axis=1) + 1e-8

    finger_indices = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
    ])

    angles = []
    for f_idx in finger_indices:
        for i in range(3):
            v1 = vectors[f_idx[i]]
            v2 = vectors[f_idx[i + 1]]

            dot_product = np.dot(v1, v2)
            norm_mult = norms[f_idx[i]] * norms[f_idx[i + 1]]

            cos_angle = np.clip(dot_product / norm_mult, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angles.append(angle)

    fingertips = np.array([4, 8, 12, 16, 20])
    wrist = landmarks[0]
    dists = np.linalg.norm(landmarks[fingertips] - wrist, axis=1)

    features = np.concatenate([np.array(angles), dists])
    return features


def preprocess_input_mlp(landmarks):
    landmarks = np.array(landmarks)
    wrist = landmarks[0, :]
    relative = landmarks - wrist
    max_val = np.max(np.abs(relative))
    normalized = relative / max_val if max_val > 0 else relative

    features = extract_features(normalized)
    combined = np.concatenate([normalized.flatten(), features])

    return combined.reshape(1, -1)


def preprocess_input_transformer(landmarks):
    landmarks = np.array(landmarks)
    wrist = landmarks[0, :]
    relative_landmarks = landmarks - wrist

    max_dist = np.max(np.linalg.norm(relative_landmarks, axis=1))
    if max_dist > 0:
        normalized_landmarks = relative_landmarks / max_dist
    else:
        normalized_landmarks = relative_landmarks

    return normalized_landmarks.reshape(1, 21, 3)


def preprocess_input_hybrid(landmarks):
    landmarks = np.array(landmarks)
    wrist = landmarks[0, :]
    relative_landmarks = landmarks - wrist

    max_dist = np.max(np.linalg.norm(relative_landmarks, axis=1))
    if max_dist > 0:
        normalized_landmarks = relative_landmarks / max_dist
    else:
        normalized_landmarks = relative_landmarks

    geo_features = compute_geometric_features(relative_landmarks)
    geo_features[15:] = geo_features[15:] / (max_dist + 1e-8)

    return normalized_landmarks.reshape(1, 21, 3), geo_features.reshape(1, -1)


def predict_gesture(processed_input, model, le, threshold=0.85, model_type="mlp"):
    model.eval()
    with torch.no_grad():
        if model_type == "hybrid":
            coords_tensor = torch.FloatTensor(processed_input[0]).to(device)
            geo_tensor = torch.FloatTensor(processed_input[1]).to(device)
            outputs = model(coords_tensor, geo_tensor)
        else:
            input_tensor = torch.FloatTensor(processed_input).to(device)
            outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        max_prob, predicted_index = torch.max(probs, 1)
        max_prob = max_prob.item()
        predicted_index = predicted_index.item()

    if max_prob < threshold:
        return "Unknown", max_prob

    label = le.inverse_transform([predicted_index])[0]
    return label, max_prob


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(1)

print("\n--- 실시간 인식 시작 (종료: 'q') ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    display_text = "Waiting..."
    color = (200, 200, 200)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmark_list = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

            try:
                if model_type == "hybrid":
                    input_data = preprocess_input_hybrid(landmark_list)
                elif model_type == "transformer":
                    input_data = preprocess_input_transformer(landmark_list)
                else:
                    input_data = preprocess_input_mlp(landmark_list)

                label, conf = predict_gesture(
                    input_data, model, le, threshold=THRESHOLD, model_type=model_type
                )

                if label == "Unknown":
                    display_text = f"Unknown ({conf*100:.1f}%)"
                    color = (0, 0, 255)
                else:
                    display_text = f"{label} ({conf*100:.1f}%)"
                    color = (0, 255, 0)

            except Exception as e:
                print(f"Error: {e}")

    cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
