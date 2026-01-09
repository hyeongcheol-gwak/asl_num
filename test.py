import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import copy
from collections import deque
import os
import warnings

# 경고 무시
warnings.filterwarnings("ignore")

# 설정
INPUT_VIDEO_PATH = "test_video.mp4"
OUTPUT_TXT_PATH = "result.txt"
LABEL_ENCODER_PATH = "label_encoder.pkl"
THRESHOLD = 0.0
MODELS_DIR = "models_final"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 실행 모드 설정
USE_ENSEMBLE = False  # True: 앙상블 모드 (Full Stacking), False: 단일 모델 모드
SINGLE_MODEL_PATH = "transformer.pth"  # 단일 모델 모드일 때 사용할 모델 경로

# ==========================================
# 1. 모델 정의 (All Classes)
# ==========================================


# 1.1 MLP (from train.py)
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


# 1.2 HybridHandModel (from train_transformer.py)
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


# 1.3 ResNet1D (from train_ultimate.py)
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


# 1.4 UltimateTransformer (from train_ultimate.py)
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


# 1.5 GCNModel (from train_ultimate.py)
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
    D = np.sum(A, axis=0)
    D_mat = np.diag(np.power(D, -0.5))
    return torch.tensor(D_mat @ A @ D_mat, dtype=torch.float32).to(DEVICE)


ADJ_MATRIX = get_adj()


class GCNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.gc1 = nn.Linear(3, 64)
        self.gc2 = nn.Linear(64, 128)
        self.gc3 = nn.Linear(128, 256)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x, geo):  # geo not used
        adj = ADJ_MATRIX
        x = F.gelu(torch.matmul(adj, self.gc1(x)))
        x = F.gelu(torch.matmul(adj, self.gc2(x)))
        x = F.gelu(torch.matmul(adj, self.gc3(x))).mean(dim=1)
        return self.head(x)


# ==========================================
# 2. 특징 추출 및 전처리 함수
# ==========================================


def compute_geometric_features(landmarks):
    # landmarks: (N, 21, 3) or (21, 3)
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
    # 기존 test.py 로직 유지
    lm_flat = landmarks.reshape(-1, 3)

    # 1. Angles (14)
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
    angles = []
    for f in fingers:
        v1 = lm_flat[f[0]] - lm_flat[f[1]]
        v2 = lm_flat[f[2]] - lm_flat[f[1]]
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        angle = np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0))) / 180.0
        angles.append(angle)

    # 2. Dists (4) - excluding thumb? Original MLP code had 4.
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

    elif model_type in ["hybrid", "ultimate_transformer", "resnet1d", "gcn", "ensemble"]:
        # Deep Learning Models: (1, 21, 3) coords AND (1, 20) geo
        geo = compute_geometric_features(normalized)  # (1, 20)

        # For ML models in ensemble, we need (1, 83) = flattened(63) + geo(20)
        # But we handle that inside the ensemble predictor.

        return (
            torch.FloatTensor(normalized).unsqueeze(0).to(DEVICE),
            torch.FloatTensor(geo).to(DEVICE),
        )

    return None


# ==========================================
# 3. Ensemble Helper
# ==========================================


class EnsemblePredictor:
    def __init__(self, models_dir="models_final", num_classes=10):
        self.models_dir = models_dir
        self.num_classes = num_classes
        self.dl_models = []  # (name, fold, model_instance)
        self.ml_models = []  # (name, fold, model_instance)
        self.meta_model = None

        self.load_models()

    def load_models(self):
        print("Loading Ensemble Models...")

        # 1. Deep Learning Models
        dl_classes = {"res": ResNet1D, "trans": UltimateTransformerModel, "gcn": GCNModel}

        for name, cls in dl_classes.items():
            for fold in range(5):
                path = os.path.join(self.models_dir, f"{name}_fold{fold}.pth")
                if os.path.exists(path):
                    model = cls(self.num_classes).to(DEVICE)
                    model.load_state_dict(torch.load(path, map_location=DEVICE))
                    model.eval()
                    self.dl_models.append((name, fold, model))
                    # print(f"Loaded {name} fold {fold}")

        # 2. Machine Learning Models
        try:
            import xgboost as xgb
            import lightgbm as lgb
            import catboost as cb
        except ImportError:
            print(
                "Warning: ML libraries (xgboost, lightgbm, catboost) not found. ML models will be skipped."
            )

        ml_names = ["xgb", "lgbm", "cat"]
        for name in ml_names:
            for fold in range(5):
                path = os.path.join(self.models_dir, f"{name}_fold{fold}.pkl")
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        model = pickle.load(f)
                    self.ml_models.append((name, fold, model))
                    # print(f"Loaded {name} fold {fold}")

        # 3. Meta Model
        meta_path = os.path.join(self.models_dir, "meta_model.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                self.meta_model = pickle.load(f)
            print("Loaded Meta Model")
        else:
            print("Warning: Meta Model not found!")

        print(f"Total DL Models: {len(self.dl_models)}, ML Models: {len(self.ml_models)}")

    def predict(self, coords, geo):
        # coords: Tensor (1, 21, 3)
        # geo: Tensor (1, 20)

        preds = {}  # key: model_name, value: list of probs

        # DL Inference
        with torch.no_grad():
            for name, _, model in self.dl_models:
                out = model(coords, geo)  # GCN ignores geo internally
                prob = F.softmax(out, dim=1).cpu().numpy()
                if name not in preds:
                    preds[name] = []
                preds[name].append(prob)

        # ML Inference
        # Prepare ML Input: (1, 83) -> Flatten coords + geo
        coords_np = coords.cpu().numpy().reshape(1, -1)  # (1, 63)
        geo_np = geo.cpu().numpy()  # (1, 20)
        ml_input = np.concatenate([coords_np, geo_np], axis=1)  # (1, 83)

        for name, _, model in self.ml_models:
            prob = model.predict_proba(ml_input)
            if name not in preds:
                preds[name] = []
            preds[name].append(prob)

        # Aggregate (Average folds)
        final_feats = []
        # Order must match train_ultimate.py: res, trans, gcn, xgb, lgbm, cat
        order = ["res", "trans", "gcn", "xgb", "lgbm", "cat"]

        for name in order:
            if name in preds and len(preds[name]) > 0:
                avg_prob = np.mean(preds[name], axis=0)  # (1, num_classes)
                final_feats.append(avg_prob)
            else:
                # If model missing, substitute with zeros (risky but handles missing files)
                final_feats.append(np.zeros((1, self.num_classes)))

        stacking_input = np.concatenate(final_feats, axis=1)  # (1, num_classes * 6)

        if self.meta_model:
            final_prob = self.meta_model.predict_proba(stacking_input)
            return torch.FloatTensor(final_prob).to(DEVICE)
        else:
            # Fallback: simple average of all available models
            return torch.FloatTensor(np.mean(final_feats, axis=0)).to(DEVICE)


# ==========================================
# 4. Main Logic
# ==========================================


def detect_model_type_from_path(path):
    # Determine model type from file content
    try:
        state = torch.load(path, map_location="cpu")
        keys = list(state.keys())
        keys_str = " ".join(keys)

        if "gc1.weight" in keys:
            return "gcn"
        if "conv.0.weight" in keys_str:
            return "resnet1d"  # ResNet keys usually have conv
        if "cls_token" in keys and "fusion_layer.0.weight" in keys:
            return "hybrid"
        if "enc.layers" in keys_str and "emb.weight" in keys:
            return "ultimate_transformer"
        if "transformer" in keys_str or "pos_embedding" in keys:
            return "transformer"  # Generic/Hybrid fallback
        return "mlp"
    except:
        return "mlp"


def main():
    print("=== ASL Recognition Test (Video) ===")

    ensemble_predictor = None
    model = None
    model_type = "mlp"

    # Label Encoder Load
    try:
        with open(LABEL_ENCODER_PATH, "rb") as f:
            le = pickle.load(f)
        num_classes = len(le.classes_)
    except Exception as e:
        print(f"Error loading label encoder: {e}")
        return

    if USE_ENSEMBLE:
        print(f"Mode: Full Ensemble (God Mode) using '{MODELS_DIR}'")
        model_type = "ensemble"
        ensemble_predictor = EnsemblePredictor(MODELS_DIR, num_classes)
        print("Ensemble Initialized.")
    else:
        path = SINGLE_MODEL_PATH
        print(f"Mode: Single Model using '{path}'")

        if not os.path.exists(path):
            print(f"Model file not found: {path}")
            return

        model_type = detect_model_type_from_path(path)
        print(f"Detected Model Type: {model_type}")

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

        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()

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
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # frame = cv2.resize(frame, (640, 480))  # Optional
            # frame = cv2.flip(
            #    frame, 1
            # )  # 좌우 반전 (필수: 학습 데이터 및 실시간 테스트와 동일하게 맞춤)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            pred_label = "None"

            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    lm_list = [[lm.x, lm.y, lm.z] for lm in hl.landmark]

                    try:
                        inp = preprocess_input(lm_list, model_type)

                        if model_type == "ensemble":
                            # inp is (coords, geo)
                            outputs = ensemble_predictor.predict(inp[0], inp[1])
                        elif model_type == "mlp":
                            outputs = model(inp)
                        else:
                            # DL models taking x, geo
                            outputs = model(inp[0], inp[1])

                        probs = F.softmax(outputs, dim=1)
                        max_prob, idx = torch.max(probs, 1)

                        if max_prob.item() > THRESHOLD:
                            pred_label = le.inverse_transform([idx.item()])[0]
                            prediction_buffer.append(pred_label + 1)  # Class 0 -> 1
                        else:
                            prediction_buffer.append("None")

                    except Exception as e:
                        print(e)
            else:
                prediction_buffer.append("None")

            # Stability Check
            if len(prediction_buffer) == 20:
                most_common = max(set(prediction_buffer), key=prediction_buffer.count)
                cnt = prediction_buffer.count(most_common)
                # 안정적인 제스처 감지 (20프레임 중 15프레임 이상)
                if cnt >= 15 and most_common != "None":
                    # 이전과 다른 제스처일 때만 출력 (연속된 같은 제스처 방지)
                    if most_common != last_output_gesture:
                        print(f"Recognized: {most_common}")
                        f.write(f"{most_common}\n")
                        f.flush()
                        last_output_gesture = most_common
                        output_triggered = True
                    else:
                        # 같은 제스처가 지속되거나, None 이후 다시 같은 제스처가 들어온 경우
                        # 출력은 하지 않지만 상태는 유지
                        output_triggered = True

                else:
                    # 제스처가 불안정하거나 손이 사라짐
                    if most_common != "None":
                        print(f"Debug - Candidate: {most_common} (Count: {cnt}/20)")

                    if most_common == "None":
                        output_triggered = False
                        # last_output_gesture는 초기화하지 않음

            cv2.imshow("Test", frame)
            if cv2.waitKey(1) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
