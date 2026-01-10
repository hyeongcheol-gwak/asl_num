import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os
import random
import warnings

# === [외부 라이브러리 체크] ===
try:
    import optuna
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
except ImportError:
    print("필수 라이브러리가 없습니다. 설치해주세요: pip install optuna xgboost lightgbm catboost")
    exit()

# 경고 무시
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# === [설정: GOD MODE] ===
CONFIG = {
    "SEED": 42,
    "N_FOLDS": 5,  # 5-Fold 교차검증
    "EPOCHS": 100,  # 딥러닝 에폭
    "BATCH_SIZE": 64,
    "DL_LR": 1e-3,  # 딥러닝 학습률
    "PATIENCE": 15,  # Early Stopping
    "N_TRIALS": 15,  # Optuna 시도 횟수 (시간이 많다면 50 이상 추천)
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CONFIG["SEED"])
print(f"ENSEMBLE MODE ACTIVATED on {CONFIG['DEVICE']}")


# ==========================================
# 1. 데이터 로드 및 기하학적 특징 추출 (Feature Engineering)
# ==========================================
def compute_geometric_features(landmarks):
    # (B, 21, 3) 입력 -> (B, 35) 특징 벡터 출력 (각도 15개 + 거리 5개 + ...)
    # 벡터화 연산을 위해 차원 유지
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

    # 각도 계산 (손가락 관절)
    finger_indices = []
    for f in range(5):
        base = f * 4
        finger_indices.extend([(base, base + 1), (base + 1, base + 2), (base + 2, base + 3)])

    v1 = vecs[:, [f[0] for f in finger_indices], :]
    v2 = vecs[:, [f[1] for f in finger_indices], :]

    dot = np.sum(v1 * v2, axis=2)
    norm_mul = norms[:, [f[0] for f in finger_indices]] * norms[:, [f[1] for f in finger_indices]]
    angles = np.arccos(np.clip(dot / norm_mul, -1.0, 1.0))

    # 손끝-손목 거리
    tips = [4, 8, 12, 16, 20]
    wrist = landmarks[:, 0, :]
    dists = np.linalg.norm(landmarks[:, tips, :] - wrist[:, None, :], axis=2)

    return np.concatenate([angles, dists], axis=1)


def load_data(path="dataset.csv"):
    df = pd.read_csv(path)
    X = df.iloc[:, 1:].values.astype(np.float32)
    y = df.iloc[:, 0].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, y_enc, le


# ==========================================
# 2. SAM Optimizer (구글 SOTA 최적화 기법)
# ==========================================
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    p.grad.norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none)


# ==========================================
# 3. 딥러닝 모델 정의 (Transformer, GCN, ResNet1D)
# ==========================================
# GCN용 인접 행렬
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
    return torch.tensor(D_mat @ A @ D_mat, dtype=torch.float32).to(CONFIG["DEVICE"])


ADJ_MATRIX = get_adj()


class HandDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = torch.LongTensor(y)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        lm = self.X[idx].reshape(21, 3).copy()
        if self.augment:
            theta = np.radians(np.random.uniform(-15, 15, 3))
            R = (
                np.array(
                    [
                        [1, 0, 0],
                        [0, np.cos(theta[0]), -np.sin(theta[0])],
                        [0, np.sin(theta[0]), np.cos(theta[0])],
                    ]
                )
                @ np.array(
                    [
                        [np.cos(theta[1]), 0, np.sin(theta[1])],
                        [0, 1, 0],
                        [-np.sin(theta[1]), 0, np.cos(theta[1])],
                    ]
                )
                @ np.array(
                    [
                        [np.cos(theta[2]), -np.sin(theta[2]), 0],
                        [np.sin(theta[2]), np.cos(theta[2]), 0],
                        [0, 0, 1],
                    ]
                )
            )
            lm = lm @ R * np.random.uniform(0.9, 1.1) + np.random.normal(0, 0.002, lm.shape)

        # 기하학 특징 추출
        lm_norm = lm - lm[0]  # 손목 원점
        lm_norm /= np.max(np.linalg.norm(lm_norm, axis=1)) + 1e-6
        geo = compute_geometric_features(lm_norm[np.newaxis, ...])[0]

        return torch.FloatTensor(lm_norm), torch.FloatTensor(geo), self.y[idx]


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


class TransformerModel(nn.Module):
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


class GCNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.gc1 = nn.Linear(3, 64)
        self.gc2 = nn.Linear(64, 128)
        self.gc3 = nn.Linear(128, 256)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x, geo):  # geo 안씀
        adj = ADJ_MATRIX
        x = F.gelu(torch.matmul(adj, self.gc1(x)))
        x = F.gelu(torch.matmul(adj, self.gc2(x)))
        x = F.gelu(torch.matmul(adj, self.gc3(x))).mean(dim=1)
        return self.head(x)


# ==========================================
# 4. 학습 파이프라인
# ==========================================
X, y_enc, le = load_data("dataset.csv")
NUM_CLASSES = len(le.classes_)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y_enc, test_size=0.1, stratify=y_enc, random_state=CONFIG["SEED"]
)

# 스태킹용 데이터 저장소 ( DL 3개 + ML 3개 = 총 6개 모델 )
stacking_train = np.zeros((len(X_train_full), NUM_CLASSES * 6))
stacking_test = np.zeros((len(X_test), NUM_CLASSES * 6))

skf = StratifiedKFold(n_splits=CONFIG["N_FOLDS"], shuffle=True, random_state=CONFIG["SEED"])

if not os.path.exists("models_final"):
    os.makedirs("models_final")

print("\n[Step 1] Deep Learning Ensemble (SAM Optimizer)")

dl_models = [("res", ResNet1D), ("trans", TransformerModel), ("gcn", GCNModel)]

for model_idx, (name, model_cls) in enumerate(dl_models):
    print(f"  Training {name.upper()}...")
    oof_preds = np.zeros((len(X_train_full), NUM_CLASSES))
    test_preds_fold = np.zeros((len(X_test), NUM_CLASSES))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        # Data Setup
        train_ds = HandDataset(X_train_full[train_idx], y_train_full[train_idx], augment=True)
        val_ds = HandDataset(X_train_full[val_idx], y_train_full[val_idx], augment=False)
        train_dl = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)

        model = model_cls(NUM_CLASSES).to(CONFIG["DEVICE"])
        optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=CONFIG["DL_LR"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer.base_optimizer, T_0=20
        )

        best_acc, patience = 0, 0
        best_state = None

        for epoch in range(CONFIG["EPOCHS"]):
            model.train()
            for x, g, y in train_dl:
                x, g, y = x.to(CONFIG["DEVICE"]), g.to(CONFIG["DEVICE"]), y.to(CONFIG["DEVICE"])
                # SAM First Step
                out = model(x, g)
                loss = nn.CrossEntropyLoss()(out, y)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                # SAM Second Step
                nn.CrossEntropyLoss()(model(x, g), y).backward()
                optimizer.second_step(zero_grad=True)

            # Validation
            model.eval()
            corr, tot = 0, 0
            with torch.no_grad():
                for x, g, y in val_dl:
                    x, g, y = x.to(CONFIG["DEVICE"]), g.to(CONFIG["DEVICE"]), y.to(CONFIG["DEVICE"])
                    out = model(x, g)
                    corr += (out.argmax(1) == y).sum().item()
                    tot += y.size(0)
            acc = corr / tot
            scheduler.step()

            if acc > best_acc:
                best_acc = acc
                patience = 0
                best_state = model.state_dict()
            else:
                patience += 1
                if patience >= CONFIG["PATIENCE"]:
                    break

        # Load Best & Save
        model.load_state_dict(best_state)
        torch.save(best_state, f"models_final/{name}_fold{fold}.pth")

        # OOF Inference
        model.eval()
        with torch.no_grad():
            # Val
            val_probs = []
            for x, g, _ in val_dl:
                x, g = x.to(CONFIG["DEVICE"]), g.to(CONFIG["DEVICE"])
                val_probs.append(F.softmax(model(x, g), dim=1).cpu().numpy())
            oof_preds[val_idx] = np.concatenate(val_probs)

            # Test
            test_probs = []
            test_ds = HandDataset(X_test, y_test, augment=False)
            test_ldr = DataLoader(test_ds, batch_size=CONFIG["BATCH_SIZE"])
            for x, g, _ in test_ldr:
                x, g = x.to(CONFIG["DEVICE"]), g.to(CONFIG["DEVICE"])
                test_probs.append(F.softmax(model(x, g), dim=1).cpu().numpy())
            test_preds_fold += np.concatenate(test_probs) / CONFIG["N_FOLDS"]

    # Stacking Feature 저장
    col_start = model_idx * NUM_CLASSES
    stacking_train[:, col_start : col_start + NUM_CLASSES] = oof_preds
    stacking_test[:, col_start : col_start + NUM_CLASSES] = test_preds_fold

print("\n[Step 2] Machine Learning Tuning & Training (Optuna + GBDT)")


# ML은 Feature Engineering을 다시 해야 함 (numpy array)
def get_ml_features(X_in):
    # X_in: (N, 63)
    X_rs = X_in.reshape(-1, 21, 3)
    feats = []
    for i in range(len(X_in)):
        # Normalize
        lm = X_rs[i] - X_rs[i, 0]
        lm /= np.max(np.linalg.norm(lm, axis=1)) + 1e-6
        geo = compute_geometric_features(lm[np.newaxis, ...])[0]  # (20,)
        feats.append(np.concatenate([X_in[i], geo]))  # 좌표(63) + 기하(20) = 83
    return np.array(feats)


X_tr_ml = get_ml_features(X_train_full)
X_te_ml = get_ml_features(X_test)

ml_models_config = {
    "xgb": (xgb.XGBClassifier, {"tree_method": "hist", "verbosity": 0}),
    "lgbm": (lgb.LGBMClassifier, {"verbosity": -1}),
    "cat": (cb.CatBoostClassifier, {"verbose": 0}),
}

offset = 3 * NUM_CLASSES  # DL 모델 이후부터 저장

for ml_idx, (name, (cls, base_params)) in enumerate(ml_models_config.items()):
    print(f"  Tuning & Training {name.upper()}...")

    def objective(trial):
        params = base_params.copy()
        if name == "xgb":
            params.update(
                {
                    "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                    "max_depth": trial.suggest_int("max_depth", 3, 9),
                }
            )
        elif name == "lgbm":
            params.update(
                {
                    "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                }
            )
        elif name == "cat":
            params.update(
                {
                    "iterations": trial.suggest_int("iterations", 300, 1000),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                    "depth": trial.suggest_int("depth", 4, 8),
                }
            )

        # Quick 3-Fold for Tuning
        skf_tune = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        accs = []
        for tr_i, va_i in skf_tune.split(X_tr_ml, y_train_full):
            m = cls(**params)
            m.fit(X_tr_ml[tr_i], y_train_full[tr_i])
            accs.append(accuracy_score(y_train_full[va_i], m.predict(X_tr_ml[va_i])))
        return np.mean(accs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=CONFIG["N_TRIALS"])
    best_params = study.best_params
    best_params.update(base_params)
    print(f"    -> Best Params: {best_params}")

    # 5-Fold Training with Best Params
    oof = np.zeros((len(X_train_full), NUM_CLASSES))
    test_pred = np.zeros((len(X_test), NUM_CLASSES))

    for fold, (tr_i, va_i) in enumerate(skf.split(X_tr_ml, y_train_full)):
        model = cls(**best_params)
        model.fit(X_tr_ml[tr_i], y_train_full[tr_i])
        oof[va_i] = model.predict_proba(X_tr_ml[va_i])
        test_pred += model.predict_proba(X_te_ml) / CONFIG["N_FOLDS"]

        with open(f"models_final/{name}_fold{fold}.pkl", "wb") as f:
            pickle.dump(model, f)

    stacking_train[:, offset + ml_idx * NUM_CLASSES : offset + (ml_idx + 1) * NUM_CLASSES] = oof
    stacking_test[:, offset + ml_idx * NUM_CLASSES : offset + (ml_idx + 1) * NUM_CLASSES] = (
        test_pred
    )

print("\n[Step 3] Final Stacking Meta-Learner")

meta_model = LogisticRegression(max_iter=2000, C=1.0)
meta_model.fit(stacking_train, y_train_full)
final_acc = accuracy_score(y_test, meta_model.predict(stacking_test))

print(f"======== FINAL ENSEMBLE ACCURACY: {final_acc*100:.4f}% ========")

with open("models_final/meta_model.pkl", "wb") as f:
    pickle.dump(meta_model, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("모든 학습이 완료되었습니다. 이제 test.py에서 이 모델들을 사용하여 추론할 수 있습니다.")
