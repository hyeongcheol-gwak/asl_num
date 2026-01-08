import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import copy

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

try:
    data = pd.read_csv("dataset.csv")
except FileNotFoundError:
    print("오류: 'dataset.csv' 파일을 찾을 수 없습니다.")
    exit()

if data.empty:
    print("오류: 데이터 파일이 비어있습니다.")
    exit()

X_raw = data.iloc[:, 1:].values.astype(np.float32)
y_raw = data.iloc[:, 0].values

if X_raw.shape[1] != 63:
    print(f"오류: 예상된 좌표 수(63)와 다릅니다. 현재: {X_raw.shape[1]}")
    exit()

le = LabelEncoder()
y_int = le.fit_transform(y_raw)
num_classes = len(np.unique(y_int))
y_onehot = np.eye(num_classes)[y_int]

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_onehot, test_size=0.2, random_state=RANDOM_SEED, stratify=y_int
)


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


class HandDataset(Dataset):
    def __init__(self, x_set, y_set, augment=False):
        self.x = x_set
        self.y = torch.FloatTensor(y_set)
        self.augment = augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        landmarks = self.x[idx].reshape(21, 3).copy()

        if self.augment:
            landmarks = self._augment(landmarks)

        wrist = landmarks[0, :].copy()
        relative_landmarks = landmarks - wrist
        max_dist = np.max(np.linalg.norm(relative_landmarks, axis=1))
        if max_dist > 0:
            normalized_landmarks = relative_landmarks / max_dist
        else:
            normalized_landmarks = relative_landmarks

        geo_features = compute_geometric_features(relative_landmarks)
        geo_features[15:] = geo_features[15:] / (max_dist + 1e-8)

        return (torch.FloatTensor(normalized_landmarks), torch.FloatTensor(geo_features)), self.y[
            idx
        ]

    def _augment(self, landmarks):
        theta_z = np.radians(np.random.uniform(-15, 15))
        theta_x = np.radians(np.random.uniform(-10, 10))
        theta_y = np.radians(np.random.uniform(-10, 10))

        cz, sz = np.cos(theta_z), np.sin(theta_z)
        cx, sx = np.cos(theta_x), np.sin(theta_x)
        cy, sy = np.cos(theta_y), np.sin(theta_y)

        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])

        landmarks = landmarks @ Rz @ Rx @ Ry

        scale = np.random.uniform(0.9, 1.1)
        landmarks = landmarks * scale

        noise = np.random.normal(0, 0.002, landmarks.shape)
        landmarks = landmarks + noise

        return landmarks


class HybridHandModel(nn.Module):
    def __init__(
        self, num_classes=10, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1
    ):
        super().__init__()

        self.input_projection = nn.Linear(3, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 22, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)

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


model = HybridHandModel(num_classes=num_classes, d_model=64, nhead=4, num_layers=3).to(device)


def label_smoothing_loss(pred, target, smoothing=0.1):
    num_classes = pred.size(1)
    log_probs = F.log_softmax(pred, dim=1)
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)
    return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

train_dataset = HandDataset(X_train, y_train, augment=True)
test_dataset = HandDataset(X_test, y_test, augment=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"\n모델 구조 (Hybrid: Transformer + MLP):")
print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

best_val_acc = 0.0
best_model_state = None
patience = 20
patience_counter = 0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

print("\n--- 학습 시작 (Hybrid Architecture) ---")
for epoch in range(150):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for (batch_coords, batch_geo), batch_y in train_loader:
        batch_coords = batch_coords.to(device)
        batch_geo = batch_geo.to(device)
        batch_y = batch_y.to(device)
        batch_y_int = torch.argmax(batch_y, dim=1)

        optimizer.zero_grad()
        outputs = model(batch_coords, batch_geo)

        loss = label_smoothing_loss(outputs, batch_y_int, smoothing=0.1)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y_int).sum().item()

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for (batch_coords, batch_geo), batch_y in test_loader:
            batch_coords = batch_coords.to(device)
            batch_geo = batch_geo.to(device)
            batch_y = batch_y.to(device)
            batch_y_int = torch.argmax(batch_y, dim=1)

            outputs = model(batch_coords, batch_geo)
            loss = criterion(outputs, batch_y_int)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y_int).sum().item()

    val_loss /= len(test_loader)
    val_acc = val_correct / val_total

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    current_lr = optimizer.param_groups[0]["lr"]

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(model.state_dict(), "transformer.pth")
        patience_counter = 0
        print(
            f"Epoch {epoch+1:03d} | Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% [Best] LR: {current_lr:.6f}"
        )
    else:
        patience_counter += 1
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:03d} | Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% LR: {current_lr:.6f}"
            )
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

model.eval()
final_correct = 0
final_total = 0

with torch.no_grad():
    for (batch_coords, batch_geo), batch_y in test_loader:
        batch_coords = batch_coords.to(device)
        batch_geo = batch_geo.to(device)
        batch_y = batch_y.to(device)
        batch_y_int = torch.argmax(batch_y, dim=1)

        outputs = model(batch_coords, batch_geo)
        _, predicted = torch.max(outputs.data, 1)
        final_total += batch_y.size(0)
        final_correct += (predicted == batch_y_int).sum().item()

final_acc = final_correct / final_total
print(f"\n최종 테스트 정확도: {final_acc*100:.2f}%")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

with open("training_history.pkl", "wb") as f:
    pickle.dump(history, f)
