import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
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
except Exception as e:
    print(f"오류: 데이터 파일 읽기 실패 - {e}")
    exit()

if data.empty:
    print("오류: 데이터 파일이 비어있습니다.")
    exit()

X_raw = data.iloc[:, 1:].values.astype(np.float32)
y_raw = data.iloc[:, 0].values

if X_raw.shape[1] != 63:
    print(f"경고: 예상된 좌표 수와 다릅니다. 현재: {X_raw.shape[1]}")

le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
num_classes = len(np.unique(y_encoded))

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y_encoded, test_size=0.2, random_state=RANDOM_SEED, stratify=y_encoded
)


def rotate_landmarks(landmarks, theta):
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    reshaped = landmarks.reshape(-1, 21, 3)
    rotated = np.dot(reshaped, rotation_matrix.T)
    return rotated.reshape(-1, 63)


def augment_data_coordinates(X, y):
    augmented_X = [X]
    augmented_y = [y]

    for angle in [-15, 15]:
        X_rot = rotate_landmarks(X, angle)
        augmented_X.append(X_rot)
        augmented_y.append(y)

    base_X = np.concatenate(augmented_X)
    base_y = np.concatenate(augmented_y)

    noise = np.random.normal(0, 0.02, base_X.shape)
    scale = np.random.uniform(0.9, 1.1, (base_X.shape[0], 1))

    X_aug = base_X * scale + noise

    final_X = np.concatenate([base_X, X_aug])
    final_y = np.concatenate([base_y, base_y])

    return final_X.astype(np.float32), final_y


def extract_features(X_data):
    landmarks = X_data.reshape(-1, 21, 3)

    wrist = landmarks[:, 0:1, :]
    relative = landmarks - wrist
    max_val = np.max(np.linalg.norm(relative, axis=2), axis=1, keepdims=True)
    max_val = np.where(max_val > 0, max_val, 1.0)
    normalized = relative / max_val.reshape(-1, 1, 1)

    features_list = []

    features_list.append(normalized.reshape(-1, 63))

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

    angle_feats = []
    for f in fingers:
        v1 = normalized[:, f[0], :] - normalized[:, f[1], :]
        v2 = normalized[:, f[2], :] - normalized[:, f[1], :]

        v1_norm = np.linalg.norm(v1, axis=1)
        v2_norm = np.linalg.norm(v2, axis=1)

        dot = np.sum(v1 * v2, axis=1)
        cosine = np.clip(dot / (v1_norm * v2_norm + 1e-8), -1.0, 1.0)
        angle = np.arccos(cosine) / np.pi
        angle_feats.append(angle)

    features_list.append(np.stack(angle_feats, axis=1))

    thumb_tip = normalized[:, 4, :]
    dist_feats = []
    for i in [8, 12, 16, 20]:
        dist = np.linalg.norm(thumb_tip - normalized[:, i, :], axis=1)
        dist_feats.append(dist)

    features_list.append(np.stack(dist_feats, axis=1))

    return np.concatenate(features_list, axis=1).astype(np.float32)


print("데이터 증강 및 전처리 진행 중...")

X_train_aug_coords, y_train_aug = augment_data_coordinates(X_train_raw, y_train)
X_train = extract_features(X_train_aug_coords)

X_test = extract_features(X_test_raw)

print(f"학습 데이터 형태: {X_train.shape} (원본+증강)")
print(f"테스트 데이터 형태: {X_test.shape}")

input_dim = X_train.shape[1]  # 81


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

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.model(x)


model = MLP(input_dim, num_classes).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.LongTensor(y_train_aug).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

best_val_acc = 0.0
best_model_state = None
patience = 20
patience_counter = 0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

print("\n=== 모델 학습 시작 ===")
epochs = 150

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor).item()
        _, val_predicted = torch.max(val_outputs.data, 1)
        val_acc = (val_predicted == y_test_tensor).sum().item() / len(y_test_tensor)

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    current_lr = optimizer.param_groups[0]["lr"]

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
        torch.save(model.state_dict(), "mlp.pth")
        print(
            f"Epoch {epoch+1:03d} | Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% [Best] LR: {current_lr:.6f}"
        )
    else:
        patience_counter += 1
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:03d} | Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | LR: {current_lr:.6f}"
            )

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

model.eval()
with torch.no_grad():
    final_outputs = model(X_test_tensor)
    final_loss = criterion(final_outputs, y_test_tensor).item()
    _, final_predicted = torch.max(final_outputs.data, 1)
    final_acc = (final_predicted == y_test_tensor).sum().item() / len(y_test_tensor)

print(f"\n최종 정확도: {final_acc*100:.2f}%")
print(f"최종 손실: {final_loss:.4f}")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

with open("training_history.pkl", "wb") as f:
    pickle.dump(history, f)

print("학습 완료! 모델(mlp.pth), 라벨 인코더, 히스토리가 저장되었습니다.")
