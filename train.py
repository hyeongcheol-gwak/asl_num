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
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 장치: {device}")

try:
    data = pd.read_csv('dataset.csv')
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

def extract_features_vectorized(landmarks_batch):
    features_list = []
    
    fingers = np.array([
        [1, 2, 3], [2, 3, 4],
        [0, 5, 6], [5, 6, 7], [6, 7, 8],
        [0, 9, 10], [9, 10, 11], [10, 11, 12],
        [0, 13, 14], [13, 14, 15], [14, 15, 16],
        [0, 17, 18], [17, 18, 19], [18, 19, 20]
    ])
    
    for f in fingers:
        p1 = landmarks_batch[:, f[0], :]
        p2 = landmarks_batch[:, f[1], :]
        p3 = landmarks_batch[:, f[2], :]
        v1 = p1 - p2
        v2 = p3 - p2
        
        dot_product = np.sum(v1 * v2, axis=1)
        norm_v1 = np.linalg.norm(v1, axis=1)
        norm_v2 = np.linalg.norm(v2, axis=1)
        
        cos_theta = dot_product / (norm_v1 * norm_v2 + 1e-8)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta)) / 180.0
        features_list.append(angle)
    
    thumb_tip = landmarks_batch[:, 4, :]
    tips = [8, 12, 16, 20]
    for tip_idx in tips:
        tip_pos = landmarks_batch[:, tip_idx, :]
        dist = np.linalg.norm(thumb_tip - tip_pos, axis=1)
        features_list.append(dist)
    
    return np.column_stack(features_list)

def preprocess_coordinates_with_features(X_data):
    landmarks_batch = X_data.reshape(-1, 21, 3)
    
    wrist = landmarks_batch[:, 0:1, :]
    relative_landmarks = landmarks_batch - wrist
    
    max_val = np.max(np.abs(relative_landmarks), axis=(1, 2), keepdims=True)
    max_val = np.where(max_val > 0, max_val, 1.0)
    normalized_landmarks = relative_landmarks / max_val
    
    geometric_features = extract_features_vectorized(normalized_landmarks)
    
    flattened = normalized_landmarks.reshape(-1, 63)
    combined = np.concatenate([flattened, geometric_features], axis=1)
    
    return combined

print("데이터 전처리 및 특성 추출 중...")
X = preprocess_coordinates_with_features(X_raw)
X = X.astype(np.float32)
input_dim = X.shape[1]
print(f"모델 입력 차원(Features): {input_dim}")

if input_dim != 81:
    print(f"경고: 예상된 입력 차원(81)과 다릅니다. 현재: {input_dim}")

le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(np.unique(y))

print(f"\n데이터셋 정보:")
print(f"  총 샘플 수: {len(X)}")
print(f"  클래스 수: {num_classes}")
print(f"  클래스 분포:")
unique, counts = np.unique(y, return_counts=True)
for cls, cnt in zip(unique, counts):
    label = le.inverse_transform([cls])[0]
    print(f"    클래스 {cls} ({label}): {cnt}개 ({cnt/len(y)*100:.1f}%)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

def augment_data(X, y, noise_level=0.01, scale_range=0.05):
    augmented_X = []
    augmented_y = []
    
    noise = np.random.normal(0, noise_level, X.shape)
    X_noise = X + noise
    augmented_X.append(X_noise)
    augmented_y.append(y)
    
    num_coords = 63
    num_angles = 14
    
    scale_factor = np.random.uniform(1 - scale_range, 1 + scale_range, (X.shape[0], 1))
    
    X_coords = X[:, :num_coords] * scale_factor
    
    X_angles = X[:, num_coords:num_coords+num_angles] 
    
    X_dists = X[:, num_coords+num_angles:] * scale_factor
    
    X_scaled = np.concatenate([X_coords, X_angles, X_dists], axis=1)
    
    augmented_X.append(X_scaled)
    augmented_y.append(y)
    
    return np.concatenate([X] + augmented_X), np.concatenate([y] + augmented_y)

print(f"증강 전: {X_train.shape}")
X_train, y_train = augment_data(X_train, y_train)
print(f"증강 후: {X_train.shape}")

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
            
            nn.Linear(64, num_classes)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        return self.model(x)

model = MLP(input_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True)

X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

best_val_acc = 0.0
best_model_state = None
patience = 20
patience_counter = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print("모델 학습 시작...")
for epoch in range(150):
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
    
    scheduler.step(val_loss)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
        torch.save(model.state_dict(), 'mlp.pth')
        print(f"Epoch {epoch+1}/150 - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}% [Best]")
    else:
        patience_counter += 1
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/150 - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
    
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

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

with open('training_history.pkl', 'wb') as f:
    pickle.dump(history, f)
print("학습 히스토리가 'training_history.pkl'에 저장되었습니다.")
