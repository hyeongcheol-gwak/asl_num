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
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 장치: {device}")

try:
    data = pd.read_csv('dataset.csv')
except FileNotFoundError:
    print("오류: 'dataset.csv' 파일을 찾을 수 없습니다.")
    exit()

X_raw = data.iloc[:, 1:].values.astype(np.float32)
y_raw = data.iloc[:, 0].values

def preprocess_coordinates(X_data):
    processed_data = []
    for row in X_data:
        landmarks = row.reshape(21, 3)
        
        wrist = landmarks[0, :]
        relative_landmarks = landmarks - wrist
        
        max_dist = np.max(np.linalg.norm(relative_landmarks, axis=1))
        if max_dist > 0:
            normalized_landmarks = relative_landmarks / max_dist
        else:
            normalized_landmarks = relative_landmarks
            
        processed_data.append(normalized_landmarks)
    
    return np.array(processed_data)

X = preprocess_coordinates(X_raw)

le = LabelEncoder()
y_int = le.fit_transform(y_raw)
num_classes = len(np.unique(y_int))
y_onehot = np.eye(num_classes)[y_int]

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=RANDOM_SEED, stratify=y_int)

class HandDataset(Dataset):
    def __init__(self, x_set, y_set, augment=False):
        self.x = torch.FloatTensor(x_set)
        self.y = torch.FloatTensor(y_set)
        self.augment = augment
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx].clone()
        y = self.y[idx]
        
        if self.augment:
            x = self._augment(x)
        
        return x, y
    
    def _augment(self, x):
        aug_x = x.clone().numpy()
        
        theta_z = np.radians(np.random.uniform(-15, 15))
        theta_x = np.radians(np.random.uniform(-5, 5))
        
        cz, sz = np.cos(theta_z), np.sin(theta_z)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        
        cx, sx = np.cos(theta_x), np.sin(theta_x)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        
        aug_x = aug_x @ Rz @ Rx
        
        scale = np.random.uniform(0.9, 1.1)
        aug_x = aug_x * scale
        
        noise = np.random.normal(0, 0.002, aug_x.shape)
        aug_x = aug_x + noise
        
        if np.random.random() < 0.3:
            num_mask = np.random.randint(1, 4)
            mask_indices = np.random.choice(21, num_mask, replace=False)
            aug_x[mask_indices, :] = 0.0
        
        return torch.FloatTensor(aug_x)

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
    def __init__(self, input_dim=3, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, num_classes=10, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Embedding(21, d_model)
        
        encoder_layer = TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
        self.transformer = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout_final = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
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

model = TransformerModel(input_dim=3, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, num_classes=num_classes, dropout=0.1).to(device)

def label_smoothing_loss(pred, target, smoothing=0.1):
    num_classes = pred.size(1)
    log_probs = F.log_softmax(pred, dim=1)
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)
    return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6, verbose=True)

train_dataset = HandDataset(X_train, y_train, augment=True)
test_dataset = HandDataset(X_test, y_test, augment=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"\n모델 구조:")
print(model)
print(f"\n파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

best_val_acc = 0.0
best_model_state = None
patience = 20
patience_counter = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print("\n--- 학습 시작 (Transformer + Full Augmentation) ---")
for epoch in range(150):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_y_int = torch.argmax(batch_y, dim=1)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        
        loss = label_smoothing_loss(outputs, batch_y_int, smoothing=0.1)
        loss.backward()
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
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_y_int = torch.argmax(batch_y, dim=1)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y_int)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y_int).sum().item()
    
    val_loss /= len(test_loader)
    val_acc = val_correct / val_total
    
    scheduler.step(val_loss)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
        torch.save(model.state_dict(), 'transformer.pth')
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
final_loss = 0.0
final_correct = 0
final_total = 0

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_y_int = torch.argmax(batch_y, dim=1)
        
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y_int)
        
        final_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        final_total += batch_y.size(0)
        final_correct += (predicted == batch_y_int).sum().item()

final_loss /= len(test_loader)
final_acc = final_correct / final_total

print(f"\n최종 테스트 정확도: {final_acc*100:.2f}%")

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

with open('training_history.pkl', 'wb') as f:
    pickle.dump(history, f)
