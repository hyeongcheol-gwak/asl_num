import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pickle
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

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
    print(f"경고: 예상된 좌표 수(63)와 다릅니다. 현재: {X_raw.shape[1]}")

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


model = Sequential([
    Dense(512, input_shape=(input_dim,), activation='swish', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(256, activation='swish', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation='swish', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='swish', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(num_classes, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('mlp.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)

print("모델 학습 시작...")
history = model.fit(
    X_train, y_train, 
    epochs=150, 
    batch_size=64, 
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n최종 정확도: {acc*100:.2f}%")
print(f"최종 손실: {loss:.4f}")

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print("학습 히스토리가 'training_history.pkl'에 저장되었습니다.")
