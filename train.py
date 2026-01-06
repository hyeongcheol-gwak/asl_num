import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pickle

try:
    data = pd.read_csv('dataset.csv')
except FileNotFoundError:
    print("오류: 'dataset.csv' 파일을 찾을 수 없습니다.")
    exit()

X_raw = data.iloc[:, 1:].values
y_raw = data.iloc[:, 0].values

def get_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    angle = np.arccos(cos_theta)
    return np.degrees(angle) / 180.0

def extract_features(landmarks):
    features = []
    
    fingers = [
        [1, 2, 3], [2, 3, 4],
        [0, 5, 6], [5, 6, 7], [6, 7, 8],
        [0, 9, 10], [9, 10, 11], [10, 11, 12],
        [0, 13, 14], [13, 14, 15], [14, 15, 16],
        [0, 17, 18], [17, 18, 19], [18, 19, 20]
    ]
    
    for f in fingers:
        p1, p2, p3 = landmarks[f[0]], landmarks[f[1]], landmarks[f[2]]
        v1 = p1 - p2
        v2 = p3 - p2
        angle = get_angle(v1, v2)
        features.append(angle)

    thumb_tip = landmarks[4]
    tips = [8, 12, 16, 20]
    
    for tip_idx in tips:
        dist = np.linalg.norm(thumb_tip - landmarks[tip_idx])
        features.append(dist)

    return np.array(features)

def extract_features_vectorized(landmarks_batch):
    batch_size = landmarks_batch.shape[0]
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
input_dim = X.shape[1]
print(f"모델 입력 차원(Features): {input_dim}")

le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def augment_data(X, y, noise_level=0.01, scale_range=0.05):
    augmented_X = []
    augmented_y = []
    
    noise = np.random.normal(0, noise_level, X.shape)
    X_noise = X + noise
    augmented_X.append(X_noise)
    augmented_y.append(y)
    
    scale_factor = np.random.uniform(1 - scale_range, 1 + scale_range, (X.shape[0], 1))
    X_scaled = X * scale_factor
    augmented_X.append(X_scaled)
    augmented_y.append(y)
    
    return np.concatenate([X] + augmented_X), np.concatenate([y] + augmented_y)

print(f"증강 전: {X_train.shape}")
X_train, y_train = augment_data(X_train, y_train, noise_level=0.01, scale_range=0.05)
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
cosine_annealing = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 0.001 * (np.cos(np.pi * epoch / 150) + 1) / 2
)

print("모델 학습 시작...")
history = model.fit(
    X_train, y_train, 
    epochs=150, 
    batch_size=64, 
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stopping, reduce_lr, cosine_annealing]
)

loss, acc = model.evaluate(X_test, y_test)
print(f"\n최종 정확도: {acc*100:.2f}%")

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
