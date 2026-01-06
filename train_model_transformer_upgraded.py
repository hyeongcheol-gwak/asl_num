import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. 데이터 로드
try:
    data = pd.read_csv('dataset_combined.csv')
except FileNotFoundError:
    print("오류: 'dataset_combined.csv' 파일을 찾을 수 없습니다.")
    exit()

X_raw = data.iloc[:, 1:].values
y_raw = data.iloc[:, 0].values

# --- [전처리] (21, 3) 구조 및 정규화 ---
def preprocess_coordinates(X_data):
    processed_data = []
    for row in X_data:
        landmarks = row.reshape(21, 3)
        
        # 1. 중심화 (손목 0번 기준)
        wrist = landmarks[0, :]
        relative_landmarks = landmarks - wrist
        
        # 2. 정규화 (최대 거리로 나누기)
        max_dist = np.max(np.linalg.norm(relative_landmarks, axis=1))
        if max_dist > 0:
            normalized_landmarks = relative_landmarks / max_dist
        else:
            normalized_landmarks = relative_landmarks
            
        processed_data.append(normalized_landmarks)
    
    return np.array(processed_data)

X = preprocess_coordinates(X_raw)

# 레이블 인코딩 & 원-핫 인코딩 (Label Smoothing을 위해 필수)
le = LabelEncoder()
y_int = le.fit_transform(y_raw)
num_classes = len(np.unique(y_int))
y = to_categorical(y_int, num_classes=num_classes) # 원-핫 변환

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_int)

# --- [업그레이드] 데이터 증강 클래스 (회전 + 마스킹 포함) ---
class DataAugmenter(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, shuffle=True, augment=False):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_x = self.x[indexes]
        batch_y = self.y[indexes]

        if self.augment:
            batch_x = self._augment_batch(batch_x)
            
        return batch_x, batch_y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.x))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def _augment_batch(self, batch_x):
        augmented_batch = []
        for x in batch_x:
            # 복사본 생성
            aug_x = x.copy()
            
            # 1. Random Rotation (실제 3D 회전 적용)
            # Z축(손목 회전) 뿐만 아니라 X, Y축도 미세하게 회전
            theta_z = np.radians(np.random.uniform(-15, 15)) 
            theta_x = np.radians(np.random.uniform(-5, 5)) # 깊이 방향 미세 변화
            
            # Z축 회전 행렬
            cz, sz = np.cos(theta_z), np.sin(theta_z)
            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            
            # X축 회전 행렬
            cx, sx = np.cos(theta_x), np.sin(theta_x)
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            
            # 회전 적용
            aug_x = aug_x @ Rz @ Rx
            
            # 2. Scaling (손 크기 변화)
            scale = np.random.uniform(0.9, 1.1)
            aug_x = aug_x * scale
            
            # 3. Noise (센서 노이즈)
            noise = np.random.normal(0, 0.002, aug_x.shape)
            aug_x = aug_x + noise
            
            # 4. Masking (관절 가림 현상 시뮬레이션)
            # 30% 확률로 1~3개의 관절 좌표를 0으로 지움
            if np.random.random() < 0.3:
                num_mask = np.random.randint(1, 4)
                mask_indices = np.random.choice(21, num_mask, replace=False)
                aug_x[mask_indices, :] = 0.0
            
            augmented_batch.append(aug_x)
            
        return np.array(augmented_batch)

# --- Transformer Encoder Block ---
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="gelu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

# --- 모델 빌드 ---
def build_transformer_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # Projection & Positional Encoding
    x = layers.Dense(64)(inputs)
    positions = tf.range(start=0, limit=21, delta=1)
    position_embedding = layers.Embedding(input_dim=21, output_dim=64)(positions)
    x = x + position_embedding

    # Deep Transformer Blocks (3단으로 깊게)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)

    # Classification Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x) # Classifier 직전 Dropout은 조금 강하게
    x = layers.Dense(64, activation="gelu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)

model = build_transformer_model(input_shape=(21, 3), num_classes=num_classes)

# 컴파일 (Label Smoothing 추가: 0.1)
# 정답을 1.0이 아닌 0.9로, 오답을 0.0이 아닌 0.1/N 으로 설정하여 과적합 방지
model.compile(
    optimizer=optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), 
    metrics=["accuracy"]
)

model.summary()

# 제너레이터 (학습 데이터에만 증강 적용)
train_gen = DataAugmenter(X_train, y_train, batch_size=32, augment=True)
test_gen = DataAugmenter(X_test, y_test, batch_size=32, augment=False)

# 콜백
callbacks_list = [
    callbacks.ModelCheckpoint('best_transformer_model.keras', save_best_only=True, monitor='val_accuracy', mode='max'),
    callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
]

print("\n--- 학습 시작 (Transformer + Full Augmentation) ---")
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=150,
    callbacks=callbacks_list
)

# 평가
loss, acc = model.evaluate(test_gen)
print(f"\n최종 테스트 정확도: {acc*100:.2f}%")

import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)