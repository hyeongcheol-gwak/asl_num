import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
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

# --- [핵심 개선 1] 고급 전처리: (21, 3) 구조 유지 ---
def preprocess_coordinates(X_data):
    processed_data = []
    for row in X_data:
        landmarks = row.reshape(21, 3)
        
        # 1. 중심화 (손목 0번 기준)
        wrist = landmarks[0, :]
        relative_landmarks = landmarks - wrist
        
        # 2. 강건한 정규화 (최대 거리로 나누기)
        max_dist = np.max(np.linalg.norm(relative_landmarks, axis=1))
        if max_dist > 0:
            normalized_landmarks = relative_landmarks / max_dist
        else:
            normalized_landmarks = relative_landmarks
            
        processed_data.append(normalized_landmarks) # (21, 3) 형태로 저장
    
    return np.array(processed_data)

X = preprocess_coordinates(X_raw)

# 레이블 인코딩
le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- [핵심 개선 2] 데이터 증강 (Data Augmentation) 클래스 ---
# 학습 시 실시간으로 데이터에 변형을 주어 과적합을 막고 일반화 성능 극대화
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
        # 랜덤 회전 (Z축 기준 - 손목을 돌리는 행위)
        theta = np.random.uniform(-0.3, 0.3, size=batch_x.shape[0]) # 약 -15~15도
        c, s = np.cos(theta), np.sin(theta)
        # 회전 행렬 적용은 간단하게 노이즈와 스케일링으로 대체하거나 복잡한 행렬 연산 수행
        # 여기서는 성능을 위해 '노이즈'와 '스케일링'에 집중
        
        # 1. 노이즈 추가 (손 떨림 보정)
        noise = np.random.normal(0, 0.02, batch_x.shape)
        
        # 2. 스케일링 (손 크기 변화)
        scale = np.random.uniform(0.9, 1.1, size=(batch_x.shape[0], 1, 1))
        
        return (batch_x * scale) + noise

# --- [핵심 개선 3] Transformer Encoder Block 정의 ---
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention Layer
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs # Skip Connection

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="gelu")(x) # Conv1D가 Dense보다 시계열 처리에 효율적
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res # Skip Connection

# --- 모델 빌드 ---
def build_transformer_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # 입력 투영 (Projection)
    x = layers.Dense(64)(inputs) # (21, 3) -> (21, 64) 차원 확장
    
    # Positional Encoding (손가락 순서 정보를 학습)
    # 21개의 점은 순서가 고정되어 있으므로 이를 모델에 알려줌
    positions = tf.range(start=0, limit=21, delta=1)
    position_embedding = layers.Embedding(input_dim=21, output_dim=64)(positions)
    x = x + position_embedding

    # Transformer Blocks (깊게 쌓음)
    for _ in range(2): # 블록 2개 적층
        x = transformer_encoder(x, head_size=64, num_heads=2, ff_dim=64, dropout=0.4)

    # Classification Head
    x = layers.GlobalAveragePooling1D()(x) # (21, 64) -> (64,) 로 압축
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="gelu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)

# 모델 생성
model = build_transformer_model(input_shape=(21, 3), num_classes=num_classes)

# 컴파일 (AdamW 사용: 일반화 성능 우수)
model.compile(
    optimizer=optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 제너레이터 생성
train_gen = DataAugmenter(X_train, y_train, batch_size=8, augment=True) # 학습시 증강 ON
test_gen = DataAugmenter(X_test, y_test, batch_size=8, augment=False)   # 테스트시 증강 OFF

# 콜백 설정
callbacks_list = [
    callbacks.ModelCheckpoint('transformer_model.h5', save_best_only=True, monitor='val_accuracy', mode='max'),
    callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

print("\n--- 학습 시작 (Transformer) ---")
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=100,
    callbacks=callbacks_list
)

# 최종 평가
loss, acc = model.evaluate(test_gen)
print(f"\n최종 테스트 정확도: {acc*100:.2f}%")

# 라벨 저장
import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)