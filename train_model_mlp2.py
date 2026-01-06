import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.initializers import HeNormal

# 1. 데이터 로드
try:
    data = pd.read_csv('dataset.csv')
except FileNotFoundError:
    print("오류: 'dataset.csv' 파일을 찾을 수 없습니다.")
    exit()

X_raw = data.iloc[:, 1:].values
y_raw = data.iloc[:, 0].values

# --- [전처리] 상대 좌표 변환 및 정규화 ---
def preprocess_coordinates(X_data):
    processed_data = []
    for row in X_data:
        landmarks = row.reshape(21, 3)
        wrist = landmarks[0, :]
        relative_landmarks = landmarks - wrist # 상대 좌표
        
        max_val = np.max(np.abs(relative_landmarks))
        if max_val > 0:
            normalized_landmarks = relative_landmarks / max_val
        else:
            normalized_landmarks = relative_landmarks
            
        processed_data.append(normalized_landmarks.flatten())
    return np.array(processed_data)

print("데이터 전처리 중...")
X = preprocess_coordinates(X_raw)

# 레이블 인코딩
le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(np.unique(y))

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- [핵심 개선 1] 데이터 증강 (Data Augmentation) ---
# 좌표 데이터에 미세한 노이즈를 섞어 학습 데이터를 인위적으로 늘림 (과적합 방지, 성능 향상)
def augment_data(X, y, noise_level=0.02):
    noise = np.random.normal(0, noise_level, X.shape)
    X_augmented = X + noise
    return np.concatenate([X, X_augmented]), np.concatenate([y, y])

print(f"증강 전 학습 데이터: {X_train.shape}")
# 학습 데이터에만 적용 (테스트 데이터는 순수하게 유지)
X_train, y_train = augment_data(X_train, y_train, noise_level=0.02)
print(f"증강 후 학습 데이터: {X_train.shape}")


# 2. 모델 설계 (Swish 활성화 함수 + He 초기화 + 용량 증대)
model = Sequential([
    # 입력층 확장: 128 -> 256 (초반 정보 손실 최소화)
    # kernel_initializer='he_normal': 가중치 초기화를 최적화
    # activation='swish': ReLU보다 깊은 층에서 학습 효율이 좋음
    Dense(256, input_shape=(63,), activation='swish', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.4), # 용량이 커진 만큼 Dropout 비율 상향

    Dense(128, activation='swish', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='swish', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(num_classes, activation='softmax')
])

# 3. 모델 컴파일
# Label Smoothing을 적용하고 싶다면 sparse 대신 categorical_crossentropy로 변경 필요하지만,
# 구조 유지를 위해 Adam optimizer의 학습률을 조금 낮춰서 정교하게 시작
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. 콜백 설정
checkpoint = ModelCheckpoint('mlp2.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True) # patience 증가
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

print("모델 학습 시작 (향상된 MLP)...")
history = model.fit(
    X_train, y_train, 
    epochs=150, # 데이터가 늘어났으므로 에포크 충분히
    batch_size=64, # 배치는 조금 더 크게 (안정적 학습)
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# 결과 확인
loss, acc = model.evaluate(X_test, y_test)
print(f"\n🚀 최종 업그레이드 모델 정확도: {acc*100:.2f}%")

import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)