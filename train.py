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

# --- [특성 공학 함수] ---
def get_angle(v1, v2):
    """두 벡터 사이의 각도를 계산 (0~180도)"""
    # 벡터의 내적과 노름(크기)을 이용한 코사인 법칙
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # 0으로 나누기 방지
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    cos_theta = dot_product / (norm_v1 * norm_v2)
    # 부동소수점 오차로 인해 -1 ~ 1 범위를 벗어나는 것 방지
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    angle = np.arccos(cos_theta)
    return np.degrees(angle) / 180.0  # 0~1 사이로 정규화해서 반환

def extract_features(landmarks):
    """63개 좌표에서 추가적인 기하학적 특성 추출"""
    # landmarks shape: (21, 3)
    features = []
    
    # [특성 1] 손가락 관절 각도 (Finger Joint Angles)
    # 각 손가락의 3개 관절(MCP, PIP, DIP) 각도를 계산
    # 엄지(1-2-3, 2-3-4), 검지~새끼(0-5-6, 5-6-7, 6-7-8, ...)
    # 단순화를 위해 각 손가락의 주요 굽힘 각도만 추출
    
    # 손가락별 관절 인덱스 정의
    fingers = [
        [1, 2, 3], [2, 3, 4],             # 엄지 (2개 각도)
        [0, 5, 6], [5, 6, 7], [6, 7, 8],  # 검지 (3개 각도)
        [0, 9, 10], [9, 10, 11], [10, 11, 12], # 중지
        [0, 13, 14], [13, 14, 15], [14, 15, 16], # 약지
        [0, 17, 18], [17, 18, 19], [18, 19, 20]  # 새끼
    ]
    
    for f in fingers:
        # 벡터 A: p1 -> p2, 벡터 B: p3 -> p2 (중심점 p2 기준)
        p1, p2, p3 = landmarks[f[0]], landmarks[f[1]], landmarks[f[2]]
        v1 = p1 - p2
        v2 = p3 - p2
        angle = get_angle(v1, v2)
        features.append(angle) # 총 14개 각도 추가

    # [특성 2] 손가락 끝과 엄지 끝 사이의 거리 (Thumb-Tip Distances)
    # 핀치 제스처(OK 등) 인식에 중요
    thumb_tip = landmarks[4]
    tips = [8, 12, 16, 20] # 검지, 중지, 약지, 새끼 끝
    
    for tip_idx in tips:
        dist = np.linalg.norm(thumb_tip - landmarks[tip_idx])
        features.append(dist) # 총 4개 거리 추가

    return np.array(features)

# --- [전처리 통합] ---
def preprocess_coordinates_with_features(X_data):
    processed_data = []
    for row in X_data:
        landmarks = row.reshape(21, 3)
        
        # 1. 상대 좌표 변환
        wrist = landmarks[0, :]
        relative_landmarks = landmarks - wrist
        
        # 2. 좌표 정규화
        max_val = np.max(np.abs(relative_landmarks))
        if max_val > 0:
            normalized_landmarks = relative_landmarks / max_val
        else:
            normalized_landmarks = relative_landmarks
        
        # 3. 특성 공학 (추가 정보 추출) - 정규화된 좌표 기반
        geometric_features = extract_features(normalized_landmarks)
        
        # 원본 좌표(63개) + 추가 특성(18개) = 총 81개 피처
        combined = np.concatenate([normalized_landmarks.flatten(), geometric_features])
        processed_data.append(combined)
    
    return np.array(processed_data)

print("데이터 전처리 및 특성 추출 중...")
X = preprocess_coordinates_with_features(X_raw)
input_dim = X.shape[1] # 입력 차원 자동 계산 (아마 81개)
print(f"모델 입력 차원(Features): {input_dim}")

# 레이블 인코딩
le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(np.unique(y))

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# [데이터 증강] (이전 단계에서 적용한 내용 유지)
def augment_data(X, y, noise_level=0.01): # 특성이 정교해졌으므로 노이즈는 살짝 줄임
    noise = np.random.normal(0, noise_level, X.shape)
    X_augmented = X + noise
    return np.concatenate([X, X_augmented]), np.concatenate([y, y])

print(f"증강 전: {X_train.shape}")
X_train, y_train = augment_data(X_train, y_train)
print(f"증강 후: {X_train.shape}")


# 2. 모델 설계 (입력 차원 변경 반영)
model = Sequential([
    # input_shape를 고정값(63) 대신 계산된 input_dim으로 변경
    Dense(256, input_shape=(input_dim,), activation='swish', kernel_initializer='he_normal'),
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

# 3. 컴파일 및 학습
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('mlp.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

print("특성 공학이 적용된 모델 학습 시작...")
history = model.fit(
    X_train, y_train, 
    epochs=150, 
    batch_size=64, 
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

loss, acc = model.evaluate(X_test, y_test)
print(f"\n✨ 특성 공학 적용 최종 정확도: {acc*100:.2f}%")

# 라벨 저장
import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)