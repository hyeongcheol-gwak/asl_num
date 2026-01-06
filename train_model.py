import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# 1. 데이터 로드
try:
    data = pd.read_csv('dataset_combined.csv')
except FileNotFoundError:
    print("오류: 'dataset_combined.csv' 파일을 찾을 수 없습니다.")
    exit()

# 입력(X)와 정답(y) 분리
X_raw = data.iloc[:, 1:].values
y_raw = data.iloc[:, 0].values

# --- [개선 1] 데이터 전처리: 상대 좌표 변환 및 정규화 ---
def preprocess_coordinates(X_data):
    processed_data = []
    for row in X_data:
        # 1차원 배열(63,)을 (21, 3)으로 변환하여 작업 용이하게 함
        landmarks = row.reshape(21, 3)
        
        # 1. 상대 좌표 변환: 모든 좌표에서 손목(0번 인덱스) 좌표 빼기
        wrist = landmarks[0, :]
        relative_landmarks = landmarks - wrist
        
        # 2. 정규화: 좌표값의 최대 절대값으로 나누어 -1 ~ 1 사이로 스케일링
        max_val = np.max(np.abs(relative_landmarks))
        if max_val > 0:
            normalized_landmarks = relative_landmarks / max_val
        else:
            normalized_landmarks = relative_landmarks
            
        # 다시 1차원(63,)으로 펼쳐서 리스트에 추가
        processed_data.append(normalized_landmarks.flatten())
    
    return np.array(processed_data)

print("데이터 전처리 중 (상대 좌표 변환)...")
X = preprocess_coordinates(X_raw)

# --- [개선 2] 레이블 인코딩 (문자열 라벨일 경우 대비) ---
le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(np.unique(y)) # 클래스 개수 자동 파악

# 학습용과 테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 모델 설계 (BatchNormalization 추가)
model = Sequential([
    Dense(128, input_shape=(63,), activation='relu'),
    BatchNormalization(), # 학습 안정화 및 가속
    Dropout(0.3),         # 드롭아웃 비율 약간 상향
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    BatchNormalization(),
    
    Dense(num_classes, activation='softmax')
])

# 3. 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- [개선 3] 콜백(Callbacks) 설정 ---
# 학습 중 가장 성능이 좋은 모델만 저장
checkpoint = ModelCheckpoint('asl_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# 성능이 향상되지 않으면 조기에 학습 종료 (과적합 방지)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
# 성능이 정체되면 학습률을 낮춤 (세밀한 학습)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

print("모델 학습을 시작합니다...")
history = model.fit(
    X_train, y_train, 
    epochs=100,            # EarlyStopping이 있으므로 에포크를 넉넉히 설정
    batch_size=32, 
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stopping, reduce_lr] # 콜백 추가
)

# 4. 결과 확인
loss, acc = model.evaluate(X_test, y_test)
print(f"\n최종 테스트 세트 정확도: {acc*100:.2f}%")

# 라벨 정보 저장 (추후 예측 시 필요)
import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("라벨 인코더가 'label_encoder.pkl'로 저장되었습니다.")