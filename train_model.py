import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. 데이터 로드
data = pd.read_csv('dataset_combined.csv')

# 입력(X)와 정답(y) 분리
# iloc[:, 1:] -> 첫 번째 열(label)을 제외한 모든 좌표 데이터
# iloc[:, 0] -> 첫 번째 열(label) 정답 데이터
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# 학습용과 테스트용 데이터 분리 (8:2 비율)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 모델 설계 (Deep Learning Architecture)
model = Sequential([
    # 입력층: 21개 랜드마크 * 3개 좌표(x,y,z) = 63개 노드
    Dense(128, input_shape=(63,), activation='relu'),
    Dropout(0.2), # 과적합 방지 (일부 뉴런 끄기)
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    # 출력층: 0~9까지 총 10개의 클래스 분류
    Dense(10, activation='softmax')
])

# 3. 모델 컴파일 및 학습
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("모델 학습을 시작합니다...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 4. 모델 저장
model.save('asl_model.h5')
print("학습 완료! 'asl_model.h5' 파일이 저장되었습니다.")

# 정확도 평가
loss, acc = model.evaluate(X_test, y_test)
print(f"테스트 세트 정확도: {acc*100:.2f}%")