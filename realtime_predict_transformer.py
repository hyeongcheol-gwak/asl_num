import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras import layers, models

# ==========================================
# 1. 설정 및 모델 구조 정의 (오류 방지용)
# ==========================================
MODEL_PATH = 'transformer_model.h5'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

# --- Transformer 모델 구조 (학습 코드와 동일) ---
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

# ==========================================
# 2. 초기화 및 리소스 로드
# ==========================================
print("--- 리소스 로딩 중... ---")

# 라벨 인코더 로드
try:
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    classes = le.classes_
    print(f"클래스 목록: {classes}")
except Exception as e:
    print(f"라벨 인코더 로드 실패: {e}")
    exit()

# 모델 빌드 및 가중치 로드
try:
    model = build_transformer_model(input_shape=(21, 3), num_classes=len(classes))
    model.load_weights(MODEL_PATH)
    print("모델 로드 완료!")
except Exception as e:
    print(f"모델 로드 실패: {e}")
    exit()

# MediaPipe 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,            # 손 하나만 인식 (필요시 변경)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================
# 3. 실시간 예측 루프
# ==========================================
def preprocess_single_frame(landmarks_list):
    """학습 때와 동일한 전처리 (중심화 + 정규화)"""
    landmarks = np.array(landmarks_list).reshape(21, 3)
    
    # 1. 중심화 (손목 기준)
    wrist = landmarks[0, :]
    relative_landmarks = landmarks - wrist
    
    # 2. 정규화
    max_dist = np.max(np.linalg.norm(relative_landmarks, axis=1))
    if max_dist > 0:
        normalized = relative_landmarks / max_dist
    else:
        normalized = relative_landmarks
        
    return normalized.reshape(1, 21, 3) # 모델 입력 형태 (Batch, 21, 3)

cap = cv2.VideoCapture(0) # 웹캠 1번

print("--- 웹캠 시작 ('q'를 누르면 종료) ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 이미지 처리 (좌우 반전 + RGB 변환)
    image = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. 손 인식 수행
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 3. 랜드마크 추출 (x, y, z)
            # MediaPipe는 0~1 사이 값으로 주지만, 우리는 상대 좌표를 쓰므로 그대로 써도 무방하나
            # 학습 데이터가 픽셀 좌표였다면 픽셀로 변환해야 함.
            # (작성해주신 학습 코드는 원본 값(보통 픽셀)을 받아 처리했으므로, 
            # 여기서도 화면 해상도 곱해서 픽셀 좌표로 만들어주는 것이 안전합니다.)
            
            h, w, c = image.shape
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.append([lm.x * w, lm.y * h, lm.z * w]) # z도 w 비율로 맞춤
            
            # 4. 전처리
            input_data = preprocess_single_frame(landmark_list)
            
            # 5. 모델 예측
            prediction = model.predict(input_data, verbose=0)
            pred_idx = np.argmax(prediction)
            confidence = np.max(prediction)
            
            label = classes[pred_idx]
            
            # 6. 화면에 그리기
            # 손 뼈대 그리기
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 결과 텍스트 표시
            text = f"{label} ({confidence*100:.1f}%)"
            
            # 배경 박스 (가독성 위해)
            cv2.rectangle(image, (10, 10), (300, 60), (0, 0, 0), -1)
            
            # 확신도가 낮으면 빨간색, 높으면 초록색
            color = (0, 255, 0) if confidence > 0.7 else (0, 0, 255)
            cv2.putText(image, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Recognition (Transformer)', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()