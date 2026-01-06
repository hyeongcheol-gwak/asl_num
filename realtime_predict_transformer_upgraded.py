import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle

# --- 1. 모델 구조 정의 (학습 코드와 동일하게 복사) ---
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
    
    # Projection & Positional Encoding
    x = layers.Dense(64)(inputs)
    positions = tf.range(start=0, limit=21, delta=1)
    position_embedding = layers.Embedding(input_dim=21, output_dim=64)(positions)
    x = x + position_embedding

    # Deep Transformer Blocks
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)

    # Classification Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="gelu")(x)
    x = layers.Dropout(0.2)(x)
    # 출력층 (num_classes는 나중에 로드한 라벨 개수로 설정해도 되지만, 
    # 가중치 로드를 위해 구조를 먼저 잡아야 하므로 미리 값을 알거나 라벨 인코더에서 가져와야 함)
    # 임시로 None으로 두고 밑에서 다시 설정하거나, 라벨 인코더를 먼저 로드해야 합니다.
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)

# --- 2. 설정 및 리소스 로드 ---
MODEL_PATH = 'best_transformer_model.h5' # 또는 .keras
LABEL_ENCODER_PATH = 'label_encoder.pkl'

# (1) 라벨 인코더 먼저 로드 (클래스 개수를 알기 위해)
print(f"라벨 인코더 로드 중: {LABEL_ENCODER_PATH}...")
try:
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    num_classes = len(le.classes_)
    print(f"라벨 인코더 로드 완료. (클래스 개수: {num_classes})")
except Exception as e:
    print(f"라벨 인코더 로드 실패: {e}")
    exit()

# (2) 모델 구조 빌드 및 가중치 로드
print(f"모델 생성 및 가중치 로드 중: {MODEL_PATH}...")
try:
    # 빈 껍데기 모델 생성
    model = build_transformer_model(input_shape=(21, 3), num_classes=num_classes)
    
    # 가중치 파일 로드 (load_model 대신 load_weights 사용)
    model.load_weights(MODEL_PATH)
    print("모델 로드 완료.")
except Exception as e:
    print(f"모델 로드 실패: {e}")
    # 만약 .keras 포맷이고 load_weights가 안 되면 아래 방식 시도
    # model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'transformer_encoder': transformer_encoder})
    exit()

# --- 3. 전처리 함수 ---
def preprocess_single_sample(landmarks):
    landmarks = np.array(landmarks)
    wrist = landmarks[0, :]
    relative_landmarks = landmarks - wrist
    max_dist = np.max(np.linalg.norm(relative_landmarks, axis=1))
    
    if max_dist > 0:
        normalized_landmarks = relative_landmarks / max_dist
    else:
        normalized_landmarks = relative_landmarks
    return normalized_landmarks

# --- 4. MediaPipe 설정 및 실행 ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

print("\n--- 실시간 예측 시작 ('q'로 종료) ---")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("카메라를 찾을 수 없습니다.")
        continue

    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    image.flags.writeable = True
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.append([lm.x, lm.y, lm.z])
            
            processed_lm = preprocess_single_sample(landmark_list)
            input_data = np.expand_dims(processed_lm, axis=0)
            
            # 예측
            prediction = model.predict(input_data, verbose=0)
            predicted_index = np.argmax(prediction)
            confidence = np.max(prediction)
            
            label_text = le.inverse_transform([predicted_index])[0]
            
            # 시각화
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            h, w, c = image.shape
            cx, cy = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
            
            text = f"{label_text} ({confidence*100:.1f}%)"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(image, (cx, cy - 30), (cx + text_w, cy), (0, 0, 0), -1)
            cv2.putText(image, text, (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Hand Gesture Recognition', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()