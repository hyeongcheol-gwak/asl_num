import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

# 1. 모델 및 라벨 인코더 로드
try:
    # 학습된 모델 파일명 확인 (feature engineering된 모델이어야 함)
    model = tf.keras.models.load_model('mlp3.h5') 
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    print("모델과 라벨 인코더를 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print("오류: 모델 파일(.h5) 또는 'label_encoder.pkl'을 찾을 수 없습니다.")
    exit()

# --- [핵심 추가] 특성 공학 함수들 (학습 코드와 동일해야 함) ---
def get_angle(v1, v2):
    """두 벡터 사이의 각도를 계산"""
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
    """63개 좌표에서 기하학적 특성(각도, 거리) 추출"""
    features = []
    
    # 1. 손가락 관절 각도 (14개)
    fingers = [
        [1, 2, 3], [2, 3, 4],             # 엄지
        [0, 5, 6], [5, 6, 7], [6, 7, 8],  # 검지
        [0, 9, 10], [9, 10, 11], [10, 11, 12], # 중지
        [0, 13, 14], [13, 14, 15], [14, 15, 16], # 약지
        [0, 17, 18], [17, 18, 19], [18, 19, 20]  # 새끼
    ]
    
    for f in fingers:
        p1, p2, p3 = landmarks[f[0]], landmarks[f[1]], landmarks[f[2]]
        v1 = p1 - p2
        v2 = p3 - p2
        angle = get_angle(v1, v2)
        features.append(angle)

    # 2. 손가락 끝과 엄지 사이 거리 (4개)
    thumb_tip = landmarks[4]
    tips = [8, 12, 16, 20]
    for tip_idx in tips:
        dist = np.linalg.norm(thumb_tip - landmarks[tip_idx])
        features.append(dist)

    return np.array(features)

# --- [수정된 전처리 함수] ---
def preprocess_input(landmarks):
    """
    입력: 랜드마크 리스트 (21, 3) (픽셀 좌표 혹은 정규 좌표)
    출력: (1, 81) 형태의 모델 입력 벡터
    """
    landmarks = np.array(landmarks)
    
    # 1. 상대 좌표 변환
    wrist = landmarks[0, :]
    relative_landmarks = landmarks - wrist
    
    # 2. 정규화
    max_val = np.max(np.abs(relative_landmarks))
    if max_val > 0:
        normalized_landmarks = relative_landmarks / max_val
    else:
        normalized_landmarks = relative_landmarks
        
    # 3. 특성 추출 (여기서 18개 특성이 추가됨)
    # normalized_landmarks 형태는 (21, 3)이어야 함
    geometric_features = extract_features(normalized_landmarks)
    
    # 4. 결합: 좌표(63) + 특성(18) = 81
    combined = np.concatenate([normalized_landmarks.flatten(), geometric_features])
    
    return combined.reshape(1, -1) # (1, 81)로 변환

# 3. MediaPipe 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# 4. 웹캠 실행
cap = cv2.VideoCapture(0)

print("\n--- 실시간 예측 시작 (종료 'q') ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 좌표 추출
            landmark_list = []
            h, w, c = frame.shape
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([cx, cy, lm.z]) # z 포함

            # [수정됨] 전처리 및 예측
            try:
                input_data = preprocess_input(landmark_list) # (1, 81) 반환
                
                prediction = model.predict(input_data, verbose=0)
                predicted_index = np.argmax(prediction)
                predicted_label = le.inverse_transform([predicted_index])[0]
                confidence = prediction[0][predicted_index]
                
                # 임계값 설정 (예: 80% 이상일 때만 표시)
                if confidence > 0.8:
                    display_text = f"{predicted_label} ({confidence*100:.1f}%)"
                    color = (0, 255, 0)
                else:
                    display_text = "Unknown"
                    color = (0, 0, 255)

                cv2.putText(frame, display_text, (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                           
            except Exception as e:
                print(f"예측 에러: {e}")

    cv2.imshow('Hand Gesture Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()