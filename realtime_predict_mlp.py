import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

# 1. 모델 및 라벨 인코더 로드
try:
    model = tf.keras.models.load_model('mlp.h5')
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    print("모델과 라벨 인코더를 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print("오류: 'mlp_model.h5' 또는 'label_encoder.pkl' 파일을 찾을 수 없습니다.")
    print("먼저 학습 코드를 실행하여 모델을 생성해주세요.")
    exit()

# 2. 전처리 함수 정의 (학습 코드와 100% 동일한 로직이어야 함)
def preprocess_input(landmarks):
    """
    입력된 랜드마크(21, 3)를 학습 때와 동일하게 전처리합니다.
    1. 상대 좌표 변환 (손목 기준)
    2. 정규화 (절대값 최대치로 나눔)
    3. 1차원 벡터로 변환
    """
    # numpy 배열로 변환
    landmarks = np.array(landmarks)
    
    # 1. 상대 좌표 변환: 손목(0번 인덱스) 좌표 빼기
    wrist = landmarks[0, :]
    relative_landmarks = landmarks - wrist
    
    # 2. 정규화
    max_val = np.max(np.abs(relative_landmarks))
    if max_val > 0:
        normalized_landmarks = relative_landmarks / max_val
    else:
        normalized_landmarks = relative_landmarks
        
    # 3. 모델 입력 형태(1, 63)로 변환
    return normalized_landmarks.flatten().reshape(1, 63)

# 3. MediaPipe Hands 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,                # 손은 하나만 인식
    min_detection_confidence=0.7,   # 탐지 임계값
    min_tracking_confidence=0.5     # 추적 임계값
)

# 4. 웹캠 캡처 시작
cap = cv2.VideoCapture(0)

print("\n--- 실시간 예측 시작 (종료하려면 'q'를 누르세요) ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지를 좌우 반전 (거울 모드) 및 BGR -> RGB 변환
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 손 인식 수행
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # 랜드마크 그리기
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 랜드마크 좌표 추출 (x, y, z)
            landmark_list = []
            h, w, c = frame.shape
            
            for lm in hand_landmarks.landmark:
                # MediaPipe는 0~1 사이의 정규화된 값을 줍니다.
                # 학습 데이터가 픽셀 좌표였다면 아래처럼 변환이 필요하고,
                # 학습 데이터도 0~1이었다면 lm.x, lm.y를 그대로 쓰면 됩니다.
                # 일반적인 CSV 수집 방식(픽셀 좌표)을 가정하여 변환합니다.
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([cx, cy, lm.z]) # z는 화면 깊이 비율

            # 전처리 수행
            input_data = preprocess_input(landmark_list)
            
            # 예측 수행
            prediction = model.predict(input_data, verbose=0)
            predicted_class_index = np.argmax(prediction)
            predicted_label = le.inverse_transform([predicted_class_index])[0]
            confidence = prediction[0][predicted_class_index]
            
            # 결과 화면 출력
            text = f"{predicted_label} ({confidence*100:.1f}%)"
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # (디버깅용) 콘솔 출력
            # print(f"예측: {predicted_label}, 확률: {confidence:.4f}")

    cv2.imshow('Hand Gesture Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()