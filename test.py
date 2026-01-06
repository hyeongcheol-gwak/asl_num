import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# 1. 설정 및 초기화
INPUT_VIDEO_PATH = 'test_video.mp4' # 제공된 영상 파일 경로
OUTPUT_TXT_PATH = 'result.txt'
MODEL_PATH = 'asl_model.h5' # 학습시킨 모델 (3번 항목 참고)

# MediaPipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# 모델 로드 (학습된 모델이 필요합니다)
# 모델이 아직 없다면 이 부분은 주석 처리하고 더미 로직으로 테스트하세요.
try:
    model = load_model(MODEL_PATH)
except:
    print("모델 파일이 없습니다. 먼저 모델을 학습시켜주세요.")
    exit()

# 클래스 정의 (0은 제외하거나 10으로 매핑, 데이터셋에 따라 조정)
# 예: 모델 출력이 0~9 인덱스라면 -> 실제 숫자 1~10으로 매핑
CLASSES = {
    0: '1', 1: '2', 2: '3', 3: '4', 4: '5',
    5: '6', 6: '7', 7: '8', 8: '9', 9: '10'
}

# 2. 안정화 로직 변수
prediction_buffer = deque(maxlen=10) # 최근 10프레임의 예측값 저장
STABILITY_THRESHOLD = 8 # 10개 중 8개 이상이 동일해야 '안정'으로 판단
output_triggered = False # 현재 제스처에 대해 출력을 했는지 여부

f = open(OUTPUT_TXT_PATH, 'w') # 결과 파일 열기

cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 전처리 (BGR -> RGB)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    current_prediction = None

    # 손이 감지된 경우
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 1. 데이터 전처리 (모델 입력 형태로 변환)
            # 랜드마크 (x, y, z) 좌표 추출 -> 1차원 리스트
            lm_list = []
            for lm in hand_landmarks.landmark:
                lm_list.append(lm.x)
                lm_list.append(lm.y)
                lm_list.append(lm.z) # z좌표까지 사용하면 방향 변화에 더 강인함
            
            # 모델 예측
            input_data = np.array([lm_list])
            pred_prob = model.predict(input_data, verbose=0)
            pred_idx = np.argmax(pred_prob)
            current_prediction = CLASSES[pred_idx]
            
            # 버퍼에 추가
            prediction_buffer.append(current_prediction)

    else:
        # 손이 감지되지 않으면 버퍼를 비우거나 특정 값으로 채움
        prediction_buffer.append("None")

    # 3. 안정화 및 출력 로직 (가장 중요한 부분)
    # 버퍼가 꽉 찼을 때 분석
    if len(prediction_buffer) == prediction_buffer.maxlen:
        # 버퍼에서 가장 많이 등장한 값 찾기
        most_common = max(set(prediction_buffer), key=prediction_buffer.count)
        count = prediction_buffer.count(most_common)

        # 조건 1: 데이터가 안정적인가? (임계값 이상 동일한 값)
        # 조건 2: 손이 없는 상태('None')가 아닌가?
        if count >= STABILITY_THRESHOLD and most_common != "None":
            stable_gesture = most_common
            
            # 조건 3: 아직 출력하지 않은 새로운 안정 제스처인가?
            if not output_triggered:
                print(f"인식됨: {stable_gesture}")
                f.write(f"{stable_gesture}\n") # 파일 쓰기
                output_triggered = True # 출력 잠금 (LOCK)
            
            # (옵션) 만약 같은 숫자가 계속 유지되면 아무것도 안 함 (LOCK 상태 유지)
            
        else:
            # 손이 사라지거나, 제스처가 바뀌는 과도기(불안정) 상태
            # 여기서 LOCK을 풀어주어 다음 제스처(혹은 동일한 1->1)를 인식할 준비를 함
            if most_common == "None" or count < (STABILITY_THRESHOLD - 2):
                output_triggered = False 

    # 화면 디스플레이 (옵션)
    cv2.imshow('ASL Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
f.close()
cv2.destroyAllWindows()