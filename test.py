import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

INPUT_VIDEO_PATH = 'test_video.mp4'
OUTPUT_TXT_PATH = 'result.txt'
MODEL_PATH = 'asl_model.h5'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

try:
    model = load_model(MODEL_PATH)
except:
    print("모델 파일이 없습니다. 먼저 모델을 학습시켜주세요.")
    exit()

CLASSES = {
    0: '1', 1: '2', 2: '3', 3: '4', 4: '5',
    5: '6', 6: '7', 7: '8', 8: '9', 9: '10'
}

prediction_buffer = deque(maxlen=10)
STABILITY_THRESHOLD = 8
output_triggered = False

f = open(OUTPUT_TXT_PATH, 'w')

cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    current_prediction = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for lm in hand_landmarks.landmark:
                lm_list.append(lm.x)
                lm_list.append(lm.y)
                lm_list.append(lm.z)
            
            input_data = np.array([lm_list])
            pred_prob = model.predict(input_data, verbose=0)
            pred_idx = np.argmax(pred_prob)
            current_prediction = CLASSES[pred_idx]
            
            prediction_buffer.append(current_prediction)

    else:
        prediction_buffer.append("None")

    if len(prediction_buffer) == prediction_buffer.maxlen:
        most_common = max(set(prediction_buffer), key=prediction_buffer.count)
        count = prediction_buffer.count(most_common)

        if count >= STABILITY_THRESHOLD and most_common != "None":
            stable_gesture = most_common
            
            if not output_triggered:
                print(f"인식됨: {stable_gesture}")
                f.write(f"{stable_gesture}\n")
                output_triggered = True
            
        else:
            if most_common == "None" or count < (STABILITY_THRESHOLD - 2):
                output_triggered = False 

    cv2.imshow('ASL Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
f.close()
cv2.destroyAllWindows()
