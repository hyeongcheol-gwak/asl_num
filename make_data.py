import cv2
import mediapipe as mp
import numpy as np
import csv

# 저장할 파일 이름
file_path = "asl_dataset.csv"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# CSV 파일 헤더 생성 (라벨 + 21개 랜드마크의 x,y,z 좌표)
header = ["label"]
for i in range(21):
    header.extend([f"x{i}", f"y{i}", f"z{i}"])

# 파일 초기화 (처음 실행 시 헤더 작성)
with open(file_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

cap = cv2.VideoCapture(1)  # 웹캠 실행

print("수집 시작! 0~9 키를 눌러 데이터를 저장하세요.")
print("예: 숫자 1 동작 -> 키보드 '0' 누름 / 숫자 10 동작 -> 키보드 '9' 누름")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 변환 및 전처리
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # 랜드마크 그리기
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 좌표 추출
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            # 키보드 입력 대기 및 저장
            key = cv2.waitKey(1)
            if 48 <= key <= 57:  # 키보드 숫자 0 ~ 9
                label = key - 48  # 0을 누르면 라벨 0, 9를 누르면 라벨 9

                # CSV에 저장 [라벨, x0, y0, z0, ... ]
                with open(file_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([label] + row)
                print(f"Label {label} 저장됨")

    cv2.imshow("Data Collection", image)
    if cv2.waitKey(1) == 27:  # ESC 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
