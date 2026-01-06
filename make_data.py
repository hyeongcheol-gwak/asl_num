import cv2
import mediapipe as mp
import numpy as np
import csv
import os  # [추가됨] 파일 존재 여부 확인을 위해 필요

file_path = 'dataset_temp.csv'

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

header = ['label']
for i in range(21):
    header.extend([f'x{i}', f'y{i}', f'z{i}'])

# [수정됨] 파일이 존재하지 않을 때만 헤더를 작성합니다.
if not os.path.exists(file_path):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

cap = cv2.VideoCapture(0) # 웹캠 번호 확인 (0 또는 1)

print("0~9 키를 눌러 데이터를 저장하세요.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 화면 좌우 반전
    frame = cv2.flip(frame, 1)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 현재 프레임에서 저장할 랜드마크 데이터 임시 저장용 변수
    current_row = None 

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 랜드마크 데이터 추출
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            
            # 손이 감지되었으므로 데이터를 current_row에 담아둠
            current_row = row

    cv2.imshow('Data Collection', image)

    # 2. 키 입력을 여기서 딱 한 번만 받음
    key = cv2.waitKey(1)

    # ESC 키 (종료)
    if key == 27:
        break

    # 숫자 키 (0~9) 입력 AND 손 데이터가 존재할 때만 저장
    if 48 <= key <= 57:
        if current_row is not None:
            label = key - 48
            # 여기는 이미 'a' (append) 모드이므로 이어쓰기가 잘 됩니다.
            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([label] + current_row)
            print(f"Label {label} 저장됨")
        else:
            print("손이 감지되지 않아 저장할 수 없습니다.")

input_file = 'dataset_temp.csv'        # 원본 데이터 파일 (오른손)
output_file = 'dataset.csv'  # 결과 파일 (오른손 + 왼손)

def create_left_hand_data():
    with open(input_file, 'r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader) # 헤더 읽기
        
        data_list = []
        
        # 원본 데이터 읽기
        for row in reader:
            # 1. 원본 데이터 (오른손) 그대로 추가
            data_list.append(row)
            
            # 2. 반전 데이터 (가상의 왼손) 생성 및 추가
            original_label = row[0]
            coords = row[1:] # 라벨 제외한 좌표값들
            
            flipped_coords = []
            # 좌표는 (x, y, z) 순서로 21개 랜드마크가 나열되어 있음
            # len(coords)는 63 (21 * 3)
            for i in range(0, len(coords), 3):
                x = float(coords[i])
                y = float(coords[i+1])
                z = float(coords[i+2])
                
                # 핵심: x 좌표 반전 (1.0 - x)
                new_x = 1.0 - x
                
                # y, z는 그대로 유지 (단, 문자열 포맷을 맞추기 위해 변환)
                flipped_coords.extend([new_x, y, z])
            
            # [라벨] + [반전된 좌표들]
            flipped_row = [original_label] + flipped_coords
            data_list.append(flipped_row)

    # 결과를 새 파일에 저장
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header) # 헤더 쓰기
        writer.writerows(data_list) # 데이터 쓰기

    print(f"변환 완료! '{output_file}'에 총 {len(data_list)}개의 데이터가 저장되었습니다.")
    print(f"(원본 {len(data_list)//2}개 + 반전 {len(data_list)//2}개)")

create_left_hand_data()

cap.release()
cv2.destroyAllWindows()