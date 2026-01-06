import cv2
import mediapipe as mp
import numpy as np
import csv
import os

file_path = 'dataset_temp.csv'

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

header = ['label']
for i in range(21):
    header.extend([f'x{i}', f'y{i}', f'z{i}'])

if not os.path.exists(file_path):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

cap = cv2.VideoCapture(0)

print("0~9 키를 눌러 데이터를 저장하세요.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    current_row = None 

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            
            current_row = row

    cv2.imshow('Data Collection', image)

    key = cv2.waitKey(1)

    if key == 27:
        break

    if 48 <= key <= 57:
        if current_row is not None:
            label = key - 48
            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([label] + current_row)
            print(f"Label {label} 저장됨")
        else:
            print("손이 감지되지 않아 저장할 수 없습니다.")

input_file = 'dataset_temp.csv'
output_file = 'dataset.csv'

def create_left_hand_data():
    with open(input_file, 'r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        
        data_list = []
        
        for row in reader:
            data_list.append(row)
            
            original_label = row[0]
            coords = row[1:]
            
            flipped_coords = []
            for i in range(0, len(coords), 3):
                x = float(coords[i])
                y = float(coords[i+1])
                z = float(coords[i+2])
                
                new_x = 1.0 - x
                
                flipped_coords.extend([new_x, y, z])
            
            flipped_row = [original_label] + flipped_coords
            data_list.append(flipped_row)

    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(data_list)

    print(f"변환 완료! '{output_file}'에 총 {len(data_list)}개의 데이터가 저장되었습니다.")
    print(f"(원본 {len(data_list)//2}개 + 반전 {len(data_list)//2}개)")

create_left_hand_data()

cap.release()
cv2.destroyAllWindows()
