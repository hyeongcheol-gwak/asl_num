"""
ASL 손동작 데이터 수집 스크립트
- 웹캠으로 손 이미지를 촬영하고 정답 라벨을 입력하여 데이터 쌍을 생성합니다.
- Space 키를 눌러 현재 프레임을 캡처합니다.
- 0-9 키를 눌러 해당 손동작의 라벨을 지정합니다.
- ESC 키를 눌러 종료합니다.
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime

# 설정
OUTPUT_DIR = "collected_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# 웹캠 초기화
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("오류: 웹캠을 열 수 없습니다.")
    exit()

print("=" * 60)
print("ASL 손동작 데이터 수집 프로그램")
print("=" * 60)
print("사용 방법:")
print("  - Space: 현재 프레임 캡처 준비")
print("  - 0-9: 라벨 지정 및 저장")
print("  - ESC: 종료")
print("=" * 60)

frame_count = 0
captured_frame = None
captured_landmarks = None
current_label = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 좌우 반전 (거울 효과)
    frame = cv2.flip(frame, 1)
    
    # RGB로 변환하여 MediaPipe 처리
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    # 화면에 표시할 이미지
    display_frame = frame.copy()
    
    # 손 감지 여부
    hand_detected = False
    landmarks_data = None
    
    if results.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            # 손 랜드마크 그리기
            mp_drawing.draw_landmarks(
                display_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # 랜드마크 데이터 저장
            landmarks_data = []
            for lm in hand_landmarks.landmark:
                landmarks_data.append({
                    'x': float(lm.x),
                    'y': float(lm.y),
                    'z': float(lm.z)
                })
    
    # 상태 표시
    status_text = "Hand Detected" if hand_detected else "No Hand"
    color = (0, 255, 0) if hand_detected else (0, 0, 255)
    cv2.putText(display_frame, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # 캡처 모드 표시
    if captured_frame is not None:
        cv2.putText(display_frame, "Captured! Press 0-9 to label", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # 현재 저장된 데이터 개수 표시
    saved_count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')])
    cv2.putText(display_frame, f"Saved Samples: {saved_count}", (10, display_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Data Collection", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # ESC: 종료
    if key == 27:
        break
    
    # Space: 프레임 캡처
    elif key == 32:  # Space
        if hand_detected and landmarks_data is not None:
            captured_frame = frame.copy()
            captured_landmarks = landmarks_data
            print(f"프레임 캡처됨! 이제 0-9 키를 눌러 라벨을 지정하세요.")
        else:
            print("손이 감지되지 않았습니다. 손을 화면에 표시해주세요.")
    
    # 0-9: 라벨 지정 및 저장
    elif 48 <= key <= 57:  # 0-9
        label = key - 48
        
        if captured_frame is not None and captured_landmarks is not None:
            # 파일명 생성 (타임스탬프 사용)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"label_{label}_{timestamp}"
            
            # 이미지 저장
            image_path = os.path.join(OUTPUT_DIR, f"{filename}.jpg")
            cv2.imwrite(image_path, captured_frame)
            
            # 메타데이터 저장 (JSON)
            metadata = {
                'label': label,
                'landmarks': captured_landmarks,
                'timestamp': timestamp,
                'image_file': f"{filename}.jpg"
            }
            
            metadata_path = os.path.join(OUTPUT_DIR, f"{filename}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            frame_count += 1
            print(f"[저장 완료] 라벨: {label}, 파일: {filename}, 총 {frame_count}개")
            
            # 캡처 초기화
            captured_frame = None
            captured_landmarks = None
        else:
            print("먼저 Space 키를 눌러 프레임을 캡처해주세요.")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print(f"데이터 수집 완료! 총 {frame_count}개의 샘플이 저장되었습니다.")
print(f"저장 위치: {OUTPUT_DIR}")
print("=" * 60)
