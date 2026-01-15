"""
MLP 모델을 사용하여 테스트 영상을 평가하는 스크립트
- mlp.pth 모델을 로드하여 영상의 손동작을 인식합니다.
- 프레임 단위로 이미지 변경을 감지하고, 변경 시에만 제스처를 인식합니다.
- 예측 결과를 prediction.txt에 저장하고 정확도를 계산합니다.
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os

# 설정
INPUT_VIDEO_PATH = "test_video.mp4"
OUTPUT_TXT_PATH = "prediction.txt"
GROUND_TRUTH_PATH = "ground_truth.txt"
LABEL_ENCODER_PATH = "../train/label_encoder.pkl"
MODEL_PATH = "../train/mlp.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 변경 감지 설정
IMAGE_CHANGE_THRESHOLD = 0.01  # 프레임 간 이미지 차이 임계값 (0~1)
# 같은 이미지 그룹 내 노이즈: ~0.001-0.002, 다른 이미지 전환: ~0.027
# 기준 프레임 방식이므로 0.01이 적절 (노이즈는 스킵, 실제 변경은 감지)

# ==========================================
# 1. MLP 모델 정의
# ==========================================

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.model(x)

# ==========================================
# 2. 특징 추출 함수
# ==========================================

def extract_features_mlp(landmarks):
    """MLP용 81차원 특징 추출 (normalized coords 63 + angles 14 + dists 4)"""
    lm_flat = landmarks.reshape(-1, 3)

    # 1. Angles (14)
    fingers = [
        [1, 2, 3], [2, 3, 4],
        [0, 5, 6], [5, 6, 7], [6, 7, 8],
        [0, 9, 10], [9, 10, 11], [10, 11, 12],
        [0, 13, 14], [13, 14, 15], [14, 15, 16],
        [0, 17, 18], [17, 18, 19], [18, 19, 20],
    ]
    angles = []
    for f in fingers:
        v1 = lm_flat[f[0]] - lm_flat[f[1]]
        v2 = lm_flat[f[2]] - lm_flat[f[1]]
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        angle = np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0))) / 180.0
        angles.append(angle)

    # 2. Distances (4) - thumb tip to other fingertips
    thumb_tip = lm_flat[4]
    dists = []
    for t in [8, 12, 16, 20]:
        dists.append(np.linalg.norm(thumb_tip - lm_flat[t]))

    return np.concatenate([lm_flat.flatten(), angles, dists])

def preprocess_input(landmarks):
    """랜드마크를 MLP 입력 형식으로 전처리"""
    landmarks = np.array(landmarks)
    wrist = landmarks[0]

    # 손목 기준 정규화
    relative = landmarks - wrist
    max_val = np.max(np.linalg.norm(relative, axis=1))
    if max_val > 0:
        normalized = relative / max_val
    else:
        normalized = relative

    # 특징 추출
    features = extract_features_mlp(normalized)
    return torch.FloatTensor(features).unsqueeze(0).to(DEVICE), normalized

def calculate_frame_difference(frame1, frame2):
    """두 프레임 간의 차이를 계산 (0~1 범위)"""
    if frame1 is None or frame2 is None:
        return 1.0  # 최대 차이로 간주
    
    # 그레이스케일로 변환하여 비교
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2
    
    # 프레임 크기 맞추기
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
    
    # 정규화된 픽셀 차이 계산
    diff = cv2.absdiff(gray1, gray2)
    mean_diff = np.mean(diff) / 255.0  # 0~1 범위로 정규화
    
    return mean_diff

def is_new_image_group(curr_frame, reference_frames, threshold=IMAGE_CHANGE_THRESHOLD, debug=False):
    """현재 프레임이 기존 이미지 그룹에 속하는지 판단"""
    if len(reference_frames) == 0:
        return True  # 첫 프레임은 항상 새로운 이미지 그룹
    
    # 모든 기준 프레임들과 비교하여 가장 비슷한 것 찾기
    min_diff = float('inf')
    for ref_frame in reference_frames:
        diff = calculate_frame_difference(ref_frame, curr_frame)
        if diff < min_diff:
            min_diff = diff
    
    if debug:
        print(f"  프레임 차이 (최소): {min_diff:.6f}, 임계값: {threshold}, 새로운 이미지: {min_diff > threshold}")
    
    # 가장 비슷한 기준 프레임과의 차이도 임계값보다 크면 새로운 이미지 그룹
    return min_diff > threshold

# ==========================================
# 3. 메인 함수
# ==========================================

def load_ground_truth():
    """정답 파일 로드"""
    if not os.path.exists(GROUND_TRUTH_PATH):
        return []
    
    with open(GROUND_TRUTH_PATH, 'r') as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    return labels

def main():
    print("=" * 60)
    print("ASL 영상 평가 프로그램 (MLP 모델)")
    print("=" * 60)

    # Label Encoder 로드
    try:
        with open(LABEL_ENCODER_PATH, "rb") as f:
            le = pickle.load(f)
        num_classes = len(le.classes_)
        print(f"✓ Label Encoder 로드 완료 (클래스 수: {num_classes})")
    except Exception as e:
        print(f"오류: Label Encoder 로드 실패: {e}")
        return

    # 모델 로드
    if not os.path.exists(MODEL_PATH):
        print(f"오류: 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return

    model = MLP(81, num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✓ MLP 모델 로드 완료: {MODEL_PATH}")

    # 영상 로드
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"오류: 영상 파일을 찾을 수 없습니다: {INPUT_VIDEO_PATH}")
        print("먼저 2_create_video.py를 실행하여 영상을 생성해주세요.")
        return

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"오류: 영상을 열 수 없습니다: {INPUT_VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"✓ 영상 로드 완료: {INPUT_VIDEO_PATH} (총 {total_frames}프레임)")

    # MediaPipe 초기화
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    # 제스처 감지 변수
    detected_gestures = []
    reference_frames = []  # 각 이미지 그룹의 기준 프레임들 (첫 프레임들)
    frame_count = 0

    print("\n영상 처리 중...")
    print("-" * 60)

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 모든 기준 프레임들과 비교하여 새로운 이미지 그룹인지 확인
            is_new_group = is_new_image_group(frame, reference_frames, debug=(frame_count <= 20))
            
            if is_new_group:
                # MediaPipe 처리
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                hand_detected = False
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # 랜드마크 추출
                        lm_list = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

                        try:
                            # 전처리 및 예측
                            features, normalized_lm = preprocess_input(lm_list)
                            outputs = model(features)
                            probs = F.softmax(outputs, dim=1)
                            max_prob, idx = torch.max(probs, 1)

                            current_prediction = le.inverse_transform([idx.item()])[0]
                            
                            # 제스처 감지 및 저장
                            detected_gestures.append(current_prediction)
                            print(f"✓ 제스처 감지: {current_prediction} (프레임 {frame_count})")
                            hand_detected = True

                        except Exception as e:
                            print(f"프레임 {frame_count} 처리 중 오류: {e}")
                else:
                    # 손이 감지되지 않은 경우
                    print(f"  프레임 {frame_count}: 손이 감지되지 않음")
                
                # 새로운 이미지 그룹의 기준 프레임으로 추가
                reference_frames.append(frame.copy())
            else:
                # 기존 이미지 그룹에 속함 - 스킵
                if frame_count <= 20:
                    print(f"  프레임 {frame_count}: 기존 이미지 그룹에 속함 (스킵)")

            # 진행 상황 표시 (매 100프레임마다)
            if frame_count % 100 == 0:
                print(f"  처리 중... {frame_count}/{total_frames} 프레임 (기준 프레임 수: {len(reference_frames)})")

    cap.release()
    hands.close()

    print("-" * 60)
    print(f"✓ 영상 처리 완료 (총 {frame_count}프레임)")
    print(f"✓ 감지된 제스처 수: {len(detected_gestures)}개")
    print(f"  제스처: {detected_gestures}")

    # 예측 결과 저장
    with open(OUTPUT_TXT_PATH, 'w') as f:
        for gesture in detected_gestures:
            f.write(f"{gesture}\n")
    
    print(f"✓ 예측 결과 저장 완료: {OUTPUT_TXT_PATH}")

    # 정답과 비교
    ground_truth = load_ground_truth()
    
    if ground_truth:
        print("\n" + "=" * 60)
        print("정확도 평가")
        print("=" * 60)
        print(f"정답 (Ground Truth): {ground_truth}")
        print(f"예측 (Prediction):   {detected_gestures}")
        
        # 정확도 계산
        min_len = min(len(ground_truth), len(detected_gestures))
        correct = sum(1 for i in range(min_len) if ground_truth[i] == detected_gestures[i])
        
        accuracy = (correct / len(ground_truth)) * 100 if ground_truth else 0
        
        print(f"\n정확도: {correct}/{len(ground_truth)} = {accuracy:.2f}%")
        
        # 세부 비교
        if len(ground_truth) != len(detected_gestures):
            print(f"\n⚠ 경고: 제스처 개수 불일치 (정답: {len(ground_truth)}, 예측: {len(detected_gestures)})")
        
        print("\n상세 비교:")
        max_len = max(len(ground_truth), len(detected_gestures))
        for i in range(max_len):
            gt = ground_truth[i] if i < len(ground_truth) else "N/A"
            pred = detected_gestures[i] if i < len(detected_gestures) else "N/A"
            match = "✓" if gt == pred else "✗"
            print(f"  {i+1}. 정답: {gt}, 예측: {pred} {match}")
    
    print("\n" + "=" * 60)
    print("평가 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()
