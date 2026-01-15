import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
from collections import Counter
from skimage.metrics import structural_similarity as ssim

# ==========================================
# 1. 설정 (Paths)
# ==========================================
OUTPUT_TXT_PATH = "prediction.txt"
GROUND_TRUTH_PATH = "ground_truth.txt"
LABEL_ENCODER_PATH = "../train/label_encoder.pkl"
MODEL_PATH = "../train/mlp.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOTE_COUNT = 10  # 한 이미지당 추론 횟수

# ==========================================
# 2. 모델 및 전처리 함수 (기존 코드 유지)
# ==========================================
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
    def forward(self, x): return self.model(x)

def extract_features_mlp(landmarks):
    lm_flat = landmarks.reshape(-1, 3)
    fingers = [[1, 2, 3], [2, 3, 4], [0, 5, 6], [5, 6, 7], [6, 7, 8], [0, 9, 10], [9, 10, 11], [10, 11, 12], [0, 13, 14], [13, 14, 15], [14, 15, 16], [0, 17, 18], [17, 18, 19], [18, 19, 20]]
    angles = []
    for f in fingers:
        v1, v2 = lm_flat[f[0]] - lm_flat[f[1]], lm_flat[f[2]] - lm_flat[f[1]]
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        angles.append(np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0))) / 180.0)
    thumb_tip = lm_flat[4]
    dists = [np.linalg.norm(thumb_tip - lm_flat[t]) for t in [8, 12, 16, 20]]
    return np.concatenate([lm_flat.flatten(), angles, dists])

def preprocess_input(landmarks):
    landmarks = np.array(landmarks)
    relative = landmarks - landmarks[0]
    max_val = np.max(np.linalg.norm(relative, axis=1))
    normalized = relative / max_val if max_val > 0 else relative
    features = extract_features_mlp(normalized)
    return torch.FloatTensor(features).unsqueeze(0).to(DEVICE)

# ==========================================
# 3. 비디오에서 프레임 추출 (메모리에 저장)
# ==========================================
def extract_frames_from_video(video_path, threshold=0.95):
    """
    비디오에서 정지 장면을 감지하여 프레임 리스트를 반환합니다.
    디스크에 저장하지 않고 메모리에 보관합니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("영상을 열 수 없습니다.")
        return []

    prev_frame = None
    frame_idx = 0
    extracted_frames = []
    
    # 현재 감지된 '정지 장면'의 프레임들을 담는 리스트
    current_scene_frames = []

    print("분석 시작... (정밀 분석을 위해 시간이 다소 소요될 수 있습니다)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 처리 속도와 정확도 균형을 위해 그레이스케일 변환
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            # 두 프레임 간의 구조적 유사도(SSIM) 계산
            # 1.0에 가까울수록 동일한 이미지임
            score, _ = ssim(prev_frame, gray_frame, full=True)

            if score < threshold:
                # 유사도가 임계값보다 낮으면 새로운 이미지가 시작된 것으로 간주
                if current_scene_frames:
                    # 이전 장면의 중간 프레임을 저장 (가장 안정적인 프레임)
                    mid_idx = len(current_scene_frames) // 2
                    best_frame = current_scene_frames[mid_idx]
                    extracted_frames.append(best_frame)
                    print(f"프레임 추출됨: {len(extracted_frames)}번째 (구간 프레임 수: {len(current_scene_frames)})")
                    current_scene_frames = []
            
            current_scene_frames.append(frame)
        else:
            current_scene_frames.append(frame)

        prev_frame = gray_frame
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"{frame_idx} 프레임 분석 중...")

    # 마지막 장면 처리
    if current_scene_frames:
        mid_idx = len(current_scene_frames) // 2
        extracted_frames.append(current_scene_frames[mid_idx])

    cap.release()
    print(f"완료! 총 {len(extracted_frames)}개의 프레임을 추출했습니다.")
    return extracted_frames

# ==========================================
# 4. 메인 평가 로직 (통합)
# ==========================================
def main():
    # 1. 모델 및 라벨 엔코더 로드
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    
    model = MLP(81, len(le.classes_)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    # 다수결 시 변동성을 주기 위해 Dropout을 활성화하려면 model.train()을 쓸 수 있으나, 
    # 일반적인 평가를 위해 eval() 모드를 사용합니다.
    model.eval() 

    # 2. 비디오에서 프레임 추출 (메모리에 저장)
    video_file = 'test_video.mp4'  # 영상 파일 경로
    extracted_frames = extract_frames_from_video(video_file, threshold=0.90)
    
    if not extracted_frames:
        print("오류: 프레임을 추출할 수 없습니다.")
        return

    # 3. MediaPipe 초기화
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    detected_gestures = []
    print(f"총 {len(extracted_frames)}개의 프레임을 분석합니다 (다수결 횟수: {VOTE_COUNT}회)...")

    for idx, frame in enumerate(extracted_frames):
        # 프레임을 RGB로 변환
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            lm_list = [[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]
            features = preprocess_input(lm_list)

            # --- 다수결 투표 시작 ---
            votes = []
            with torch.no_grad():
                for _ in range(VOTE_COUNT):
                    outputs = model(features)
                    idx_pred = torch.max(outputs, 1)[1].item()
                    votes.append(le.inverse_transform([idx_pred])[0])
            
            # 가장 많이 나온 라벨 선택
            final_prediction = Counter(votes).most_common(1)[0][0]
            detected_gestures.append(final_prediction)
            print(f"[프레임 {idx:03d}] 예측 결과: {final_prediction} (투표 분포: {Counter(votes)})")
            # -----------------------
        else:
            print(f"[프레임 {idx:03d}] 손을 감지하지 못했습니다.")
            detected_gestures.append("Unknown")

    # 4. 결과 저장 및 Ground Truth 비교
    with open(OUTPUT_TXT_PATH, 'w') as f:
        for res in detected_gestures:
            f.write(f"{res}\n")

    if os.path.exists(GROUND_TRUTH_PATH):
        with open(GROUND_TRUTH_PATH, 'r') as f:
            gt = [line.strip() for line in f if line.strip()]
        
        print("\n" + "="*30)
        print("정확도 분석 결과")
        print("="*30)
        
        correct = 0
        for i in range(min(len(gt), len(detected_gestures))):
            match = "✓" if str(gt[i]) == str(detected_gestures[i]) else "✗"
            if match == "✓": correct += 1
            print(f"이미지 {i:03d}: 정답({gt[i]}) | 예측({detected_gestures[i]}) {match}")
        
        accuracy = (correct / len(gt)) * 100 if gt else 0
        print(f"\n최종 정확도: {accuracy:.2f}% ({correct}/{len(gt)})")
    
    hands.close()

if __name__ == "__main__":
    main()
