import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
from collections import Counter

# ==========================================
# 1. 설정 (Paths)
# ==========================================
EXTRACTED_DIR = 'extracted_images'   # 첫 번째 코드가 이미지를 저장한 폴더
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
# 3. 메인 평가 로직
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

    # 2. 이미지 파일 목록 가져오기 (이름순 정렬)
    img_files = sorted([f for f in os.listdir(EXTRACTED_DIR) if f.endswith(('.png', '.jpg'))])
    if not img_files:
        print(f"오류: {EXTRACTED_DIR} 폴더에 이미지가 없습니다.")
        return

    # 3. MediaPipe 초기화
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    detected_gestures = []
    print(f"총 {len(img_files)}개의 이미지를 분석합니다 (다수결 횟수: {VOTE_COUNT}회)...")

    for img_name in img_files:
        img_path = os.path.join(EXTRACTED_DIR, img_name)
        frame = cv2.imread(img_path)
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
                    idx = torch.max(outputs, 1)[1].item()
                    votes.append(le.inverse_transform([idx])[0])
            
            # 가장 많이 나온 라벨 선택
            final_prediction = Counter(votes).most_common(1)[0][0]
            detected_gestures.append(final_prediction)
            print(f"[{img_name}] 예측 결과: {final_prediction} (투표 분포: {Counter(votes)})")
            # -----------------------
        else:
            print(f"[{img_name}] 손을 감지하지 못했습니다.")
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