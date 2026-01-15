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
import warnings

# 경고 무시
warnings.filterwarnings("ignore")

# ==========================================
# 1. 설정 (Paths & Config)
# ==========================================
OUTPUT_TXT_PATH = "prediction.txt"
GROUND_TRUTH_PATH = "ground_truth.txt"
LABEL_ENCODER_PATH = "label_encoder.pkl"
MODEL_PATH = "transformer.pth"  # HybridHandModel 가중치 파일
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOTE_COUNT = 20  # 이미지당 다수결 투표 횟수

# ==========================================
# 2. HybridHandModel 정의 (제공된 test.py 기준)
# ==========================================
class HybridHandModel(nn.Module):
    def __init__(self, num_classes=10, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(3, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 22, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.geo_mlp = nn.Sequential(
            nn.Linear(20, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.GELU()
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model + 32, 64), nn.BatchNorm1d(64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, num_classes)
        )

    def forward(self, x_coords, x_geo):
        batch_size = x_coords.size(0)
        x_emb = self.input_projection(x_coords)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat((cls_tokens, x_emb), dim=1)
        x_seq = x_seq + self.pos_embedding[:, : x_seq.size(1), :]
        t_out = self.transformer(x_seq)
        t_feature = t_out[:, 0, :]
        g_feature = self.geo_mlp(x_geo)
        combined = torch.cat((t_feature, g_feature), dim=1)
        return self.fusion_layer(combined)

# ==========================================
# 3. 특징 추출 및 전처리 (제공된 test.py 기준)
# ==========================================
def compute_geometric_features(landmarks):
    if landmarks.ndim == 2:
        landmarks = landmarks[np.newaxis, ...]
    connections = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),
                   (10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
    vecs = landmarks[:, [c[1] for c in connections], :] - landmarks[:, [c[0] for c in connections], :]
    norms = np.linalg.norm(vecs, axis=2) + 1e-8
    
    finger_indices = []
    for f in range(5):
        base = f * 4
        finger_indices.extend([(base, base + 1), (base + 1, base + 2), (base + 2, base + 3)])
    
    v1, v2 = vecs[:, [f[0] for f in finger_indices], :], vecs[:, [f[1] for f in finger_indices], :]
    dot = np.sum(v1 * v2, axis=2)
    norm_mul = norms[:, [f[0] for f in finger_indices]] * norms[:, [f[1] for f in finger_indices]]
    angles = np.arccos(np.clip(dot / norm_mul, -1.0, 1.0))
    
    wrist = landmarks[:, 0, :]
    dists = np.linalg.norm(landmarks[:, [4,8,12,16,20], :] - wrist[:, None, :], axis=2)
    return np.concatenate([angles, dists], axis=1)

def preprocess_hybrid(landmarks):
    landmarks = np.array(landmarks)
    relative = landmarks - landmarks[0] # wrist normalization
    max_val = np.max(np.linalg.norm(relative, axis=1))
    normalized = relative / max_val if max_val > 0 else relative
    
    geo = compute_geometric_features(normalized)
    return (
        torch.FloatTensor(normalized).unsqueeze(0).to(DEVICE),
        torch.FloatTensor(geo).to(DEVICE)
    )

# ==========================================
# 4. 비디오에서 프레임 추출 (메모리에 저장)
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

    print("--------------- 분석 시작... ---------------")

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
                    print(f"이미지 추출됨: {len(extracted_frames)}번째 [구간 프레임 수: {len(current_scene_frames)}]")
                    current_scene_frames = []
            
            current_scene_frames.append(frame)
        else:
            current_scene_frames.append(frame)

        prev_frame = gray_frame
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"---------- {frame_idx} 프레임 분석 중... ----------")

    # 마지막 장면 처리
    if current_scene_frames:
        mid_idx = len(current_scene_frames) // 2
        extracted_frames.append(current_scene_frames[mid_idx])

    cap.release()
    print(f"총 {len(extracted_frames)}개의 이미지 추출 완료\n")
    return extracted_frames

# ==========================================
# 5. 메인 평가 로직
# ==========================================
def main():
    # 1. 모델 및 라벨 로드
    try:
        with open(LABEL_ENCODER_PATH, "rb") as f:
            le = pickle.load(f)
        num_classes = len(le.classes_)
        
        model = HybridHandModel(num_classes=num_classes).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print("트랜스포머 모델 로드 완료")
    except Exception as e:
        print(f"로드 실패: {e}")
        return

    # 2. 비디오에서 프레임 추출 (메모리에 저장)
    video_file = 'test_video.mp4'  # 영상 파일 경로
    extracted_frames = extract_frames_from_video(video_file, threshold=0.90)
    
    if not extracted_frames:
        print("오류: 프레임을 추출할 수 없습니다.")
        return

    # 3. MediaPipe 초기화
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

    detected_gestures = []
    print(f"----- 총 {len(extracted_frames)}개의 이미지를 분석합니다 [다수결 횟수: {VOTE_COUNT}회] -----")

    with torch.no_grad():
        for idx, frame in enumerate(extracted_frames):
            # 프레임을 RGB로 변환
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                hl = results.multi_hand_landmarks[0]
                lm_list = [[lm.x, lm.y, lm.z] for lm in hl.landmark]
                
                # Hybrid 입력 전처리
                coords, geo = preprocess_hybrid(lm_list)
                
                # 다수결 투표
                votes = []
                for _ in range(VOTE_COUNT):
                    outputs = model(coords, geo)
                    idx_pred = torch.max(outputs, 1)[1].item()
                    votes.append(le.inverse_transform([idx_pred])[0])
                
                final_pred = Counter(votes).most_common(1)[0][0]
                detected_gestures.append(final_pred)
                print(f"이미지[{idx:03d}] | 예측 결과: {final_pred} | 투표 분포: {Counter(votes)}")
            else:
                print(f"이미지[{idx:03d}] 손 감지 실패")
                detected_gestures.append("None")

    # 4. 결과 저장 및 비교
    with open(OUTPUT_TXT_PATH, "w") as f:
        for g in detected_gestures:
            # Class 0 -> 1 변환이 필요하면 여기서 처리 (test.py logic: pred_label + 1)
            f.write(f"{g}\n")

    if os.path.exists(GROUND_TRUTH_PATH):
        with open(GROUND_TRUTH_PATH, 'r') as f:
            gt = [line.strip() for line in f if line.strip()]
        
        print("\n"+"="*15 + " 정확도 분석 결과 " + "="*15)
        
        correct = 0
        for i in range(min(len(gt), len(detected_gestures))):
            match = "✓" if str(gt[i]) == str(detected_gestures[i]) else "✗"
            if match == "✓": correct += 1
            print(f"이미지[{i:03d}]: 정답[{gt[i]}] | 예측[{detected_gestures[i]}] | 결과: {match}")
        
        acc = (correct / len(gt)) * 100 if gt else 0
        print(f"\n정확도: {acc:.2f}% ({correct}/{len(gt)})")

    hands.close()

if __name__ == "__main__":
    main()
