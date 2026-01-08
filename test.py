import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import copy
from collections import deque
import os

INPUT_VIDEO_PATH = 'test_video.mp4'
OUTPUT_TXT_PATH = 'result.txt'
MODEL_PATH = 'mlp.pth'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
THRESHOLD = 0.85

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(src2)
        
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src

class TransformerModel(nn.Module):
    def __init__(self, input_dim=3, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, num_classes=10, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Embedding(21, d_model)
        
        encoder_layer = TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
        self.transformer = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout_final = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        x = self.input_projection(x)
        
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb
        
        for layer in self.transformer:
            x = layer(x)
        
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.dropout_final(x)
        x = self.classifier(x)
        
        return x

def detect_model_type(model_path):
    if 'transformer' in model_path.lower():
        return 'transformer'
    elif 'mlp' in model_path.lower():
        return 'mlp'
    else:
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            for key in state_dict.keys():
                if 'transformer' in key or 'pos_embedding' in key:
                    return 'transformer'
            return 'mlp'
        except:
            return 'mlp'

try:
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    
    num_classes = len(le.classes_)
    model_type = detect_model_type(MODEL_PATH)
    
    if model_type == 'transformer':
        model = TransformerModel(input_dim=3, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, num_classes=num_classes, dropout=0.1).to(device)
        print(f"Transformer 모델 로드: {MODEL_PATH}")
    else:
        model = MLP(81, num_classes).to(device)
        print(f"MLP 모델 로드: {MODEL_PATH}")
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"모델 로드 성공: {MODEL_PATH}")
except FileNotFoundError:
    print("오류: 모델 파일이나 라벨 인코더를 찾을 수 없습니다.")
    exit()
except Exception as e:
    print(f"모델 로드 오류: {e}")
    exit()

def get_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0.0
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta)) / 180.0

def extract_features(landmarks):
    features = []
    fingers = [
        [1, 2, 3], [2, 3, 4], [0, 5, 6], [5, 6, 7], [6, 7, 8],
        [0, 9, 10], [9, 10, 11], [10, 11, 12], [0, 13, 14], [13, 14, 15],
        [14, 15, 16], [0, 17, 18], [17, 18, 19], [18, 19, 20]
    ]
    for f in fingers:
        v1 = landmarks[f[0]] - landmarks[f[1]]
        v2 = landmarks[f[2]] - landmarks[f[1]]
        features.append(get_angle(v1, v2))
        
    thumb_tip = landmarks[4]
    for tip_idx in [8, 12, 16, 20]:
        features.append(np.linalg.norm(thumb_tip - landmarks[tip_idx]))
        
    return np.array(features)

def preprocess_input_mlp(landmarks):
    landmarks = np.array(landmarks)
    wrist = landmarks[0, :]
    relative = landmarks - wrist
    max_val = np.max(np.abs(relative))
    normalized = relative / max_val if max_val > 0 else relative
    
    features = extract_features(normalized)
    combined = np.concatenate([normalized.flatten(), features])
    
    return combined.reshape(1, -1)

def preprocess_input_transformer(landmarks):
    landmarks = np.array(landmarks)
    wrist = landmarks[0, :]
    relative_landmarks = landmarks - wrist
    
    max_dist = np.max(np.linalg.norm(relative_landmarks, axis=1))
    if max_dist > 0:
        normalized_landmarks = relative_landmarks / max_dist
    else:
        normalized_landmarks = relative_landmarks
    
    return normalized_landmarks.reshape(1, 21, 3)

def predict_gesture(processed_input, model, le, threshold=0.85, model_type='mlp'):
    model.eval()
    with torch.no_grad():
        if model_type == 'transformer':
            input_tensor = torch.FloatTensor(processed_input).to(device)
        else:
            input_tensor = torch.FloatTensor(processed_input).to(device)
        
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        max_prob, predicted_index = torch.max(probs, 1)
        max_prob = max_prob.item()
        predicted_index = predicted_index.item()
    
    if max_prob < threshold:
        return "Unknown", max_prob
    
    label = le.inverse_transform([predicted_index])[0]
    return label, max_prob

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

prediction_buffer = deque(maxlen=10)
STABILITY_THRESHOLD = 8
output_triggered = False

cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

if not cap.isOpened():
    print(f"오류: 비디오 파일 '{INPUT_VIDEO_PATH}'을 열 수 없습니다.")
    exit()

with open(OUTPUT_TXT_PATH, 'w') as f:
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
            landmark_list = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            
            try:
                if model_type == 'transformer':
                    input_data = preprocess_input_transformer(landmark_list)
                else:
                    input_data = preprocess_input_mlp(landmark_list)
                
                label, conf = predict_gesture(input_data, model, le, threshold=THRESHOLD, model_type=model_type)
                
                if label != "Unknown":
                    current_prediction = label
                    prediction_buffer.append(current_prediction)
                else:
                    prediction_buffer.append("None")
                    
            except Exception as e:
                print(f"Error: {e}")
                prediction_buffer.append("None")

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
                f.flush()
                output_triggered = True
            
        else:
            if most_common == "None" or count < (STABILITY_THRESHOLD - 2):
                output_triggered = False 

        cv2.imshow('ASL Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
