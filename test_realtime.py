import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

MODEL_PATH = 'mlp.h5'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
THRESHOLD = 0.85

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    print(f"모델 로드 성공: {MODEL_PATH}")
except FileNotFoundError:
    print("오류: 모델 파일이나 라벨 인코더를 찾을 수 없습니다.")
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

def preprocess_input(landmarks):
    landmarks = np.array(landmarks)
    wrist = landmarks[0, :]
    relative = landmarks - wrist
    max_val = np.max(np.abs(relative))
    normalized = relative / max_val if max_val > 0 else relative
    
    features = extract_features(normalized)
    combined = np.concatenate([normalized.flatten(), features])
    
    return combined.reshape(1, -1)

def predict_gesture(processed_input, model, le, threshold=0.85):
    prediction = model.predict(processed_input, verbose=0)
    max_prob = np.max(prediction)
    predicted_index = np.argmax(prediction)
    
    if max_prob < threshold:
        return "Unknown", max_prob
    
    label = le.inverse_transform([predicted_index])[0]
    return label, max_prob

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

print("\n--- 실시간 인식 시작 (종료: 'q') ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    display_text = "Waiting..."
    color = (200, 200, 200)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            h, w, _ = frame.shape
            landmark_list = [[lm.x * w, lm.y * h, lm.z] for lm in hand_landmarks.landmark]
            
            try:
                input_data = preprocess_input(landmark_list)
                
                label, conf = predict_gesture(input_data, model, le, threshold=THRESHOLD)
                
                if label == "Unknown":
                    display_text = f"Unknown ({conf*100:.1f}%)"
                    color = (0, 0, 255)
                else:
                    display_text = f"{label} ({conf*100:.1f}%)"
                    color = (0, 255, 0)
                    
            except Exception as e:
                print(f"Error: {e}")

    cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, color, 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
