import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def extract_original_images(video_path, output_dir, threshold=0.95):
    # 출력 폴더 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("영상을 열 수 없습니다.")
        return

    prev_frame = None
    image_count = 0
    frame_idx = 0
    
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
                    
                    save_path = os.path.join(output_dir, f"image_{image_count:03d}.png")
                    cv2.imwrite(save_path, best_frame)
                    print(f"저장됨: {save_path} (구간 프레임 수: {len(current_scene_frames)})")
                    
                    image_count += 1
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
        cv2.imwrite(os.path.join(output_dir, f"image_{image_count:03d}.png"), current_scene_frames[mid_idx])

    cap.release()
    print(f"완료! 총 {image_count + 1}개의 원본 이미지를 추출했습니다.")

# 사용 예시
video_file = 'test_video.mp4'  # 영상 파일 경로
output_folder = 'extracted_images' # 저장할 폴더
extract_original_images(video_file, output_folder, threshold=0.90)