"""
수집된 데이터로부터 테스트 영상 생성 스크립트
- collected_data 폴더의 이미지들을 사용하여 영상을 생성합니다.
- 각 이미지는 15프레임씩 반복되어 영상에 포함됩니다.
- 영상과 함께 정답(ground truth) 텍스트 파일도 생성됩니다.
"""

import cv2
import json
import os
from pathlib import Path
import numpy as np

# 설정
INPUT_DIR = "collected_data"
OUTPUT_VIDEO = "test_video.mp4"
OUTPUT_GROUND_TRUTH = "ground_truth.txt"
FRAMES_PER_IMAGE = 15  # 각 이미지를 15프레임 동안 표시
FPS = 30  # 초당 프레임 수
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

def load_data_samples():
    """collected_data 폴더에서 데이터 샘플들을 로드합니다."""
    samples = []
    
    if not os.path.exists(INPUT_DIR):
        print(f"오류: '{INPUT_DIR}' 폴더가 존재하지 않습니다.")
        print("먼저 1_collect_data.py를 실행하여 데이터를 수집해주세요.")
        return samples
    
    # JSON 파일 찾기
    json_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.json')])
    
    if not json_files:
        print(f"오류: '{INPUT_DIR}' 폴더에 데이터가 없습니다.")
        return samples
    
    for json_file in json_files:
        json_path = os.path.join(INPUT_DIR, json_file)
        
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            
            image_file = metadata.get('image_file')
            image_path = os.path.join(INPUT_DIR, image_file)
            
            if not os.path.exists(image_path):
                print(f"경고: 이미지 파일을 찾을 수 없습니다: {image_file}")
                continue
            
            samples.append({
                'image_path': image_path,
                'label': metadata['label'],
                'timestamp': metadata.get('timestamp', ''),
                'landmarks': metadata.get('landmarks', [])
            })
        
        except Exception as e:
            print(f"경고: {json_file} 로드 중 오류: {e}")
            continue
    
    return samples

def create_video(samples):
    """샘플들로부터 영상을 생성합니다."""
    if not samples:
        print("생성할 샘플이 없습니다.")
        return False
    
    # VideoWriter 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))
    
    ground_truth_labels = []
    
    print(f"\n영상 생성 중... (총 {len(samples)}개 샘플)")
    print(f"각 샘플당 {FRAMES_PER_IMAGE}프레임 = 총 {len(samples) * FRAMES_PER_IMAGE}프레임")
    print("-" * 60)
    
    for idx, sample in enumerate(samples):
        # 이미지 로드
        img = cv2.imread(sample['image_path'])
        
        if img is None:
            print(f"경고: 이미지를 읽을 수 없습니다: {sample['image_path']}")
            continue
        
        # 이미지 크기 조정
        img_resized = cv2.resize(img, (VIDEO_WIDTH, VIDEO_HEIGHT))
        
        # 라벨 정보를 이미지에 표시 (선택사항)
        label = sample['label']
        cv2.putText(img_resized, f"Label: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img_resized, f"Sample {idx+1}/{len(samples)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 동일한 프레임을 FRAMES_PER_IMAGE번 반복
        for _ in range(FRAMES_PER_IMAGE):
            out.write(img_resized)
        
        # Ground truth에 라벨 추가
        ground_truth_labels.append(label)
        
        print(f"  샘플 {idx+1}: 라벨={label}, 프레임={FRAMES_PER_IMAGE}개 추가")
    
    out.release()
    
    # Ground truth 텍스트 파일 저장
    with open(OUTPUT_GROUND_TRUTH, 'w') as f:
        for label in ground_truth_labels:
            f.write(f"{label}\n")
    
    print("-" * 60)
    print(f"✓ 영상 생성 완료: {OUTPUT_VIDEO}")
    print(f"✓ 정답 파일 생성 완료: {OUTPUT_GROUND_TRUTH}")
    print(f"  - 총 프레임: {len(samples) * FRAMES_PER_IMAGE}프레임")
    print(f"  - 영상 길이: {len(samples) * FRAMES_PER_IMAGE / FPS:.2f}초")
    print(f"  - 제스처 수: {len(ground_truth_labels)}개")
    
    return True

def main():
    print("=" * 60)
    print("ASL 테스트 영상 생성 프로그램")
    print("=" * 60)
    
    # 데이터 로드
    samples = load_data_samples()
    
    if not samples:
        print("\n데이터 샘플을 찾을 수 없습니다.")
        return
    
    print(f"\n로드된 샘플: {len(samples)}개")
    
    # 라벨 분포 출력
    label_counts = {}
    for sample in samples:
        label = sample['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\n라벨 분포:")
    for label in sorted(label_counts.keys()):
        print(f"  라벨 {label}: {label_counts[label]}개")
    
    # 영상 생성
    success = create_video(samples)
    
    if success:
        print("\n" + "=" * 60)
        print("모든 작업이 완료되었습니다!")
        print("다음 단계: 3_evaluate_video.py를 실행하여 영상을 평가하세요.")
        print("=" * 60)
    else:
        print("\n영상 생성에 실패했습니다.")

if __name__ == "__main__":
    main()
