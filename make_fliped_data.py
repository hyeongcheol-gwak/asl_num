import csv

input_file = 'asl_dataset.csv'        # 원본 데이터 파일 (오른손)
output_file = 'dataset_combined.csv'  # 결과 파일 (오른손 + 왼손)

def create_left_hand_data():
    with open(input_file, 'r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader) # 헤더 읽기
        
        data_list = []
        
        # 원본 데이터 읽기
        for row in reader:
            # 1. 원본 데이터 (오른손) 그대로 추가
            data_list.append(row)
            
            # 2. 반전 데이터 (가상의 왼손) 생성 및 추가
            original_label = row[0]
            coords = row[1:] # 라벨 제외한 좌표값들
            
            flipped_coords = []
            # 좌표는 (x, y, z) 순서로 21개 랜드마크가 나열되어 있음
            # len(coords)는 63 (21 * 3)
            for i in range(0, len(coords), 3):
                x = float(coords[i])
                y = float(coords[i+1])
                z = float(coords[i+2])
                
                # 핵심: x 좌표 반전 (1.0 - x)
                new_x = 1.0 - x
                
                # y, z는 그대로 유지 (단, 문자열 포맷을 맞추기 위해 변환)
                flipped_coords.extend([new_x, y, z])
            
            # [라벨] + [반전된 좌표들]
            flipped_row = [original_label] + flipped_coords
            data_list.append(flipped_row)

    # 결과를 새 파일에 저장
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header) # 헤더 쓰기
        writer.writerows(data_list) # 데이터 쓰기

    print(f"변환 완료! '{output_file}'에 총 {len(data_list)}개의 데이터가 저장되었습니다.")
    print(f"(원본 {len(data_list)//2}개 + 반전 {len(data_list)//2}개)")

if __name__ == "__main__":
    try:
        create_left_hand_data()
    except FileNotFoundError:
        print(f"오류: '{input_file}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")