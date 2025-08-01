import cv2
import numpy as np
import os


def generate_single_marker(dictionary, marker_id, marker_size=200, border_bits=1, filename=None):
    """
    단일 ArUco 마커를 생성하는 함수
    
    Args:
        dictionary: ArUco 딕셔너리
        marker_id: 마커 ID (0부터 시작)
        marker_size: 마커 이미지 크기 (픽셀)
        border_bits: 테두리 크기 (비트 단위)
        filename: 저장할 파일명 (None이면 자동 생성)
    
    Returns:
        marker_image: 생성된 마커 이미지
    """
    # 마커 이미지 생성
    marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size, border_bits)
    
    # 파일명 설정
    if filename is None:
        filename = f'marker_{marker_id}.png'
    
    # 이미지 저장
    cv2.imwrite(filename, marker_image)
    print(f"마커 ID {marker_id} 생성됨: {filename}")
    
    return marker_image


def generate_multiple_markers(dictionary_type=cv2.aruco.DICT_5X5_100, 
                            marker_ids=[0, 1, 2, 3, 4], 
                            marker_size=200, 
                            border_bits=1,
                            output_dir="markers"):
    """
    여러 ArUco 마커를 한번에 생성하는 함수
    
    Args:
        dictionary_type: ArUco 딕셔너리 타입
        marker_ids: 생성할 마커 ID 리스트
        marker_size: 마커 이미지 크기 (픽셀)
        border_bits: 테두리 크기 (비트 단위)
        output_dir: 출력 폴더명
    """
    # 출력 폴더 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ArUco 딕셔너리 설정
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
    
    print(f"ArUco 마커 생성 시작...")
    print(f"딕셔너리 타입: {dictionary_type}")
    print(f"마커 크기: {marker_size}x{marker_size} 픽셀")
    print(f"테두리 크기: {border_bits} 비트")
    print(f"출력 폴더: {output_dir}")
    print("-" * 50)
    
    generated_markers = []
    
    for marker_id in marker_ids:
        try:
            filename = os.path.join(output_dir, f'marker_{marker_id}.png')
            marker_image = generate_single_marker(aruco_dict, marker_id, marker_size, border_bits, filename)
            generated_markers.append((marker_id, filename, marker_image))
        except Exception as e:
            print(f"마커 ID {marker_id} 생성 실패: {e}")
    
    print("-" * 50)
    print(f"총 {len(generated_markers)}개 마커 생성 완료!")
    
    return generated_markers


def generate_marker_board(dictionary_type=cv2.aruco.DICT_5X5_100,
                         marker_ids=[0, 1, 2, 3],
                         markers_x=2, markers_y=2,
                         marker_length=100, marker_separation=20,
                         filename="marker_board.png"):
    """
    여러 마커를 하나의 보드에 배치하여 생성하는 함수
    
    Args:
        dictionary_type: ArUco 딕셔너리 타입
        marker_ids: 사용할 마커 ID 리스트
        markers_x: 가로 마커 개수
        markers_y: 세로 마커 개수
        marker_length: 각 마커의 크기 (픽셀)
        marker_separation: 마커 간 간격 (픽셀)
        filename: 저장할 파일명
    """
    # ArUco 딕셔너리 설정
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
    
    try:
        # OpenCV 4.12+ 버전용 (numpy array로 변환)
        marker_ids_array = np.array(marker_ids, dtype=np.int32)
        board = cv2.aruco.GridBoard((markers_x, markers_y), marker_length, marker_separation, aruco_dict, marker_ids_array)
    except:
        try:
            # 이전 버전 호환성
            board = cv2.aruco.GridBoard_create(markers_x, markers_y, marker_length, marker_separation, aruco_dict, marker_ids[0])
        except:
            # 수동으로 보드 이미지 생성
            print("GridBoard 함수를 사용할 수 없어 수동으로 보드를 생성합니다...")
            return generate_manual_board(aruco_dict, marker_ids, markers_x, markers_y, marker_length, marker_separation, filename)
    
    # 보드 이미지 크기 계산
    board_width = markers_x * marker_length + (markers_x - 1) * marker_separation + 2 * marker_separation
    board_height = markers_y * marker_length + (markers_y - 1) * marker_separation + 2 * marker_separation
    
    # 보드 이미지 생성
    board_image = board.generateImage((board_width, board_height))
    
    # 이미지 저장
    cv2.imwrite(filename, board_image)
    print(f"마커 보드 생성됨: {filename} ({board_width}x{board_height} 픽셀)")
    
    return board_image


def generate_manual_board(aruco_dict, marker_ids, markers_x, markers_y, marker_length, marker_separation, filename):
    """
    수동으로 마커 보드를 생성하는 함수 (호환성 문제 해결용)
    """
    # 보드 이미지 크기 계산
    board_width = markers_x * marker_length + (markers_x - 1) * marker_separation + 2 * marker_separation
    board_height = markers_y * marker_length + (markers_y - 1) * marker_separation + 2 * marker_separation
    
    # 빈 보드 이미지 생성 (흰색 배경)
    board_image = np.ones((board_height, board_width), dtype=np.uint8) * 255
    
    # 각 위치에 마커 배치
    marker_idx = 0
    for row in range(markers_y):
        for col in range(markers_x):
            if marker_idx < len(marker_ids):
                # 마커 생성
                marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_ids[marker_idx], marker_length)
                
                # 마커 위치 계산
                x_pos = marker_separation + col * (marker_length + marker_separation)
                y_pos = marker_separation + row * (marker_length + marker_separation)
                
                # 마커를 보드에 배치
                board_image[y_pos:y_pos+marker_length, x_pos:x_pos+marker_length] = marker_img
                
                marker_idx += 1
    
    # 이미지 저장
    cv2.imwrite(filename, board_image)
    print(f"수동 마커 보드 생성됨: {filename} ({board_width}x{board_height} 픽셀)")
    
    return board_image


def display_markers(marker_list, window_name="Generated Markers"):
    """생성된 마커들을 화면에 표시하는 함수"""
    if not marker_list:
        print("표시할 마커가 없습니다.")
        return
    
    # 마커들을 가로로 나열
    combined_image = np.hstack([marker[2] for marker in marker_list[:4]])  # 최대 4개만 표시
    
    cv2.imshow(window_name, combined_image)
    print("마커 이미지가 표시되었습니다. 아무 키나 누르면 닫힙니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    print("=== ArUco 마커 생성기 ===")
    
    # 사용 가능한 딕셔너리 타입들
    dictionary_options = {
        1: (cv2.aruco.DICT_4X4_50, "DICT_4X4_50 (50개 마커)"),
        2: (cv2.aruco.DICT_4X4_100, "DICT_4X4_100 (100개 마커)"),
        3: (cv2.aruco.DICT_4X4_250, "DICT_4X4_250 (250개 마커)"),
        4: (cv2.aruco.DICT_5X5_50, "DICT_5X5_50 (50개 마커)"),
        5: (cv2.aruco.DICT_5X5_100, "DICT_5X5_100 (100개 마커)"),
        6: (cv2.aruco.DICT_5X5_250, "DICT_5X5_250 (250개 마커)"),
        7: (cv2.aruco.DICT_6X6_50, "DICT_6X6_50 (50개 마커)"),
        8: (cv2.aruco.DICT_6X6_100, "DICT_6X6_100 (100개 마커)"),
        9: (cv2.aruco.DICT_6X6_250, "DICT_6X6_250 (250개 마커)"),
    }
    
    print("\n사용 가능한 ArUco 딕셔너리:")
    for key, (_, desc) in dictionary_options.items():
        print(f"{key}. {desc}")
    
    # 딕셔너리 선택
    try:
        dict_choice = int(input("\n딕셔너리 번호를 선택하세요 (기본값: 5): ") or 5)
        if dict_choice not in dictionary_options:
            dict_choice = 5
        dictionary_type, dict_name = dictionary_options[dict_choice]
        print(f"선택된 딕셔너리: {dict_name}")
    except ValueError:
        dictionary_type = cv2.aruco.DICT_5X5_100
        print("기본 딕셔너리 사용: DICT_5X5_100")
    
    # 생성 모드 선택
    print("\n생성 모드:")
    print("1. 개별 마커 생성")
    print("2. 마커 보드 생성")
    
    try:
        mode = int(input("모드를 선택하세요 (기본값: 1): ") or 1)
    except ValueError:
        mode = 1
    
    if mode == 1:
        # 개별 마커 생성
        marker_ids_input = input("\n생성할 마커 ID를 입력하세요 (예: 0,1,2,3 또는 0-5): ") or "0,1,2,3"
        
        # ID 파싱
        marker_ids = []
        if '-' in marker_ids_input:
            # 범위 입력 (예: 0-5)
            start, end = map(int, marker_ids_input.split('-'))
            marker_ids = list(range(start, end + 1))
        else:
            # 개별 입력 (예: 0,1,2,3)
            marker_ids = [int(x.strip()) for x in marker_ids_input.split(',')]
        
        print(f"생성할 마커 ID: {marker_ids}")
        
        # 마커 크기 입력
        try:
            marker_size = int(input("마커 크기를 입력하세요 (픽셀, 기본값: 200): ") or 200)
        except ValueError:
            marker_size = 200
        
        # 마커 생성
        generated_markers = generate_multiple_markers(
            dictionary_type=dictionary_type,
            marker_ids=marker_ids,
            marker_size=marker_size
        )
        
        # 생성된 마커 표시 여부
        show_markers = input("\n생성된 마커를 화면에 표시하시겠습니까? (y/n): ").lower() == 'y'
        if show_markers:
            display_markers(generated_markers)
    
    elif mode == 2:
        # 마커 보드 생성
        try:
            markers_x = int(input("가로 마커 개수 (기본값: 2): ") or 2)
            markers_y = int(input("세로 마커 개수 (기본값: 2): ") or 2)
            marker_length = int(input("각 마커 크기 (픽셀, 기본값: 100): ") or 100)
            marker_separation = int(input("마커 간 간격 (픽셀, 기본값: 20): ") or 20)
        except ValueError:
            markers_x, markers_y = 2, 2
            marker_length, marker_separation = 100, 20
        
        # 사용할 마커 ID 생성
        total_markers = markers_x * markers_y
        marker_ids = list(range(total_markers))
        
        # 마커 보드 생성
        board_image = generate_marker_board(
            dictionary_type=dictionary_type,
            marker_ids=marker_ids,
            markers_x=markers_x,
            markers_y=markers_y,
            marker_length=marker_length,
            marker_separation=marker_separation
        )
        
        # 생성된 보드 표시 여부
        show_board = input("생성된 마커 보드를 화면에 표시하시겠습니까? (y/n): ").lower() == 'y'
        if show_board:
            cv2.imshow('Generated Marker Board', board_image)
            print("마커 보드가 표시되었습니다. 아무 키나 누르면 닫힙니다.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    print("\n마커 생성이 완료되었습니다!")
    print("생성된 이미지를 인쇄할 때는 '실제 크기'로 인쇄하세요.")


if __name__ == "__main__":
    main()