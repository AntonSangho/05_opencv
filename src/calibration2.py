import cv2
import numpy as np
import os
import glob
import pickle

def calibrate_camera():
    # 체커보드의 차원 정의
    CHECKERBOARD = (7,10)  # 체커보드 행과 열당 내부 코너 수
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # 각 체커보드 이미지에 대한 3D 점 벡터를 저장할 벡터 생성
    objpoints = []
    # 각 체커보드 이미지에 대한 2D 점 벡터를 저장할 벡터 생성
    imgpoints = [] 
    
    # 3D 점의 세계 좌표 정의
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    
    # 다양한 이미지 형식과 경로 시도
    image_paths = [
        '../img/*.png',
        '../img/*.jpg',
        '../img/*.jpeg',
        './img/*.png',
        './img/*.jpg',
        './img/*.jpeg',
        'img/*.png',
        'img/*.jpg',
        'img/*.jpeg',
        '*.png',
        '*.jpg',
        '*.jpeg'
    ]
    
    images = []
    for path_pattern in image_paths:
        found_images = glob.glob(path_pattern)
        if found_images:
            images.extend(found_images)
            print(f"이미지 {len(found_images)}개 발견: {path_pattern}")
    
    # 중복 제거
    images = list(set(images))
    
    if not images:
        print("에러: 체커보드 이미지를 찾을 수 없습니다!")
        print("다음 중 하나의 방법을 시도해보세요:")
        print("1. ../img/ 폴더에 체커보드 이미지 파일들을 넣어주세요")
        print("2. ./img/ 폴더에 체커보드 이미지 파일들을 넣어주세요")
        print("3. 현재 디렉토리에 체커보드 이미지 파일들을 넣어주세요")
        print("4. 아래 함수를 사용해 체커보드 이미지를 직접 촬영해보세요")
        
        # 체커보드 이미지 촬영 옵션 제공
        choice = input("체커보드 이미지를 지금 촬영하시겠습니까? (y/n): ")
        if choice.lower() == 'y':
            capture_checkerboard_images()
            # 다시 이미지 검색
            for path_pattern in image_paths:
                found_images = glob.glob(path_pattern)
                if found_images:
                    images.extend(found_images)
            images = list(set(images))
        
        if not images:
            return None
    
    print(f"총 {len(images)}개의 이미지를 처리합니다...")
    
    successful_detections = 0
    
    for i, fname in enumerate(images):
        print(f"처리 중: {fname} ({i+1}/{len(images)})")
        
        img = cv2.imread(fname)
        if img is None:
            print(f"경고: {fname} 이미지를 읽을 수 없습니다.")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 체커보드 코너 찾기
        ret, corners = cv2.findChessboardCorners(gray,
                                               CHECKERBOARD,
                                               cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_FAST_CHECK +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            successful_detections += 1
            
            # 코너 그리기 및 표시
            img_corners = cv2.drawChessboardCorners(img.copy(), CHECKERBOARD, corners2, ret)
            
            # 이미지 크기 조정 (화면에 맞게)
            height, width = img_corners.shape[:2]
            if height > 800 or width > 1200:
                scale = min(800/height, 1200/width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img_corners = cv2.resize(img_corners, (new_width, new_height))
            
            cv2.imshow('Checkerboard Corners', img_corners)
            cv2.waitKey(500)  # 0.5초 대기
            print(f"  ✓ 체커보드 검출 성공")
        else:
            print(f"  ✗ 체커보드 검출 실패")
    
    cv2.destroyAllWindows()
    
    print(f"\n총 {successful_detections}개 이미지에서 체커보드 검출 성공")
    
    if successful_detections < 3:
        print("에러: 캘리브레이션을 위해서는 최소 3개 이상의 성공적인 체커보드 검출이 필요합니다.")
        print("체커보드 이미지의 품질을 확인하거나 더 많은 이미지를 추가해주세요.")
        return None
    
    # 카메라 캘리브레이션 수행
    print("카메라 캘리브레이션을 수행 중...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                      gray.shape[::-1], None, None)
    
    if ret:
        print("캘리브레이션 성공!")
        # 결과 출력
        print("Camera matrix : \n")
        print(mtx)
        print("\ndist : \n")
        print(dist)
        
        # 캘리브레이션 결과를 파일로 저장
        calibration_data = {
            'camera_matrix': mtx,
            'dist_coeffs': dist,
            'rvecs': rvecs,
            'tvecs': tvecs
        }
        
        with open('camera_calibration.pkl', 'wb') as f:
            pickle.dump(calibration_data, f)
        
        print("캘리브레이션 데이터가 'camera_calibration.pkl' 파일로 저장되었습니다.")
        return calibration_data
    else:
        print("캘리브레이션 실패!")
        return None

def capture_checkerboard_images():
    """웹캠을 사용해 체커보드 이미지를 촬영하는 함수"""
    print("체커보드 이미지 촬영 모드")
    print("스페이스바를 눌러 이미지를 저장하고, 'q'를 눌러 종료하세요.")
    
    # img 폴더 생성
    if not os.path.exists('img'):
        os.makedirs('img')
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    image_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 체커보드 검출 시도
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_check, corners = cv2.findChessboardCorners(gray, (7,10), None)
        
        # 검출 결과 표시
        if ret_check:
            cv2.drawChessboardCorners(frame, (7,10), corners, ret_check)
            cv2.putText(frame, "Checkerboard Detected - Press SPACE to save", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Checkerboard NOT detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, f"Images captured: {image_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Capture Checkerboard Images', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and ret_check:  # 스페이스바를 누르고 체커보드가 검출된 경우
            filename = f'img/checkerboard_{image_count:02d}.jpg'
            cv2.imwrite(filename, frame)
            image_count += 1
            print(f"이미지 저장됨: {filename}")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"총 {image_count}개의 이미지가 저장되었습니다.")

def live_video_correction(calibration_data):
    if calibration_data is None:
        print("캘리브레이션 데이터가 없습니다.")
        return
    
    mtx = calibration_data['camera_matrix']
    dist = calibration_data['dist_coeffs']
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    print("실시간 왜곡 보정을 시작합니다. 'q'를 눌러 종료하세요.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 크기 가져오기
        h, w = frame.shape[:2]
        
        # 최적의 카메라 행렬 구하기
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        
        # 왜곡 보정
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        
        # ROI로 이미지 자르기
        x, y, w_roi, h_roi = roi
        if all(v > 0 for v in [x, y, w_roi, h_roi]):
            dst = dst[y:y+h_roi, x:x+w_roi]
        
        # 원본과 보정된 이미지를 나란히 표시
        try:
            original = cv2.resize(frame, (640, 480))
            corrected = cv2.resize(dst, (640, 480))
            combined = np.hstack((original, corrected))
            
            # 텍스트 추가
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Corrected", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 결과 표시
            cv2.imshow('Original | Corrected', combined)
        except Exception as e:
            print(f"화면 표시 오류: {e}")
            cv2.imshow('Original', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=== 카메라 캘리브레이션 프로그램 ===")
    
    # 이미 캘리브레이션 파일이 있는지 확인
    if os.path.exists('camera_calibration.pkl'):
        print("기존 캘리브레이션 데이터를 발견했습니다.")
        choice = input("기존 데이터를 사용하시겠습니까? (y/n): ")
        if choice.lower() == 'y':
            print("기존 캘리브레이션 데이터를 로드합니다...")
            with open('camera_calibration.pkl', 'rb') as f:
                calibration_data = pickle.load(f)
        else:
            print("새로운 캘리브레이션을 수행합니다...")
            calibration_data = calibrate_camera()
    else:
        print("새로운 캘리브레이션을 수행합니다...")
        calibration_data = calibrate_camera()
    
    # 캘리브레이션이 성공한 경우에만 실시간 보정 실행
    if calibration_data is not None:
        print("\n실시간 비디오 보정을 시작합니다...")
        live_video_correction(calibration_data)
    else:
        print("캘리브레이션에 실패했습니다. 프로그램을 종료합니다.")