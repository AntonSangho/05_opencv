import cv2
import numpy as np
import os
import glob
import pickle

def test_different_checkerboard_sizes(img_path):
    """다양한 체커보드 크기로 테스트해보는 함수"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 일반적인 체커보드 크기들
    checkerboard_sizes = [
        (7, 10), (10, 7),   # 원래 설정
        (6, 9), (9, 6),     # 8x10 체커보드
        (5, 8), (8, 5),     # 6x9 체커보드
        (4, 7), (7, 4),     # 5x8 체커보드
        (6, 8), (8, 6),     # 7x9 체커보드
        (5, 7), (7, 5),     # 6x8 체커보드
        (4, 6), (6, 4),     # 5x7 체커보드
        (3, 5), (5, 3),     # 4x6 체커보드
    ]
    
    print(f"\n=== {os.path.basename(img_path)} 체커보드 크기 테스트 ===")
    
    successful_sizes = []
    
    for size in checkerboard_sizes:
        ret, corners = cv2.findChessboardCorners(gray, size, 
                                               cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_FAST_CHECK +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            print(f"✓ {size} 크기로 체커보드 검출 성공!")
            successful_sizes.append(size)
        else:
            print(f"✗ {size} 크기로 검출 실패")
    
    return successful_sizes

def analyze_image_quality(img_path):
    """이미지 품질 분석 함수"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"이미지를 읽을 수 없습니다: {img_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(f"\n=== {os.path.basename(img_path)} 이미지 분석 ===")
    print(f"이미지 크기: {img.shape[1]} x {img.shape[0]}")
    print(f"평균 밝기: {np.mean(gray):.1f}")
    print(f"밝기 표준편차: {np.std(gray):.1f}")
    
    # 대비 분석
    contrast = gray.max() - gray.min()
    print(f"대비: {contrast}")
    
    # 블러 정도 분석 (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"선명도 (높을수록 좋음): {laplacian_var:.1f}")
    
    if laplacian_var < 100:
        print("⚠️  이미지가 흐릿할 수 있습니다.")
    if contrast < 100:
        print("⚠️  이미지 대비가 낮습니다.")
    if np.mean(gray) < 50:
        print("⚠️  이미지가 너무 어둡습니다.")
    elif np.mean(gray) > 200:
        print("⚠️  이미지가 너무 밝습니다.")

def show_preprocessed_image(img_path, checkerboard_size=(7, 10)):
    """전처리된 이미지를 보여주는 함수"""
    img = cv2.imread(img_path)
    if img is None:
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 다양한 전처리 방법들
    # 1. 히스토그램 평활화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)
    
    # 2. 가우시안 블러
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. 이진화
    _, gray_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. 적응적 이진화
    gray_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
    
    # 각각에 대해 체커보드 검출 시도
    methods = [
        ("Original", gray),
        ("CLAHE", gray_clahe),
        ("Gaussian Blur", gray_blur),
        ("Threshold", gray_thresh),
        ("Adaptive Threshold", gray_adaptive)
    ]
    
    print(f"\n=== {os.path.basename(img_path)} 전처리 방법별 테스트 ===")
    
    best_result = None
    best_method = None
    
    for method_name, processed_img in methods:
        ret, corners = cv2.findChessboardCorners(processed_img, checkerboard_size,
                                               cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_FAST_CHECK +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret:
            print(f"✓ {method_name}: 체커보드 검출 성공!")
            if best_result is None:
                best_result = (processed_img, corners)
                best_method = method_name
        else:
            print(f"✗ {method_name}: 체커보드 검출 실패")
    
    # 결과 시각화
    if best_result is not None:
        processed_img, corners = best_result
        result_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(result_img, checkerboard_size, corners, True)
        
        # 이미지 크기 조정
        height, width = result_img.shape[:2]
        if height > 600 or width > 800:
            scale = min(600/height, 800/width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            result_img = cv2.resize(result_img, (new_width, new_height))
        
        cv2.imshow(f'Best Result - {best_method}', result_img)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    
    return best_result is not None

def calibrate_camera_flexible():
    """유연한 체커보드 검출을 위한 캘리브레이션 함수"""
    
    # 다양한 이미지 형식과 경로 시도
    image_paths = [
        '../img/*.png', '../img/*.jpg', '../img/*.jpeg',
        './img/*.png', './img/*.jpg', './img/*.jpeg',
        'img/*.png', 'img/*.jpg', 'img/*.jpeg',
        '*.png', '*.jpg', '*.jpeg'
    ]
    
    images = []
    for path_pattern in image_paths:
        found_images = glob.glob(path_pattern)
        if found_images:
            images.extend(found_images)
    
    images = list(set(images))
    
    if not images:
        print("체커보드 이미지를 찾을 수 없습니다!")
        return None
    
    print(f"총 {len(images)}개의 이미지를 발견했습니다.")
    
    # 첫 번째 이미지로 체커보드 크기 자동 감지
    print("\n=== 체커보드 크기 자동 감지 ===")
    first_image = images[0]
    successful_sizes = test_different_checkerboard_sizes(first_image)
    
    if not successful_sizes:
        print("첫 번째 이미지에서 체커보드를 찾을 수 없습니다.")
        print("이미지 품질을 분석합니다...")
        analyze_image_quality(first_image)
        
        # 전처리 방법 테스트
        print("다양한 전처리 방법을 테스트합니다...")
        if show_preprocessed_image(first_image):
            print("전처리를 통해 검출이 가능할 수 있습니다.")
        
        return None
    
    # 가장 많이 검출된 크기 선택
    CHECKERBOARD = successful_sizes[0]
    print(f"선택된 체커보드 크기: {CHECKERBOARD}")
    
    # 캘리브레이션 진행
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []
    imgpoints = []
    
    # 3D 점의 세계 좌표 정의
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    
    successful_detections = 0
    
    for i, fname in enumerate(images):
        print(f"처리 중: {os.path.basename(fname)} ({i+1}/{len(images)})")
        
        img = cv2.imread(fname)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 여러 전처리 방법 시도
        preprocessing_methods = [
            ("original", gray),
            ("clahe", cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)),
            ("blur", cv2.GaussianBlur(gray, (3, 3), 0))
        ]
        
        corners_found = False
        for method_name, processed_gray in preprocessing_methods:
            ret, corners = cv2.findChessboardCorners(processed_gray, CHECKERBOARD,
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                   cv2.CALIB_CB_FAST_CHECK +
                                                   cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(processed_gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)
                successful_detections += 1
                corners_found = True
                
                print(f"  ✓ 체커보드 검출 성공 ({method_name})")
                
                # 결과 시각화
                img_corners = cv2.drawChessboardCorners(img.copy(), CHECKERBOARD, corners2, ret)
                height, width = img_corners.shape[:2]
                if height > 600 or width > 800:
                    scale = min(600/height, 800/width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img_corners = cv2.resize(img_corners, (new_width, new_height))
                
                cv2.imshow('Checkerboard Detection', img_corners)
                cv2.waitKey(300)
                break
        
        if not corners_found:
            print(f"  ✗ 체커보드 검출 실패")
    
    cv2.destroyAllWindows()
    
    print(f"\n총 {successful_detections}개 이미지에서 체커보드 검출 성공")
    
    if successful_detections < 3:
        print("캘리브레이션을 위해서는 최소 3개 이상의 성공적인 검출이 필요합니다.")
        return None
    
    # 카메라 캘리브레이션 수행
    print("카메라 캘리브레이션을 수행 중...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                      gray.shape[::-1], None, None)
    
    if ret:
        print("캘리브레이션 성공!")
        print("Camera matrix:")
        print(mtx)
        print("\nDistortion coefficients:")
        print(dist)
        
        calibration_data = {
            'camera_matrix': mtx,
            'dist_coeffs': dist,
            'rvecs': rvecs,
            'tvecs': tvecs,
            'checkerboard_size': CHECKERBOARD
        }
        
        with open('camera_calibration.pkl', 'wb') as f:
            pickle.dump(calibration_data, f)
        
        print("캘리브레이션 데이터가 저장되었습니다.")
        return calibration_data
    else:
        print("캘리브레이션 실패!")
        return None

def live_video_correction(calibration_data):
    """실시간 비디오 왜곡 보정"""
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
        
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        
        x, y, w_roi, h_roi = roi
        if all(v > 0 for v in [x, y, w_roi, h_roi]):
            dst = dst[y:y+h_roi, x:x+w_roi]
        
        try:
            original = cv2.resize(frame, (640, 480))
            corrected = cv2.resize(dst, (640, 480))
            combined = np.hstack((original, corrected))
            
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Corrected", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Camera Calibration Result', combined)
        except:
            cv2.imshow('Original', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=== 향상된 카메라 캘리브레이션 프로그램 ===")
    
    if os.path.exists('camera_calibration.pkl'):
        choice = input("기존 캘리브레이션 데이터를 사용하시겠습니까? (y/n): ")
        if choice.lower() == 'y':
            with open('camera_calibration.pkl', 'rb') as f:
                calibration_data = pickle.load(f)
        else:
            calibration_data = calibrate_camera_flexible()
    else:
        calibration_data = calibrate_camera_flexible()
    
    if calibration_data is not None:
        print("\n실시간 비디오 보정을 시작합니다...")
        live_video_correction(calibration_data)
    else:
        print("\n캘리브레이션에 실패했습니다.")
        print("다음 사항을 확인해보세요:")
        print("1. 체커보드가 명확하게 보이는 이미지인지 확인")
        print("2. 체커보드의 모든 코너가 이미지 안에 포함되어 있는지 확인")
        print("3. 이미지가 너무 흐리거나 어둡지 않은지 확인")
        print("4. 다양한 각도에서 촬영된 이미지들인지 확인")