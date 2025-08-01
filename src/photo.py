import cv2
import datetime

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0) 

while True:

    # 카메라로부터 프레임을 읽음
    ret, frame = cap.read()
    if not ret:
        print("프레임 X")  # 프레임 읽기 실패 시 메시지 출력
        break

    # 읽은 프레임을 화면에 표시
    cv2.imshow("Video", frame)

    # 키 입력을 기다림 (1ms 대기 후 다음 프레임으로 이동)
    key = cv2.waitKey(1) & 0xFF

    # 'a' 키가 눌리면 현재 프레임을 저장
    if key == ord('a'):
        # 파일 이름을 현재 날짜 및 시간으로 설정
        filename = datetime.datetime.now().strftime("../img/capture_%Y%m%d_%H%M%S.png")
        # 프레임을 이미지 파일로 저장
        cv2.imwrite(filename, frame)
        print(f"{filename}")  # 저장된 파일 이름 출력

    # 'q' 키가 눌리면 루프를 종료
    elif key == ord('q'):
        break

# 자원 해제 (카메라 및 모든 OpenCV 창 닫기)
cap.release()
cv2.destroyAllWindows()