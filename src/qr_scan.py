import cv2 
import matplotlib.pyplot as plt
import pyzbar.pyzbar as pyzbar

#img = cv2.imread('../img/frame.png')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    ret ,img = cap.read()

    if not ret:
        continue

    #img = cv2.imread('../img/frame.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    decoded = pyzbar.decode(gray)

    for d in decoded:
        x, y, w, h = d.rect 

        barcode_data = d.data.decode('utf-8')
        barcode_type = d.type
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        #print(d.data.decode('utf-8'))
        #barcode_data = d.data.decode('utf-8')
        #print(d.type)
        #barcode_type = d.type

        text = '%s (%s)' % (barcode_data, barcode_type)

        #cv2.rectangle(img, (d.rect[0], d.rect[1]), (d.rect[0] + d.rect[2], d.rect[1] + d.rect[3]), (0, 255, 0 ), 20)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0 ), 2)
        #cv2.putText(img, text, (d.rect[0], d.rect[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA )
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA )
    
    cv2.imshow('camera', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows()
#plt.imshow(img)
#plt.imshow(gray, cmap='gray')
#plt.show()

# 디코딩 
# decoded = pyzbar.decode(gray)
# print(decoded)

# for d in decoded:
#     print(d.data.decode('utf-8'))
#     barcode_data = d.data.decode('utf-8')
#     print(d.type)
#     barcode_type = d.type

#     text = '%s (%s)' % (barcode_data, barcode_type)

#     cv2.rectangle(img, (d.rect[0], d.rect[1]), (d.rect[0] + d.rect[2], d.rect[1] + d.rect[3]), (0, 255, 0 ), 20)
#     cv2.putText(img, text, (d.rect[0], d.rect[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA )

# plt.imshow(img)
# plt.show()

"""
img: 사각형을 그릴 이미지입니다.
pt1: 사각형의 왼쪽 상단 꼭지점 좌표입니다. (x, y) 형식의 튜플이어야 합니다.
pt2: 사각형의 오른쪽 하단 꼭지점 좌표입니다. (x, y) 형식의 튜플이어야 합니다.
color: 사각형의 색상입니다. (B, G, R) 형식의 튜플이나 스칼라 값으로 지정할 수 있습니다.
thickness: 선택적으로 사각형의 선 두께를 지정합니다. 기본값은 1입니다. 음수 값을 전달하면 내부를 채웁니다.
lineType: 선택적으로 선의 형태를 지정합니다. 기본값은 cv2.LINE_8입니다.
shift: 선택적으로 좌표값의 소수 부분을 비트 시프트할 양을 지정합니다.
"""
# cv2.waitKey(0)
# cv2.destroyAllWindows()