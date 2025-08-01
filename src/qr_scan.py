import cv2 
import matplotlib.pyplot as plt
import pyzbar.pyzbar as pyzbar

img = cv2.imread('../img/frame.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#plt.imshow(img)
plt.imshow(gray, cmap='gray')
plt.show()

# 디코딩 
decoded = pyzbar.decode(gray)
print(decoded)

cv2.waitKey(0)
cv2.destroyAllWindows()