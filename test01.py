import cv2
import numpy as np

image = np.zeros((200,200), dtype=np.uint8)
for r in range(200):
    for c in range(200):
        if r >= 80 and r <= 120 and c >= 80 and c <= 120:
            image[r,c] = 0
        else:
            image[r,c] = 255
cv2.imshow("test", image)
cv2.waitKey()