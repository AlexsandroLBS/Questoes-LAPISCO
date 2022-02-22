import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
video = cv2.VideoWriter('videoquestao3.avi', fourcc, 20.0, (640, 480))

while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #Green 
    low_green = np.array([25, 52, 72])
    high_green = np.array([102, 255, 255])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    green = cv2.bitwise_and(frame, frame, mask=green_mask)
    
    
    cv2.imshow("Green", green)
    video.write(green)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release() 
video.release()  
cv2.destroyAllWindows() 