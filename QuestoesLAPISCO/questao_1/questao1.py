import cv2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
video = cv2.VideoWriter('questao_1/videoquestao1.avi', fourcc, 20.0, (640, 480))

while True:
    _, frame = cap.read()

    
    cv2.imshow("frame", frame)
    video.write(frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release() 
video.release()  
cv2.destroyAllWindows() 