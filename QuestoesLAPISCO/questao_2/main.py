import time
import cv2


#CORES 
colors = [(255, 0, 0), (255, 255, 0), (0, 255, 255), (0, 0 , 255), (0, 255, 0), (255, 0, 255)]

#CLASSES
class_names = []
with open('questao_2/coco.names', 'r') as f:
    class_names = [cname.strip() for cname in f.readlines()]

#Vídeo
cap = cv2.VideoCapture(0)#WEBCAM
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
video = cv2.VideoWriter('questao_2/videoquestao2.avi', fourcc, 20.0, (640, 480))


#PESOS
net = cv2.dnn.readNet('questao_2/yolov4-tiny.weights', 'questao_2/yolov4-tiny.cfg')

#REDE NEURAL
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)


#FRAMES
while True:

    #CAPTURA DE FRAME
    _, frame = cap.read()

    # #CONTAGEM FPS
    # start = time.time()

    #DETEC
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    # #FIM CONT FPS
    # end = time.time()

    for (classid, score, box) in zip(classes, scores, boxes):
        #Cor por classes
        color = colors[int(classid) % len(colors)]

        #ID
        label = f'{class_names[classid]} : {score}'

        #Box de deteccao
        cv2.rectangle(frame, box, color, 2)

        #Colocando as labels nos objetos
        cv2.putText(frame, label, (box[0], box[1]- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
    # #Calc FPS
    #fps_label = f'FPS:{round((1-(end+start)), 2)}'

    # #PRINTING FPS
    # cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    # cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 130, 0), 1)
        
    #IMAGEM
    cv2.imshow('detections', frame)
    video.write(frame)

    if cv2.waitKey(1) == 27:
        break


cap.release()
video.release()

cv2.destroyAllWindows()