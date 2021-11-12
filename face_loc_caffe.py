import cv2,time
import numpy as np

prototxtPath = "./detector/deploy.prototxt"
weightsPath ="./detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath,weightsPath)
cap=cv2.VideoCapture(0)

while True:
    _,frame=cap.read()
    start_time=time.time()
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >= 0.2:  # precision
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            cv2.rectangle(frame,(startX,startY),(endX,endY),(0,0,255),2)
            
    cv2.imshow('frame',frame)
    end_time = time.time()
    print(round((1/(end_time-start_time)),2))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
