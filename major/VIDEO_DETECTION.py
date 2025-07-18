import numpy as np 
import cv2

prototxt = 'Model/MobileNetSSD_deploy.prototxt'
model= 'Model/MobileNetSSD_deploy.caffemodel'
min_confidence= 0.2

classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

np.random.seed(984343)
color = np.random.uniform(0,255, size = (len(classes), 3))

net = cv2.dnn.readNetFromCaffe(prototxt,model)

cap = cv2.VideoCapture(0)

while True:

    _,img = cap.read()

    heigth,width = img.shape[0],img.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(img,(300,300)), 0.007,(300,300), 130)

    net.setInput(blob)
    detected_objects = net.forward()

    for i in range(detected_objects.shape[2]):

        confidence = detected_objects[0][0][i][2]

        if confidence > min_confidence:
            class_index = int(detected_objects[0,0,i,1])

            upper_left_x = int(detected_objects[0,0,i,3] * width)
            upper_left_y = int(detected_objects[0,0,i,4] * heigth)
            lower_right_x = int(detected_objects[0,0,i,5] * width)
            lower_right_y = int(detected_objects[0,0,i,6] * heigth)

            perdiction_text = f"{classes[class_index]}: {confidence:.2f}%"
            cv2.rectangle(img, (upper_left_x,upper_left_y),( lower_right_x,lower_right_y),color[class_index],3)
            cv2.putText(img, perdiction_text, (upper_left_x,
                        upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color[class_index],2)
    cv2.imshow("Oject Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
cap.release()
