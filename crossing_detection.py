import numpy as np
import cv2
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from numpy import expand_dims
import datetime
from PIL import Image
from numpy import asarray
import requests

url = "https://kairosapi-karios-v1.p.rapidapi.com/detect"



# Grabing the camera

cap = cv2.VideoCapture(0)

# instancing the face detector

detector = MTCNN()

# start clock

temp = datetime.datetime.now()

# counts how many crossings there was

count = 0

from keras_vggface.vggface import VGGFace
# create a vggface2 model
model = VGGFace(model='resnet50')
        
# start camera loop
while(True):
    ret, frame = cap.read()

    # width and height from the image

    width = cap.get(3)
    height = cap.get(4)
    
    # face detection with MTCNN
    faces = detector.detect_faces(frame)
    for face in faces:

        # face detection box
        box = face.get('box')

        # face landmarks
        left_eye = face.get('keypoints').get('left_eye')
        right_eye = face.get('keypoints').get('right_eye')
        mouth_left = face.get('keypoints').get('mouth_left')
        mouth_right = face.get('keypoints').get('mouth_right')
        nose = face.get('keypoints').get('nose')

        # it's drawing time
        cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color = (255,0,0))
        cv2.circle(frame, left_eye, 4, color = (255,0,0))
        cv2.circle(frame, right_eye, 4, color = (255,0,0))
        cv2.circle(frame, mouth_left, 4, color = (255,0,0))
        cv2.circle(frame, mouth_right, 4, color = (255,0,0))
        cv2.circle(frame, nose, 4, color = (255,0,0))
        
        # end clock

        temp2 = datetime.datetime.now()
        
        # creating a time diff variable
         
        dateTimeDifference = temp2 - temp
        
        # imaginary line

        cv2.line(frame, (int(width/2), 0), (int(width/2), int(height)), color = (0,0,255))
        
        # if the time diff is < 2 show red clock

        if dateTimeDifference.total_seconds() < 2:
            fontColor = (0,0,255) 
        else:
            fontColor = (255, 255, 255)
        
        
        # general formatting and putting the text into the image
    
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,100)
        fontScale              = 1
        lineType               = 2

        cv2.putText(frame, str(dateTimeDifference.total_seconds()),
        (20, 200),
        font,
        fontScale,
        fontColor,
        lineType)

        cv2.putText(frame, str(count),
        (20, 100),
        font,
        fontScale,
        (255, 255, 255),
        lineType)

        px_tol = 10
        sec_tol = 2
        
        # creating a 20 pixels wide boundary that will detect the crossing (20px because of framerate reasons)
        # only count one more crossing if timediff > 2, so it can't duplicate crossings

        if (nose[0] <= int(width/2) + px_tol and nose[0] >= int(width/2) - px_tol and (dateTimeDifference.total_seconds() > sec_tol)):
            count+= 1
            temp = datetime.datetime.now()
            cv2.imwrite("frame%d.jpg" % count, frame)
        points = [left_eye, right_eye, mouth_left, mouth_right, nose]
        dists = []
        #(left_eye[1] - right_eye[1])**2 + left_eye[1] - 

        

        # para x

        x_total = 0
        y_total = 0
        for point in points:
            x_1 = point[0]
            y_1 = point[1]
            for other in points:
                x_2 = other[0]
                x_total += (x_1-x_2)**2
                y_2 = other[1]
                y_total += (y_1-y_2)**2
            print("-"*10)
            print("\n"*2)
            print("X_total ", np.sqrt(x_total))
            print("Y_total ", np.sqrt(y_total))
            print("Distância Euclidiana ", np.sqrt(y_total) + np.sqrt(x_total))

        
        # example of creating a face embedding
        # summarize input and output shape
        
        #payload = "{    \"image\":\""+frame+"\",    \"selector\":\"ROLL\"}"
                
    # displaying the image

    cv2.imshow('frame',frame)
    
    # press q to break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()