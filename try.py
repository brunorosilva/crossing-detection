import numpy as np
import cv2
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from numpy import expand_dims
import datetime
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
cap = cv2.VideoCapture(0)

detector = MTCNN()
'''
    pixels = frame
    detector = MTCNN()
    results = detector.detect_faces(frame)
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize((224,224))
    face_array = asarray(image)
    pixels = pixels.astype('float32')
    samples = expand_dims(pixels, axis=0)
    samples = preprocess_input(samples, version=2)
    model = VGGFace(model='resnet50')
    yhat = model.predict(samples)
    results = decode_predictions(yhat)
    
    for result in results[0]:
	    print('%s: %.3f%%' % (result[0], result[1]*100))
'''

temp = datetime.datetime.now()
count = 0
while(True):
    ret, frame = cap.read()

    width = cap.get(3)
    height = cap.get(4)
    faces = detector.detect_faces(frame)
    for face in faces:
        box = face.get('box')
        left_eye = face.get('keypoints').get('left_eye')
        right_eye = face.get('keypoints').get('right_eye')
        mouth_left = face.get('keypoints').get('mouth_left')
        mouth_right = face.get('keypoints').get('mouth_right')
        nose = face.get('keypoints').get('nose')
        cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color = (255,0,0))
        cv2.circle(frame, left_eye, 4, color = (255,0,0))
        cv2.circle(frame, right_eye, 4, color = (255,0,0))
        cv2.circle(frame, mouth_left, 4, color = (255,0,0))
        cv2.circle(frame, mouth_right, 4, color = (255,0,0))
        cv2.circle(frame, nose, 4, color = (255,0,0))
        

        temp2 = datetime.datetime.now()
        dateTimeDifference = temp2 - temp
        cv2.line(frame, (int(width/2), 0), (int(width/2), int(height)), color = (0,0,255))
        if (nose[0] <= int(width/2) + 10 and nose[0] >= int(width/2) - 10 and (dateTimeDifference.total_seconds() > 5))   :
            count+= 1
            temp = datetime.datetime.now()
    
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,100)
        fontScale              = 1
        
        if dateTimeDifference.total_seconds() < 5:
            fontColor = (0,0,255) 
        else:
            fontColor = (255, 255, 255)
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



    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()