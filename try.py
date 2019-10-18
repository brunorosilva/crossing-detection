import numpy as np
import cv2
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from numpy import expand_dims

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
while(True):
    ret, frame = cap.read()
    faces = detector.detect_faces(frame)
    for face in faces:
        box = face.get('box')
        cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color = (255,0,0))
        
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()