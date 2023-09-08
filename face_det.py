import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

network_loaded = load_model('models/emodet.h5')
network_loaded.compile(loss = 'categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
face_detector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

img_type = int(input("Enter the input type (1=single face image), (2=multi face image), (3=video) : "))

if img_type == 1:
    image = cv2.imread('data/Images/gabriel.png')
    original_image = image.copy()
    faces = face_detector.detectMultiScale(original_image)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        roi = image[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi / 255
        roi = np.expand_dims(roi, axis = 0)
        prediction = network_loaded.predict(roi, verbose=0)
        cv2.putText(image, emotions[np.argmax(prediction)], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow('abc',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif img_type == 2:
    image = cv2.imread('data/Images/faces_emotions.png')
    faces = face_detector.detectMultiScale(image)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        roi = image[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi / 255
        roi = np.expand_dims(roi, axis = 0)
        prediction = network_loaded.predict(roi, verbose=0)
        cv2.putText(image, emotions[np.argmax(prediction)], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow('abc',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif img_type == 3:
    cap = cv2.VideoCapture('data/Videos/emotion_test01.mp4')
    connected, video = cap.read()
    save_path = 'results/Videos/emotion_test01_result.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 24
    output_video = cv2.VideoWriter(save_path, fourcc, fps, (video.shape[1], video.shape[0]))
    while (cv2.waitKey(1) < 0):
        connected, frame = cap.read()
        if not connected:
            break
        faces = face_detector.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5, minSize=(30,30))
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = frame[y:y + h, x:x + w]
                roi = cv2.resize(roi, (48, 48))
                roi = roi / 255
                roi = np.expand_dims(roi, axis = 0)
                prediction = network_loaded.predict(roi, verbose=0)

                if prediction is not None:
                    result = np.argmax(prediction)
                    cv2.putText(frame, emotions[result], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        output_video.write(frame)

    print('End')
    output_video.release()
    cv2.destroyAllWindows()