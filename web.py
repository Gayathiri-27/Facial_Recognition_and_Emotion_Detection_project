#!/usr/bin/env python
import os
import shutil
from flask import Flask, render_template, request, \
    Response, send_file, redirect, url_for
from camera import Camera
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt


app = Flask(__name__)
camera = None

detection_model_path = 'haarcascade_frontalface_default.xml'


face_detection = cv2.CascadeClassifier(detection_model_path)

emotion_classifier = load_model("model.hdf5")

emotions = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']


def get_camera():
    global camera
    if not camera:
        camera = Camera()

    return camera


@app.route('/')
def root():
    return redirect(url_for('index'))

@app.route('/index/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_feed()
        if frame:
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            raise RuntimeError("No frame captured.")

@app.route('/video_feed/')
def video_feed():
    camera = get_camera()
    return Response(gen(camera),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture/')
def capture():
    camera = get_camera()
    stamp = camera.capture()
    return redirect(url_for('show_capture', timestamp=stamp))

def stamp_file(timestamp):
    return 'captures/' + timestamp +".jpg"

@app.route('/capture/image/<timestamp>', methods=['POST', 'GET'])
def show_capture(timestamp):
    path = stamp_file(timestamp)
    image_path = "static/"+path
    print(image_path)
    color_frame = cv2.imread(image_path)
    gray_frame = cv2.imread(image_path, 0)
    detected_faces = face_detection.detectMultiScale(color_frame, scaleFactor=1.1, minNeighbors=5, 
                                        minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    print('Number of faces detected : ', len(detected_faces))
    if len(detected_faces)>0:
        detected_faces = sorted(detected_faces, reverse=True, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))[0] # if more than one faces
        (fx, fy, fw, fh) = detected_faces
        im = gray_frame[fy:fy+fh, fx:fx+fw]
        im = cv2.resize(im, (64,64))  # the model is trained on 48*48 pixel image
        im = im.astype("float")/255.0
        im = img_to_array(im)
        im = np.expand_dims(im, axis=0)
        preds = emotion_classifier.predict(im)[0]
        print(emotions)
        print(preds)
        emotion_probability = np.max(preds)
        
        label = emotions[preds.argmax()]
        print("Predicted Result "+label)
        cv2.putText(color_frame, label, (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (153, 38, 240), 2)
        cv2.rectangle(color_frame, (fx, fy), (fx + fw, fy + fh),(153, 38, 240), 2)
    cv2.imwrite("static/result/"+timestamp+".jpg", color_frame)
    cv2.waitKey(1)
    path="result/"+timestamp+".jpg"
    return render_template('capture.html',
        stamp=timestamp, path=path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
