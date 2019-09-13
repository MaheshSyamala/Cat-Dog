# -*- coding: utf-8 -*-

"""
Created on Sat Apr 20 09:38:21 2019

@author: mahesh.s.reddy315
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import time

#img=cv2.imread(r'E:\Data\Machine Learning A-Z™ Hands-On Python & R In Data Science\Practice\Machine Learning A-Z Template Folder\Part 3 - Classification\Kaggle\Dogs and Cats\train\cat.0.jpg',1)
#
#img=cv2.imread(r'.\train\cat.4.jpg',1)
#cv2.imshow('Cat',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

def __draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

video=cv2.VideoCapture(0)
a=1
results=0
while True:
    a=a+1
    check,frame=video.read()
    print(frame)
    if(a%30==0):
        frame=cv2.flip(frame,1)
        resized_image=cv2.resize(frame,(128,128))
        test=resized_image/255
        test=np.expand_dims(test,axis=0)
        results=classifier.predict(test)
    if(results>0.5):
        pred='Dog'+str(results)
    else:
        pred='Cat'+str(results)
    __draw_label(frame, pred, (50,50), (0,255,0))
    cv2.imshow('Capture',frame)
    key=cv2.waitKey(1)
    if(key==ord('q')):
        break

video.release()
cv2.destroyAllWindows()

#plt.imshow(frame)


from keras.models import Sequential
from keras.layers import Dense,Convolution2D,MaxPooling2D,Flatten

classifier=Sequential()

#Model Begin
classifier.add(Convolution2D(32,3,3,input_shape=(128,128,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(64,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(128,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(256,3,3,activation='relu'))


classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim=4096,activation='relu'))
classifier.add(Dense(output_dim=2048,activation='relu'))
classifier.add(Dense(output_dim=512,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))

classifier.summary()


classifier.load_weights('Model.h5')



'''Image Grab'''
from PIL import ImageGrab


def __draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

last_time = time.time()
while(True):
    # 800x600 windowed mode
    printscreen =  np.array(ImageGrab.grab(bbox=(0,40,500,400)))
    print('loop took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    resized_image=cv2.resize(printscreen,(128,128))
    test=resized_image/255
    test=np.expand_dims(test,axis=0)
    results=classifier.predict(test)
    if(results>0.5):
        pred='Dog'+str(results)
    else:
        pred='Cat'+str(results)
    __draw_label(printscreen, pred, (50,50), (0,255,0))
    cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break



video=cv2.VideoCapture(0)
a=1
results=0
while True:
    a=a+1
    check,frame=video.read()
    print(frame)
    if(a%30==0):
        frame=cv2.flip(frame,1)
        resized_image=cv2.resize(frame,(128,128))
        test=resized_image/255
        test=np.expand_dims(test,axis=0)
        results=classifier.predict(test)
    if(results>0.5):
        pred='Dog'+str(results)
    else:
        pred='Cat'+str(results)
    __draw_label(frame, pred, (50,50), (0,255,0))
    cv2.imshow('Capture',frame)
    key=cv2.waitKey(1)
    if(key==ord('q')):
        break

video.release()
cv2.destroyAllWindows()

