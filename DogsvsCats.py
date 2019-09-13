# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 23:04:37 2019

@author: mahesh.s.reddy315
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

dirct=os.listdir('./train')

labels=[]
file=[]
for i in dirct:
    a=i.split('.')
    labels.append(a[0])
    file.append(i)
    
d={'Labels':labels,'file':file}
dataframe=pd.DataFrame(data=d)



import keras
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense

model=Sequential()

model.add(Convolution2D(32,5,5,input_shape=(128,128,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dense(output_dim=512,activation='relu'))
model.add(Dense(output_dim=256,activation='relu'))
model.add(Dense(output_dim=128,activation='relu'))
model.add(Dense(output_dim=1,activation='sigmoid'))

model.summary()


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255)
training_set=train_datagen.flow_from_dataframe(dataframe,directory='./train',x_col="file",y_col="Labels"
                                               ,target_size=(128,128),class_mode="binary")

model.fit_generator(training_set,samples_per_epoch=10000,epochs=6)


import pickle

with open('DogsVsCats','wb') as f:
    pickle.dump(model,f)
    
with open('DogsVsCats','rb') as f:
    pickle.load(f)

m=training_set.class_indices

y_pred=model.predict(training_set)
y_pred=model.predict_generator(training_set,steps=1)
y_pred2=model.predict_generator(training_set,steps=2)

m=y_pred>0.5
df=pd.Series(data=m.reshape(25000))
mp=df.map({True:'cat',False:'dog'})

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(dataframe['Labels'],mp)

'''Predictions'''
from keras.preprocessing import image


test_image=image.load_img('./train/cat.45.jpg',target_size=(128,128))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)

model.predict(test_image)


all_files=[]
for i in dirct:
    a='./train/'+i
    all_files.append(a)

y_pred_train=[]
for i in all_files:
    test_image=image.load_img(i,target_size=(128,128))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)

    pred=model.predict(test_image)
    y_pred_train.append(pred)




'''Test Data'''

dirct_t=os.listdir('./yy/test')
dat=pd.DataFrame({'file':dirct_t})

test_set=train_datagen.flow_from_directory(directory='./yy',target_size=(128,128))

test_set=train_datagen.flow_from_dataframe(dat,directory='./test',x_col="file"
                                               ,target_size=(128,128),class_mode='binary')

y_pred_t=model.predict_generator(test_set,steps=391)
df_t=pd.Series(data=y_pred_t.reshape(12500))
df_t.to_csv('DogsVsCats.csv',index=False)



'''testdata'''

all_files_t=[]
for i in dirct_t:
    a='./yy/test/'+i
    all_files_t.append(a)

y_pred_test=[]
for i in all_files_t:
    test_image=image.load_img(i,target_size=(128,128))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)

    pred=model.predict(test_image)
    mm=pred.reshape(1)
    mm=mm.tolist()
    y_pred_test.append(mm)
    print(i)
    
values=[]
for i in y_pred_test:
    values.append(i[0])
    
df_t=pd.Series(data=values)
df_t.to_csv('DogsVsCats5.csv',index=False)



'''Testing''''


test_set=train_datagen.flow_from_directory(directory='./yy',target_size=(128,128),batch_size=1)

y_pred1=model.predict(test_set)
y_pred_t=model.predict_generator(test_set,steps=12500,verbose=1)

x=test_set.filenames
x1=test_set.image_data_generator


labels_t=[]
for i in x:
    a=i.split('.')
    labels_t.append(a[0])

labels_t1=[]
for i in labels_t:
    a=i.split('\\')
    labels_t1.append(a[1])
    

df_t=pd.Series(data=labels_t1)
df_t.to_csv('DogsVsCatslables1.csv',index=False)

df_t=pd.Series(data=y_pred_t.reshape(12500))
df_t.to_csv('DogsVsCats4.csv',index=False)




'''OpenCV'''

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

files=os.listdir('./train')

labels=[]
for i in files:
    a=i.split('.')
    labels.append(a[0])
    
d={'Labels':labels}
dataframe=pd.DataFrame(data=d)
    
dire=[]
for i in files:
    files1='./train/'+i
    dire.append(files1)

dataset=np.empty([25000,128,128,3])
count=0
for i1 in dire:
    img=cv2.imread(i1,1)
    resized_image=cv2.resize(img,(128,128))
    dataset[count]=resized_image/255.
    count=count+1


import keras
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense

model=Sequential()

model.add(Convolution2D(32,5,5,input_shape=(128,128,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dense(output_dim=512,activation='relu'))
model.add(Dense(output_dim=256,activation='relu'))
model.add(Dense(output_dim=128,activation='relu'))
model.add(Dense(output_dim=1,activation='sigmoid'))

model.summary()


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



y_train=dataframe.iloc[:,0].map({'cat':0,'dog':1})


model.fit(dataset,y_train,epochs=6,batch_size=96)

model.save_weights('model.h5')

'''TestData''''
files_test=os.listdir('./yy/test')

labels_test=[]
for i in files_test:
    a=i.split('.')
    labels_test.append(a[0])
    
d={'Labels':labels}
dataframe=pd.DataFrame(data=d)
    
dire_test=[]
for i in files_test:
    files1='./yy/test/'+i
    dire_test.append(files1)

dataset_test=np.empty([12500,128,128,3])
count=0
for i1 in dire_test:
    img=cv2.imread(i1,1)
    resized_image=cv2.resize(img,(128,128))
    dataset_test[count]=resized_image/255.
    count=count+1

#Train Data test
   
y_pred_train=model.predict(dataset)
    
y_pred_t=(y_pred_train>0.5)
y_train_t=y_train>0.5


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred_t,y_train_t)


#Test Data
y_pred=model.predict(dataset_test)


sub_label=pd.Series(files_test)
sub_pred=pd.Series(y_pred.reshape(12500,))

sub_label.to_csv('DogsVsCatslablesFinal.csv',index=False)

sub_pred.to_csv('DogsVsCatsPredFinal.csv',index=False)







'''OpenCV for more accuracy'''


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

files=os.listdir('./train')

labels=[]
for i in files:
    a=i.split('.')
    labels.append(a[0])
    
d={'Labels':labels}
dataframe=pd.DataFrame(data=d)
    
dire=[]
for i in files:
    files1='./train/'+i
    dire.append(files1)

dataset=np.empty([25000,128,128,3])
count=0
for i1 in dire:
    img=cv2.imread(i1,1)
    resized_image=cv2.resize(img,(128,128))
    dataset[count]=resized_image/255.
    count=count+1


import keras
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense

model=Sequential()

model.add(Convolution2D(32,5,5,input_shape=(128,128,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(64,5,5,input_shape=(128,128,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(128,5,5,input_shape=(128,128,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dense(output_dim=2048,activation='relu'))
model.add(Dense(output_dim=512,activation='relu'))
model.add(Dense(output_dim=128,activation='relu'))
model.add(Dense(output_dim=1,activation='sigmoid'))

model.summary()


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



y_train=dataframe.iloc[:,0].map({'cat':0,'dog':1})


model.fit(dataset,y_train,epochs=6,batch_size=96)

model.save_weights('model_1.h5')

'''TestData''''
files_test=os.listdir('./yy/test')

labels_test=[]
for i in files_test:
    a=i.split('.')
    labels_test.append(a[0])
    
d={'Labels':labels}
dataframe=pd.DataFrame(data=d)
    
dire_test=[]
for i in files_test:
    files1='./yy/test/'+i
    dire_test.append(files1)

dataset_test=np.empty([12500,128,128,3])
count=0
for i1 in dire_test:
    img=cv2.imread(i1,1)
    resized_image=cv2.resize(img,(128,128))
    dataset_test[count]=resized_image/255.
    count=count+1

#Train Data test
   
y_pred_train=model.predict(dataset)
    
y_pred_t=(y_pred_train>0.5)
y_train_t=y_train>0.5


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred_t,y_train_t)


#Test Data
y_pred=model.predict(dataset_test)


sub_label=pd.Series(files_test)
sub_pred=pd.Series(y_pred.reshape(12500,))


sub_label.to_csv('DogsVsCatslablesFinal.csv',index=False)


sub_pred.to_csv('DogsVsCatsPredFinal.csv',index=False)















'''OpenCV2 Kaggle'''

import numpy as np 
import pandas as pd
import cv2
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.utils import to_categorical
import matplotlib.pyplot as plt 
from random import shuffle
np.set_printoptions(suppress=True)
import os

IMG_SIZE = 120
X_Train_orig = []
Y_Train_orig = []
for i in os.listdir('./train/'):
    label = i.split('.')[-3]
    if label == 'cat':
        label = 0
    elif label == 'dog':
        label = 1
    img = cv2.imread('./train/'+i, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    #img = img / 255
    X_Train_orig.append([np.array(img), np.array(label)])

np.save('Training_Data.npy', X_Train_orig)


X = np.array([i[0] for i in X_Train_orig]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = np.array([i[1] for i in X_Train_orig])
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.1)
print('Shape of X_train is :', X_train.shape)
print('Shape of Y_train is :', Y_train.shape)
print('Shape of X_val is :', X_val.shape)
print('Shape of Y_val is :', Y_val.shape)


import matplotlib.pyplot as plt 
plt.figure(figsize=(20,20))   
for i in range(5):         
    plt.subplot(5, 10, i+1) 
    plt.imshow(X_val[i,:,:,:]) 
    plt.title('DOG' if Y_val[i] == 1 else 'CAT')   
    plt.axis('off') 
    
    
    
def Keras_Model(input_shape):    
    
    X_input = Input(input_shape)
    
    # First Layer
    X = Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', name = 'conv0')(X_input) 
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X) 
    
    X = Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', name = 'conv1')(X) 
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((3, 3), name='max_pool_0')(X)
    X = Dropout(0.3)(X)
    
    # Second Layer
    X = Conv2D(128, (3, 3), strides = (1, 1), padding = 'same', name = 'conv3')(X) 
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(128, (3, 3), strides = (1, 1), padding = 'same', name = 'conv4')(X) 
    X = BatchNormalization(axis = 3, name = 'bn4')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(128, (3, 3), strides = (1, 1), padding = 'same', name = 'conv5')(X) 
    X = BatchNormalization(axis = 3, name = 'bn5')(X)
    X = Activation('relu')(X)
     
    X = MaxPooling2D((3, 3), name='max_pool_1')(X)
    X = Dropout(0.3)(X)
    
    # Fourth Convolutional Layer
    X = Conv2D(256, (3, 3), strides = (1, 1), padding = 'same', name = 'conv6')(X) 
    X = BatchNormalization(axis = 3, name = 'bn6')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(256, (3, 3), strides = (1, 1), padding = 'same', name = 'conv7')(X) 
    X = BatchNormalization(axis = 3, name = 'bn7')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(256, (3, 3), strides = (1, 1), padding = 'same', name = 'conv8')(X) 
    X = BatchNormalization(axis = 3, name = 'bn8')(X)
    X = Activation('relu')(X)

 
    X = MaxPooling2D((3, 3), name='max_pool_2')(X)
    X = Dropout(0.3)(X)
    
    X = Conv2D(512, (3, 3), strides = (1, 1), padding = 'same', name = 'conv10')(X) 
    X = BatchNormalization(axis = 3, name = 'bn10')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(512, (3, 3), strides = (1, 1), padding = 'same', name = 'conv11')(X) 
    X = BatchNormalization(axis = 3, name = 'bn11')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(512, (3, 3), strides = (1, 1), padding = 'same', name = 'conv12')(X) 
    X = BatchNormalization(axis = 3, name = 'bn12')(X)
    X = Activation('relu')(X)

    
    X = MaxPooling2D((3, 3), name='max_pool_3')(X)
    X = Dropout(0.3)(X)
    
    # Flatten the data.
    X = Flatten()(X)
    # Dense Layer
    X = Dense(4096, activation='relu', name='fc1')(X)
    X = Dropout(0.5)(X)
    X = Dense(1024, activation='relu', name='fc2')(X)
    X = Dropout(0.5)(X)
    X = Dense(256, activation='relu', name='fc3')(X)
    # Using softmax function to get the output
    X = Dense(1, activation='sigmoid', name='fc4')(X)
    
    model = Model(inputs = X_input, outputs = X, name='model')
    
    return model



Keras_Model = Keras_Model(X_train.shape[1:4])

from keras.optimizers import Adam
epochs = 20
batch_size = 64
lrate = 0.001
decay = lrate/epochs
optimizer = Adam(lr=lrate, epsilon=1e-08, decay = decay)


Keras_Model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=1, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0000001)

preds = Keras_Model.evaluate(X_train, Y_train)
print ("Loss = " + str(preds[0]))
print ("Train set Accuracy = " + str(preds[1]))


preds = Keras_Model.evaluate(X_train, Y_train)
print ("Loss = " + str(preds[0]))
print ("Train set Accuracy = " + str(preds[1]))

preds_val = Keras_Model.evaluate(X_val, Y_val)
print ("Loss = " + str(preds_val[0]))
print ("Validation Set Accuracy = " + str(preds_val[1]))



X_Test_orig = []
for i in os.listdir('../input/test/'):
    label = i.split('.')[-2]
    img = cv2.imread('../input/test/'+i, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE), interpolation = cv2.INTER_CUBIC)
    #img = img / 255
    X_Test_orig.append([np.array(img), np.array(label)])

np.save('Test_Data.npy', X_Train_orig)
X_test = np.array([i[0] for i in X_Test_orig]).reshape(-1,IMG_SIZE, IMG_SIZE, 3)
Label = np.array([i[1] for i in X_Test_orig])
X_test = X_test / 255
classes = Keras_Model.predict(X_test, batch_size = batch_size)
prediction = pd.DataFrame()
prediction['id'] = Label
prediction['label'] = classes

prediction.to_csv('submission.csv', index = False)



'''TEST SUbmission'''

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

files_test=os.listdir('./yy/test')

labels_test=[]
for i in files_test:
    a=i.split('.')
    labels_test.append(a[0])
    
d={'Labels':labels}
dataframe=pd.DataFrame(data=d)
    
dire_test=[]
for i in files_test:
    files1='./yy/test/'+i
    dire_test.append(files1)

dataset_test=np.empty([12500,128,128,3])
count=0
for i1 in dire_test:
    img=cv2.imread(i1,1)
    resized_image=cv2.resize(img,(128,128))
    dataset_test[count]=resized_image/255.
    count=count+1

plt.imshow(dataset_test[901])
    
results=classifier.predict(dataset_test)

id=[]
for i in files_test:
    a=i.split('.')
    id.append(a[0])
    
prediction = pd.DataFrame()
prediction['id'] = labels_test
prediction['label'] = results
    
prediction.to_csv('submission10.csv', index = False)



