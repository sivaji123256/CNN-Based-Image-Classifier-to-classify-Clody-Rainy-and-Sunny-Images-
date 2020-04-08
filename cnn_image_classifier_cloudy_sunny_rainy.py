
# Import libraries
import os,cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import train_test_split
from keras import backend as K
#K.set_image_dim_ordering('th')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.layers import LSTM
from keras.layers import Embedding

PATH = os.getcwd()
print(PATH)
# Define data path
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)

img_rows=128
img_cols=128
num_channel=1
num_epoch=1
# Define the number of classes
num_classes = 3

labels_name={'cloudy':0,'rain':1,'sunrise':2}

img_data_list=[]
labels_list = []

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loading the images of dataset-'+'{}\n'.format(dataset))
	label = labels_name[dataset]
	print(label)
	for img in img_list:
		input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
		#print(input_img.shape)
		try:
		   input_img = cv2.resize(input_img,(400,600),interpolation=cv2.INTER_CUBIC)
		   print(input_img.shape)
		#plt.imshow(input_img)
		#input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
		#input_img_resize=cv2.resize(input_img,(128,128))
		   img_data_list.append(input_img)
		   labels_list.append(label)
		except:
			None
print(len(img_data_list))
img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
#img_data /= 255
#exit()
labels = np.array(labels_list)
# print the count of number of samples for different classes
print(np.unique(labels,return_counts=True))
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)
#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Defining the model
input_shape=img_data[0].shape
print(input_shape)
#X_train=np.array(X_train)
#y_train=np.array(y_train)
#x_shape=X_train.shape
#print(x_shape)
#y_shape=y_train.shape
#print(y_shape)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#X_train=X_train.reshape(X_train.shape[0],input_shape)
#y_train=y_train.reshape(y_train.shape[0],input_shape)
#input_shape=np.array(input_shape)
#input_shape=np.reshape(X_train[0],input_shape[0],input_shape[1],input_shape[2])					
model = Sequential()

model.add(Convolution2D(32, (3,3),border_mode='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
#model.add(Convolution2D(64, 3, 3))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
#print("model outputs:",model.outputs)
#LSTM
#max_features = 1024
#timesteps=148
#data_dim=98
#model.add(Embedding(max_features, output_dim=256))
#model.add(LSTM(32,return_sequences=False,input_shape=(148,98)))
#model.add(LSTM(32, return_sequences=True,
               #input_shape=(timesteps, data_dim)))
#model.add(LSTM(max_features,output_dim=256))
#model.add(LSTM(128,return_sequences=True))
#model.add(LSTM(32, return_sequences=True,
           #input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
#model.add(LSTM(32, return_sequences=False))  # returns a sequence of vectors of dimension 32
#model.add(Dense(10, activation='softmax'))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])



# Training
hist = model.fit(X_train, y_train, batch_size=16, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test))

#hist = model.fit(X_train, y_train, batch_size=32, nb_epoch=20,verbose=1, validation_split=0.2)


# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(num_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

#%%

# Evaluating the model

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

#test_image = X_test[0:1]
#cv2.imshow(test_image)
#print (test_image.shape)

#print(model.predict(test_image))
#print(model.predict_classes(test_image))
#print(y_test[0:1])

# Testing a new image
test_image = cv2.imread('test.jpg')
#test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
#test_image=cv2.resize(test_image,(128,128))
test_image = cv2.resize(test_image,(400,600),interpolation=cv2.INTER_CUBIC)
test_image=test_image.reshape(1,test_image.shape[0],test_image.shape[1],3)
print(test_image.shape)
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.shape)
   
# Predicting the test image
print((model.predict(test_image)))
print(model.predict_classes(test_image))

