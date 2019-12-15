# importing the datasets

from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
import glob

def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['targets'] , 133))
    return dog_files , dog_targets

# load train , test and validation dataset

train_files , train_target = load_dataset('dogImages/train')
valid_files , valid_targets = load_dataset('dogImages/valid')
test_files , test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))


# import human dataset
import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))


#=====================================================================

# detect humans

import cv2
import matplotlib.pyplot as plt

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[3])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x, y, w, h) in faces:
    # add bounding box to color image
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()


# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

#=====================================================================



# dog detection

from keras.applications.resnet50 import resnet50

resnet50_model = resnet50(weights='imagenet')

# pre-process the data
from keras.preprocessing import image
import tqdm

def path_to_tensor(img_path):
    img = image.load_img(img_path , target_size=(224,224))

    # convert the image to 3D tensor (224,224,3)
    x = image.img_to_array(img)

    # convert the 3D tensor to 4D tensor (1,224,224,3) and retuen 4D tensor
    return np.expand_dims(x,axis=0)

def paths_to_file(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


# making predictions with resnet50
from keras.applications.resnet50 import preprocess_input , decode_predictions

def resnet50_predict_labels(img_path):
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(resnet50_model.predict(img))

# writing the dog-detector function
# imagenet dictionary contains keys for dogbreed => [151,268]

def dog_detector(img_path):
    prediction = resnet50_predict_labels(img_path)
    return ((prediction<=268) & (prediction<=151))


# creating the CNN to classify the dog breeds

#pre-process the images

train_tensors = paths_to_file(train_files).astype('float32')/255
valid_tensors = paths_to_file(valid_files).astype('float32')/255
test_tensors = paths_to_file(test_files).astype('float32')/255

# creating the model architecture

from keras.models import Sequential
from keras.layers import Conv2D ,MaxPool2D , Flatten , Dense , Dropout , GlobalAveragePooling2D

model = Sequential()

model.add(Conv2D(filters=16 , kernel_size=2 , strides=1 , padding='valid' , activation='relu' , input_shape=(224,224,3)))
model.add(MaxPool2D(pool_size=2 , padding='valid'))
model.add(Conv2D(filters=32 , kernel_size=2 , activation='relu'))
model.add(MaxPool2D(pool_size=2 , padding='valid'))
model.add(Conv2D(filters=64 , kernel_size=2 , activation='relu'))
model.add(MaxPool2D(pool_size=2 , padding='valid'))
model.add(GlobalAveragePooling2D())
model.add(Dense(133 , activation='softmax'))

model.summary()


# compile the model
model.compile(loss='categorical_crossentropy' , optimizer='remprop' , metrics=['accuracy'])


# train your model

from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(filepath='dog_breed.best.hdf5' , verbose=2 , save_best_only=True)

hist = model.fit(train_tensors , train_target , batch_size=32 , epochs=21 , validation_data=(valid_tensors,valid_targets) , callbacks=[checkpoint] , verbose=2 , shuffle=True )

# load the model with best weights
model.load_weights('dog_breed.best.hdf5')

# evaluate the model
score = model.evaluate(test_tensors , test_targets , verbose=0)
accuracy = 100*score[1]

print("Test Accuracy = " , accuracy)
