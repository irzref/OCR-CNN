################################################################################
### test with pytesseract
################################################################################

from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\Admin\\Documents\\S2_SS18\\DL\\CNN'
# Include the above line, if you don't have tesseract executable in your PATH
# Example tesseract_cmd: 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'

# Simple image to string
print(pytesseract.image_to_string(Image.open('ocr.jpg')))

# French text image to string
print(pytesseract.image_to_string(Image.open('ha.png'), lang='fra'))

# Get bounding box estimates
print(pytesseract.image_to_boxes(Image.open('ocr.jpg')))

# Get verbose data including boxes, confidences, line and page numbers
print(pytesseract.image_to_data(Image.open('ocr.jpg')))



pytesseract.image_to_string(Image.open('ocr.jpg'))

imag = Image.open('ocr2.jpg')
pytesseract.image_to_string(imag,lang='eng')


################################################################################
### convert dataset from byte to csv
################################################################################

#path to folder with data
path = '/home/irza/Projects/gzip/'
def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

#converting dataset for letters

convert(path+"emnist-letters-train-images-idx3-ubyte", path+"emnist-letters-train-labels-idx1-ubyte",
        path+"letter_mnist_train.csv", 60000)

convert(path+"emnist-letters-test-images-idx3-ubyte", path+"emnist-letters-test-labels-idx1-ubyte",
        path+"letter_mnist_test.csv", 10000)

#converting dataset for digits

convert(path+"emnist-digits-train-images-idx3-ubyte", path+"emnist-digits-train-labels-idx1-ubyte",
        path+"digits_mnist_train.csv", 60000)
convert(path+"emnist-digits-test-images-idx3-ubyte", path+"emnist-digits-test-labels-idx1-ubyte",
        path+"digits_mnist_test.csv", 10000)


################################################################################
### building the model
################################################################################

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#data source
#https://pjreddie.com/projects/mnist-in-csv/
df_train = pd.read_csv('/home/irza/Projects/gzip/letter_mnist_train.csv')
df_test = pd.read_csv('/home/irza/Projects/gzip/letter_mnist_test.csv')
#df_train = pd.read_csv('./input/mnist_train.csv')
#df_test = pd.read_csv('./input/mnist_test.csv')

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df_train.head()


#every columns but the first
df_train_x = df_train.iloc[:,1:] 
#only the first column
df_train_y = df_train.iloc[:,:1] 

df_test_x = df_test.iloc[:,1:] 
df_test_y = df_test.iloc[:,:1] 


ax = plt.subplots(1,8)
for i in range(0,8):   #validate the first 5 records
    j = i+50
    ax[1][i].imshow(df_train_x.values[j].reshape(28,28), cmap='gray')
    ax[1][i].set_title(df_train_y.values[j])



def cnn_model(result_class_size):
    model = Sequential()
    #use Conv2D to create our first convolutional layer, with 32 filters, 5x5 filter size, 
    #input_shape = input image with (height, width, channels), activate ReLU to turn negative to zero
    model.add(Conv2D(32, (5, 5), input_shape=(28,28,1), activation='relu'))
    #add a pooling layer for down sampling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # add another conv layer with 16 filters, 3x3 filter size, 
    model.add(Conv2D(16, (3, 3), activation='relu'))
    #set 20% of the layer's activation to zero, to void overfit
    model.add(Dropout(0.2))
    #convert a 2D matrix in a vector
    model.add(Flatten())
    #add fully-connected layers, and ReLU activation
    model.add(Dense(130, activation='relu'))
    model.add(Dense(50, activation='relu'))
    #add a fully-connected layer with softmax function to squash values to 0...1 
    model.add(Dense(result_class_size, activation='softmax'))   
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model



#turn the label to 42000 binary class matrix 
arr_train_y = np_utils.to_categorical(df_train_y.iloc[:,0].values)
model = cnn_model(arr_train_y.shape[1])
model.summary()


#normalize 255 grey scale to values between 0 and 1 
df_test_x = df_test_x / 255
df_train_x = df_train_x / 255

#reshape training X and text x to (number, height, width, channels)
arr_train_x_28x28 = np.reshape(df_train_x.values, (df_train_x.values.shape[0], 28, 28, 1))
arr_test_x_28x28 = np.reshape(df_test_x.values, (df_test_x.values.shape[0], 28, 28, 1))

random_seed = 3
#validate size = 8%
split_train_x, split_val_x, split_train_y, split_val_y, = train_test_split(arr_train_x_28x28, arr_train_y, test_size = 0.08, random_state=random_seed)

reduce_lr = ReduceLROnPlateau(monitor='val_acc',factor=0.5,patience=3,min_lr=0.00001)


datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range 
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally
        height_shift_range=0.1  # randomly shift images vertically
        )

datagen.fit(split_train_x)


model.fit_generator(datagen.flow(split_train_x,split_train_y, batch_size=64),
                              epochs = 10, validation_data = (split_val_x,split_val_y),
                              verbose = 2, steps_per_epoch=700 
                              , callbacks=[reduce_lr])



prediction = model.predict_classes(arr_test_x_28x28, verbose=0)
data_to_submit = pd.DataFrame({"ImageId": list(range(1,len(prediction)+1)), "Label": prediction})
data_to_submit.to_csv("result.csv", header=True, index = False)


from random import randrange
#pick 10 images from testing data set
start_idx = randrange(df_test_x.shape[0]-10) 


fig, ax = plt.subplots(2,5, figsize=(15,8))
for j in range(0,2): 
  for i in range(0,5):
     ax[j][i].imshow(df_test_x.values[start_idx].reshape(28,28), cmap='gray')
     ax[j][i].set_title("Index:{} \nPrediction:{}".format(start_idx, prediction[start_idx]))
     start_idx +=1
     

#evaluation metrix for testing data is still missing