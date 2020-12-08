# Importing modules 
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
import csv
from sklearn.model_selection import train_test_split
from kerastuner.tuners import RandomSearch
import kerastuner.engine.hyperparameters
import time
from tqdm import tqdm

#LOG_DIR = f"trails/{int(time.time())}"

train_images = []
img_names=[]
img_names2=[]
available_files = []
train_labels = []
shape = (100,100)  
train_path = 'JPEGImages/'


def load_data(data):
    X = []
    y = []
    #To traverse every blood cell type folders in data 
    for bloodcell_type in os.listdir(data):
        if not bloodcell_type.startswith('.'):
            if bloodcell_type in ['NEUTROPHIL']:
                label = 0
            elif bloodcell_type in ['EOSINOPHIL']:
                label = 1
            elif bloodcell_type in ['MONOCYTE']:  
                label = 2
            elif bloodcell_type in ['LYMPHOCYTE']:
                label = 3
    
            #This loop is to traverse to every blood cell type in train data
            for filename in tqdm(os.listdir(data + '/' + bloodcell_type)):
                #To read every image in every folder
                image = cv2.imread(data +'/'+ bloodcell_type + '/' + filename)
                
                #If the image is found
                if image is not None:
                    #To resize the random sized images into a fixed size of 100x100x3
                    image = cv2.resize(image, shape)
                    #Changing the datatype into array to process through the cnn algorithm
                    image_data_as_arr = np.asarray(image)
                    #Appending the data in the empty lists of X and y
                    X.append(image_data_as_arr)
                    y.append(label)
    X = np.asarray(X)
    y = pd.get_dummies(y).values
    #y = np.asarray(y)
    
    return X,y


#Loading the train and test data
X_train, Y_train = load_data(r'dataset2-master/dataset2-master/images/TRAIN/')
X_val, Y_val = load_data(r'dataset2-master/dataset2-master/images/TEST/')
X_test, Y_test = load_data(r'dataset2-master/dataset2-master/images/TEST_SIMPLE/')

'''
f = open('labels.csv')
csv_f = csv.reader(f, delimiter=",", quotechar='"')

for filename in os.listdir(train_path):
    available_files.append(filename)

for row in csv_f:
    if row[0] in available_files:
        train_labels.append(row[1])
        img_names.append(row[0])

for filename in img_names:
    img_names2.append(filename)
    img = cv2.imread(os.path.join(train_path,filename))
    # Resize all images to a specific shape
    img = cv2.resize(img,shape) 
    train_images.append(img)


label_types = {}
count = 0

for label in train_labels:
    if label in label_types:
        continue
    else:
        label_types[count] = label
        count+=1

labels = train_labels

# Converting labels into One Hot encoded sparse matrix
train_labels = pd.get_dummies(train_labels).values

# Converting train_images to array
train_images = np.array(train_images)

# Splitting Training data into train and validation dataset
x_train,x_dev,y_train,y_dev = train_test_split(train_images,train_labels, test_size=0.203, random_state=1)

x_val, X_test = np.split(x_dev, 2)
y_val, Y_test = np.split(y_dev, 2)


# Visualizing Training data
#print(train_labels[0])
#plt.imshow(train_images[1]) 
'''

def CNN(imgs,img_labels,test_imgs,test_labels,stride):

    #lineralreg = l2(0.010)
    #Number of classes (4)
    num_classes = len(img_labels[0])
    
    epochs = 30
    
    #Size of image
    img_rows,img_cols=imgs.shape[1],imgs.shape[2]
    input_shape = (img_rows, img_cols, 3)
    
    #Creating the model
    model = Sequential()
    
    #convolution layer
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape,
                     strides=stride))
    
    model.add(MaxPool2D(pool_size=(2,2)))
    #Dropout(0.20)
    
    #convolution layer#convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))       
    #model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    #model.add(Dropout(0.20))
    
    #convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    #BatchNormalization()
    model.add(MaxPool2D(pool_size=(2, 2)))
    #model.add(Dropout(0.40))
    
    #Convert the matrix to a fully connected layer
    model.add(Flatten())
    
    model.add(Dense(128, activation = 'relu'))
    
    #Final dense layer on which softmax function is performed
    model.add(Dense(num_classes, activation='softmax'))
    
    #Model parameters
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    
    #Evaluate the model on the test data before training your model
    score = model.evaluate(test_imgs,test_labels, verbose=1)
    
    print('\nKeras CNN Validation accuracy before training:', score[1],'\n')
    
    #The model details
    history = model.fit(imgs,img_labels,
                        shuffle = True, 
                        epochs=epochs, 
                        validation_data = (test_imgs, test_labels))
    
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    #Evaluate the model on the test data after training your model
    score = model.evaluate(test_imgs,test_labels, verbose=1)
    print('\nKeras CNN Validation accuracy after training:', score[1],'\n')
    
    #Predict the labels from test data
    y_pred = model.predict(test_imgs)
    Y_pred_classes = np.argmax(y_pred,axis=1) 
    Y_true = np.argmax(test_labels,axis=1)
    
    #Correct labels
    for i in range(len(Y_true)):
        if(Y_pred_classes[i] == Y_true[i]):
            print("The predicted class is : " , Y_pred_classes[i])
            print("The real class is : " , Y_true[i])
            break
            
    #The confusion matrix made from the real Y values and the predicted Y values
    #confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    #print("The confusion matrix is : \n",confusion_mtx)
    
    #Summary of the model
    model.summary()
    
    return model
   
model = CNN(X_train,Y_train,X_val,Y_val,1);

evaluate = model.evaluate(X_val,Y_val, verbose=1)
print('Val LOSS: ', evaluate[0])
print('Val ACCURACY: ', evaluate[1])

# Testing predictions and the actual label
#checkImage = X_test[0:1]
#checklabel = Y_test[0:1]

evaluate = model.evaluate(X_test,Y_test, verbose=1)
print('TEST LOSS: ', evaluate[0])
print('TEST ACCURACY: ', evaluate[1])

predict = model.predict(np.array(X_test))
pred = np.argmax(predict, axis = 1)[:] 
label = np.argmax(Y_test,axis = 1)[:] 

print("Prediction:")
print(pred[:20]) 
print("Actual:")
print(label[:20])


#print("Actual :- ",checklabel)
#print("Predicted :- ",label_types[np.argmax(predict)])
'''

def build_model(hp):
    # Creating a Sequential model 'relu', 'tanh', 
    model= Sequential()
    model.add(Conv2D(kernel_size=(3,3), filters=hp.Int("input_units", min_value=2, max_value=20, step=2), activation=hp.Choice('conv_activation_top',values=['sigmoid']), input_shape=(100,100,3)))
    model.add(MaxPool2D(2,2))
    for i in range(hp.Int("n_layers",1,2)):
        model.add(Conv2D(kernel_size=(hp.Int(f"kernel_units{i}", min_value=1, max_value=4, step=1)), filters=hp.Int(f"conv_{i}_units", min_value=64, max_value=128, step=32), activation=hp.Choice(f"conv_activation_{i}",values=['tanh'])))
        model.add(MaxPool2D(2,2))
        
    model.add(Flatten())
    
    for i in range(hp.Int("n_layers",1,2)):
        model.add(Dense(hp.Int(f"dense_{i}_units", min_value=8, max_value=32, step=4), activation=hp.Choice(f"dense_activation_{i}",values=['relu'])))
        
    
    model.add(Dense(5,activation = 'softmax'))
    
    model.compile(
                  loss='categorical_crossentropy', 
                  metrics=['acc'],
                  optimizer='adam'
                 )
    
    # Model Summary
    model.summary()
    return model

# Training the model
tuner = RandomSearch(
    build_model,
    objective = "val_acc",
    max_trials = 10,
    executions_per_trial = 1,
    directory = LOG_DIR
    )

tuner.search(x = x_train,
             y = y_train,
             epochs=50,
             batch_size=25,
             validation_data = (x_val,y_val))

import pickle

with open(f"tuner_{int(time.time())}.pkl", "wb") as f:
    pickle.dump(tuner,f)

'''