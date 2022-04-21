import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import seaborn as sn
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.losses import categorical_crossentropy
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

EPOCHS = 100
EPOCHS_LENET = 10
IMG_WIDTH = 128
IMG_HEIGHT = 128
NUM_CATEGORIES = 16
INPUT_IMAGE_DIR="images/v-i_images/valid_images/"
INPUT_MODEL_DIR="models"
OUTPUT_MODEL_DIR="models"
SAVE_DIR="images"

def cnn_main(model=None):
    available_options=[1,2,3,4]
    print("\nChoose an option below: \n" ,
        "\n(1) Train Lenet model with MNIST dataset ",
        "\n(2) Train VI-Lenet model with binary VI images (GPU recommended)",
        "\n(3) Evaluate perfomance of VI- Lenet model with test images",
        "\n(4) Exit\n")
    while True:
        try:
            ans = int(input("Option: "))
            if ans in available_options:
                break
            print("Invalid option.")
        except ValueError:
            print("Invalid option.")       
    if ans==1:
        available_options=['Y','N']
        print('\nTraining Lenet model on MNIST dataset...')
        x_train,y_train,x_test,y_test=process_data_MNIST()
        lenet_model=LeNet_model(x_train)
        if not os.path.exists(f"{OUTPUT_MODEL_DIR}/models_architecures"):
            os.makedirs(f"{OUTPUT_MODEL_DIR}/models_architecures/") 
        tf.keras.utils.plot_model(
            lenet_model,
            to_file=f"{OUTPUT_MODEL_DIR}/models_architecures/lenet_model.png",
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=96,
            )
        if not os.path.exists(f"{OUTPUT_MODEL_DIR}/models_architecures"):
            os.makedirs(f"{OUTPUT_MODEL_DIR}/models_architecures/") 
        save_summary(lenet_model,'lenet')
        print(f"Model architecture saved in '{OUTPUT_MODEL_DIR}/models_architecures/'\n")
        history = [lenet_model.fit(x_train, y_train, epochs=EPOCHS_LENET)]
        plot_training_results(lenet_model,history,EPOCHS_LENET,'history_lenet')
        save_model(lenet_model)
    if ans==2:
        """ 
        Requires LeNet model trained and saved as .h5 file in '.../models/' directory.
        OBS: due to long process of training (100 epochs and 4 times with K-FOLD),
        it's recommended use of GPU. If not available on your machine, try online
        platforms that offers GPU services online (Kaggle, Google Colab, ...). Just paste
        cnn.py content on editor of platform and change directories only to get trained model, then save it 
        to local directory (".../models/"). Maybe will need to upload V-I images for training.

        Instructions for training and saving model with Kaggle's GPU:
        1) Go to https://www.kaggle.com/ and create user account;
        2) Click on 'Code' in left menu and then on 'New Notebook';
        3) Click on plus sign '+' to create a new cell and paste the content of this file there;
        4) Zip the V-I trajectories file and upload to Kaggle by clicking on '+ Add data' on right side menu
        and then 'Upload a Dataset';
        5) Give it a name (example: 'VI_images') and click on 'Create';
        6) Upload the .h5 file of lenet model located in '.../model/' directory the same way;
        7) On the right side menu, below 'Input Data', click on your dataset and locate 'valid_images' directory;
        8) On the right of 'valid_images', click on 'Copy file path' and paste it to 'INPUT_IMAGE_DIR' in code;
        9) Also copy the path of lenet model and paste it to 'INPUT_MODEL_DIR' in code;
        10) Copy the path of 'kaggle/working' below 'Output' on the right side menu and 
        paste it to 'SAVE_DIR' and 'OUTPUT_MODEL_DIR' in code; 
        11) In 'Settings', choose 'GPU' on 'Accelerator' menu;  
        12) Execute code by clicking on 'Run all';
        13) After code execution, give the model a name and save it;
        14) Model will be save inside 'kaggle/working' as .h5 file, download it to '../models/' directory on local machine.
        """

        print('\nImporting Lenet model...')
        lenet_model=import_model()
        x_train,y_train,x_test,y_test,le ,labels_literal=process_data_VI_Images(k_folds=True)
        # Removes last layer of Lenet
        lenet_model.pop()
        for layer in lenet_model.layers:
            layer.trainable=False        
        VI_model=get_model(x_train[0])  
        output_vi=VI_model.layers[-1].output 
        output= lenet_model(inputs=output_vi)        
        output=layers.Dense(units=NUM_CATEGORIES,activation='softmax',name="last_layer")(output)
        complete_model = tf.keras.Model(inputs = VI_model.input, outputs = output)
        complete_model.summary() 
        save_summary(complete_model,'V-I_lenet')
        if not os.path.exists(f"{OUTPUT_MODEL_DIR}/models_architecures"):
            os.makedirs(f"{OUTPUT_MODEL_DIR}/models_architecures/") 
        tf.keras.utils.plot_model(
            complete_model,
            to_file=f"{OUTPUT_MODEL_DIR}/models_architecures/V-I_model.png",
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=96,
            )
        print(f"Model architecture saved in '{OUTPUT_MODEL_DIR}/models_architecures/'\n")
        complete_model.compile(
            optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
            )
        history=[]
        scores=[]
        for i in range(len(x_train)):
            history.append(complete_model.fit(x_train[i], y_train[i], validation_data = (x_test[i], y_test[i]), epochs=EPOCHS))           
            scores.append(complete_model.evaluate(x_test[i], y_test[i], verbose=0))      
        plot_training_results(model,history,EPOCHS,filename='history_VI_lenet_')
        print(scores)
        save_model(complete_model)
    if ans==3:
        VI_lenet_model=import_model()
        prediction_uni=[]
        label_uni=[]
        print(f'Plotting Confusion Matrix...')
        x_train,y_train,x_test,y_test,le,labels_literal=process_data_VI_Images(k_folds=False) 
        prediction=VI_lenet_model.predict(x_test[0])
        for j in range(len(prediction)):
            prediction_uni.append(np.argmax([prediction[j]]))
            label_uni.append(np.argmax([y_test[0][j]]))
    
        confusionMatrix = tf.math.confusion_matrix(labels=label_uni, predictions=prediction_uni,num_classes=16)
        df_cm = pd.DataFrame(confusionMatrix, index = [i for i in labels_literal],
                  columns = [i for i in labels_literal])
        plt.figure(figsize = (12,8))
        sn.heatmap(df_cm, annot=True, fmt='g')
        plt.tight_layout()
        if not os.path.exists(f"{SAVE_DIR}/confusion_matrix"):
            os.makedirs(f"{SAVE_DIR}/confusion_matrix/") 
        plt.savefig(f"{SAVE_DIR}/confusion_matrix/confusion_matrix.png",dpi=128)
        print(f"Confusion Matrix saved in '{SAVE_DIR}/confusion_matrix/'\n")
    if ans==4:
        exit()

def import_model():
    while True:
        filename = input("\nModel name: ")
        try:
            model = tf.keras.models.load_model(f"{INPUT_MODEL_DIR}/{filename}.h5")
            model.summary()
            print('Model imported.')
            break
        except OSError:
            print(f"Model with name {filename} does not exist in current directory")
            cnn_main()
    return model

def save_model(model=None):
    available_options=['Y','N']
    print('\nModel Trained\n\nDo you wish to save model (Y/N)?')
    while True:
        ans = input("").upper()
        if ans in available_options:
            break
        print("Invalid option.")
    if ans=='Y':
        filename = input("\nModel name: ")  
        if not os.path.exists(f"{OUTPUT_MODEL_DIR}"):
            os.makedirs(f"{OUTPUT_MODEL_DIR}/")         
        model.save(f"{OUTPUT_MODEL_DIR}/{filename}.h5",save_format='h5')
        print(f"Model saved in '{OUTPUT_MODEL_DIR}'\n")
        cnn_main(model)
    else:
        cnn_main()   

def save_summary(model,filename):
    if not os.path.exists(f"{OUTPUT_MODEL_DIR}/models_summaries"):
            os.makedirs(f"{OUTPUT_MODEL_DIR}/models_summaries/") 
    with open(f'{OUTPUT_MODEL_DIR}/summary_{filename}.txt', 'a') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"Model summary saved in '{OUTPUT_MODEL_DIR}/models_summaries/'\n")
    
def load_data(data_dir):   
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category.\
    Inside each category directory will be some number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 1. `labels` should
    be a list of strings (ex: 'Laptop', 'Blender', ...), \
    representing the categories for each of the
    corresponding `images/loads`.   
    """
    images=[]
    labels=[]
    labels_literal=[]
    for category in os.listdir(data_dir):  
        for img_file in os.listdir(os.path.join(data_dir,category)):
            img = cv2.imread(os.path.join(data_dir,category,img_file))
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = np.array(img)
            images.append(img)
            labels.append(str(category)) 
        labels_literal.append(str(category))
    return (images,labels),labels_literal

def process_data_MNIST():
    # Load MNIST dataset and separate images and labels into train and test samples
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Check train and test samples dimensions
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    # Normalize images pixels to be between 0 and 1 in grayscale
    x_train = x_train/255
    x_test = x_test/255
    # Add one dimension for color channel (in this case, \
    # black and white images have 1 channel), because\
    # convolution layers require input images to be in the format:\
    # (n_images, height, width, n_channels)
    x_train = tf.expand_dims(x_train, axis=3, name=None)
    x_test = tf.expand_dims(x_test, axis=3, name=None)
    # Lenet has 10 outputs (0-9 digits)
    num_classes=10
    # Change labels to be in the format of categorical arrays.
    # Ex: [0 0 0 1 0 0 0 0 0 0 0] = 3
    #     [0 0 0 0 0 0 0 0 0 0 1] = 9
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return x_train,y_train,x_test,y_test

def process_data_VI_Images(k_folds=True):
    le = preprocessing.LabelEncoder()
    (images, labels), labels_literal = load_data(INPUT_IMAGE_DIR)
    X=np.array(images)
    Y=np.array(labels)
    skf = StratifiedKFold(n_splits=4,shuffle=True)
    i=0
    X_train=[]
    X_test=[]
    Y_train=[]
    Y_test=[]
    Y_train_literal=[]
    Y_test_literal=[]
    for train_index, test_index in skf.split(X,Y):
        X_train.append(X[train_index]) 
        X_test.append(X[test_index])
        Y_train_literal.append(Y[train_index]) 
        Y_test_literal.append(Y[test_index])
        X_train[i] = X_train[i].reshape(X_train[i].shape[0], IMG_HEIGHT, IMG_WIDTH, 1)
        X_test[i] = X_test[i].reshape(X_test[i].shape[0], IMG_HEIGHT, IMG_WIDTH, 1)        
        le.fit(Y_test_literal[i])
        Y_test.append(le.transform(Y_test_literal[i]))  
        le.fit(Y_train_literal[i])
        Y_train.append(le.transform(Y_train_literal[i]))    
        num_classes=16
        Y_train[i] = to_categorical(Y_train[i], num_classes).astype('int')
        Y_test[i] = to_categorical(Y_test[i], num_classes).astype('int')
        i+=1
        if k_folds==False:
            break
        
    return X_train,Y_train,X_test,Y_test,le,labels_literal

def get_model(x_train):
    model = models.Sequential([
        tf.keras.Input(shape=x_train[0].shape,name='Input'),  
        layers.experimental.preprocessing.Rescaling(1./255,name='Rescaling'),    
        layers.Conv2D(6, kernel_size=(3,3),strides=(1,1), activation='tanh',input_shape=x_train[0].shape,padding='valid',name='Conv_1'), #128x128
        layers.SpatialDropout2D(1/6,name='Dropout'),
        layers.Conv2D(42, kernel_size=(3,3),strides=(1,1), activation='tanh',input_shape=x_train[0].shape,padding='valid',name='Conv_2'),#126x126
        layers.MaxPooling2D(pool_size=(2,2), padding='valid',name='Pooling_1'),                                   #124x124
        layers.Conv2D(84, kernel_size=(3,3),strides=(1,1), activation='tanh',padding='valid',name='Conv_3'),  #62x62
        layers.MaxPooling2D(pool_size=(2,2), padding='valid',name='Pooling_2'),                                   #60x60
        layers.Conv2D(1, kernel_size=(3,3),strides=(1,1), activation='tanh',padding='valid',name='Conv_4'),  #30x30
    ],name="LeNet Extension")
    model.summary()

    return model

def LeNet_model(x_train):
    """
    Returns a compiled convolutional neural network LeNet model.
    """
    print(x_train[0].shape)
    model = models.Sequential([
        layers.InputLayer(input_shape=x_train[0].shape,name='Input_Lenet'),
        layers.Conv2D(6, kernel_size=(5,5),strides=(1,1), activation='tanh',padding='same',name='Conv1_Lenet'),
        layers.AveragePooling2D(pool_size=(2,2),strides=(2, 2), padding='valid',name='Pooling_1_Lenet'),
        layers.Conv2D(16, kernel_size=(5,5),strides=(1,1), activation='tanh',padding='valid',name='Conv_2_Lenet'),
        layers.AveragePooling2D(pool_size=(2,2),strides=(2, 2), padding='valid',name='Pooling_2_Lenet'),
        layers.Conv2D(120, kernel_size=(5,5),strides=(1,1), activation='tanh',padding='valid',name='Conv_3_Lenet'),
        layers.Flatten(name="Flatten"),
        layers.Dense(84, activation='tanh',name='Dense_Lenet'),
        layers.Dense(10, activation='softmax',name='Lenet_Output')
    ],name="LeNet")
    
    model.summary()
    model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])

    return model

def plot_training_results(model,history,epochs,filename):    
    plt.figure(figsize=(16, 8))
    subtitles=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']
    j=1
    acc=[]
    loss=[]
    for i in range(len(history)):  
        acc.append(history[i].history['accuracy'])
        loss.append(history[i].history['loss'])
        epochs_range = range(epochs)
        plt.subplot(len(history), 2, j)
        plt.title(subtitles[j-1],fontsize=10,pad=10)
        plt.plot(epochs_range, acc[i], label=f'Training Accuracy {i+1}')      
        plt.legend(loc='lower right')
        j+=1
        plt.subplot(len(history), 2, j)
        plt.title(subtitles[j-1],fontsize=10,pad=10)
        plt.plot(epochs_range, loss[i], label=f'Training Loss {i+1}')
        plt.legend(loc='upper right')
        j+=1
    plt.tight_layout()    
    if not os.path.exists(f"{OUTPUT_MODEL_DIR}/training_history"):
        os.makedirs(f"{OUTPUT_MODEL_DIR}/training_history/")
    plt.savefig(f"{OUTPUT_MODEL_DIR}/training_history/{filename}.png",dpi=128)
    print(f"Training history saved in '{OUTPUT_MODEL_DIR}/training_history/'\n")

if __name__ == "__main__":
    cnn_main()






