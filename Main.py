from imutils import paths
from fastai.vision import *
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator

'''
Class Name: myModel
Info: This class contains the logic of my model
'''


class myModel:
    '''

    Init the constructor with your : learning-rate:(lr), num of epochs:(ep) and batch size:(bt)
    '''

    def __init__(self, lr, ep, bt):
        self.lr = lr
        self.epochs = ep
        self.batch_size = bt

    '''
    This method will train our model
    args for this method: train and test data - trainAug, trainX, testX, trainY, testY
    '''

    def train(self, trainA, trainX, testX, trainY, testY):
        inputs = Input(shape=(224, 224, 3))

        # First conv block
        x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        # Second conv block
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        # Third conv block
        x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        # Fourth conv block
        x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Dropout(rate=0.2)(x)

        # Fifth conv block
        x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Dropout(rate=0.2)(x)

        # FC layer
        x = Flatten()(x)
        x = Dense(units=512, activation='relu')(x)
        x = Dropout(rate=0.7)(x)
        x = Dense(units=128, activation='relu')(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(units=64, activation='relu')(x)
        x = Dropout(rate=0.3)(x)

        # Output layer
        output = Dense(units=2, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=output)
        opt = Adam(lr=self.lr, decay=self.lr / self.epochs)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        H = model.fit_generator(
            trainA.flow(trainX, trainY, batch_size=self.batch_size),
            steps_per_epoch=len(trainX) // self.batch_size,
            validation_data=(testX, testY),
            validation_steps=len(testX) // self.batch_size,
            epochs=self.epochs)
        self.plot(self.epochs, H)  # plot our results
        return H

    def plot(self, epochs, H):
        N = epochs
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
        plt.title("Loss and Accuracy on COVID-19 Dataset")
        plt.xlabel("Epoch")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower right")
        plt.savefig("plot.png")


class Data:

    def __init__(self):
        self.path_set = "./dataset"

    def ceildiv(self, a, b):
        return -(-a // b)

    '''
    Here we plot our images from data set directory
    
    '''

    def plot(self, imspaths, figsize=(300, 10), rows=1, titles=None, maintitle=None):

        f = plt.figure(figsize=figsize)
        if maintitle is not None:
            plt.suptitle(maintitle, fontsize=10)

        for i in range(len(imspaths)):
            sp = f.add_subplot(rows, self.ceildiv(len(imspaths), rows), i + 1)
            sp.axis('off')
            if titles is not None: sp.set_title(titles[i], fontsize=16)
            img = plt.imread(imspaths[i])
            plt.imshow(img)

    '''
    Create an Directory that will contain our data set for training and test 
    '''

    def prepare(self):
        sample = 50
        covid_set = '/home/bariudin77/PycharmProjects/Covid19_CT_Scan/covid-chestxray-dataset-master/'
        metadata = pd.read_csv(covid_set + 'metadata.csv')
        for (i, row) in metadata.iterrows():
            if row["finding"] != "COVID-19" or row["view"] != "PA":
                continue
            img_path = os.path.sep.join([covid_set, "images", row["filename"]])
            if not os.path.exists(img_path):
                continue
            filename = row["filename"].split(os.path.sep)[-1]
            output = os.path.sep.join([f"{self.path_set}/covid", filename])
            shutil.copy2(img_path, output)

        pneumonia_dataset_path = '/home/bariudin77/PycharmProjects/Covid19_CT_Scan/chest-xray-pneumonia/chest_xray'
        base_path = os.path.sep.join([pneumonia_dataset_path, "train", "NORMAL"])
        img_paths = list(paths.list_images(base_path))
        random.seed(50)
        random.shuffle(img_paths)
        img_paths = img_paths[:sample]
        for (i, img_path) in enumerate(img_paths):
            filename = img_path.split(os.path.sep)[-1]
            output = os.path.sep.join([f"{self.path_set}/normal", filename])
            shutil.copy2(img_path, output)
        normal = list(paths.list_images(f"{self.path_set}/normal"))
        covid = list(paths.list_images(f"{self.path_set}/covid"))
        self.plot(normal, rows=5, maintitle="Normal X-ray images")
        self.plot(covid, rows=5, maintitle="Covid-19 X-ray images")

    '''
    Process the data, label and order it 
    
    '''

    def process(self):
        img_paths = list(paths.list_images(self.path_set))
        data = []
        labels = []
        for img_path in img_paths:
            label = img_path.split(os.path.sep)[-2]
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            data.append(image)
            labels.append(label)
        data = np.array(data) / 255.0
        labels = np.array(labels)

        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)
        labels = to_categorical(labels)

        (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels,
                                                          random_state=42)
        trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")
        return (trainAug, trainX, testX, trainY, testY)


if __name__ == "__main__":
    lr = 0.00001
    epoches = 10
    batch_size = 6
    data = Data()
    data.prepare()
    (trainAug, trainX, testX, trainY, testY) = data.process()
    myModel = myModel(lr, epoches, batch_size)
    myModel.train(trainAug, trainX, testX, trainY, testY)
