from tensorflow import lite
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import random\
    , os
import cv2
import shutil
import keras
import matplotlib.pyplot as plt
from matplotlib.image import imread
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_accuracy
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
from sklearn.utils import shuffle


class preprocess:
    def generate_images_dataset(self, data_path):
        image_paths = []
        labels = []

        for class_label, class_name in enumerate(os.listdir(data_path)):
            class_path = os.path.join(data_path, class_name)
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                image_paths.append(file_path)
                labels.append(class_label)
            print('label{}]n'.format(class_path))
            print('Number of train images : {} \n'.format(len(image_paths)))
            print('Number of train images labels: {} \n'.format(len(labels)))
        print('Number of train images : {} \n'.format(len(image_paths)))
        print('Number of train images labels: {} \n'.format(len(labels)))
        df = pd.DataFrame({"Image_Path": image_paths, "Label": labels})
        print("Dataset Information:")
        print(df.head())
        return df
    def load_data(self):
        df = pd.read_csv("diabetic_retinopathy_dataset.csv")
        diagnosis_binary = {
            0: 'DR',
            1: 'DR',
            2: 'NO_DR',
            3: 'DR',
            4: 'DR',
        }
        diagnosis_classes = {
            2: 'No_DR',
            0: 'Mild',
            1: 'Moderate',
            4: 'Severe',
            3: 'Proliferate_DR',
        }

        df["binary"] = df["Label"].map(diagnosis_binary.get)
        df["type"] = df["Label"].map(diagnosis_classes.get)
        print(df.head())
        return df
    def count_plot(self,df):
        # Create a count plot using seaborn
        plt.figure(figsize=(10, 6))
        sns.countplot(x='type', data=df)
        plt.title('Distribution of Image Labels')
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.show()

        sns.countplot(x='binary', data=df)
        plt.title('Distribution of Image Labels')
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.show()
    def preprocess(self):
        data = []
        labels = []
        width, height = 224, 224

        imagePaths = list(paths.list_images('C:/Users/Bhaskar Marapelli/Downloads/gaussian_filtered_images/gaussian_filtered_images'))

        data = []
        labels = []

        for imagePath in imagePaths:
            label = imagePath.split(os.path.sep)[-2]
            image = load_img(imagePath, target_size=(width, height))
            image = img_to_array(image)
            data.append(image)
            labels.append(label)

        data = np.array(data, dtype="float32")
        labels = np.array(labels)

        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)

        data, labels = shuffle(data, labels)

        print(data.shape)
        print(labels.shape)
        #normalize
        data = data / 255.0
        return data,labels

    def data_split(self,data,labels):
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=.2)

        print("Train images:", x_train.shape)
        print("Test images:", x_test.shape)
        print("Train label:", y_train.shape)
        print("Test label:", y_test.shape)
        return x_train,y_train
    def train_valid_split(self,x_train,y_train):
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.2)

        print("Train images:", x_train.shape)
        print("Test images:", x_val.shape)
        print("Train label:", y_train.shape)
        print("Test label:", y_val.shape)
        return x_train, x_val, y_train, y_val


