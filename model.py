from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.applications.vgg16 import VGG16
from keras import layers
import matplotlib.pyplot as plt


class Model:
    def model(self):
        # Create VGG16 base model with pre-trained weights (excluding top layers) and set input shape
        Vgg16 = VGG16(include_top=False, input_shape=(224, 224, 3))

        # Freeze the weights of the VGG16 base model
        Vgg16.trainable = False

        # Create a Sequential model
        model = Sequential()

        # Add the pre-trained VGG16 base model to the Sequential model
        model.add(Vgg16)

        # Add a Dropout layer to reduce overfitting
        model.add(Dropout(0.25))

        # Flatten the output of the VGG16 base model
        model.add(Flatten())

        # Add a Dense layer with 64 neurons and ReLU activation function
        model.add(Dense(64, activation="relu"))

        # Add another Dropout layer
        model.add(Dropout(0.25))

        # Add a Dense layer with 32 neurons and ReLU activation function
        model.add(Dense(32, activation="relu"))

        # Add the output layer with 5 neurons and Sigmoid activation function
        model.add(Dense(5, activation="sigmoid"))

        # Compile the model with the Adam optimizer, binary crossentropy loss, and binary accuracy metric
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])

        # Display a summary of the model architecture
        model.summary()

        return model

    def train_model(self, model, x_train, x_val, y_train, y_val):
        callbacks = [EarlyStopping(monitor='val_binary_accuracy', patience=10, restore_best_weights=True)]

        history = model.fit(x_train, y_train, epochs=50, batch_size=64,
                            validation_data=(x_val, y_val), verbose=1, callbacks=callbacks)
        return history

    def plot_curves(history):
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        accuracy = history.history["binary_accuracy"]
        val_accuracy = history.history["val_binary_accuracy"]

        epochs = range(len(history.history["loss"]))

        # plot loss
        plt.plot(epochs, loss, label="training_loss")
        plt.plot(epochs, val_loss, label="val_loss")
        plt.title("loss")
        plt.xlabel("epochs")
        plt.legend()

        # plot accuracy
        plt.figure()
        plt.plot(epochs, accuracy, label="training_accuracy")
        plt.plot(epochs, val_accuracy, label="val_accuracy")
        plt.title("accuracy")
        plt.xlabel("epochs")
        plt.legend()