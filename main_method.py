from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.applications.vgg16 import VGG16
from keras import layers
import preprocess as pr
import model as m

obj=pr.preprocess()
data_path = r"C:\Users\Bhaskar Marapelli\Downloads\gaussian_filtered_images\gaussian_filtered_images"
dataset_df = obj.generate_images_dataset(data_path)

# Save the DataFrame to a CSV file
csv_filename = "diabetic_retinopathy_dataset.csv"
dataset_df.to_csv(csv_filename, index=False)
df=obj.load_data()
obj.count_plot(df)
data,labels=obj.preprocess()
x_train,y_train=obj.data_split(data,labels)
x_train, x_val, y_train, y_val=obj.train_valid_split(x_train,y_train)

obj2=m.Model()
model=obj2.model()

history=obj2.train_model(model,x_train, x_val, y_train, y_val)

obj2.plot_curves(history)
