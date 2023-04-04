from dataloader import make_dataset
from model import create_cnn_agent
import pandas as pd
import numpy as np
import tensorflow as tf
df=pd.read_csv("data.csv")
#drop the column "select" because it is not used in the game
df.drop(columns=["Select button"],inplace=True)
#make the dataset
print("Making dataset...")
frames,inputs=make_dataset(df)


#normalize the frames
print("Normalizing frames...")
frames_normalized = tf.image.convert_image_dtype(frames, dtype=tf.float32)
#convert the inputs to binary 
print("Converting inputs to binary...")
#inputs_binary= tf.one_hot(inputs, depth=7)

#train the model
input_shape=frames[0].shape
print(inputs[0].shape)
model=create_cnn_agent(input_shape, 7)
model.summary()
print("Training model...")
#check if there is a gpu and use it
if tf.test.is_gpu_available():
    print("Using GPU")
    with tf.device('/device:GPU:0'):
        model.fit(frames_normalized, inputs, batch_size=32, epochs=10, validation_split=0.2)
else:
    print("Using CPU")
    model.fit(frames_normalized, inputs, batch_size=32, epochs=10, validation_split=0.2)
