import tensorflow as tf
from tensorflow.keras import layers

# Define the CNN agent
def create_cnn_agent(input_shape, num_actions):
    # Define the input layer
    inputs = layers.Input(shape=input_shape)

    # Define the convolutional layers
    x = layers.Conv2D(32, 8, strides=4, activation='relu')(inputs)
    x = layers.Conv2D(64, 4, strides=2, activation='relu')(x)
    x = layers.Conv2D(64, 3, strides=1, activation='relu')(x)

    # Define the fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(num_actions, activation='softmax')(x)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model