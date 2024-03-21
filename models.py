from pickletools import optimize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from contrastive_loss import SupervisedContrastiveLoss

def create_encoder(input_shape):
    """
    This will be the encoder of the contrastive learning model.
    """
    encoder = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Flatten(),
        layers.Dense(256, activation='relu')
    ])

    # Set the encoder name
    encoder._name = 'ofmano_encoder'

    return encoder


def create_encoder_with_projection_head(encoder, input_shape, projection_units=128, temperature=0.05):
    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="cifar-encoder_with_projection-head"
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=SupervisedContrastiveLoss(temperature=temperature),
    )

    return model


def create_classifier(encoder, input_shape, trainable=True):
    """
    This will be the classifier of the contrastive learning model.
    """
    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=input_shape)
    x = encoder(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs, name='ofmano_classifier')
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=0.0001),
        loss = keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.Recall(), keras.metrics.Precision()]
    )

    return model

