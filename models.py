import tensorflow as tf
from tensorflow.keras import layers

from model_utils import ClipConstraint
# algo for image output corrections
const = ClipConstraint(0.01)
#changing filter sizes makes you lose checkpoints
filter_size = (3, 3)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(32 * 32 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Reshape((32, 32, 256)))
    assert model.output_shape == (None, 32, 32, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, filter_size, strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 128)
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(64, filter_size, strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 64)
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(1, filter_size, strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 128, 128, 1)

    return model


def make_discriminator_model(normalize):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, filter_size, strides=(2, 2), padding='same',
                            input_shape=[128, 128, 1], kernel_constraint=const))
    if normalize:
        model.add(layers.LayerNormalization())

    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(128, filter_size, strides=(2, 2), padding='same', kernel_constraint=const))

    if normalize:
        model.add(layers.LayerNormalization())

    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
