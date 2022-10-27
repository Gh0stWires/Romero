import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from matplotlib import pyplot as plt

import envirment_utils
from image_utils import generate_and_save_images, invert_pixel_colors
from model_utils import ClipConstraint
# algo for image output corrections
# const = ClipConstraint(0.01)
#changing filter sizes makes you lose checkpoints
filter_size = (4, 4)
strides = (2, 2)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8 * 8 * 1024, use_bias=False, input_shape=(100,)))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Reshape((8, 8, 1024)))
    assert model.output_shape == (None, 8, 8, 1024)  # Note: None is the batch size

    # print(model.output_shape)
    model.add(layers.Conv2DTranspose(512, filter_size, strides=strides, padding='same', use_bias=False))
    # print(model.output_shape)
    assert model.output_shape == (None, 16, 16, 512)
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(256, filter_size, strides=strides, padding='same', use_bias=False))
    # print(model.output_shape)
    assert model.output_shape == (None, 32, 32, 256)
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(128, filter_size, strides=strides, padding='same', use_bias=False))
    # print(model.output_shape)
    assert model.output_shape == (None, 64, 64, 128)
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(1, filter_size, strides=strides, padding='same', use_bias=False))
    # print(model.output_shape)
    assert model.output_shape == (None, 128, 128, 1)

    return model


def make_discriminator_model(normalize):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, filter_size, strides=strides, padding='same',
                            input_shape=[128, 128, 1]))
    print(model.output_shape)
    if normalize:
        model.add(layers.LayerNormalization())

    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(256, filter_size, strides=strides, padding='same'))
    print(model.output_shape)

    if normalize:
        model.add(layers.LayerNormalization())

    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(512, filter_size, strides=strides, padding='same'))
    print(model.output_shape)

    if normalize:
        model.add(layers.LayerNormalization())

    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(1024, filter_size, strides=strides, padding='same'))
    print(model.output_shape)

    if normalize:
        model.add(layers.LayerNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    print(model.output_shape)

    return model


def load_model(model, trial_num):
    weights = f'./weights/{trial_num}.h5'
    model.load_weights(weights)
    print('model loaded')
    return model


def prediction(input_for_pred, model):
    return model.predict(input_for_pred)


def prediction():
    noise_dim = 100
    num_examples_to_generate = 16
    batch_size = envirment_utils.batch_size
    noise = tf.random.normal([batch_size, noise_dim])
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    model = load_model(make_generator_model(), '777')
    predictions = model(noise, training=False)
    rnd = tf.math.round(predictions)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        # plt.subplot(4, 4, i + 1)
        plt.imshow(rnd[i, :, :, 0] * 255, cmap='gray', vmin=0, vmax=255)
        # plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.show()

if __name__ == "__main__":
    # prediction()
    invert_pixel_colors(image_path='./output/trial-777/batch-2022-10-20 01.56.14.792504/image_at_epoch_100000.png',
                        file_path='./output/inverted-150k-transfer-learn.png')
#
