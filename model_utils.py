import tensorflow as tf
import numpy as np
import envirment_utils

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def add_label_noise(labels, noise_val):
    def random_val():
        return np.random.randint(0, row_size)

    row_size, column_size = labels.shape
    assert row_size == column_size
    np_arr = labels.numpy()

    for i in range(0, row_size):
        np_arr[random_val(), random_val()] = noise_val

    return tf.convert_to_tesnsor(np_arr)


def discriminator_loss(real_output, fake_output):
    real_loss = None
    fake_loss = None

    if envirment_utils.add_label_noise:
        real_loss = cross_entropy(add_label_noise(tf.ones_like(real_output), 0), real_output)
        fake_loss = cross_entropy(add_label_noise(tf.zeros_like(fake_output), 1), fake_output)
    else:
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    total_loss = real_loss + fake_loss

    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
