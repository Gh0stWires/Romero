import tensorflow as tf
import numpy as np
import envirment_utils

generator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.RMSprop(5e-5)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def add_label_noise(labels, noise_val):
    print(labels.shape)
    def random_val():
        return np.random.randint(0, row_size)

    batch_size, row_size, column_size, alpha = labels.shape
    assert row_size == column_size
    np_arr = labels.numpy

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


def w_generator_loss(fake_output):
    return tf.reduce_mean(fake_output)


def w_discriminator_loss(real_output, fake_output, discriminator, images, generated_images):
    real_loss = None
    fake_loss = None



    d_regularizer = gradient_penalty(discriminator, images, generated_images)

    penalty_coeffcient = 10
    total_loss =  (
            tf.reduce_mean(real_output)
            - tf.reduce_mean(fake_output)
            + d_regularizer * penalty_coeffcient
        )

    return total_loss


def gradient_penalty(discriminator, images, generated_images):
    '''
       1. get epsilon
       2. calculate the x_hat
       3. get gradient
       4. regularizer

       input:
           x: real data
           x_fake: fake data

           x and x_fake must have the same batch size

       '''

    print(images.shape)
    x = discriminator(images, training=True)
    x_fake = discriminator(generated_images, training=True)

    epsilon = tf.random.uniform([images.shape[0], 1, 1, 1], 0.0, 1.0)
    x_hat = epsilon * images + (1 - epsilon) * generated_images
    with tf.GradientTape() as t:
        t.watch(x_hat)
        d_hat = discriminator(x_hat)
    gradients = t.gradient(d_hat, x_hat)
    ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
    d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
    return d_regularizer


# clip model weights to a given hypercube
class ClipConstraint(tf.keras.constraints.Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return tf.keras.backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}