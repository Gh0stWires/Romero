import tensorflow as tf
import numpy as np

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def add_label_noise(labels, noise_val):
    row_size, column_size = labels.shape
    assert row_size == column_size
    random_val = lambda: np.random.randint(0, row_size)
    np_arr = labels.numpy()
    
    for i in range(0, row_size):
        np_arr[random_val(), random_val()] = noise_val
    
    return tf.convert_to_tensor(np_arr)
    


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(add_label_noise(tf.ones_like(real_output), 0) , real_output)
    fake_loss = cross_entropy(add_label_noise(tf.ones_like(real_output), 1), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(add_label_noise(tf.ones_like(fake_output)), fake_output)