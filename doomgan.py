import os
import time
import tensorflow as tf
import envirment_utils
# from IPython import display

from image_utils import generate_and_save_images
from model_utils import generator_optimizer, discriminator_optimizer, discriminator_loss, generator_loss
from models import make_generator_model, make_discriminator_model
import envirment_utils

# Constants
BUFFER_SIZE = 60000
BATCH_SIZE = envirment_utils.batch_size

# Models
generator = make_generator_model()
discriminator = make_discriminator_model()

# Checkpoint info
checkpoint_prefix = os.path.join(envirment_utils.checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

#  Training setup
EPOCHS = envirment_utils.epochs
noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Load image data
data = tf.keras.utils.image_dataset_from_directory(
    envirment_utils.processed_directory, labels=None, label_mode=None,
    class_names=None, color_mode='grayscale', batch_size=32, image_size=(128,
                                                                   128))


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    print('gen loss {} disc loss {}'.format(gen_loss, disc_loss))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)


        # Produce images for the GIF as you go
        # display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    # display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epochs,
                             seed)


if __name__ == '__main__':
    train(data, int(envirment_utils.epochs))