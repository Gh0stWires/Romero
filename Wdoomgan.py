import os
from tabnanny import check
import time
import tensorflow as tf
import envirment_utils
# from IPython import display

from image_utils import generate_and_save_images
from model_utils import generator_optimizer, discriminator_optimizer, w_discriminator_loss, w_generator_loss
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
                                 discriminator=discriminator,
                                 )
tf_step_counter = tf.Variable(1)
manager = tf.train.CheckpointManager(
    checkpoint,
    directory=envirment_utils.checkpoint_dir,
    max_to_keep=envirment_utils.max_checkpoints,
    checkpoint_interval=envirment_utils.checkpoint_interval,
    step_counter=tf_step_counter,
)

# Setup Tensorboard and metrics
train_summary_writer = tf.summary.create_file_writer(envirment_utils.tensorboard_directory)
gen_loss_metric = tf.keras.metrics.Mean('gen_loss', dtype=tf.float32)
disc_loss_metric = tf.keras.metrics.Mean('disc_loss', dtype=tf.float32)

#  Training setup
EPOCHS = envirment_utils.epochs
noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Load image data
data = tf.keras.utils.image_dataset_from_directory(
    envirment_utils.processed_directory, labels=None, label_mode=None,
    class_names=None, color_mode='grayscale', batch_size=BATCH_SIZE, image_size=(128, 128))


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = w_generator_loss(fake_output)
        disc_loss = w_discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    gen_loss_metric(gen_loss)
    disc_loss_metric(disc_loss)
    tf.print("\nGen loss:", gen_loss)
    tf.print("Disc loss:", disc_loss)


def train(dataset, epochs):
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Checkpoints restored from {}".format(manager.latest_checkpoint))
    else:
        print("Training from scratch.")

    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        with train_summary_writer.as_default():
            tf.summary.scalar('gen_loss', gen_loss_metric.result(), step=epoch)
            tf.summary.scalar('disc_loss', disc_loss_metric.result(), step=epoch)

        # Save the model every N epochs
        manager.save()

        if (epoch + 1) % envirment_utils.image_interval == 0:
            generate_and_save_images(generator, epoch + 1, seed)

        print(f'Time for epoch {epoch + 1} is {time.time() - start} sec')
        tf_step_counter.assign_add(1)

        #  Reset metrics
        gen_loss_metric.reset_states()
        disc_loss_metric.reset_states()

    # Generate after the final epoch
    # display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


if __name__ == '__main__':
    train(data, int(envirment_utils.epochs))
