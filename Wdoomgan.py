import datetime
import time
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import itertools

import image_utils
from image_utils import generate_and_save_images
from model_utils import discriminator_optimizer, w_discriminator_loss, w_generator_loss
from models import make_generator_model, make_discriminator_model
import envirment_utils
import model_utils
import file_utils
import os
import json


class Count(tf.Module):
  def __init__(self):
    self.count = None

  @tf.function
  def __call__(self):
    if self.count is None:
        self.count = tf.Variable(0)
    self.count.assign_add(1)
    return self.count


tf_step_counter = Count()
hparam_training = False # Set to true to perform Hyper parameter trial runs for training
now = datetime.datetime.now()

def load_data(directory, batch_size):
    # Load image data
    working_data = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels=None,
        label_mode=None,
        class_names=None,
        color_mode="grayscale",
        batch_size=None,
        image_size=(128, 128),
        shuffle=True,
    )

    # Rebatch data so batch sizes are consistent
    return working_data.batch(batch_size, drop_remainder=True)


def generate_checkpoint_manager(generator, discriminator, generator_optimizer, trial_num):
    # Checkpoint info
    step_counter = tf_step_counter()

    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )
    print(f'{envirment_utils.checkpoint_dir}/trial-{trial_num}/')
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=f'{envirment_utils.checkpoint_dir}trial-{trial_num}/',
        max_to_keep=envirment_utils.max_checkpoints,
        checkpoint_interval=envirment_utils.checkpoint_interval,
        step_counter=step_counter,
    )
    return step_counter, checkpoint, manager


def train_step(generator, discriminator, images, batch_size, generator_optimizer):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        # moi_z_images = model_utils.add_label_noise(generated_images, 0)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = w_generator_loss(fake_output)
        disc_loss = w_discriminator_loss(
            real_output, fake_output, discriminator, images, generated_images
        )

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )

    return gen_loss, disc_loss


def train(dataset, batch_size, seed, hparams, trial_num):
    # Setup models, optimizers, etc.
    generator = make_generator_model()
    discriminator = make_discriminator_model(hparams[HP_NORMALIZE_DISCRIMINATOR])
    g_optimizer = model_utils.generator_optimizer(hparams[HP_GENERATOR_OPTIMIZER])

    # Setup checkpoint manager
    step_counter, checkpoint, manager = generate_checkpoint_manager(
        generator, discriminator, g_optimizer, trial_num
    )

    disc_loss_metric, gen_loss_metric, train_summary_writer = setup_tensorboard(step_counter)

    checkpoint.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        print(f"Checkpoints restored from {manager.latest_checkpoint}")
    else:
        print("Training from scratch.")

    for epoch in range(hparams[HP_EPOCHS] + 1):
        start = time.time()

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(generator, discriminator, image_batch, batch_size, g_optimizer)
            tf.print("\nGen loss:", gen_loss)
            tf.print("Disc loss:", disc_loss)
            gen_loss_metric(gen_loss)
            disc_loss_metric(disc_loss)

        # with train_summary_writer.as_default():
        if hparam_training:
            write_hparam_metrics(train_summary_writer, hparams, epoch, gen_loss_metric, disc_loss_metric)

        # Save the model every N epochs
        manager.save()

        if (epoch + 1) % envirment_utils.image_interval == 0:
            generate_and_save_images(generator, epoch + 1, seed, trial_num, now)

        if epoch + 1 == hparams[HP_EPOCHS]:
            break

        print(f"Time for epoch {epoch + 1} is {time.time() - start} sec")
        #  Reset metrics
        gen_loss_metric.reset_states()
        disc_loss_metric.reset_states()

    # Generate after the final epoch
    # display.clear_output(wait=True)
    generate_and_save_images(generator, hparams[HP_EPOCHS], seed, trial_num, now)


def write_hparam_metrics(train_summary_writer, hparams, step_counter, gen_loss_metric, disc_loss_metric):
    with train_summary_writer.as_default():
        hp.hparams(hparams)
        # record the values used in this trial
        tf.summary.scalar("loss/gen_loss", gen_loss_metric.result(), step=step_counter)
        tf.summary.scalar("loss/disc_loss", disc_loss_metric.result(), step=step_counter)


def write_trial_notes(trial_num, hparams, skip_descrip):
    meta_file = f'trial_{trial_num}_meta.json'
    path = f'{envirment_utils.checkpoint_dir}trial-{trial_num}/'
    target_image_dir = envirment_utils.output_directory + f'trial-{trial_num}'

    if os.path.exists(path + meta_file):
        print('Trial meta found. Moving on...')
        return

    if not os.path.exists(target_image_dir):
        os.makedirs(target_image_dir)

    description = ''
    if not skip_descrip:
        describe = input('Leave a trial description (press return to skip): ')
        description = describe

    meta_data = {
        'description': f'trial_{trial_num}' if description == '' else description,
        'trial_num': trial_num,
        "hparams": dict(
            HP_NORMALIZE_DISCRIMINATOR=hparams[HP_NORMALIZE_DISCRIMINATOR],
            HP_GENERATOR_OPTIMIZER=hparams[HP_GENERATOR_OPTIMIZER],
            HP_EPOCHS=hparams[HP_EPOCHS]
        ),
        "training_env": dict(
            ouput_directory=envirment_utils.output_directory,
            processed_directory=envirment_utils.processed_directory,
            checkpoint_dir=envirment_utils.checkpoint_dir,
            tensorboard_directory=envirment_utils.tensorboard_directory,
            add_label_noise=envirment_utils.add_label_noise,
            show_plots=envirment_utils.show_plots,
            batch_size=envirment_utils.batch_size,
            epochs=envirment_utils.epochs,
            max_checkpoints=envirment_utils.max_checkpoints,
            checkpoint_interval=envirment_utils.checkpoint_interval,
            image_interval=envirment_utils.image_interval,
            wads=envirment_utils.wads,
            mapdata=envirment_utils.mapdata
        )}
    #Create checkpoint folder
    if not os.path.exists(path):
        os.makedirs(path)
    #Create meta file for that checkpoint folder
    if not os.path.exists(path + meta_file):
        with open(path + meta_file, 'w') as f:
            json.dump(meta_data, f)


def training_setup(trial_num, hparams, skip_description=True):
    trial_name = f'trial-{trial_num}'

    print('Setting up trial:', trial_name)
    file_utils.batch_check()
    image_utils.create_output_batch_dir(trial_num, now)
    write_trial_notes(trial_num, hparams, skip_description)
    print('\nTrial meta data setup complete!')

def setup_tensorboard(step_count):
    # Setup Tensorboard and metrics
    train_summary_writer = tf.summary.create_file_writer(
        f"{envirment_utils.tensorboard_directory}/trial-{step_count}"
    )
    gen_loss_metric = tf.keras.metrics.Mean("gen_loss", dtype=tf.float32)
    disc_loss_metric = tf.keras.metrics.Mean("disc_loss", dtype=tf.float32)

    return disc_loss_metric, gen_loss_metric, train_summary_writer


HP_EPOCHS = hp.HParam("epochs", hp.Discrete([10, 100]))
HP_NORMALIZE_DISCRIMINATOR = hp.HParam(
    "normalize_discriminator", hp.Discrete([True, False])
)
HP_GENERATOR_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adam", "sgd"]))


METRIC_ACCURACY = "accuracy"
selected_hparams = [HP_EPOCHS, HP_NORMALIZE_DISCRIMINATOR, HP_GENERATOR_OPTIMIZER]
with tf.summary.create_file_writer("logs/hparam_tuning").as_default():
    hp.hparams_config(
        hparams=selected_hparams,
        metrics=[hp.Metric(METRIC_ACCURACY, display_name="Accuracy")],
    )

# Models
if __name__ == "__main__":
    batch_size = envirment_utils.batch_size
    data = load_data(envirment_utils.processed_directory, batch_size)

    #  Training setup
    noise_dim = 100
    num_examples_to_generate = 16
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    hparam_domains = [i.domain.values for i in selected_hparams]
    # TODO: Handle for this making sense with other trials being present and hparams is on.
    trial_num = 14  # Set this to run a specific trial with Hparams turned off


    if hparam_training:
        trial_num = 0
        for epochs, discriminator, optimizer in itertools.product(*hparam_domains):
            generated_hparams = {
                HP_EPOCHS: epochs,
                HP_NORMALIZE_DISCRIMINATOR: discriminator,
                HP_GENERATOR_OPTIMIZER: optimizer,
            }

            trial_num += 1
            training_setup(trial_num, generated_hparams)
            print(f"--- Starting trial: {trial_num}")
            train(data, batch_size, seed, generated_hparams, trial_num)


    else:
        # Set up hparams manually and target trial to preserve checkpoints data
        print(f'Training for {envirment_utils.epochs} epochs!')
        hparams = {
            HP_EPOCHS: envirment_utils.epochs,
            HP_NORMALIZE_DISCRIMINATOR: True,
            HP_GENERATOR_OPTIMIZER: 'adam'
        }
        training_setup(trial_num, hparams, skip_description=False)
        print(f"--- Starting trial: {trial_num}")
        train(data, batch_size, seed, hparams, trial_num=trial_num)

    print(f"Finished {'HP' if hparam_training else 'training'} loop")
