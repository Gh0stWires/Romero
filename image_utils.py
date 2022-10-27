import PIL
from matplotlib import pyplot as plt
import envirment_utils
import os
import tensorflow as tf
import cv2


# output
 # unique-identifier-name-folder <--
  # trial number
   # images-batches


def create_output_batch_dir(trial_num, date, hparam_target_dir=''):

    path = envirment_utils.output_directory +  hparam_target_dir + f'trial-{trial_num}'
    batch_folder = image_batch_gen(path, date)

    if not os.path.exists(batch_folder):
        os.makedirs(batch_folder)


def generate_and_save_images(model, epoch, test_input, trial_num, date, hparam_target_dir=''):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    rnd = tf.math.round(predictions)
#    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        #plt.subplot(4, 4, i + 1)
        plt.imshow(rnd[i, :, :, 0] * 255, cmap='gray', vmin=0, vmax=255)
        #plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(get_path_for_epoch_img(epoch, trial_num, date, hparam_target_dir=hparam_target_dir))
    
    if envirment_utils.show_plots:
        #We don't want this happening in the Docker container
        plt.show()


def image_batch_gen(target_image_dir, date):
    mod_dir = f'{target_image_dir}/batch-'
    return f'{mod_dir}{str(date).replace(":", ".")}/'


def invert_pixel_colors(image_path, file_path):
    img = cv2.imread(image_path)
    invert = cv2.bitwise_not(img)
    cv2.imwrite(img=invert, filename=file_path)
# Display a single image using the epoch number
# def display_image(epoch_no, trial_num):
#     return PIL.Image.open(get_path_for_epoch_img(epoch_no, trial_num))


def get_path_for_epoch_img(epoch, trial_num, date, hparam_target_dir=''):
    trial_dir = f'trial-{trial_num}'
    return image_batch_gen(envirment_utils.output_directory + hparam_target_dir + trial_dir, date) + 'image_at_epoch_{:04d}.png'.format(epoch)
