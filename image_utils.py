import PIL
from matplotlib import pyplot as plt
import envirment_utils
import os
import tensorflow as tf

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    rnd = tf.math.round(predictions)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(rnd[i, :, :, 0] * 255, cmap='gray', vmin=0, vmax=255)
        # plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    if not os.path.exists(envirment_utils.output_directory):
        os.makedirs(envirment_utils.output_directory)
    plt.savefig(get_path_for_epoch_img(epoch))
    
    if envirment_utils.show_plots:
        #We don't want this happening in the Docker container
        plt.show()


# Display a single image using the epoch number
def display_image(epoch_no):
    return PIL.Image.open(get_path_for_epoch_img(epoch_no))

def get_path_for_epoch_img(epoch):
    return envirment_utils.output_directory + 'image_at_epoch_{:04d}.png'.format(epoch)
