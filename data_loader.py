import pathlib
import numpy as np
import PIL
import tensorflow as tf

debug = True


def pre_process(directory):
    data_dir = pathlib.Path(directory)
    files = list(data_dir.glob('*wallmap.png'))
    print(f"Image count: {len(files)}")
    process_image(files[0])


def process_image(image):
    img = PIL.Image.open(str(image))
    # img.show() if debug else None
    img_array = np.array(img)/255.0
    print(f"Image shape: {img_array.shape}") if debug else None
    img_array = img_array.reshape(img_array.shape[0], img_array.shape[1], 1)
    print(f"Image shape: {img_array.shape}") if debug else None
    tf_img = tf.image.resize_with_crop_or_pad(img_array, 128, 128)
    print(F"Image tensor shape: {tf_img.shape}") if debug else None

#     TODO - load image.


if __name__ == '__main__':
    pre_process("./map-data/")
