import pathlib
import numpy as np
import PIL
import tensorflow as tf
from pathlib import Path

debug = True


def pre_process(directory):
    data_dir = pathlib.Path(directory)
    files = list(data_dir.glob('*wallmap.png'))
    print(f"Image count: {len(files)}")
    for file in files:
        process_image(file)


def process_image(image):
    img = PIL.Image.open(str(image))
    # img.show() if debug else None
    img_array = np.array(img)/255.0
    print(f"Image shape: {img_array.shape}") if debug else None
    img_array = img_array.reshape(img_array.shape[0], img_array.shape[1], 1)
    print(f"Image shape: {img_array.shape}") if debug else None
    tf_img = tf.image.resize_with_crop_or_pad(img_array, target_height=128, target_width=128)
    print(F"Image tensor shape: {tf_img.shape}") if debug else None
    out_array = tf_img.numpy()
    print(f"Output shape: {out_array.shape}") if debug else None

    test = np.squeeze(out_array, axis=2)

    im = PIL.Image.fromarray(test * 255)
    im = im.convert("L")
    im.save(Path('/Users/sam.grogan/Documents/Romero/processed-images/' + image.name))



#     TODO - load image.


if __name__ == '__main__':
    pre_process("./map-data/")
