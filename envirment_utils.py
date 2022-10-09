import os
from file_utils import trailing_slash_check

output_directory = trailing_slash_check(os.environ['DOOM_OUTPUT_DIRECTORY'])
processed_directory = trailing_slash_check(os.environ['PROCESSED_IMAGES'])
checkpoint_dir = trailing_slash_check(os.environ['CHECKPOINT_DIR'])
tensorboard_directory = trailing_slash_check(os.environ['TENSORBOARD_DIR'])
wads = trailing_slash_check(os.environ['WADS'])
mapdata = trailing_slash_check(os.environ['MAP_DATA'])
add_label_noise = os.environ['ADD_LABEL_NOISE'] == 'true'
show_plots = os.environ['SHOW_PLOTS'] == 'true'
batch_size = int(os.environ['BATCH_SIZE'])
epochs = int(os.environ['EPOCHS'])
max_checkpoints = int(os.environ['MAX_CHECKPOINTS'])
checkpoint_interval = int(os.environ['CHECKPOINT_INTERVAL'])
image_interval = int(os.environ['IMAGE_INTERVAL'])

