import os

output_directory = os.environ['DOOM_OUTPUT_DIRECTORY']
processed_directory = os.environ['PROCESSED_IMAGES']
checkpoint_dir = os.environ['CHECKPOINT_DIR']
tensorboard_directory = os.environ['TENSORBOARD_DIR']
add_label_noise = os.environ['ADD_LABEL_NOISE'] == 'true'
show_plots = os.environ['SHOW_PLOTS'] == 'true'
batch_size = int(os.environ['BATCH_SIZE'])
epochs = int(os.environ['EPOCHS'])
max_checkpoints = int(os.environ['MAX_CHECKPOINTS'])
checkpoint_interval = int(os.environ['CHECKPOINT_INTERVAL'])
image_interval = int(os.environ['IMAGE_INTERVAL'])
