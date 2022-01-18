import os

output_directory = os.environ['DOOM_OUTPUT_DIRECTORY']
processed_directory = os.environ['PROCESSED_IMAGES']
epochs = os.environ['EPOCHS']
checkpoint_dir = os.environ['CHECKPOINT_DIR']
batch_size = os.environ['BATCH_SIZE']
add_label_noise = os.environ['ADD_LABEL_NOISE'] == 'true'
