# Where's Wally Image Segmentation

This repository handles the segmentation of the full size puzzles into smaller sub-images.

### Python packages

Python dependencies are handled through Miniconda3 which can be found here

To set up the environment install Minicoinda3 and run the following command in the Anaconda terminal:

 `conda create --name  wheres-wally --file wally-env.txt`

### Running the scripts

Follow these steps to generate the sub-images with labels (all commands in Anaconda Terminal).

+ Unzip images.zip.
+ Import any required dependencies (listed below).
+ Run this command to create the targets from the bounding box xml files.

`python create_targets.py`
+ Run this command to apply required preprocessing to the images. This will create the wally/not wally labels.

`python preprocessing.py`
+ Run this command to create npy files that contain the broken down wally/not wally images

`python generate_subimages.py`

