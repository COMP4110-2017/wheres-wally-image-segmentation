# Where's Wally

## Python Packages

Python dependencies are handled through Anaconda. To set up the environment install Anaconda and run the following command in the Anaconda terminal:

    conda create --name  wheres-wally --file wally-env.txt

## Configuration

To setup the system, copy the ```env.py.example``` file to ```env.py``` in the root folder.

    cp env.py.example env.py

You can then adjust the parameters to your liking.

    # Training parameters
    EPOCHS = 2000
    STEPS_PER_EPOCH = 6
    SPLIT = 0.67
    CHARACTER = "wanda"

    # Predicting model (just the filename)
    MODEL_NAME = "wanda_2000_6_0.67"

## Building The Model

Run this command in Anaconda Terminal to build the model, using the configuration parameters in the ```env.py``` file.

    python run_everything.py

## Predicting On The Built Model

Run this command in Anaconda Terminal to start predicting on the built model with the name as specified in the ```env.py``` file.

    python predict.py

Images with prediction masks will be stored in the ```output``` folder.