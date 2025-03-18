# C-Vision: a U-Net Model for Semantic Segmentation

This repository contains the implementation of a U-Net model for semantic segmentation, specifically trained on the Oxford-IIIT Pet dataset. The U-Net architecture is well-suited for image segmentation tasks, where the goal is to assign each pixel in the image to a specific class.

## Model Architecture

The U-Net model consists of an encoder-decoder architecture with skip connections. The encoder extracts features from the input image, while the decoder reconstructs the segmentation map using these features. Skip connections help in preserving spatial information, which is crucial for accurate segmentation.

### Encoder

The encoder is composed of multiple convolutional layers followed by max-pooling layers. Each convolutional block contains two convolutional layers with ReLU activation.

### Decoder

The decoder consists of upsampling layers followed by convolutional layers. Each upsampling block contains a transposed convolutional layer followed by two convolutional layers with ReLU activation.

### Final Layer

The final layer is a 1x1 convolution that maps the output to the desired number of classes.

## Training

The model is trained using the CrossEntropyLoss function, which is well-suited for multi-class classification problems. The Adam optimizer is used for training.

### Training Script

The training script (`u-net.py`) includes the following steps:

1. Dataset preparation: Download and preprocess the Oxford-IIIT Pet dataset.
2. Model initialization: Define the U-Net model architecture.
3. Training loop: Train the model for a specified number of epochs.
4. Evaluation: Evaluate the model on the validation set.
5. Visualization: Save the segmentation results as images.

### Validation

The validation script (`validate_unet.py`) loads the trained model from a checkpoint and performs segmentation on the validation dataset. The results are saved as images in the `results/` directory.

### Validation Results

Below is an example of the validation results:

![Validation Results](https://github.com/Vlasenko2006/c-vision/blob/main/validation_sample.png)

## Environment Setup

To set up the environment for running the code, you can use the provided `env-config.yaml` file. This file contains the necessary dependencies for the project. To create the environment, follow these steps:

1. Install Anaconda or Miniconda if you haven't already.
2. Create the environment using the `env-config.yaml` file:

```sh
conda env create -f env-config.yaml
```

3. Activate the environment:

```sh
conda activate unet-env
```

## Files

- `main.py`:The top level wrapper that runs the model.
- `u-net.py`:The U-Net model.
- `train_and_evaluate.py`:Trainer for U-Net model.
- `visualise_predictions`: Visualize predictions on the validation set
- `utilities.py`: contains the dataloaers and I/O routines.
- `validate_unet.py`: Validation script for the U-Net model.
- `checkpoints/`: Directory containing model checkpoints.
- `results/`: Directory containing validation results.

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

