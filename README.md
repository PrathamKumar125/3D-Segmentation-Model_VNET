# 3D Segmentation Model_VNET
Dataset Used: https://zenodo.org/records/7860267
---

## Overview

This project focuses on 3D medical image segmentation using the VNet architecture. The primary objective is to accurately segment anatomical structures from volumetric medical scans, such as MRI or CT images. The VNet model is particularly well-suited for this task due to its ability to capture spatial hierarchies and its efficient handling of 3D data. This project aims to provide a robust framework for training, validating, and inferring 3D segmentations, complete with performance metrics and visualization tools.

## Setup Instructions

### Prerequisites

- Python 3.7+
- CUDA-enabled GPU (optional but recommended for faster training)
- Python packages:
  - TensorFlow/PyTorch (depending on the framework used in the notebook)
  - NumPy
  - SciPy
  - Matplotlib
  - Nibabel (for handling medical images)
  - Scikit-learn
  - OpenCV (for 3D visualization)
  - Other dependencies listed in `requirements.txt` (if available)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/VNET_3DSegmentation.git
   cd VNET_3DSegmentation
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**:
   Place your 3D medical image dataset in the `data/` directory. Ensure that the data is in NIfTI format (`.nii` or `.nii.gz`).

5. **Run the notebook**:
   Open the Jupyter notebook and run all cells to start training the model:
   ```bash
   jupyter notebook VNET_3DSegmentation.ipynb
   ```

## Model Architecture

The chosen model for this project is **VNet**, a 3D segmentation model. Key architectural features include:

- **Encoder-Decoder Structure**: The model uses a series of 3D convolutional layers with downsampling to encode the input volume, followed by upsampling layers to decode the segmentation map.
- **Residual Blocks**: Each block in the VNet contains residual connections to enhance gradient flow during training.
- **Skip Connections**: Skip connections between corresponding encoder and decoder layers help in recovering spatial resolution lost during downsampling.
- **Input & Output**: The model accepts 3D volumes as input and outputs a 3D segmentation map of the same size.

### Convolutional Block

The `conv_block` function defines a standard convolutional block used throughout the network:

- **Convolutional Layers**: Two 3D convolutional layers with ReLU activation.
- **Batch Normalization**: Optional batch normalization applied after each convolutional layer.
- **Skip Connections**: Add the input of the block to its output (residual connections).
<br>

- **Encoder**:
  - **c1**: Initial 3D convolutional layer with 16 filters.
  - **c2**: Downsampling with 3D convolutional layer and max pooling.
  - **c3**: Convolutional block with 32 filters.
  - **p3**: Further downsampling with convolutional layer and dropout.
  - **c4**: Convolutional block with 64 filters.
  - **p4**: Additional downsampling with convolutional layer and dropout.
  - **c5**: Convolutional block with 128 filters.
  - **p6**: Further downsampling with convolutional layer and dropout.
  - **p7**: Final convolutional block with 128 filters.

- **Decoder**:
  - **u6**: Upsampling with 3D transposed convolution, concatenated with `c5`.
  - **c7**: Convolutional block with 128 filters.
  - **u7**: Upsampling with 3D transposed convolution, concatenated with `c4`.
  - **c8**: Convolutional block with 64 filters.
  - **u9**: Upsampling with 3D transposed convolution, concatenated with `c3`.
  - **c9**: Convolutional block with 32 filters.
  - **u10**: Upsampling with 3D transposed convolution, concatenated with `c1`.
  - **c10**: Final convolutional block with 16 filters and additional dropout.

- **Output Layer**: A final 3D convolutional layer with 4 filters and softmax activation for multi-class segmentation.

## Training

### Loss Functions

- **Dice Loss**: Custom loss function to handle class imbalance with weights `[0.26, 22.53, 22.53, 26.21]`.
- **Focal Loss**: Used to address class imbalance by focusing more on hard examples.
- **Total Loss**: Combination of Dice Loss and Focal Loss.

### Metrics

- **Accuracy**: General classification accuracy.
- **IOU Score**: Intersection over Union score with a threshold of 0.5.
- **Mean IoU**: Mean Intersection over Union metric for multi-class segmentation.

### Optimizer

- **Adam Optimizer** with a learning rate of 0.0001.


## Validation

- **Dice Score Calculation**: The Dice score is computed between the predicted segmentation and the ground truth for each organ. This metric is crucial for evaluating the overlap between the prediction and the actual structure.
- **Cross-Validation**: The dataset is split into training and validation sets, with k-fold cross-validation employed to ensure robust performance evaluation.

