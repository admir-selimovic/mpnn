# Message-Passing Neural Networks: Stability of Hidden States under 3D Rotations

This project is an implementation of the Message Passing Neural Networks (MPNN) for 3D model processing. The project utilizes the PyTorch Geometric library and PyTorch Lightning for efficient neural network management and training.

## Source
The MPNN model implemented in this project is based on the paper:
- **Neural Message Passing for Quantum Chemistry** (2017)  
  [Link to Paper](https://arxiv.org/pdf/1704.01212v2.pdf)

## Features
- Utilizes Message Passing Neural Networks (MPNNs).
- Offers multiple configurations:
  - Basic: Without any edge attributes.
  - Dist: Uses Euclidean distance as edge embeddings.
  - Relpos: Uses relative position vectors as edge embeddings.
- Integrates PyTorch Lightning for training management.
- Loads custom datasets from the ModelNet10 dataset that has been preprocessed.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch and PyTorch Geometric
- PyTorch Lightning
- YAML for configuration management

### Setup

1. Clone this repository.
```bash
git clone https://github.com/admir-selimovic/mpnn.git
cd mpnn
```

2. Install the necessary libraries.
```bash
pip install torch torch_geometric pytorch_lightning yaml
```

3. Modify the `configs/paths.yml` and `configs/hyperparameters.yml` to suit your dataset paths and desired hyperparameters respectively.

### Usage

Execute the preprocessing script:

```bash
python data_preprocessing.py
```

Execute the training script:

```bash
python train_model.py
```

When prompted, input the model type you want to train (`basic`, `dist`, or `relpos`).

## Structure

The project contains the following key components:

- `models/`: Contains the MPNN and MNISTClassifier implementations.
- `datasets/`: Houses the custom dataset loader for the ModelNet10 dataset.
- `configs/`: Contains YAML configurations for hyperparameters and dataset paths.

## Contributions

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## Acknowledgments

- [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)
- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
