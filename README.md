
# Graph Neural Networks for 3D Model Classification

This repository contains an implementation of Graph Neural Networks (GNNs) for classifying 3D models using the ModelNet10 dataset. The project utilizes the PyTorch Geometric library and PyTorch Lightning for efficient neural network management and training.

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

## License

This project is licensed under the MIT License. See `LICENSE` for more details.

## Acknowledgments

- [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)
- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)

```

Please replace the placeholders `<repository-url>`, `<repository-name>`, and `<script-name>.py` with the actual details corresponding to your project. You can also expand on sections like "Usage" or "Structure" to provide more specific details about your project. 

This README provides a good starting point and can be further enriched with sections like "Future Work", "Known Issues", or any other information that you find important.
