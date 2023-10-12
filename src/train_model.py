import torch_geometric as tg
import pytorch_lightning as pl
import yaml
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch.nn as nn
from torch_geometric.transforms import Distance, Cartesian
from datasets.modelnet_dataset import ModelNet10Processed
from models.mpnn import MPNN
from models.modelnet10_classifier import ModelNet10Classifier
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar
from pathlib import Path

def load_config(file_path):
    '''Load YAML configuration from the given file path'''
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Setup directory paths for configuration and data
current_script_path = Path(os.path.abspath(__file__))
BASE_DIR = current_script_path.parent.parent
hyperparameters_path = BASE_DIR / "configs" / "hyperparameters.yml"
paths_path = BASE_DIR / "configs" / "paths.yml"

# Load hyperparameters and paths from the config files
hyperparameters = load_config(hyperparameters_path)
paths = load_config(paths_path)

# Extract hyperparameters from the loaded configurations
n_epochs = hyperparameters['n_epochs']
learning_rate = hyperparameters['learning_rate']
batch_size = hyperparameters['batch_size']
node_features = hyperparameters['node_features']
hidden_features = hyperparameters['hidden_features']
out_features = hyperparameters['out_features']
num_layers = hyperparameters['num_layers']
aggr = hyperparameters['aggr']
pool = hyperparameters['pool']
act = nn.ReLU

# Setup paths for the custom datasets
TRAIN_DATA_FOLDER = os.path.join(BASE_DIR, 'data', 'processed_train')
TRAIN_DATA_LIST_FILE = os.path.join(
    BASE_DIR, 'data', 'processed_train_file_name_list.txt'
)
VAL_DATA_FOLDER = os.path.join(BASE_DIR, 'data', 'processed_val')
VAL_DATA_LIST_FILE = os.path.join(
    BASE_DIR, 'data', 'processed_val_file_name_list.txt'
)

# Load datasets with different configurations
train_set = ModelNet10Processed(
    data_folder=TRAIN_DATA_FOLDER, 
    data_list_filename=TRAIN_DATA_LIST_FILE
)
val_set = ModelNet10Processed(
    data_folder=VAL_DATA_FOLDER, 
    data_list_filename=VAL_DATA_LIST_FILE
)
train_loader = tg.loader.DataLoader(
    train_set, batch_size=batch_size, num_workers=0, shuffle=True
)
val_loader = tg.loader.DataLoader(
    val_set, batch_size=batch_size, num_workers=0, shuffle=False
)

# Load datasets with distance edge attributes
dist_transforms = Distance()
train_set_dist = ModelNet10Processed(
    data_folder=TRAIN_DATA_FOLDER, 
    data_list_filename=TRAIN_DATA_LIST_FILE, 
    transform=dist_transforms
)
val_set_dist = ModelNet10Processed(
    data_folder=TRAIN_DATA_FOLDER, 
    data_list_filename=TRAIN_DATA_LIST_FILE, 
    transform=dist_transforms
)
train_loader_dist = tg.loader.DataLoader(
    train_set_dist, batch_size=batch_size, shuffle=True
)
val_loader_dist = tg.loader.DataLoader(
    val_set_dist, batch_size=batch_size, shuffle=False
)

# Load datasets with relative position edge attributes
relpos_transforms = Cartesian()
train_set_relpos = ModelNet10Processed(
    data_folder=TRAIN_DATA_FOLDER, 
    data_list_filename=TRAIN_DATA_LIST_FILE, 
    transform=relpos_transforms
)
val_set_relpos = ModelNet10Processed(
    data_folder=TRAIN_DATA_FOLDER, 
    data_list_filename=TRAIN_DATA_LIST_FILE, 
    transform=relpos_transforms
)
train_loader_relpos = tg.loader.DataLoader(
    train_set_relpos, batch_size=batch_size, shuffle=True
)
val_loader_relpos = tg.loader.DataLoader(
    val_set_relpos, batch_size=batch_size, shuffle=False
)

# Define configurations for each type of model
model_configs = {
    "basic": {
        "edge_features": 0,
        "version": 1,
        "description": "Type of edge embeddings: no edge attributes.",
        "train_loader": train_loader,
        "val_loader": val_loader
    },
    "dist": {
        "edge_features": 1,
        "version": 2,
        "description": "Type of edge embeddings: Euclidean distance.",
        "train_loader": train_loader_dist,
        "val_loader": val_loader_dist
    },
    "relpos": {
        "edge_features": 3,
        "version": 3,
        "description": "Type of edge embeddings: relative distance vector.",
        "train_loader": train_loader_relpos,
        "val_loader": val_loader_relpos
    }
}

# Get user input for the model type to train
model_name = input("Which model do you want to train (basic, dist, relpos)? ")

config = model_configs.get(model_name)

# Check if user input corresponds to a valid model type
if config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_DIR = os.path.join(BASE_DIR, '..', '..', 'checkpoints')
    progress_bar = RichProgressBar()
    
    print(config["description"])
    logger = TensorBoardLogger(
        save_dir=os.path.join(BASE_DIR, '..', '..', "logs"), 
        version=config["version"], 
        name="lightning_logs"
    )
    
    mpnn_model = MPNN(
        node_features=node_features,
        edge_features=config["edge_features"],
        hidden_features=hidden_features,
        out_features=out_features,
        num_layers=num_layers,
        aggr=aggr,
        pool=pool,
        act=act
    )

    trainer = pl.Trainer(
        logger=logger, 
        gpus=1, 
        max_epochs=n_epochs, 
        callbacks=[progress_bar]
    )
    trainer.fit(
        model=ModelNet10Classifier(model=mpnn_model, lr=learning_rate),
        train_dataloaders=config["train_loader"],
        val_dataloaders=config["val_loader"]
    )
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_mpnn.ckpt")
    trainer.save_checkpoint(checkpoint_path)

    # To load from checkpoint, uncomment the following lines:
    # trainer.fit(
    #     model=ModelNet10Classifier(model=mpnn_model, lr=learning_rate), 
    #     train_dataloaders=config["train_loader"], 
    #     val_dataloaders=config["val_loader"], 
    #     ckpt_path=checkpoint_path
    # )
else:
    # Handle invalid model type input
    print(f"Invalid model name: {model_name}. Please select from 'basic', 'dist', or 'relpos'.")