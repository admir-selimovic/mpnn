import os
from pathlib import Path
import itertools
import numpy as np
import torch
import torch_geometric as tg
import open3d as o3d
import yaml
from torch_geometric.transforms import BaseTransform, Compose, RadiusGraph, SamplePoints, Constant
from natsort import natsorted
from tqdm import tqdm

# ------------------------ CONSTANTS ------------------------

CURRENT_SCRIPT_PATH = Path(os.path.abspath(__file__))
BASE_DIR = CURRENT_SCRIPT_PATH.parent.parent.parent
RADIUS = 0.5
SAMPLE_POINTS_NUM = 3000
MAX_NUM_NEIGHBORS = 128

# ------------------------ TRANSFORMATIONS ------------------------

class NormPos(BaseTransform):
    '''Normalize position of the graph nodes.'''
    
    def __call__(self, graph):
        x, y, z = graph.pos[:, 0], graph.pos[:, 1], graph.pos[:, 2]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        z_min, z_max = z.min(), z.max()

        pos_all = [
            [
                2 * ((pos[0] - x_min) / (x_max - x_min)) - 1,
                2 * ((pos[1] - y_min) / (y_max - y_min)) - 1,
                2 * ((pos[2] - z_min) / (z_max - z_min)) - 1
            ]
            for pos in graph.pos
        ]

        graph.pos = torch.FloatTensor(pos_all)
        return graph

class AddNodesFeat():
    '''Add node features based on distance buckets.'''
    
    def __init__(self):
        self.buckets = [
            (0.04, 1.),
            (0.05, 0.99),
            (0.06, 0.98),
            (0.07, 0.97),
            (0.08, 0.96),
            (0.09, 0.95),
            (0.10, 0.94),
            (0.125, 0.9),
            (0.15, 0.8),
            (0.20, 0.5),
            (float('inf'), 0.0)  # To cover everything larger than 0.20
        ]

    def get_bucket_value(self, value):
        for limit, bucket_val in self.buckets:
            if value <= limit:
                return bucket_val
        return 0.  # default, shouldn't be reached

    def calculate_distance(self, p1, p2):
        # Convert tensors to numpy arrays if they are not already
        p1 = p1.numpy() if isinstance(p1, torch.Tensor) else p1
        p2 = p2.numpy() if isinstance(p2, torch.Tensor) else p2
        squared_dist = np.sum((p1 - p2) ** 2, axis=0)
        return np.sqrt(squared_dist)

    def __call__(self, graph):
        x = np.linspace(-1, 1, 8)
        y = np.linspace(-1, 1, 8)
        z = np.linspace(-1, 1, 8)
        pos_new = torch.FloatTensor(np.array(list(itertools.product(x, y, z))))

        pos_new2 = np.tile(pos_new, (len(graph.pos), 1))
        graph_pos2 = np.repeat(graph.pos, len(pos_new), axis=0)

        distances = self.calculate_distance(pos_new2.T, graph_pos2.T)
        min_distances = distances.reshape(len(pos_new), -1).min(axis=1)

        min_dist_lst_bucketed = [self.get_bucket_value(dist) for dist in min_distances]

        graph.x = torch.FloatTensor([[item] for item in min_dist_lst_bucketed])
        graph.pos = pos_new

        return graph
    
# ------------------------ UTILITIES ------------------------

def load_config(file_path):
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_data_directory(directory):
    """Create data directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_files(data_folder):
    """Retrieve file names from a folder."""
    return natsorted([entry.name for entry in os.scandir(data_folder) if entry.is_file()])

def export_file_names_to_txt(data_folder, file_name):
    """Export file names from a folder to a .txt file."""
    with open(file_name, 'w') as f:
        for line in get_files(data_folder):
            f.write(line)
            f.write('\n')

# ------------------------ MAIN PROCESSING ------------------------

paths_path = BASE_DIR / "configs" / "paths.yml"
paths = load_config(paths_path)

# Define transformations
transforms = Compose([
    SamplePoints(SAMPLE_POINTS_NUM),
    NormPos(),
    Constant(),
    AddNodesFeat(),
    RadiusGraph(r=RADIUS, max_num_neighbors=MAX_NUM_NEIGHBORS)
])

# Create data directories
create_data_directory(BASE_DIR / paths["TRAIN_DATA_FOLDER"])
create_data_directory(BASE_DIR / paths["VAL_DATA_FOLDER"])

# Load original datasets
data_root = (BASE_DIR / "../../data").resolve()
train_set_orig = tg.datasets.ModelNet(root=str(data_root), transform=transforms, train=True)
val_set_orig = tg.datasets.ModelNet(root=str(data_root), transform=transforms, train=False)

# Process and export training data
train_data_folder = BASE_DIR / paths["TRAIN_DATA_FOLDER"]
for idx, data in enumerate(tqdm(train_set_orig)):
    if len(data.pos) == 512 and len(data.x) == 512:
        torch.save(data, train_data_folder / f'{idx}.pt')

# Process and export validation data
val_data_folder = BASE_DIR / paths["VAL_DATA_FOLDER"]
for idx, data in enumerate(tqdm(val_set_orig)):
    if len(data.pos) == 512 and len(data.x) == 512:
        torch.save(data, val_data_folder / f'{idx}.pt')

# Export processed file names to .txt
export_file_names_to_txt(str(train_data_folder), str(BASE_DIR / paths["TRAIN_DATA_LIST_FILE"]))
export_file_names_to_txt(str(val_data_folder), str(BASE_DIR / paths["VAL_DATA_LIST_FILE"]))