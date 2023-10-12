import os
import torch
from torch_geometric.data.dataset import Dataset

# ------------------------ CUSTOM DATASET ------------------------

class ModelNet10Processed(Dataset):
    '''Custom dataset class for the processed ModelNet10 data.'''
    
    def __init__(self, data_folder, data_list_filename, transform=None):
        '''Initializes the dataset.

        Parameters:
        - data_folder (str): Path to the folder containing processed data files.
        - data_list_filename (str): Path to the .txt file listing filenames in data_folder.
        - transform (callable, optional): Optional transform to apply on a sample. Defaults to None.
        '''
        super().__init__(transform)
        self.data_folder = data_folder
        self.data_file_list = open(data_list_filename, 'r').read().splitlines()
        self.transform = transform
        
    def __len__(self):
        '''Gets the number of data points in the dataset.

        Returns:
        - int: Number of data points in the dataset.
        '''
        return len(self.data_file_list)

    def __getitem__(self, idx):
        '''Retrieves a data point by its index.

        Parameters:
        - idx (int): Index of the data point.

        Returns:
        - torch_geometric.data.data.Data: Graph data.
        '''
        filename = self.data_file_list[idx]
        data_path = os.path.join(self.data_folder, filename)
        data = torch.load(data_path)
        data = data if self.transform is None else self.transform(data)
        return data