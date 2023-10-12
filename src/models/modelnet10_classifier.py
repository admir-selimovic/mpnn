import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from .mpnn import MPNN

# ------------------------ LIGHTNING MODULE ------------------------

class ModelNet10Classifier(pl.LightningModule):
    '''Lightning module for classifying ModelNet10 shapes.'''
    
    def __init__(self, model, lr, **kwargs):
        '''Initialize the lightning module.

        Parameters:
        - model: Instance of the model to use for predictions.
        - lr (float): Learning rate.
        '''
        super().__init__(**kwargs)
        self.model = model
        self.lr = lr
        self.train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.valid_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, graph):
        '''Forward pass of the lightning module.

        Parameters:
        - graph (torch_geometric.data.data.Data): Input graph.

        Returns:
        - torch.Tensor: Predictions tensor.
        '''
        if isinstance(self.model, MPNN):
            return self.model(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, batch=graph.batch)

    def training_step(self, graph):
        '''Training step for the lightning module.

        Parameters:
        - graph (torch_geometric.data.data.Data): Input graph for training.

        Returns:
        - torch.Tensor: Training loss tensor.
        '''
        pred = self(graph).squeeze()
        loss = F.cross_entropy(pred, graph.y)
        self.train_metric(pred, graph.y)
        return loss

    def training_epoch_end(self, outs):
        '''Actions to perform at the end of each training epoch.'''
        self.log("train acc", self.train_metric, prog_bar=True)

    def validation_step(self, graph, batch_idx):
        '''Validation step for the lightning module.

        Parameters:
        - graph (torch_geometric.data.data.Data): Input graph for validation.
        - batch_idx (int): Index of the current batch.
        '''
        pred = self(graph).squeeze()
        self.valid_metric(pred, graph.y)

    def validation_epoch_end(self, outs):
        '''Actions to perform at the end of each validation epoch.'''
        self.log("valid acc", self.valid_metric, prog_bar=True)

    def configure_optimizers(self):
        '''Configure the optimizers for the lightning module.

        Returns:
        - torch.optim.Optimizer: Optimizer to use.
        '''
        return torch.optim.Adam(self.parameters(), lr=self.lr)