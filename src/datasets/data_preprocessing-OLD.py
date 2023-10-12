import os
import itertools
import numpy as np
import torch
import torch_geometric as tg
import open3d as o3d
from torch_geometric.transforms import BaseTransform, Compose, RadiusGraph, SamplePoints, Constant
from natsort import natsorted
from tqdm import tqdm


class NormPos(BaseTransform):
    def __call__(self, graph):
        x = graph.pos[:,0].tolist()
        x_min = min(x)
        x_max = max(x)

        y = graph.pos[:,1].tolist()
        y_min = min(y)
        y_max = max(y)

        z = graph.pos[:,2].tolist()
        z_min = min(z)
        z_max = max(z)

        pos_all = []
        for i in range(len(graph.pos)):
            pos_tmp = graph.pos[i].tolist()
            pos_tmp = [2 * ((pos_tmp[0] - x_min) / (x_max - x_min)) - 1,
                       2 * ((pos_tmp[1] - y_min) / (y_max - y_min)) - 1,
                       2 * ((pos_tmp[2] - z_min) / (z_max - z_min)) - 1]
            pos_all.append(pos_tmp)

        pos_all = torch.FloatTensor(pos_all)
        graph.pos = pos_all

        return graph
    
    
class MeshToGraph(BaseTransform):
    def __call__(self, graph):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(graph.pos.tolist()))
        mesh.triangles = o3d.cpu.pybind.utility.Vector3iVector(np.asarray(graph.face.T.tolist()))

        mesh = mesh.subdivide_loop(3)
        mesh = mesh.simplify_vertex_clustering(0.2)

        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()

        graph.pos = torch.FloatTensor(np.asarray(mesh.vertices))
        graph.face = torch.FloatTensor(np.asarray(mesh.triangles).T)
        
      
        return graph


class EdgeIndexLongTensor():
    def __call__(self, graph):
        graph.edge_index = torch.LongTensor(graph.edge_index.tolist())
        
        return graph


class AddNodesFeat():
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
    
    
radius = 0.5
batch_size = 1


if not os.path.exists('../../data/processed_train/'):
    os.makedirs('../../data/processed_train/')
if not os.path.exists('../../data/processed_val/'):
    os.makedirs('../../data/processed_val/')


transforms = Compose([SamplePoints(3000),NormPos(),Constant(),AddNodesFeat(),RadiusGraph(r=radius,max_num_neighbors=128)])

train_set_orig = tg.datasets.ModelNet(root='../../data', transform=transforms, train=True)
val_set_orig = tg.datasets.ModelNet(root='../../data', transform=transforms, train=False)


# export processed train data to .pt
for idx, data in enumerate(tqdm(train_set_orig)):
    if len(data.pos) == 512 and len(data.x) == 512:
        torch.save(data, '../../data/processed_train/{}.pt'.format(idx))
        
# export processed val data to .pt
for idx, data in enumerate(tqdm(val_set_orig)):
    if len(data.pos) == 512 and len(data.x) == 512:
        torch.save(data, '../../data/processed_val/{}.pt'.format(idx))
        
    
# export .pt file names to .txt
def get_files(data_folder):
    fnames_lst = []
    dir_entries = os.scandir(data_folder)
    for entry in dir_entries:
        if entry.is_file():
            info = entry.stat()
            fnames_lst.append(entry.name)
    return fnames_lst


# train file names
data_folder = '../../data/processed_train/'
fnames_lst = get_files(data_folder)
natsort_file_names = natsorted(fnames_lst)

# export file names list .txt
with open('../../data/processed_train_file_name_list.txt', 'w') as f:
    for line in natsort_file_names:
        f.write(line)
        f.write('\n')


# val file names
data_folder = '../../data/processed_val/'
fnames_lst = get_files(data_folder)
natsort_file_names = natsorted(fnames_lst)

# export file names list .txt
with open('../../data/processed_val_file_name_list.txt', 'w') as f:
    for line in natsort_file_names:
        f.write(line)
        f.write('\n')