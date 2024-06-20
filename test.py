from torchdrug import transforms
from torchdrug import datasets
import time
from torchdrug import core
from torchdrug import datasets, transforms,layers, tasks, models
from torchdrug.core import Registry as R
from torchdrug.layers import geometry
from torchdrug import core

import torch
import torchdrug
from torchdrug import data

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')



truncate_transform = transforms.TruncateProtein(max_length=350, random=False)
protein_view_transform = transforms.ProteinView(view="residue")
transform = transforms.Compose([truncate_transform, protein_view_transform])



EnzymeCommission = R.search("datasets.EnzymeCommission")
PV = R.search("transforms.ProteinView")
trans = PV(view = "residue")
dataset = EnzymeCommission("~/scratch/protein-datasets/", test_cutoff=0.95, 
                           atom_feature="full", bond_feature="full", verbose=1, transform = trans)

train_set, valid_set, test_set = dataset.split()

gearnet = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512], num_relation=7,
                         batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")

gearnet_edge = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512], 
                              num_relation=7, edge_input_dim=59, num_angle_bin=8,
                              batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")


graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()], 
                                                    edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                 geometry.KNNEdge(k=10, min_distance=5),
                                                                 geometry.SequentialEdge(max_distance=2)],
                                                    edge_feature="gearnet")

task = tasks.MultipleBinaryClassification(gearnet, graph_construction_model=graph_construction_model, num_mlp_layer=3,
                                          task=[_ for _ in range(len(dataset.tasks))], criterion="bce", metric=["auprc@micro", "f1_max"])



optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                     gpus=[0], batch_size=4)
solver.train(num_epoch=2)
solver.evaluate("valid")

print(solver.evaluate("test"))