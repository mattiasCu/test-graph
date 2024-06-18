from torchdrug import core
from torchdrug import datasets, transforms
from torchdrug.core import Registry as R

import torchdrug
from torchdrug import data

import matplotlib as mpl
import matplotlib.pyplot as plt

EnzymeCommission = R.search("datasets.EnzymeCommission")
PV = R.search("transforms.ProteinView")
trans = PV(view = "residue")
dataset = EnzymeCommission("~/scratch/protein-datasets/", test_cutoff=0.95, 
                           atom_feature="full", bond_feature="full", verbose=1, transform = trans)

dataset.visualize()
print(dataset.data.node_feature.shape)  # torch.Size([6, 69])
print(dataset.data.edge_feature.shape)  # torch.Size([12, 19])
plt.show()



