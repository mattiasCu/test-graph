from torchdrug import core
from torchdrug import datasets, transforms,layers
from torchdrug.core import Registry as R
from torchdrug.layers import geometry

import torchdrug
from torchdrug import data

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')


EnzymeCommission = R.search("datasets.EnzymeCommission")
PV = R.search("transforms.ProteinView")
trans = PV(view = "residue")
dataset = EnzymeCommission("~/scratch/protein-datasets/", test_cutoff=0.95, 
                           atom_feature="full", bond_feature="full", verbose=1, transform = trans)

# 数据集第一个样本，前两个残基的原子
protein = dataset[0]["graph"]
is_first_two = (protein.residue_number == 1) | (protein.residue_number == 2)
first_two = protein.residue_mask(is_first_two, compact=True)

#first_two.visualize()
#plt.savefig("fig/first_two.png")

graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()], 
                                                    edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                 geometry.KNNEdge(k=10, min_distance=5),
                                                                 geometry.SequentialEdge(max_distance=2)
                                                                 ],
                                                    #edge_feature="gearnet"
                                                    )

_protein = data.Protein.pack([protein])
protein_ = graph_construction_model(_protein)
print("Graph before: ", _protein)
print("Graph after: ", protein_)

"""
is_first_two = (graph.residue_number == 1) | (graph.residue_number == 2)
first_two = graph.residue_mask(is_first_two, compact=True)
first_two.visualize()
plt.savefig("fig/test_protein.png")
"""


degree = protein_.degree_in + protein_.degree_out
print("Average degree: ", degree.mean())
print("Maximum degree: ", degree.max())
print("Minimum degree: ", degree.min())




