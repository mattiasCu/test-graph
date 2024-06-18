import torchdrug
from torchdrug import data

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')


# 创建一个示例图
mol = data.Molecule.from_smiles("C1=CC=CC=C1")
mol.visualize()
print(mol.node_feature.shape)  # torch.Size([6, 69])
print(mol.edge_feature.shape)  # torch.Size([12, 19])
plt.show()

