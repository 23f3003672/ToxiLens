from typing import List, Optional, Tuple
import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

# Supported atom types for one-hot encoding
COMMON_ATOMS = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C, N, O, F, P, S, Cl, Br, I


def atom_to_feature_vector(atom: Chem.Atom) -> List[float]:
    """
    Encode atom into a numerical feature vector:
    - One-hot atomic number (9 features + 1 other)
    - Degree (number of bonded neighbors)
    - Formal charge
    - Is aromatic (0 or 1)
    - Total Num Hydrogens
    Total features = 10 + 4 = 14
    """
    atomic_num = atom.GetAtomicNum()
    one_hot = [1.0 if atomic_num == a else 0.0 for a in COMMON_ATOMS]
    other = [1.0 if atomic_num not in COMMON_ATOMS else 0.0]

    extras = [
        float(atom.GetTotalDegree()),
        float(atom.GetFormalCharge()),
        1.0 if atom.GetIsAromatic() else 0.0,
        float(atom.GetTotalNumHs()),
    ]
    return one_hot + other + extras


def smiles_to_graph(smiles: str, label: Optional[float] = None) -> Optional[Data]:
    """
    Convert a SMILES string into a PyTorch Geometric Data graph object.
    Returns None if SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Compute node features
    node_features = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
    if len(node_features) == 0:
        return None
    x = torch.tensor(node_features, dtype=torch.float32)

    # Compute edge indices (undirected / bidirectional)
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])

    if len(edge_indices) == 0:
        # Molecule has no bonds (e.g. single atom)
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float32)

    return data


class GINNet(nn.Module):
    """
    Graph Isomorphism Network (GIN) for graph-level binary toxicity prediction.
    """

    def __init__(self, in_features: int = 14, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        mlp1 = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs.append(GINConv(mlp1, train_eps=True))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Subsequent layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Graph-level classifier head
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        # Global graph pooling across atoms in each molecule
        h_graph = global_mean_pool(x, batch)

        out = F.relu(self.fc1(h_graph))
        out = self.fc2(out)
        return out


def train_gnn(
    train_graphs: List[Data],
    num_epochs: int = 8,
    batch_size: int = 64,
    lr: float = 0.003,
    pos_weight: Optional[float] = None,
    device: str = "cpu",
) -> GINNet:
    """
    Train GIN on a list of PyTorch Geometric molecular graphs.
    """
    loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    model = GINNet(in_features=14, hidden_dim=64, num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(num_epochs):
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()

    return model


def predict_gnn_proba(
    model: GINNet,
    graphs: List[Data],
    batch_size: int = 128,
    device: str = "cpu",
) -> np.ndarray:
    """
    Inference: Predict P(toxicity=1) across graph batches.
    """
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    probs_list = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
            probs = torch.sigmoid(logits)
            probs_list.append(probs.cpu().numpy().reshape(-1))

    return np.concatenate(probs_list) if len(probs_list) > 0 else np.array([])
