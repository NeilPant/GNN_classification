import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import joblib
import networkx as nx
from torch_geometric.nn import GCNConv

# Load the Ego-Facebook dataset again to prepare the data
file_path = '/path to/facebook_combined.txt'  # Update this path if necessary

# Create a graph from the edge list
G = nx.read_edgelist(file_path)

# Create node features and labels
num_nodes = G.number_of_nodes()
node_features = torch.eye(num_nodes)  # Identity matrix as features (one-hot encoding)
labels = []  # List to hold labels

# Assign labels based on degree (custom logic for classification)
degree_dict = dict(G.degree())
node_mapping = {node: idx for idx, node in enumerate(G.nodes())}  # Create a mapping from original node IDs to indices

for node in G.nodes():
    degree = degree_dict[node]
    if degree > 10:
        labels.append(0)  # Active
    elif 1 < degree <= 10:
        labels.append(1)  # Passive
    else:
        labels.append(2)  # Inactive

labels = torch.tensor(labels, dtype=torch.long)

# Create edge index for PyTorch Geometric
edge_index = []
for u, v in G.edges():
    edge_index.append([node_mapping[u], node_mapping[v]])  # Use the mapping to get index

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Transpose and make contiguous

# Create a PyTorch Geometric data object
data = Data(x=node_features, edge_index=edge_index, y=labels)

# Define your GNN class again or import it if defined in another module
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_nodes, 16)  # Adjust num_nodes as necessary
        self.conv2 = GCNConv(16, 3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.softmax(x, dim=1)

# Load the model
model = GNN()  # Create a new instance of your model
model.load_state_dict(torch.load("/path to/70_acc_model.pth"))  # Load the state dict
model.eval()  # Set the model to evaluation mode

# Create test masks
data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

data.test_mask[:] = True

# Evaluation function
def test(data):
    out = model(data)
    pred = out.argmax(dim=1)  # Get predicted class labels
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()  # Count correct predictions
    return int(correct) / int(data.test_mask.sum())  # Return accuracy

# Final evaluation on test set which is whole data
final_accuracy = test(data)
print(f'Final Test Accuracy: {final_accuracy:.4f}')

