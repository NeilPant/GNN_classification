import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
import networkx as nx

# Load the Ego-Facebook dataset
file_path = '/path to/facebook_combined.txt'  # Update this path to where the dataset is stored

# Create a graph from the edge list
G = nx.read_edgelist(file_path)

# Create node features and labels
num_nodes = G.number_of_nodes()
node_features = torch.eye(num_nodes)  # Identity matrix as features (one-hot encoding)
# node 0 as [1,0,0,0,...]
# node 1 as [0,1,0,0,....]
# this is known as one hot encoding
labels = []  # List to hold labels

# Assign labels based on degree (custom logic for classification)
degree_dict = dict(G.degree())
node_mapping = {node: idx for idx, node in enumerate(G.nodes())}  # Create a mapping from original node IDs to indices
# we map as it is needed for faster implementation rather than traversing the graph every time
for node in G.nodes():
    degree = degree_dict[node]
    if degree > 10:
        labels.append(0)  # Active user
    elif 1 < degree <= 10:
        labels.append(1)  # Passive user
    else:
        labels.append(2)  # Inactive user

labels = torch.tensor(labels, dtype=torch.long) # we convert it to a tensor for sending it as a training dataset as pytorch understands it

# Create edge index for PyTorch Geometric
# Ensure that edges are correctly mapped to their indices
edge_index = []
for u, v in G.edges():
    edge_index.append([node_mapping[u], node_mapping[v]])  # Use the mapping to get index

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Transpose and make contiguous
# we transpose it to make 2 rows with indexes first row contains the source node indices and second row contains the target node indices
# Making it contiguous ensures that the memory layout is optimized for performance during training and inference.
# Create a PyTorch Geometric data object
data = Data(x=node_features, edge_index=edge_index, y=labels)

# Define the GNN model
{# another bigger model with about the same accuracy
# class GNN(torch.nn.Module):
#     def __init__(self):
#         super(GNN, self).__init__()
#         self.conv1 = GCNConv(num_nodes, 64)  # First layer with 64 output features
#         self.dropout = torch.nn.Dropout(p=0.5)  # Dropout layer
#         self.conv2 = GCNConv(64, 64)          # Second layer with 64 output features
#         self.conv3 = GCNConv(64, 32)          # Third layer with 32 output features
#         self.conv4 = GCNConv(32, 3)           # Fourth layer with 3 output classes

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.dropout(x)                   # Apply dropout
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = self.dropout(x)                   # Apply dropout
#         x = self.conv3(x, edge_index)
#         x = F.relu(x)
#         x = self.dropout(x)                   # Apply dropout
#         x = self.conv4(x, edge_index)
#         return F.log_softmax(x, dim=1)
}

class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_nodes, 16)  # First layer with 16 output features
        self.conv2 = GCNConv(16, 3)          # Second layer with 3 output classes

    def forward(self, data): # function for forward propogation
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x) # using relu activation function on the first layer
        x = self.conv2(x, edge_index)
        return F.softmax(x,dim=1)# using softmax activation function on the second/output layer as it is a multi-class classification problem

# Initialize model, optimizer, and loss function
model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # adam is used as an optimizer for faster working on the gradient descent algo with the initial learning rate as 0.01
criterion = F.nll_loss # negative log likelihood loss function is used as a loss function
# NLL(y,y_cap)=-sigma^N_i=1(yi*log(yi_cap))
# here N is the no. of classes
# y is the true distribution(one hot encoded vector of true labels)
# y_cap is the predicted probability distribution (output from the model after applying softmax).
# Training the model
def train(data):
    model.train() # set model to training mode
    optimizer.zero_grad() # this resets the gradient before performing back-propagation as after initial run it will store it and do incorrect updates
    out = model(data) # the data we get after forward prop
    loss = criterion(out[data.train_mask], data.y[data.train_mask]) # we pass it through the loss function to get the loss
    loss.backward() # we do Backpropagation to get the change in weights and biases
    optimizer.step() # this updates the values of weight and biases
    return loss.item() # this allows the tracking of loss value during iterations

# Evaluation function
def test(data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)  # Get predicted class labels
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()  # Count correct predictions
    return int(correct) / int(data.test_mask.sum())  # Return accuracy

# Create train/test masks (for simplicity, we'll use a basic split here)
data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_size = int(0.8 * num_nodes)  # Use 80% of nodes for training
data.train_mask[:train_size] = True
data.test_mask[train_size:] = True

# Training loop
for epoch in range(200):  # Adjust epochs as needed
    loss = train(data)
    if epoch % 20 == 0:
        acc = test(data)
        print(f'Epoch {epoch}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')

# Final evaluation on test set
final_accuracy = test(data)
print(f'Final Test Accuracy: {final_accuracy:.4f}')

# Save only the model's state_dict
# torch.save(model.state_dict(), "70_acc_model.pth")




