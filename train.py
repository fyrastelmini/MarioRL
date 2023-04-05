from dataloader import make_dataset,preprocess_frame,preprocess_data
from model import PolicyNetwork,ValueNetwork
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, required=True, help='Number of training epochs')
args = parser.parse_args()
epochs=args.num_epochs

df=pd.read_csv("data.csv")
frames, labels = make_dataset(df[0:10000])
# Load data and labels
#data, labels = preprocess_data(frames, labels)

# Print the shapes of the data and labels tensors
#print(f"Data shape: {data.shape}")  # should be (3, 1, 84, 84)
#print(f"Labels shape: {labels.shape}")  # should be (1, 7)

def train_batch(data, target, policy_net, value_net, optimizer, gamma, epsilon_clip):
    optimizer.zero_grad()
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
        # Feed forward the data into the policy and value networks
    policy_output = policy_net(data)
    value_output = value_net(data)

    # Compute the loss
    criterion = nn.BCEWithLogitsLoss()
    #print(policy_output.shape)
    #print(target.shape)
    #print(value_output.shape)
    policy_loss = criterion(policy_output, target)
    value_loss = criterion(value_output, target)

            # Compute the total loss
    loss = policy_loss + value_loss

            # Backpropagate the loss and update the weights
    loss.backward()
    optimizer.step()

    return loss.item()

# Create an instance of the PolicyNetwork model
policy_net = PolicyNetwork(num_actions=7)
value_net = ValueNetwork(num_actions=7)
# Define hyperparameters
gamma = 0.99
epsilon_clip = 0.2
learning_rate = 1e-5
num_epochs = epochs
batch_size = 32

# Create an instance of the Adam optimizer
optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
# Convert data to TensorDataset
print("loading dataset")
print(type(labels))


#check if there's a gpu, if so train on gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
policy_net.to(device)
value_net.to(device)
#put data in gpu
preprocessed_frames, preprocessed_labels = preprocess_data(frames, labels)

# Convert data to TensorDataset
dataset = TensorDataset(torch.Tensor(preprocessed_frames), torch.Tensor(preprocessed_labels))

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



for epoch in range(num_epochs):
    for i, (batch_frames, batch_labels) in enumerate(dataloader):
        batch_frames = batch_frames.to(device)
        batch_labels = batch_labels.to(device)
        loss = train_batch(batch_frames, batch_labels, policy_net=policy_net, value_net=value_net, optimizer=optimizer, gamma=gamma, epsilon_clip=epsilon_clip)
    print(f"Epoch {epoch+1}, Loss = {loss}")


