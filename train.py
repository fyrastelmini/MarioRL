from dataloader import make_dataset,preprocess_frame,preprocess_data
from model import PolicyNetwork,ValueNetwork
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
df=pd.read_csv("data.csv")
frames, labels = make_dataset(df)
# Load data and labels
#data, labels = preprocess_data(frames, labels)

# Print the shapes of the data and labels tensors
#print(f"Data shape: {data.shape}")  # should be (3, 1, 84, 84)
#print(f"Labels shape: {labels.shape}")  # should be (1, 7)

def train_batch(frames, labels, policy_net, value_net, optimizer, gamma, epsilon_clip):
    frames_tensor = torch.tensor(frames, dtype=torch.float32).unsqueeze(1)

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Calculate action probabilities and values
    action_probs = policy_net(frames_tensor)
    state_values = value_net(frames_tensor)
    # Calculate log probabilities of chosen actions
    chosen_action_probs = action_probs.gather(1, labels_tensor)
    log_probs = torch.log(chosen_action_probs)

    # Calculate total loss
    loss = -torch.mean(log_probs * state_values)
    print(loss)
    # Backpropagate and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# Create an instance of the PolicyNetwork model
policy_net = PolicyNetwork(num_actions=7)
value_net = ValueNetwork()
# Define hyperparameters
gamma = 0.99
epsilon_clip = 0.2
learning_rate = 1e-5
num_epochs = 10
batch_size = 32

# Create an instance of the Adam optimizer
optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
# Convert data to TensorDataset
print("loading dataset")
print(type(labels))
preprocessed_frames, preprocessed_labels = preprocess_data(frames, labels)

# Convert data to TensorDataset
dataset = TensorDataset(torch.Tensor(preprocessed_frames), torch.Tensor(preprocessed_labels))

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for i, (batch_frames, batch_labels) in enumerate(dataloader):
        loss = train_batch(batch_frames, batch_labels, policy_net=policy_net, value_net=value_net, optimizer=optimizer, gamma=gamma, epsilon_clip=epsilon_clip)
    print(f"Epoch {epoch+1}, Loss = {loss}")


