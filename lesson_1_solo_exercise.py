# -*- coding: utf-8 -*-
"""
Created on Thu May 29 19:53:23 2025

@author: Gavin
"""

import torch

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from torch.utils.data import TensorDataset, DataLoader

# Generate 2D data with 3 linearly separable classes
X, y = make_classification(
    n_samples=300, 
    n_features=2, 
    n_informative=2, 
    n_redundant=0, 
    n_classes=3,
    n_clusters_per_class=1,
    class_sep=2.0, 
    random_state=0
)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Visualize the dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.title("2D Classification Dataset with 3 Classes")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

batch_size = 32

# Wrap in DataLoader
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define a simple neural network classifier
class SimpleClassifier(torch.nn.Module): 
    # TODO: Write a neural network to solve this classification problem
    
    pass

# TODO: Set the learning_rate
learning_rate = 0

model = SimpleClassifier()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with loss and accuracy tracking
epochs = 100
loss_history = []
acc_history = []

for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in loader:
        # TODO: Implement the training loop
        
        pass

    if epoch % 10 == 0 or epoch == epochs - 1:
        # TODO: Report the metrics every 10 epochs
        
        pass


# Plot loss and accuracy over time
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_history, label='Loss', color='red')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(acc_history, label='Accuracy', color='blue')
plt.title("Training Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.grid(True)

plt.tight_layout()
plt.show()
