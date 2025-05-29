# -*- coding: utf-8 -*-
"""
Created on Thu May 29 20:13:41 2025

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
    
    def __init__(self):
        super().__init__()
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 3)  # 3 output classes
        )

    def forward(self, x):
        return self.net(x)

model = SimpleClassifier()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop with loss and accuracy tracking
epochs = 100
loss_history = []
acc_history = []

for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in loader:
        logits = model(batch_X)
        loss = loss_fn(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.item() * batch_X.size(0)

        # Convert logits to probabilities, then to predictions
        probs = torch.nn.functional.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        # Track accuracy
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    epoch_loss = total_loss / total
    epoch_acc = correct / total

    loss_history.append(epoch_loss)
    acc_history.append(epoch_acc)

    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}")


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