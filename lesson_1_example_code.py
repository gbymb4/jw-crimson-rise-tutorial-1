# -*- coding: utf-8 -*-
"""
Created on Thu May 29 19:28:13 2025

@author: Gavin
"""

import time
import random
import torch

import numpy as np 

import matplotlib.pyplot as plt

# Set seed
seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Generate data (y = 2x + 10 with some noise)
X = torch.rand(100) * 10  # X values from 0 to 10

true_a = 2
true_b = 5
true_c = 10

y = true_a * X ** 2 + true_b * X + true_c + torch.randn_like(X) * 2  # add some noise

plt.scatter(X, y)
plt.show()

# Define a simple model class using torch.nn.Parameter
class LinearModel(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.gradient = torch.nn.Parameter(torch.randn(1))
        self.intercept = torch.nn.Parameter(torch.randn(1))




    def forward(self, x):
        return self.gradient * x + self.intercept



# Instantiate the model
model = LinearModel()

# Set Hyper-parameters
learning_rate = 0.01

# Define loss function and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Training loop
epochs = 1000
loss_history = []
for epoch in range(epochs):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 10 == 0:
        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, label='Actual Data')
        plt.plot(X, y_pred.detach(), color='red', label='Model Prediction')
        plt.title(f"Epoch {epoch} — Loss: {loss.item():.2f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()

    print(f"Epoch {epoch}: gradient = {model.gradient.item():.2f}, intercept = {model.intercept.item():.2f}, loss = {loss.item():.2f}")
    # time.sleep(0.1)
        
# Final results
print(f"\nLearned weight: {model.gradient.item():.2f}")
print(f"Learned bias: {model.intercept.item():.2f}")

# Plot the results
plt.scatter(X, y, label='Actual Data')
plt.plot(X, model(X).detach(), color='red', label='Fitted Line')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Fitting y = 2x + 10 with PyTorch")
plt.legend()
plt.show()

# Plot loss over time
plt.figure(figsize=(8, 5))
plt.plot(loss_history, color='purple')
plt.title("Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()