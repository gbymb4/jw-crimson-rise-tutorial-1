# -*- coding: utf-8 -*-
"""
Created on Fri May 30 21:46:09 2025

@author: Gavin
"""

# ---------------------------------------------------------------- 
# FOR THIS HOMEWORK, PLEASE COMPLETE THE 'TODO' SECTIONS I'VE LEFT IN THE CODE FOR YOU
# JUST HAVE FUN EXPERIMENTING AND TRYING DIFFERENT IDEAS
# IF YOU GET STUCK, REFER TO THE EXAMPLE WE WORKED THROUGH DURING OUR SESSION
# ----------------------------------------------------------------

import random
import torch

import numpy as np 

import matplotlib.pyplot as plt

# Set seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def generate_random_data(a, b, c, d, e):
    print(f"Polynomial Equation: y = {a}x^4 + {b}x^3 + {c}x^2 + {d}x + {e}")

    # Generate X values from -3 to 3
    X = (torch.rand(100) * 6) - 3  # Scale from 0–1 to -3–3

    # Generate y values based on the polynomial with added noise
    y = (
        a * X**4 +
        b * X**3 +
        c * X**2 +
        d * X +
        e +
        torch.randn_like(X) * 5  # Add Gaussian noise
    )
    
    return X, y

# Generate data for equation: y = 1x^4 + 1x^3 + -5x^2 + -1x + 3
X, y = generate_random_data(1, 1, -5, -1, 3)

plt.scatter(X, y)
plt.show()

# Define a simple model class using torch.nn.Parameter
class LinearModel(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # TODO: Write the parameters for the model for the equation y = 1x^4 + 1x^3 + -5x^2 + -1x + 3



    def forward(self, x):
        # TODO: Write the forward function's return using the parameters to implement y = 1x^4 + 1x^3 + -5x^2 + -1x + 3
        pass


# Instantiate the model
model = LinearModel()

# TODO: Select a learning rate that works to train the model
learning_rate = 0

# Define loss function and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Training loop
epochs = 100
loss_history = []
for epoch in range(epochs):
    # TODO: Implement the training loop and uncomment lines 78 - 90

    # loss_history.append(loss.item())

    # plt.figure(figsize=(8, 5))
    # plt.scatter(X, y, label='Actual Data')
    # plt.plot(X, y_pred.detach(), color='red', label='Model Prediction')
    # plt.title(f"Epoch {epoch} — Loss: {loss.item():.2f}")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # print(f"Epoch {epoch}: gradient = {model.gradient.item():.2f}, intercept = {model.intercept.item():.2f}, loss = {loss.item():.2f}")
    pass
    
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