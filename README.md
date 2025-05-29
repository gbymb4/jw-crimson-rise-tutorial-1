# Deep Learning with PyTorch - Session 1

## Session Timeline

| Time      | Activity                                |
| --------- | --------------------------------------- |
| 0:00 - 0:10 | 1. Rapport + Skill Check                |
| 0:10 - 0:20 | 2. Run the Full Example Together        |
| 0:20 - 0:45 | 3. Explain the Code (Discussion/Slides) |
| 0:45 - 1:15 | 4. Guided Hands-On Practice             |
| 1:15 - 1:50 | 5. Independent Challenge                |
| 1:50 - 2:00 | 6. Wrap-Up + Homework                   |

---

## 1. Rapport + Skill Check

### Goals

Let's start by getting to know each other and where you're at with Python and AI! This will help make the session more fun and make sure we go at the right pace.

We'll talk a bit about:

* What you've already learned in Python.
* How much you've heard about neural networks or machine learning, and what you've done so far.
* If you've ever used tools like PyTorch before.

This is not a test - it's just a way to understand your experience so we can have a great time learning together!

### Questions We Might Chat About

* What subjects do you enjoy the most?
* What projects have you coded in Python before?
* What types of AI or machine learning do you know about or tried?
* What do you know neural networks? Especially in PyTorch?

### Skill Diagnostic

```python
import torch
x = torch.tensor([1.0, 2.0, 3.0])
print(x * 2)
```

What do you think the output of this code will be?

a)
```
>>> torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
```

b)
```
>>> torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
```

c)
```
>>> torch.tensor([2.0, 4.0, 6.0])
```

---

## 2. Run the Full Example Together

### Goals

* Show a working model right away to demonstrate the power of PyTorch.
* Let Jeremy see the "big picture" before breaking it down.


WeРђЎre going to train a tiny machine learning model to learn the equation:

```
y = 2x + 10
```

Instead of giving it the equation, we'll let it **figure it out** from the data by adjusting two numbers - the slope (gradient) and the offset (y-intercept) - just like solving a puzzle!

This is a great way to see how machine learning really works at a basic level.

---

### Code Demo: Learn y = 2x + 10 from Random Data

Paste this into a Python or Colab notebook, or run the script `session_1_example_code.py`:

```python
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

true_gradient = 2
true_intercept = 10

y = true_gradient * X + true_intercept + torch.randn_like(X) * 2  # add some noise

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
epochs = 100
loss_history = []
for epoch in range(epochs):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

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
    time.sleep(0.1)
        
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
```

---

### What is Happening Here?

* We're **randomly generating points** along the line `y = 2x + 10`, but with a little noise to make it realistic.
* Instead of giving the model the formula, we ask it to **learn the best values for `gradient` (slope) and `intercept` (offset)** using gradient descent.
* The model learns by trying values, seeing how far off it is, then adjusting - this is **training**!
* We use a **loss function** (`MSELoss`) to measure how wrong the guesses are, and an **optimizer** (`SGD`) to help it improve.

---

## 3. Explain the Code

### Goals

* Understand how PyTorch represents models and parameters.
* Learn what each major block of code does in the linear regression example.
* Build intuition for training a model using gradient descent.
* Connect the code to the theory of fitting a line to data.

---

### Code Walkthrough

---

### 1. **Data Generation**

```python
X = torch.rand(100) * 10  # Random inputs between 0 and 10
y = true_gradient * X + true_intercept + torch.randn_like(X) * 2  # y = 2x + 10 + noise
```

* We create **100 random points** between 0 and 10 as input `X`.
* The output `y` is generated from the equation `y = 2x + 10` plus some **random noise**.
* This mimics real-world data, which is rarely perfect and always has some randomness.

---

### 2. **Model Definition**

```python
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gradient = torch.nn.Parameter(torch.randn(1))
        self.intercept = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.gradient * x + self.intercept
```

* The model is a simple **linear function**: `y = gradient * x + intercept`.
* `torch.nn.Parameter` tells PyTorch these values should be **learned during training**.
* The `forward` method defines how to compute predictions from inputs.

---

### 3. **Loss Function and Optimizer**

```python
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

* The **loss function** measures how far off the model’s predictions are from the true `y` values.
* We use **Mean Squared Error (MSE)** — average of squared differences.
* The **optimizer** updates model parameters to reduce the loss, here using **Stochastic Gradient Descent (SGD)**.

---

### 4. **Training Loop**

```python
for epoch in range(epochs):
    y_pred = model(X)                 # Make predictions
    loss = loss_fn(y_pred, y)        # Calculate loss

    optimizer.zero_grad()             # Clear previous gradients
    loss.backward()                  # Compute gradients (backpropagation)
    optimizer.step()                 # Update parameters with gradients
```

* Each **epoch** is one complete pass through the data.
* The model makes predictions for all inputs `X`.
* We calculate the loss, then run **backpropagation** to find gradients.
* The optimizer uses these gradients to adjust `gradient` and `intercept` parameters to better fit the data.
* This loop repeats, improving the model step-by-step.

---

### 5. **Visualization**

* Plotting the data and model predictions every epoch helps **see how the model learns** over time.
* The **loss curve** at the end shows how error decreases, confirming the model is improving.

---

### Key Takeaways

* We **don’t tell** the model the formula — it *learns* `gradient` and `intercept` by adjusting them to minimize error.
* The parameters start with random values and gradually move toward the true values (`2` and `10`).
* This process demonstrates **gradient descent** — the core idea behind training most machine learning models.

---


## 4. Questions and Guided Hands-On Practice

### Goals

* Encourage Jeremy to articulate his understanding.
* Identify any gaps or confusion early.
* Guide Jeremy through modifying and extending the code.
* Foster problem-solving skills by hands-on coding.

---

### **Discussion & Questions**

Start by asking Jeremy a few questions to check understanding and encourage reflection:

* “Can you explain in your own words what the model is learning?”
* “What do the gradient and intercept represent in the line `y = mx + b`?”
* “Why do we add noise to the data? What would happen if there was no noise?”
* “How does the optimizer know how to update the parameters?”
* “What would happen if we increased or decreased the learning rate?”

Give Jeremy time to answer, and offer hints or clarifications as needed.

---

### **Guided Coding Exercises**

Walk Jeremy through the following small tasks. Let him type, experiment, and observe the effects. Be ready to support and explain.

---

### Task 1: **Change the Number of Epochs**

* Ask Jeremy to increase the training epochs from 100 to 200.
* Question: “What changes do you expect to see in the loss curve and model fit?”
* Run the training and observe — discuss overfitting vs underfitting.

---

### Task 2: **Experiment with Learning Rate**

* Change the `learning_rate` variable to a higher value (e.g., 0.1) or lower value (e.g., 0.001).
* Ask: “How does changing the learning rate affect training speed and stability?”
* Run the training, look for oscillations or slow convergence.

---

### Task 3: **Add a New Parameter**

* Challenge Jeremy to add a *second feature* to the model.
* For example: generate `y = 2x ^ 2 + y`, then try fitting a model with 3 parameters (gradient, exponent, and intercept).
* This can be a stretch task — guide through expanding the model and data tensors.

---

### Task 4: **Modify the Model to Use a Different Loss**

* Introduce another loss function like Mean Absolute Error (`torch.nn.L1Loss()`).
* Ask: “How do you think the loss curve will differ?”
* Run and compare results.

---

### Tips During Practice:

* Encourage Jeremy to **print intermediate values** or use plots.
* Ask him to **predict** outcomes before running experiments.
* Prompt reflection: “What did you learn from this change?”
* Celebrate small wins to boost confidence.

---

### Optional: Mini Quiz (if time permits)

1. What PyTorch function is used to clear gradients before backpropagation?
2. Why do we call `.detach()` when plotting the predicted values?
3. What does `requires_grad=True` mean?

---

## 5. Independent Challenge

### Goal

Let Jeremy try a complete, visual, and creative classification task solo — to build confidence and reinforce what he’s learned.

---

### Task

Create a 2D dataset with **3 linearly separable classes**, train a model to classify the points, and visualize the results.

Jeremy will:

* Generate the dataset.
* Build and train a simple neural network.
* Predict and visualize class regions in 2D space.

---

### Starter Code

```python
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

```

---

### Challenge Prompt

Once the model is trained, Jeremy should:

1. **Plot the decision regions** — visualize what the model has learned.
2. **Test the model on new points** — pick new coordinates and see how the model classifies them.

---

### Bonus Task (Optional)

* Add noise or overlapping regions and see how well the model handles ambiguity.
* Try using more layers or different activation functions.

---

## 6. Wrap-Up + Homework

### Recap

Ask:

* What are tensors?
* What does the model learn to do?
* What happens during training?
* How can you use the model on new images?

### Homework Ideas

To continue building your PyTorch skills, try these projects that expand on what we did today - applying the basics to new problems and ideas:

---

#### 1. **Multiple Linear Regression**

* Create a dataset with *multiple input features* (for example, 2 or 3 variables).
* Modify the model to learn multiple weights (one for each input feature) plus an intercept.
* Train the model to fit this data.
* This helps you understand how PyTorch handles more complex inputs.

---

#### 2. **Batch Training**

* Instead of training on all data at once, split the data into *batches*.
* Update the model using batches (mini-batch gradient descent).
* Observe how this affects training speed and convergence.

---

#### 3. **Visualizing Parameters Changing**

* Add code to track and plot the parameters of your model during training.
* See how values change epoch by epoch.

---

## PyTorch Setup Instructions

### Windows (with GPU)

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Windows (CPU only)

```bash
pip3 install torch torchvision torchaudio
```

### macOS

```bash
pip3 install torch torchvision torchaudio
```

### Linux (with GPU)

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Linux (CPU only)

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Install Matplotlib

```bash
pip install matplotlib
```
