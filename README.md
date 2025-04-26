# tnt 🧨 (Tiny Neural Trainer)
A small toy library for representing layered ML architectures bundled with a **batch** GD trainer.

## Contents 
Currently contains some default implementations in `tnt.layer` for:
1. Fully connected linear layers
2. ReLU and Tanh activation layers
3. Sequential Layers

and in `tnt.loss` for:
1. MSE loss

By following these implementations, one can write their own custom layers and loss functions.

The trainer in `tnt.trainer` implements batched gradient descent training.

## Usage Example

In the following example we implement a simple regressor to approximate $f(x) = x^2$.
```python
import tnt
import numpy as np

# Create a train dataset
x_train = np.random.rand(1,400)
y_train = np.square(x_train)

# Create a test dataset
x_test = np.random.rand(1,1000)
y_test = np.square(x_test)

# Construct the model by sequencing two linear layers with tanh activation
model = tnt.layer.Sequential(tnt.layer.LinearTanh(1, 3), tnt.layer.LinearTanh(3,1)) 

# Initialize the trainer and train the model
trainer = tnt.trainer.Trainer(model, 1e-3, 300, tnt.loss.MSE())
trainer.train(x_train, y_train) 

# Evaluate the dataset
y_pred = model.predict(x_test)
print("MSE:", tnt.loss.MSE().forward(y_pred, y_test))
```

## Acknowledgements
This project is an independent, personal effort and is not directly related to [PyTorch](https://github.com/pytorch/pytorch), except for taking inspiration from its design patterns, particularly its layer and forward-pass structure.
