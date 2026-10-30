# Taxi Fare Prediction

My first Machine Learning project. A simple model that predicts taxi fares based on distance. I built it using Python and PyTorch.

## The Project

I used 5 examples to train the model:

| Distance | Price |
|---:|---:|
| 1 km | 100 |
| 2 km | 200 |
| 3 km | 300 |
| 4 km | 400 |
| 5 km | 500 |

The model uses two parameters: `weight` and `bias`. During training, it learns their values to make better predictions.

## Training

I started by creating two tensors: `distance` and `price`. They contain the training data.

Then I created `weight` and `bias` with `requires_grad=True`. This allows PyTorch to calculate their gradients. I also set up an SGD optimizer.

The training loop runs 100,000 times. Each iteration:

1. The model predicts the prices.
2. I calculate the loss using the predictions and actual prices.
3. I calculate the gradients with `loss.backward()`.
4. I update the parameters with `optimizer.step()`.

After training, I save `weight` and `bias` in `model_params.pth`.

## Making Predictions

I use a separate file, `predict.py`, to make predictions.

It loads the saved parameters and calculates the fare for a 6 km trip.

Example output:

```text
tensor(599.4065)
```

I don't need to train the model again every time I want to make a prediction.

## What I Learned

This was my first time working with PyTorch.

I learned how gradient descent works, how to calculate loss, how gradients help update model parameters, and how to save and load trained parameters.

## Project Files

```text
.
├── train.py
├── predict.py
├── model_params.pth
└── README.md
```

- `train.py` — trains the model.
- `predict.py` — loads the saved parameters and makes predictions.
- `model_params.pth` — stores the trained parameters.

## Running the Project

Install PyTorch:

```bash
pip install torch
```

Train the model:

```bash
python train.py
```

Make a prediction:

```bash
python predict.py
```