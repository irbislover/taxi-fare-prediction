import torch
parameters = torch.load("model_params.pth")
print(parameters)
distance = torch.tensor(6)
with torch.no_grad(): predict = distance * parameters["weight"] + parameters["bias"]
print(predict)