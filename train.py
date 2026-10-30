import torch
lr = 0.0001
distance = torch.tensor([1,2,3,4,5])
price =  torch.tensor([100,200,300,400,500])
weight = torch.tensor(0.0,requires_grad=True)
bias = torch.tensor(0.0,requires_grad=True)
optimizer = torch.optim.SGD([weight,bias],lr)
for i in range(100000):
    predictions = distance*weight + bias
    Errors = (predictions - price)**2
    optimizer.zero_grad()
    Loss = torch.mean(Errors)
    Loss.backward()
    optimizer.step()
    weight.grad.zero_()
    bias.grad.zero_()
predictions = distance*weight + bias  
Errors = (predictions - price)**2  
Loss = torch.mean(Errors)
print(weight,bias,Loss)

a = torch.tensor(6)
with torch.no_grad(): predict6km = a*weight + bias
print("Предикт:",predict6km)
model_params = {
    "weight": weight,
    "bias": bias
}
print(model_params)
torch.save(model_params,"model_params.pth")