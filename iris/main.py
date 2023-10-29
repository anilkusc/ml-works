import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

dataset = pd.read_csv("iris.csv")

dataset.columns = ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species"]

dataset.head()

mappings = {
   "Iris-setosa": 0,
   "Iris-versicolor": 1,
   "Iris-virginica": 2
}

dataset["Species"] = dataset["Species"].apply(lambda x: mappings[x])

dataset.head()

X = dataset.drop("Species",axis=1).values
y = dataset["Species"].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

model = nn.Sequential(
    nn.Linear(4, 25),
    nn.ReLU(),
    nn.Linear(25, 25),
    nn.ReLU(),
    nn.Linear(25, 3),
    nn.Sigmoid()
)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 1000
losses = []

for i in range(epochs):
    y_pred = model(X_train)
    loss = loss_function(y_pred, y_train)
    losses.append(loss)
    print(f'epoch: {i:2}  loss: {loss.item():10.8f}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

preds = []
with torch.no_grad():
    for val in X_test:
        y_hat = model.forward(val)
        preds.append(y_hat.argmax().item())
