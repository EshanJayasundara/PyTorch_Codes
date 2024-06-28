import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets 
from sklearn.metrics import confusion_matrix, classification_report

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
# print(y)
n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scale the feature vectors such that each having standard deviation of 1
standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train = standardScaler.transform(X_train)
X_test = standardScaler.transform(X_test)

# convert numpy arrays to tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_train = X_train.to(device=device) # change the device
X_test = torch.from_numpy(X_test.astype(np.float32))
X_test = X_test.to(device=device) # change the device
y_train = torch.from_numpy(y_train.astype(np.float32))
y_train = y_train.to(device=device) # change the device
y_test = torch.from_numpy(y_test.astype(np.float32))
y_test = y_test.to(device=device) # change the device

# change the shape of the target to a column vector
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# model
# f = wx + b, sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x)) 
        return y_pred
    
model = LogisticRegression(n_features)

# loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# training loop
epochs = 100
for epoch in range(epochs):
    # forward pass and loss
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # backward pass
    loss.backward()

    # update
    optimizer.step()

    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1} loss = {loss}')

pred = model(X_test).detach()
pred = pred.round().numpy()

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))