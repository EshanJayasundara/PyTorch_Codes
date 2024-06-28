import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ManufacturingDefectDataset(Dataset):

    def __init__(self):
        xy = np.loadtxt('data/manufacturing_defect_dataset.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, :-1])
        self.x.to(device=device)
        self.y = torch.from_numpy(xy[:, [-1]])
        self.y.to(device=device)

        self.n_samples, self.n_features = self.x.shape
    
    def __getitem__(self, index):
        return self.x[index, :], self.y[index]

    def __len__(self):
        return self.n_samples

    def __repr__(self):
        return f"x = {self.x[:2, :]}, y = {self.y[:2, :]}"

dataset = ManufacturingDefectDataset()

batch_size = 90
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# dataiter = iter(dataloader)

epochs = 10
total_samples = len(dataset)
n_iterations = total_samples/batch_size
print(total_samples, batch_size, n_iterations)

for epoch in range(epochs):
    for i, (inputs, label) in enumerate(dataloader):
        # forward pass

        # loss and backward pass

        # update

        if (i+1) % 9 == 0:
            print(f'epoch {epoch+1}/{epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')

