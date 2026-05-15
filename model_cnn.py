import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transformacja = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transformacja)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

class MojaSiecCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, 64 * 5 * 5)

        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MojaSiecCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

model.train()
for epoka in range(3):
    biezacy_loss = 0.0
    for i, (obrazy, etykiety) in enumerate(train_loader):
        optimizer.zero_grad()
        
        wyjscie = model(obrazy)
        loss = criterion(wyjscie, etykiety)
        loss.backward()
        optimizer.step()
        
        biezacy_loss += loss.item()
        if i % 100 == 99:
            print(f'Epoka {epoka + 1}, Paczka {i + 1}, Błąd: {biezacy_loss / 100:.3f}')
            biezacy_loss = 0.0

wagi = model.conv1.weight.data

plt.figure(figsize=(15, 5))
for i in range(10):

    filtr = wagi[i, 0] 
    
    plt.subplot(2, 5, i+1)
    plt.imshow(filtr, cmap='seismic') 
    plt.title(f'Filtr {i}')
    plt.axis('off')

plt.show()

torch.save(model.state_dict(), "moj_model_cnn.pth")
print("Trening CNN zakończony i model zapisany!")