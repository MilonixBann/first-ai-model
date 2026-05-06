import torch
import torch.nn as nn
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

class MojaSiec(nn.Module):
    def __init__(self):
        super().__init__()
        self.warstwa1 = nn.Linear(784, 256) 
        self.warstwa2 = nn.Linear(256, 128) 
        self.warstwa3 = nn.Linear(128, 10)  

    def forward(self, x):
        x = torch.relu(self.warstwa1(x))
        x = torch.relu(self.warstwa2(x))
        x = self.warstwa3(x)
        return x

model = MojaSiec()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)



model.train()
for epoka in range(2):
    biezacy_loss = 0.0
    for i, (obrazy, etykiety) in enumerate(train_loader):
        optimizer.zero_grad()

        obrazy = obrazy.view(-1, 28*28)

        wyjscie = model(obrazy)
        loss = criterion(wyjscie, etykiety)
        loss.backward()
        optimizer.step()

        biezacy_loss += loss.item()
        if i % 100 == 99:
            print(f'Epoka{epoka + 1}, Paczka{i + 1}, Błąd:{biezacy_loss / 100:.3f}')
            biezacy_loss = 0.0

print("Trening zakończony.")



model.eval()
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transformacja)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

obraz, etykieta = next(iter(test_loader))
zmienna_test = obraz.view(-1, 28*28)

with torch.no_grad():
    output = model(zmienna_test)
    zgadnieta = torch.argmax(output, dim=1)

print(f"Prawdziwa cyfra: {etykieta.item()}")
print(f"AI twierdzi, że to: {zgadnieta.item()}")


wagi = model.warstwa1.weight.data

plt.figure(figsize=(15, 5))
for i in range(10):
    duch = wagi[i].reshape(28, 28)
    
    plt.subplot(2, 5, i+1)
    plt.imshow(duch, cmap='seismic')
    plt.title(f'Neuron {i}')
    plt.axis('off')

plt.show()


torch.save(model.state_dict(), "moj_model.pth")
print("Model zapisany do pliku moj_model.pth!")