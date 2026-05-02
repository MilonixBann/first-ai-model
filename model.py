import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Zdefiniuj transformację - musimy zamienić obrazki na Tensory (format zrozumiały dla PyTorch)
transformacja = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 2. Pobierz zbiór treningowy (Train Set)
# Podpowiedź: użyj datasets.MNIST, ustaw download=True i train=True
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transformacja)

# 3. Stwórz DataLoader
# To narzędzie, które będzie podawać modelowi dane w małych paczkach (np. po 64 obrazki)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

print("Dane gotowe! Mamy", len(train_set), "obrazków do nauki.")