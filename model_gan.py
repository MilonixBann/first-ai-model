import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils

transformacja = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transformacja)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1, 28, 28)
    
class Dyskryminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.LeakyReLU(0.2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.model(x)
    
netG = Generator()
netD = Dyskryminator()

criterion = nn.BCELoss()

optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))


num_epochs = 10

print("Zaczynamy trening GAN...")
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        batch_size = real_images.size(0)

        label_real = torch.ones(batch_size, 1)
        label_fake = torch.zeros(batch_size, 1)

        netD.zero_grad()

        output_real = netD(real_images)
        lossD_real = criterion(output_real, label_real)

        noise = torch.randn(batch_size, 100)
        fake_images = netG(noise)

        output_fake = netD(fake_images.detach())
        lossD_fake = criterion(output_fake, label_fake)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()



        netG.zero_grad()

        output = netD(fake_images)
        lossG = criterion(output, label_real)

        lossG.backward()
        optimizerG.step()

    vutils.save_image(fake_images, f"wygenerowane_epoka_{epoch+1}.png", normalize=True)
    print(f"Epoka [{epoch+1}/{num_epochs}] | Błąd Dyskryminatora: {lossD.item():.4f} | Błąd Generatora: {lossG.item():.4f}")

print("Trening ukończony! Sprawdź wygenerowane pliki PNG w folderze.")