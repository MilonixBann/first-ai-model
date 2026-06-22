import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import os, ssl

# Ominięcie problemu z SSL
ssl._create_default_https_context = ssl._create_unverified_context

transformacja = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.CIFAR10(root='./data_cifar', train=True, download=True, transform=transformacja)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

nz = 100
ngf = 64
ndf = 64

class WGAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, nz, 1, 1)
        return self.main(x)
    
# UZYCIE KRYTYKA ZAMIAST DYSKRYMINATORA - USUNIECIE SIGMOID
class WGAN_Krytyk(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)
            # BRAK SIGMOIDA NA KONCU // WYJSCIE TO SUROWA LICZBA
        )
    
    def forward(self, x):
        return self.main(x).view(-1, 1)
    
netG = WGAN_Generator()
netC = WGAN_Krytyk()

# UZYCIE BARDZIEJ OPTYMALNEGO DLA WGAN RMSprop ZAMIAST Adam
optimizerC = optim.RMSprop(netC.parameters(), lr=0.00005)
optimizerG = optim.RMSprop(netG.parameters(), lr=0.00005)

num_epochs = 30
clipping_value = 0.01 # Wartosc przycinania wag

print("Zaczynamy trening WGAN...")
for epoch in range(num_epochs):
    # Iterator po dataloaderze, zeby moc pobierac paczki recznie
    data_iter = iter(dataloader)
    i = 0

    while i < len(dataloader):
        # Trenowanie Krytyka 5 raz
        for _ in range(5):
            if i >= len(dataloader):
                break

            real_images, _ = next(data_iter)
            i += 1
            batch_size = real_images.size(0)

            netC.zero_grad()

            # Ocena prawdziwych zdjec
            output_real = netC(real_images)
            loss_real = torch.mean(output_real)

            # Generowanie sztucznych zdjec
            noise = torch.randn(batch_size, nz)
            fake_images = netG(noise)

            # Ocena sztucznych zdjec
            output_fake = netC(fake_images.detach())
            loss_fake = torch.mean(output_fake)

            lossC = loss_fake - loss_real
            lossC.backward()
            optimizerC.step()

            # KLUCZOWY ELEMENT: Przycinanie wag Krytyka
            for p in netC.parameters():
                p.data.clamp_(-clipping_value, clipping_value)

        # Trenowanie Generatora tylko 1 raz
        netG.zero_grad()

        output = netC(fake_images)
        lossG = -torch.mean(output) # Minus, zeby odwrocic wartosc bo trzeba maksymalizowac wynik oceny

        lossG.backward()
        optimizerG.step()

    vutils.save_image(fake_images, f"wgan_kolor_epoka_{epoch+1}.png", normalize=True)
    # W WGAN wartosci loss nie oznaczaja pomylki, tylko odleglosc miedzy dystrybucjami.
    print(f"Epoka [{epoch+1}/{num_epochs}] | Dystans Wassersteina (Loss C): {-lossC.item():.4f} | Loss G: {lossG.item():.4f}")

print("Trening WGAN zakończony!")