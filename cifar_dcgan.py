import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

transformacja = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # R, G, B
])

dataset = datasets.CIFAR10(root='./data_cifar', train=True, download=True, transform=transformacja)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

nz = 100   # noise size
ngf = 64   # map size in generator
ndf = 64   # map size in discriminator

class DCGAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, nz, 1, 1)
        return self.main(x)
    

class DCGAN_Dyskryminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)
    
netG = DCGAN_Generator()
netD = DCGAN_Dyskryminator()

criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

num_epochs = 15

print("Zaczynamy trening kolorowego DCGAN...")
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        batch_size = real_images.size(0)
        label_real = torch.ones(batch_size, 1)
        label_fake = torch.zeros(batch_size, 1)

        netD.zero_grad()
        output_real = netD(real_images)
        lossD_real = criterion(output_real, label_real)

        noise = torch.randn(batch_size, nz)
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

    vutils.save_image(fake_images, f"dcgan_kolor_epoka_{epoch+1}.png", normalize=True)
    print(f"Epoka [{epoch+1}/{num_epochs}] | Loss D: {lossD.item():.4f} | Loss G: {lossG.item():.4f}")

print("Trening zakończony!")