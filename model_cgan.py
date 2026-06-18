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

num_classes = 10

class ConditionalGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 10)
        
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        c = self.label_emb(labels)
        x = torch.cat([noise, c], 1)
        return self.model(x).view(-1, 1, 28, 28)
    
class ConditionalDyskryminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 10)
        
        self.model = nn.Sequential(
            nn.Linear(794, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = img.view(-1, 784)
        c = self.label_emb(labels)
        x = torch.cat([img_flat, c], 1)
        return self.model(x)
    
netG = ConditionalGenerator()
netD = ConditionalDyskryminator()

criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

num_epochs = 12

num_epochs = 12

print("Zaczynamy trening cGAN...")
for epoch in range(num_epochs):
    for i, (real_images, labels) in enumerate(dataloader):
        batch_size = real_images.size(0)
        
        label_real = torch.ones(batch_size, 1)
        label_fake = torch.zeros(batch_size, 1)

        netD.zero_grad()

        output_real = netD(real_images, labels)
        lossD_real = criterion(output_real, label_real)

        fake_labels = torch.randint(0, num_classes, (batch_size,))
        noise = torch.randn(batch_size, 100)
        fake_images = netG(noise, fake_labels)

        output_fake = netD(fake_images.detach(), fake_labels)
        lossD_fake = criterion(output_fake, label_fake)
        
        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        netG.zero_grad()
        
        output = netD(fake_images, fake_labels)
        lossG = criterion(output, label_real)
        
        lossG.backward()
        optimizerG.step()

    with torch.no_grad():
        test_labels = torch.tensor([num for num in range(10) for _ in range(10)])
        test_noise = torch.randn(100, 100)
        sample_images = netG(test_noise, test_labels)
        
        vutils.save_image(sample_images, f"cgan_epoka_{epoch+1}.png", nrow=10, normalize=True)
        
    print(f"Epoka [{epoch+1}/{num_epochs}] | Loss D: {lossD.item():.4f} | Loss G: {lossG.item():.4f}")

print("Trening cGAN zakończony!")