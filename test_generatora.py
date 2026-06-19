import torch
import torch.nn as nn
import torchvision.utils as vutils

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
    
netG = ConditionalGenerator()
netG.load_state_dict(torch.load("generator_cyfr.pth"))
netG.eval()


def wygenerujCyfre(jaka_cyfra, nazwa_pliku="wygenerowana.png"):
    if jaka_cyfra < 0 or jaka_cyfra > 9:
        print("Mozna wygenerowac tylko cyfry od 0 do 9!")
        return
    
    szum = torch.randn(1, 100)
    rozkaz = torch.tensor([jaka_cyfra])

    with torch.no_grad():
        obrazek = netG(szum, rozkaz)

    vutils.save_image(obrazek, nazwa_pliku, normalize=True)
    print(f"Sukces! Cyfra {jaka_cyfra} zostala wygenerowana i zapisana jako {nazwa_pliku}")

wygenerujCyfre(jaka_cyfra=3, nazwa_pliku="pierwsza_cyfra_trojka.png")

# for i in range(5):
#     wygenerujCyfre(jaka_cyfra=5, nazwa_pliku=f"wygenerowana_cyfra_piec_{i+1}.png")