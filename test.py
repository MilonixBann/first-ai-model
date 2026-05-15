import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

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
model.load_state_dict(torch.load("moj_model_cnn.pth"))
model.eval()

transformacja = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def sprawdz_cyfre(sciezka_do_pliku):
    img = Image.open(sciezka_do_pliku)
    img = transformacja(img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        wynik = model(img)
        prawdopodobienstwa = F.softmax(wynik, dim=1) # Zamiana na %
        
    print(f"Prawdopodobieństwa:")
    for i, prob in enumerate(prawdopodobienstwa[0]):
        print(f"Cyfra {i}: {prob.item()*100:.2f}%")

sprawdz_cyfre("my_number.png")