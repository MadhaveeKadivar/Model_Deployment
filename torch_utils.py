import io
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.transforms as transforms 
from PIL import Image

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()

PATH = "mnist.pth"
model.load_state_dict(torch.load(PATH))
model.eval()

# image -> tensor
# image -> tensor
def transform_image(image_bytes):
    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale if needed
    transforms.Resize((28, 28)),  # Resize the image to the expected size
    transforms.ToTensor(),  # Convert to a PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize as done during training
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

# predict
def get_prediction(image_tensor):
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs.data, 1)
    return predicted
