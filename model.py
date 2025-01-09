import torch
import torch.nn as nn
from torchvision import models

class HIFF(nn.Module):
    def __init__(self):
        super(HIFF, self).__init__()
        self.efficient_net = models.efficientnet_b0(pretrained=True)
        self.fc1 = nn.Linear(1000, 64)
        self.fc2 = nn.Linear(3, 64)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x_image, x_texture):
        x1 = self.efficient_net(x_image)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc1(x1)

        x_texture = x_texture.to(x1.dtype)
        x2 = self.fc2(x_texture)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc3(x)

        return x
