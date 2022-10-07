import torch
from torch import nn
# from .. import config as cfg
from torchsummary import summary

class Cnn_Test(nn.Module):
    def __init__(self):
        super().__init__()
        # input shape = (1, 256, 256)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2) # output_shape = (16, 628, 128)
        self.relu1 = nn.ReLU() # activation

        self.maxpool1 = nn.MaxPool2d(kernel_size=2) # output_shape = (16, 214, 64)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2) #output_shape = (32,314,64)
        self.relu2 = nn.ReLU() # activation

        self.maxpool2 = nn.MaxPool2d(kernel_size=2) # output_shape = (32, 107, 32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2) #output_shape = (16,157,32)
        self.relu3 = nn.ReLU() # activation

        self.maxpool3 = nn.MaxPool2d(kernel_size=2) # output_shape = (16, 54, 16)

        self.conv4 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, stride=1, padding=2) #output_shape = (8,78,16)
        self.relu4 = nn.ReLU() # activation

        self.maxpool4 = nn.MaxPool2d(kernel_size=2) # output_shape = (8, 39, 8)
        self.flatten =nn.Flatten()
        self.fc1 = nn.Linear(8 * 39 * 8, 512)
        self.relu5 = nn.ReLU() # activation
        self.fc2 = nn.Linear(512, 3)
        self.dropout = nn.Dropout(p=0.2)
        self.output = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu1(x)
        x = self.maxpool1(x)# Max pool 1
        x = self.conv2(x) # Convolution 2
        x = self.relu2(x)
        x = self.maxpool2(x) # Max pool 2
        x = self.conv3(x) # Convolution 3
        x = self.relu3(x)
        x = self.maxpool3(x) # Max pool 3
        x = self.conv4(x) # Convolution 4
        x = self.relu4(x)
        x = self.maxpool4(x) # Max pool 4
        x = self.flatten(x)
        x = self.fc1(x) # Linear function (readout)
        x = self.dropout(x)
        x = self.fc2(x)
        # predictions = self.output(x)
        return x

if __name__ == '__main__':
    model = Cnn_Test()
    summary(model.cuda(), (1, 500, 128))
