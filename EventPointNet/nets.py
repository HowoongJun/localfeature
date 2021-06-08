import torch

class CDetectorNet(torch.nn.Module):
    def __init__(self):
        super(CDetectorNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1_1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.convKp1 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convKp2 = torch.nn.Conv2d(256, 65, kernel_size=3, stride=1, padding=1)

        self.convDsc1 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convDsc2 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool(x)
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool(x)
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.pool(x)
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))

        kpt = self.relu(self.convKp1(x))
        kpt = self.convKp2(kpt)

        desc = self.relu(self.convDsc1(x))
        desc = self.convDsc2(desc)
        descNorm = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(descNorm, 1))
        return kpt, desc

        
class CDescriptorNet(torch.nn.Module):
    def __init__(self):
        super(CDescriptorNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
                
        self.conv1_1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = torch.nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        
        desc = self.relu(self.conv4_1(x))
        descNorm = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(descNorm, 1))

        return desc