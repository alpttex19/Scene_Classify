import torch
from torch import nn
from torchvision.models import vgg16

class conv2d_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv3d_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    

class VggNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=6, init_weight=True):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv2d_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv2d_block(ch_in=64, ch_out=128)
        self.Conv3 = conv3d_block(ch_in=128, ch_out=256)
        self.Conv4 = conv3d_block(ch_in=256, ch_out=512)
        self.Conv5 = conv3d_block(ch_in=512, ch_out=512)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(in_features=7*7*512,out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096,out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096,out_features=output_ch),
        )

         # 参数初始化
        if init_weight: # 如果进行参数初始化
            for m in self.modules():  # 对于模型的每一层
                if isinstance(m, nn.Conv2d): # 如果是卷积层
                    # 使用kaiming初始化
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    # 如果bias不为空，固定为0
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):# 如果是线性层
                    # 正态初始化
                    nn.init.normal_(m.weight, 0, 0.01)
                    # bias则固定为0
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x1 = self.Conv1(x)
        x = self.Maxpool(x1)
        x2 = self.Conv2(x)
        x = self.Maxpool(x2)
        x3 = self.Conv3(x)
        x = self.Maxpool(x3)
        x4 = self.Conv4(x)
        x = self.Maxpool(x4)
        x5 = self.Conv5(x)
        x = self.Maxpool(x5)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        result = self.classifier(x)
        return result
    

if __name__ == "__main__":
    model = VggNet()
    print(model)

    input_1 = torch.randn(2, 3, 150, 150)
    input_1 = torch.nn.functional.interpolate(input_1, size=(224, 224), mode='bilinear', align_corners=False)
    print(input_1.shape)
    outputs = model(input_1)
    print(outputs.shape)

    path = "/home/xinhuang/Desktop/arapat/Scene_Classify/output/model_baseon_pretrained.pth"
    model = VggNet()
    model.load_state_dict(torch.load(path))

