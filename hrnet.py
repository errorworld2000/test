import pytorch.nn as nn
class HRNet(nn.Module):
    def __init__(self):
        super(HRNet, self).__init__()
        self.layer1=nn.Sequential(
            Bottleneck(64,64,downsample=downsample),
            Bottleneck(256,64),
            Bottleneck(256,64),
            Bottleneck(256,64),
        )
        
        
        
downsample=nn.Sequential(
    nn.Conv2d(64,256,kernel_size=1,stride=1,bias=False),
    nn.batchNorm2d(256,momentum=BN_MOMENTUM)
)


class BasicBlock(nn.Module):
    def __init__(self,input_channels,output_channels,stride=1,downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1=nn.Conv2d(input_channels,output_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(output_channels,momentum=BN_MOMENTUM)
        self.relu=nn.ReLU(inplace=relu_inplace)
        self.conv2=nn.Conv2d(output_channels,output_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(output_channels,momentum=BN_MOMENTUM)
        self.downsample=downsample
        self.stride=stride
    
    def forward(self,x):
        residual=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        if self.downsample is not None:
            residual=self.downsample(x)
        out+=residual
        out.relu_(out)
        return out

class Bottleneck(nn.Module):
    expansion =4
    def __init__(self,input_channels,output_channels,stride=1,downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1=nn.Conv2d(input_channels,output_channels,kernel_size=1,bias=False)
        self.bn1=nn.BatchNorm2d(output_channels,momentum=BN_MOMENTUM)
        self.conv2=nn.Conv2d(output_channels,output_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(output_channels,momentum=BN_MOMENTUM)
        self.conv3=nn.Conv2d(output_channels,output_channels*self.expansion,kernel_size=1,bias=False)
        self.bn3=nn.BatchNorm2d(output_channels*self.expansion,momentum=BN_MOMENTUM)
        self.relu=nn.ReLU(inplace=relu_inplace)
        self.downsample=downsample
        self.stride=stride
        
    def forward(self,x):
        residual=x
        output=self.conv1(x)
        output=self.bn1(output)
        output=self.relu(output)
        output=self.conv2(output)
        output=self.bn2(output)
        output=self.relu(output)
        output=self.conv3(output)
        output=self.bn3(output)
        if self.downsample is not None:
            residual=self.downsample(x)
        output+=residual
        output=self.relu(output)
        return output