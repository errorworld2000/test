import paddle
from paddle import nn
from paddle.nn import initializer 
from paddle.nn import functional as F
import numpy as np
BN_MOMENTUM = 0.1
class PlaceHolder(nn.Layer):
    def __init__(self):
        super(PlaceHolder, self).__init__()
    
    def forward(self, inputs):
        return inputs

class HRNetConv3_3(nn.layer):
    def __init__(self,input_channels,output_channels,stride=1,padding=0):
        super(HRNetConv3_3,self).__init__()
        self.conv=nn.Conv2D(input_channels,output_channels,kernel_size=3,stride=stride,padding=padding,bias_attr=False)
        self.bn=nn.BatchNorm2D(output_channels,momentum=BN_MOMENTUM)
        self.relu=nn.ReLU()
    def forward(self,inputs):
        x=self.conv(inputs)
        x=self.bn(x)
        x=self.relu(x)
        return x

class HRNetStem(nn.layer):
    def __init__()