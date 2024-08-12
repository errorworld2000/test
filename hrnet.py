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

class HighResolutionModule(nn.Module):
    def __init__(self,num_branches,blocks,num_blocks,input_channels,output_channels,fuse_method,multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        
    def _check_branches(self,num_branches,blocks,num_blocks,num_inchannels,num_channels):
        if num_branches!=len(num_blocks):
            err_msg='NUM_BRANCHES({})<>NUM_BLOCKS({})'.format(num_branches,len(num_blocks))
            logger.error(err_msg)
            raise ValueError(err_msg)
        if num_branches!=len(num_channels):
            err_msg='NUM_BRANCHES({})<>NUM_CHANNELS({})'.format(num_branches,len(num_channels))
            logger.error(err_msg)
            raise ValueError(err_msg)
        if num_branches!=len(num_inchannels):
            err_msg='NUM_BRANCHES({})<>NUM_INCHANNELS({})'.format(num_branches,len(num_inchannels))
            logger.error(err_msg)
            raise ValueError(err_msg)
        
    def _make_one_branch(self,branch_index,block,num_blocks,stride=1):
        downsample=None
        if stride!=1 or self.input_channels[branch_index]!=self.output_channels[branch_index]:
            downsample=nn.Sequential(
                nn.Conv2d(self.input_channels[branch_index],self.output_channels[branch_index],kernel_size=1,stride=stride,bias=False),
                nn.batchNorm2d(self.output_channels[branch_index],momentum=BN_MOMENTUM)
            )
        layers=[]
        layers.append(block(self.input_channels[branch_index],self.output_channels[branch_index],stride,downsample))
        self.input_channels[branch_index]=self.output_channels[branch_index]
        for i in range(1,len(num_blocks)):
            layers.append(block(self.input_channels[branch_index],self.output_channels[branch_index]))
        return nn.Sequential(*layers)
        
    def _make_branches(self,num_branches,block,num_blocks):
        branches=[]
        for i in range(num_branches):
            branches.append(self._make_one_branch(i,block,num_blocks))
        return nn.ModuleList(branches)
        
    def _make_fuse_layers(self):
        if self.num_branches==1:
            return None
        fuse_layers=[]
        for i in range(self.num_branches if self.multi_scale_output else 1):
            fuse_layer=[]
            for j in range(self.num_branches):
                if j>i:
                    # TODO: upsample
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(self.output_channels[j],self.output_channels[i],1,1,0,bias=False),
                        nn.BatchNorm2d(self.output_channels[j],momentum=BN_MOMENTUM),
                        )
                    )
                if j==i:
                    fuse_layer.append(None)
                else:
                    conv3x3s=[]
                    for k in range(i-j):
                        if k==i-j-1:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(self.output_channels[i-1],self.output_channels[i],3,2,1,bias=False),
                                nn.BatchNorm2d(self.output_channels[i],momentum=BN_MOMENTUM),
                            ))
                        else:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(self.output_channels[k+j],self.output_channels[k+j+1],3,2,1,bias=False),
                                nn.BatchNorm2d(self.output_channels[k+j+1],momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=relu_inplace)
                            ))
                fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return  nn.ModuleList(fuse_layers)
    
    def forward(self,x):
        if self.num_branches==1:
            return [self.branches[0](x[0])]
        
        for i in range(self.num_branches):
            x[i]=self.branches[i](x[i])
        x_fuse=[]
        for i in range(len(self.fuse_layers)):
            y=x[0] if i==0 else self.fuse_layers[i][0](x[0])
            for j in range(1,self.num_branches):
                if i==j:
                    y+=x[j]
                elif j>i:
                    width=x[i].shape[-1]
                    height=x[i].shape[-2]
                    y+=F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height,width],
                        mode=self.upsample_mode,
                        align_corners=ALIGN_CORNERS
                    )
                else:
                    y+=self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse
        
