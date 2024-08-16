import torch.nn as nn
import logging
from torch.hub import load_state_dict_from_url
from yacs.config import CfgNode as CN
import os
# import torch.utils.model_zoo as model_zoo

logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__=['hrnet18', 'hrnet32', 'hrnet48']
BN_MOMENTUM=0.1
ALIGN_CORNERS=True
RELU_INPLACE=True

class HRNet(nn.Module):
    def __init__(self):
        super(HRNet, self).__init__()
        
      
downsample=nn.Sequential(
    nn.Conv2d(64,256,kernel_size=1,stride=1,bias=False),
    nn.BatchNorm2d(256,momentum=BN_MOMENTUM)
)

class BasicBlock(nn.Module):
    def __init__(self,input_channels,output_channels,stride=1,downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1=nn.Conv2d(input_channels,output_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(output_channels,momentum=BN_MOMENTUM)
        self.relu=nn.ReLU(inplace=RELU_INPLACE)
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
        self.relu=nn.ReLU(inplace=RELU_INPLACE)
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
    """_summary_
    Constructs a HighResolutionModule which is typically part of a high-resolution network.
    Args:
        nn.Module (class): pytorch base class which use to construct the module.
    """    
    def __init__(self,num_branches,blocks,num_blocks,input_channels,output_channels,fuse_method,multi_scale_output=True, upsample_mode='bilinear'):
        """_summary_
        初始化函数，用于构建HighResolutionModule模块。
        Args:
            num_branches (int): 分支数量，用于确定模块内部的分支结构。
            blocks (class): 每个分支中的块类型，用于构建每个分支的具体结构。
            num_blocks (list of int): 一个列表，表示每个分支中的块数量。
            input_channels (list of int): 一个列表，表示每个分支的输入通道数。
            output_channels (list of int): 一个列表，表示每个分支的输出通道数。
            fuse_method (str): 融合方法，用于确定如何融合不同分支的信息。
            multi_scale_output (bool, optional): 是否使用多尺度输出，默认为True。
            upsample_mode (str, optional): 上采样模式，默认为'bilinear'（双线性插值）。
        """        
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches=num_branches,num_blocks=num_blocks,input_channels=input_channels,output_channels=output_channels)
        self.num_branches=num_branches
        self.input_channels=input_channels
        self.output_channels=output_channels
        self.fuse_method=fuse_method
        self.multi_scale_output=multi_scale_output
        self.upsample_mode=upsample_mode
        self.branches=self._make_branches(num_branches,blocks,num_blocks)
        self.fuse_layers=self._make_fuse_layers()
        self.relu=nn.ReLU(inplace=True)
        
    def _check_branches(self,num_branches,num_blocks,input_channels,output_channels):
        if num_branches!=len(num_blocks):
            err_msg='NUM_BRANCHES({})<>NUM_BLOCKS({})'.format(num_branches,len(num_blocks))
            logger.error(err_msg)
            raise ValueError(err_msg)
        if num_branches!=len(input_channels):
            err_msg='NUM_BRANCHES({})<>LEN_INPUT_CHANNELS({})'.format(num_branches,len(input_channels))
            logger.error(err_msg)
            raise ValueError(err_msg)
        # if num_branches!=len(output_channels):
        #     err_msg='NUM_BRANCHES({})<>LEN_OUTPUT_CHANNELS({})'.format(num_branches,len(output_channels))
        #     logger.error(err_msg)
        #     raise ValueError(err_msg)
        
    def _make_one_branch(self,branch_index,block,num_blocks,stride=1):
        downsample=None
        if stride!=1 or self.input_channels[branch_index]!=self.output_channels[branch_index]:
            downsample=nn.Sequential(
                nn.Conv2d(self.input_channels[branch_index],self.output_channels[branch_index],kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(self.output_channels[branch_index],momentum=BN_MOMENTUM)
            )
        layers=[]
        layers.append(block(self.input_channels[branch_index],self.output_channels[branch_index],stride,downsample))
        # self.input_channels[branch_index]=self.output_channels[branch_index]
        for i in range(1,len(num_blocks)):
            layers.append(block(self.output_channels[branch_index],self.output_channels[branch_index]))
        return nn.Sequential(*layers)
        
    def _make_branches(self,num_branches,block,num_blocks):
        branches=[]
        for i in range(num_branches):
            branches.append(self._make_one_branch(i,block,num_blocks))
        return nn.ModuleList(branches)
        
    def _make_fuse_layers(self):
        fuse_layers=[]
        if self.num_branches==1:
            return fuse_layers
        for i in range(self.num_branches if self.multi_scale_output else 1):
            fuse_layer=[]
            for j in range(self.num_branches):
                if j>i:
                    # TODO: upsample place in forward
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(self.output_channels[j],self.output_channels[i],1,1,0,bias=False),
                        nn.BatchNorm2d(self.output_channels[j],momentum=BN_MOMENTUM),
                        nn.Upsample(scale_factor=2**(j-i),mode=self.upsample_mode,align_corners=ALIGN_CORNERS)
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
                                nn.ReLU(inplace=RELU_INPLACE)
                            ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return fuse_layers
    
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
                else:
                    y+=self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse

class HighResolutionNet(nn.Module):
    def __init__(self,config,**kwargs):
        global ALIGN_CORNERS
        # extra=config.MODEL.EXTRA
        super(HighResolutionNet,self).__init__()
        
        ALIGN_CORNERS=config.MODEL.ALIGN_CORNERS
        
        self.layer1_cfg = config.LAYER1
        self.stage1=self._make_layer(self.layer1_cfg)
        self.stage2_cfg = config.STAGE2
        self.stage2=self._make_stage(self.stage2_cfg)
        self.stage3_cfg = config.STAGE3
        self.stage3=self._make_stage(self.stage3_cfg)
        self.stage4_cfg = config.STAGE4
        self.stage4=self._make_stage(self.stage4_cfg,False)
        
    def _make_layer(self,layer_config):
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        )
        layer=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(64,momentum=BN_MOMENTUM),
            nn.Conv2d(64,64,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(64,momentum=BN_MOMENTUM),
            nn.ReLU(inplace=RELU_INPLACE),
            Bottleneck(64,256,stride=1,downsample=downsample),
            Bottleneck(256,256,stride=1),
            Bottleneck(256,256,stride=1),
            Bottleneck(256,256,stride=1)
        )
        layer1=nn.Sequential(
            layer,
            nn.Conv2d(256,layer_config['TRANS_CHANNELS'][0],kernel_size=1,bias=False)
        )
        layer2=nn.Sequential(
            layer,
            nn.Conv2d(256,layer_config['TRANS_CHANNELS'][1],kernel_size=1,bias=False)
        )
        return nn.ModuleList([layer1,layer2])
          
    def _make_stage(self,stage_config,multi_scale_output=True):
        num_modules = stage_config['NUM_MODULES']
        block = stage_config['BLOCK']
        num_blocks = stage_config['NUM_BLOCKS']
        num_branches = stage_config['NUM_BRANCHES']
        input_channels = stage_config['INPUT_CHANNELS']
        output_channels = stage_config['OUTPUT_CHANNELS']
        fuse_method = stage_config['FUSE_METHOD']
        
        modules=[]
        for i in range(num_modules-1):
            modules.append(
                HighResolutionModule(num_branches=num_branches,blocks=block,num_blocks=num_blocks,input_channels=input_channels,output_channels=input_channels,fuse_method=fuse_method,multi_scale_output=False)
            )
        modules.append(
                HighResolutionModule(num_branches=num_branches,blocks=block,num_blocks=num_blocks,input_channels=input_channels,output_channels=output_channels,fuse_method=fuse_method,multi_scale_output=multi_scale_output)
            )
        return nn.Sequential(*modules)
            
    def forward(self,x):
        x=self.stage1(x)
        x=self.stage2(x)
        x=self.stage3(x)
        x=self.stage4(x)
        return x

model_urls = {
    'hrnet18_imagenet': 'https://opr0mq.dm.files.1drv.com/y4mIoWpP2n-LUohHHANpC0jrOixm1FZgO2OsUtP2DwIozH5RsoYVyv_De5wDgR6XuQmirMV3C0AljLeB-zQXevfLlnQpcNeJlT9Q8LwNYDwh3TsECkMTWXCUn3vDGJWpCxQcQWKONr5VQWO1hLEKPeJbbSZ6tgbWwJHgHF7592HY7ilmGe39o5BhHz7P9QqMYLBts6V7QGoaKrr0PL3wvvR4w',
    'hrnet32_imagenet': 'https://opr74a.dm.files.1drv.com/y4mKOuRSNGQQlp6wm_a9bF-UEQwp6a10xFCLhm4bqjDu6aSNW9yhDRM7qyx0vK0WTh42gEaniUVm3h7pg0H-W0yJff5qQtoAX7Zze4vOsqjoIthp-FW3nlfMD0-gcJi8IiVrMWqVOw2N3MbCud6uQQrTaEAvAdNjtjMpym1JghN-F060rSQKmgtq5R-wJe185IyW4-_c5_ItbhYpCyLxdqdEQ',
    'hrnet48_imagenet': 'https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ',
    'hrnet48_cityscapes': 'https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ',
    'hrnet48_ocr_cityscapes': 'https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ'
}


def _hrnet(arch,pretrained, progress,**kwargs):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(script_dir, "../config/hrnet_config.yaml"))
    cfg=CN()
    cfg.merge_from_file(config_path)
    model = HighResolutionNet(cfg.arch,**kwargs)
    if pretrained:
        model_url=model_urls[arch]
        # state_dict = model_zoo.load_url(model_url,progress=progress)
        state_dict = load_state_dict_from_url(model_url,progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model
    
def hrnet18(pretrained=True,process=True,**kwargs):
    return _hrnet('hrnet18',pretrained,process,**kwargs)

def hrnet32(pretrained=True,process=True,**kwargs):
    return _hrnet('hrnet32',pretrained,process,**kwargs)

def hrnet48(pretrained=True,process=True,**kwargs):
    return _hrnet('hrnet48',pretrained,process,**kwargs)
