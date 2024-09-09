from ast import List
import os
import logging
from dataclasses import dataclass,field
from typing import Optional,Dict,Type,List
import click
from numpy import block
from omegaconf import DictConfig, OmegaConf
from rich import print


import torch.nn as nn
from torch.hub import load_state_dict_from_url

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config/model")

logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__=['hrnet18', 'hrnet32', 'hrnet48']

class BasicBlock(nn.Module):
    def __init__(self,input_channels,output_channels,stride=1,downsample=None,momentum=0.1,relu_inplace=True):
        super(BasicBlock, self).__init__()
        self.conv1=nn.Conv2d(input_channels,output_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(output_channels,momentum=momentum)
        self.relu=nn.ReLU(inplace=relu_inplace)
        self.conv2=nn.Conv2d(output_channels,output_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(output_channels,momentum=momentum)
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
    def __init__(self,input_channels,output_channels,stride=1,downsample=None,momentum=0.1,relu_inplace=True):
        super(Bottleneck, self).__init__()
        self.conv1=nn.Conv2d(input_channels,output_channels,kernel_size=1,bias=False)
        self.bn1=nn.BatchNorm2d(output_channels,momentum=momentum)
        self.conv2=nn.Conv2d(output_channels,output_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(output_channels,momentum=momentum)
        self.conv3=nn.Conv2d(output_channels,output_channels*self.expansion,kernel_size=1,bias=False)
        self.bn3=nn.BatchNorm2d(output_channels*self.expansion,momentum=momentum)
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
    num_branches: int
    blocks: Type[nn.Module]
    num_blocks: List[int]
    input_channels: List[int]
    output_channels: List[int]
    fuse_method: str
    multi_scale_output: bool = True
    upsample_mode: str = 'bilinear'
    branches: nn.ModuleList
    fuse_layers: List[nn.ModuleList] = field(init=False)
    relu: nn.Module = field(init=False)

    def __post_init__(self):
        """
        Post-initialization to set up the internal components of the module.
        """
        self._check_branches(self.num_branches,self.num_blocks,self.input_channels,self.output_channels)
        self.branches = self._make_branches(self.num_branches,self.block,self.num_blocks)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)
        
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

@dataclass
class HighResolutionNet(nn.Module):
    def __init__(self,cfg:OmegaConf,**kwargs:dict):
        super(HighResolutionNet,self).__init__()
        self.cfg=cfg
        self.momentum=cfg.BASE.BN_MOMENTUM
        self.relu_inplace=cfg.BASE.RELU_INPLACE

        required_keys = ['LAYER1', 'STAGE2', 'STAGE3', 'STAGE4']
        if not all(key in cfg for key in required_keys):
            raise ValueError("Configuration must contain 'LAYER1', 'STAGE2', 'STAGE3', and 'STAGE4' fields.")
        self.stage1=self._make_layer(cfg.LAYER1)
        self.stage2=self._make_stage(cfg.STAGE2)
        self.stage3=self._make_stage(cfg.STAGE3)
        self.stage4=self._make_stage(cfg.STAGE4)
        
    def _make_layer(self,layer_cfg:OmegaConf):
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, momentum=self.momentum),
        )
        layer=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(64,momentum=self.momentum),
            nn.Conv2d(64,64,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(64,momentum=self.momentum),
            nn.ReLU(inplace=self.relu_inplace),
            Bottleneck(64,256,stride=1,momentum=self.momentum,relu_inplace=self.relu_inplace,downsample=downsample),
            Bottleneck(256,256,stride=1,momentum=self.momentum,relu_inplace=self.relu_inplace),
            Bottleneck(256,256,stride=1,momentum=self.momentum,relu_inplace=self.relu_inplace),
            Bottleneck(256,256,stride=1,momentum=self.momentum,relu_inplace=self.relu_inplace)
        )
        layer1=nn.Sequential(
            layer,
            nn.Conv2d(256,layer_cfg['TRANS_CHANNELS'][0],kernel_size=1,bias=False)
        )
        layer2=nn.Sequential(
            layer,
            nn.Conv2d(256,layer_cfg.TRANS_CHANNELS[1],kernel_size=1,bias=False)
        )
        return nn.ModuleList([layer1,layer2])
          
    def _make_stage(self,stage_cfg:OmegaConf):
        num_modules = stage_cfg['NUM_MODULES']
        block = stage_cfg['BLOCK']
        num_blocks = stage_cfg['NUM_BLOCKS']
        num_branches = stage_cfg['NUM_BRANCHES']
        input_channels = stage_cfg['INPUT_CHANNELS']
        output_channels = stage_cfg['OUTPUT_CHANNELS']
        fuse_method = stage_cfg['FUSE_METHOD']
        multi_scale_output=stage_cfg.get('MULTI_SCALE_OUTPUT', False)
        
        modules=[]
        for _ in range(num_modules-1):
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

def _hrnet(cfg_path: str, pretrained: bool, **kwargs: dict):
    # 加载配置文件
    try:
        cfg1=OmegaConf.load(os.path.join(CONFIG_DIR, "hrnet_base_config.yaml"))
        cfg2 = OmegaConf.load(cfg_path)
        cfg=OmegaConf.merge(cfg1,cfg2)
        logger.info(f"Configuration loaded from: {cfg_path}")
    except Exception as e:
        logger.error(f"Error loading configuration file '{cfg_path}': {e}")
        return None
    
    # 创建模型实例
    try:
        model = HighResolutionNet(cfg, **kwargs)
        logger.info(f"Model instance created: {model.__class__.__name__}")
    except Exception as e:
        logger.error(f"Error creating model instance: {e}")
        return None
    
    # 加载预训练权重
    if pretrained:
        try:
            model_url = cfg.model_url
            logger.info(f"Loading pretrained weights from: {model_url}")
            state_dict = load_state_dict_from_url(model_url)
            model.load_state_dict(state_dict, strict=False)
            logger.info("Pretrained weights loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading pretrained weights from '{model_url}': {e}")
            return None
    
    return model

def hrnet18(pretrained: bool = True, **kwargs: dict):
    # 构建配置文件路径
    config_path = os.path.join(CONFIG_DIR, "hrnet18_config.yaml")
    logger.info(f"Using configuration file: {config_path}")
    return _hrnet(config_path, pretrained, **kwargs)

def hrnet32(pretrained: bool = True, **kwargs: dict):
    # 构建配置文件路径
    config_path = os.path.join(CONFIG_DIR, "hrnet32_config.yaml")
    logger.info(f"Using configuration file: {config_path}")
    return _hrnet(config_path, pretrained, **kwargs)

def hrnet48(pretrained: bool = True, **kwargs: dict):
    # 构建配置文件路径
    config_path = os.path.join(CONFIG_DIR, "hrnet48_config.yaml")
    logger.info(f"Using configuration file: {config_path}")
    return _hrnet(config_path, pretrained, **kwargs)