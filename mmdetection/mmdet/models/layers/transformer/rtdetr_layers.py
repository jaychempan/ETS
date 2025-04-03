# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule, ModuleList
from torch import Tensor, nn

from mmdet.models.layers.transformer.detr_layers import DetrTransformerEncoder
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .dino_layers import DinoTransformerDecoder
from .utils import MLP, inverse_sigmoid
from mmdet.structures import SampleList
from typing import Tuple, Union
from .utils import inverse_sigmoid
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmcv.cnn import Linear
import torch.nn.functional as F
from mmengine import ConfigDict


class RepVGGBlock(nn.Module):
    """RepVGGBlock is a basic rep-style block, including training and deploy
    status This code is based on
    https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple): Stride of the convolution. Default: 1
        padding (int, tuple): Padding added to all four sides of
            the input. Default: 1
        dilation (int or tuple): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        padding_mode (string, optional): Default: 'zeros'
        use_se (bool): Whether to use se. Default: False
        use_alpha (bool): Whether to use `alpha` parameter at 1x1 conv.
            In PPYOLOE+ model backbone, `use_alpha` will be set to True.
            Default: False.
        use_bn_first (bool): Whether to use bn layer before conv.
            In YOLOv6 and YOLOv7, this will be set to True.
            In PPYOLOE, this will be set to False.
            Default: True.
        deploy (bool): Whether in deploy mode. Default: False
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 1,
                 dilation: Union[int, Tuple[int]] = 1,
                 groups: Optional[int] = 1,
                 padding_mode: Optional[str] = 'zeros',
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 use_se: bool = False,
                 use_alpha: bool = False,
                 use_bn_first=True,
                 deploy: bool = False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = MODELS.build(act_cfg)

        if use_se:
            raise NotImplementedError('se block not supported yet')
        else:
            self.se = nn.Identity()

        if use_alpha:
            alpha = torch.ones([
                1,
            ], dtype=torch.float32, requires_grad=True)
            self.alpha = nn.Parameter(alpha, requires_grad=True)
        else:
            self.alpha = None

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode)

        else:
            if use_bn_first and (out_channels == in_channels) and stride == 1:
                self.rbr_identity = build_norm_layer(
                    norm_cfg, num_features=in_channels)[1]
            else:
                self.rbr_identity = None

            self.rbr_dense = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.rbr_1x1 = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward process.
        Args:
            inputs (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        if self.alpha:
            return self.nonlinearity(
                self.se(
                    self.rbr_dense(inputs) +
                    self.alpha * self.rbr_1x1(inputs) + id_out))
        else:
            return self.nonlinearity(
                self.se(
                    self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        """Derives the equivalent kernel and bias in a differentiable way.

        Returns:
            tuple: Equivalent kernel and bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        if self.alpha:
            return kernel3x3 + self.alpha * self._pad_1x1_to_3x3_tensor(
                kernel1x1) + kernelid, bias3x3 + self.alpha * bias1x1 + biasid
        else:
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(
                kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pad 1x1 tensor to 3x3.
        Args:
            kernel1x1 (Tensor): The input 1x1 kernel need to be padded.

        Returns:
            Tensor: 3x3 kernel after padded.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: nn.Module) -> Tuple[np.ndarray, Tensor]:
        """Derives the equivalent kernel and bias of a specific branch layer.

        Args:
            branch (nn.Module): The layer that needs to be equivalently
                transformed, which can be nn.Sequential or nn.Batchnorm2d

        Returns:
            tuple: Equivalent kernel and bias
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvModule):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, (nn.SyncBatchNorm, nn.BatchNorm2d))
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """Switch to deploy mode."""
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class CSPLayer(BaseModule):
    """CSPLayer from RTDETR.

    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 1.0.
        num_blocks (int): Number of blocks. Defaults to 3.
        conv_cfg (:obj:`ConfigDict`, optional): Config dict for convolution
            layer. Defaults to None, which means using conv2d.
        norm_cfg (:obj:`ConfigDict`, optional): Config dict for normalization
            layer. Defaults to dict(type='BN', requires_grad=True)
        act_cfg (:obj:`ConfigDict`, optional): Config dict for activation
            layer. Defaults to dict(type='SiLU', inplace=True)
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 1.0,
                 num_blocks: int = 3,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.short_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks = nn.Sequential(*[
            RepVGGBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                use_bn_first=False) for _ in range(num_blocks)
        ])
        if mid_channels != out_channels:
            self.final_conv = ConvModule(
                mid_channels,
                out_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.final_conv = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)
        return self.final_conv(x_main + x_short)


class MLP_WO_ACT(BaseModule):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = ModuleList(
            Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x) if i < self.num_layers - 1 else layer(x)
        return x


class TextGuideVTFusion(BaseModule):
    def __init__(self,
                 text_channels: int=512,
                 embed_channels: int=384,
                 num_heads: int = 8,
                 init_cfg: OptMultiConfig = None,
                 norm_cfg: OptConfigType = dict(type='SyncBN', requires_grad=True)
                 ):
        super().__init__(init_cfg=init_cfg)
        self.num_heads = num_heads
        self.text_channels = text_channels
        self.embed_channels = embed_channels
        
        self.hidden_channel = self.text_channels
        self.head_channels = self.hidden_channel // num_heads
        
        # self.img_proj = Linear(self.embed_channels, self.text_channels)
        # self.text_fc = Linear(self.text_channels, self.embed_channels, bias=False)
        self.img_proj = nn.Sequential(
                    ConvModule(in_channels=self.embed_channels,
                               out_channels=self.embed_channels*2,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=norm_cfg,
                               act_cfg=None),
                    ConvModule(in_channels=self.embed_channels*2,
                               out_channels=self.text_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=norm_cfg,
                               act_cfg=None),
                    nn.Conv2d(in_channels=self.text_channels,
                              out_channels=self.text_channels,
                              kernel_size=1))
        
        self.query = nn.Sequential(nn.LayerNorm(self.hidden_channel),
                                   Linear(self.hidden_channel, self.hidden_channel))
        self.key = nn.Sequential(nn.LayerNorm(self.hidden_channel),
                                 Linear(self.hidden_channel, self.hidden_channel))
        self.value = nn.Sequential(nn.LayerNorm(self.hidden_channel),
                                   Linear(self.hidden_channel, self.hidden_channel))
        self.proj = Linear(self.hidden_channel, self.embed_channels)
    
    def forward(self, img_feat: Tensor, txt_feat: Tensor) -> Tensor:
        B, _, H, W = img_feat.shape
        # txt_feat = self.text_fc(txt_feat)
        
        img_feat_tmp = self.img_proj(img_feat)
        img_feat_tmp = img_feat_tmp.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        
        
        q = self.query(img_feat_tmp)
        k = self.key(txt_feat)
        v = self.value(txt_feat)
        
        q = q.reshape(B, -1, self.num_heads, self.head_channels)
        k = k.reshape(B, -1, self.num_heads, self.head_channels)
        v = v.reshape(B, -1, self.num_heads, self.head_channels)
        
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        attn_weight = torch.matmul(q, k)
        B_F_weight = attn_weight
        
        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = F.softmax(attn_weight, dim=-1)
        
        B_F_weight = B_F_weight.max(dim=-1)[0]
        B_F_weight = B_F_weight / (self.head_channels**0.5)
        B_F_weight = B_F_weight.sigmoid()
        
        v = v.permute(0, 2, 1, 3)
        aug_v = torch.matmul(attn_weight, v)
        aug_v = aug_v.permute(0, 2, 1, 3).reshape(B, -1, self.hidden_channel)

        aug_v = self.proj(aug_v)
        
        aug_v = aug_v.permute(0, 2, 1).reshape(B, self.num_heads, self.embed_channels//self.num_heads, H, W)
        aug_v = aug_v * B_F_weight.reshape(B, self.num_heads, H, W).unsqueeze(2)
        aug_v = aug_v.reshape(B, -1, H, W)
        return img_feat + aug_v
    

class TextGuideVTFusion_Decoder(BaseModule):
    def __init__(self,
                 text_channels: int=512,
                 embed_channels: int=256,
                 num_heads: int = 8,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.num_heads = num_heads
        self.text_channels = text_channels
        self.embed_channels = embed_channels

        self.hidden_channel = text_channels
        self.head_channels = self.hidden_channel // num_heads
        
        # self.img_proj = Linear(self.embed_channels, self.text_channels)
        # self.text_fc = Linear(self.text_channels, self.embed_channels, bias=False)
        self.img_proj = MLP_WO_ACT(self.embed_channels, self.embed_channels*2, self.text_channels, 2)
        
        self.query = nn.Sequential(nn.LayerNorm(self.hidden_channel),
                                   Linear(self.hidden_channel, self.hidden_channel))
        self.key = nn.Sequential(nn.LayerNorm(self.hidden_channel),
                                 Linear(self.hidden_channel, self.hidden_channel))
        self.value = nn.Sequential(nn.LayerNorm(self.hidden_channel),
                                   Linear(self.hidden_channel, self.hidden_channel))
        self.proj = Linear(self.hidden_channel, self.embed_channels)
    
    def forward(self, img_feat: Tensor, txt_feat: Tensor) -> Tensor:
        B, num_k, _, = img_feat.shape
        img_feat_tmp = self.img_proj(img_feat)
        
        q = self.query(img_feat_tmp)
        k = self.key(txt_feat)
        v = self.value(txt_feat)
        
        q = q.reshape(B, -1, self.num_heads, self.head_channels)
        k = k.reshape(B, -1, self.num_heads, self.head_channels)
        v = v.reshape(B, -1, self.num_heads, self.head_channels)
        
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        attn_weight = torch.matmul(q, k)
        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = F.softmax(attn_weight, dim=-1)
        
        v = v.permute(0, 2, 1, 3)
        aug_v = torch.matmul(attn_weight, v)
        aug_v = aug_v.permute(0, 2, 1, 3).reshape(B, -1, self.hidden_channel)
        
        aug_v = self.proj(aug_v)
        return img_feat + aug_v

    
@MODELS.register_module()
class Vision_aug_Text(BaseModule):
    def __init__(self,
                 text_channels: int=512,
                 embed_channels: int=256,
                 num_heads: int = 8,
                 init_cfg: OptMultiConfig = None,
                 norm_cfg: OptConfigType = dict(type='SyncBN', requires_grad=True)
                 ):
        super().__init__(init_cfg=init_cfg)
        self.num_heads = num_heads
        self.text_channels = text_channels
        self.embed_channels = embed_channels

        self.hidden_channel = self.text_channels
        self.head_channels = self.hidden_channel // num_heads
        
        # self.img_proj = Linear(self.embed_channels, self.text_channels)
        # self.text_fc = Linear(self.text_channels, self.embed_channels, bias=False)
        self.img_proj = nn.Sequential(
                    ConvModule(in_channels=self.embed_channels,
                               out_channels=self.embed_channels*2,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=norm_cfg,
                               act_cfg=None),
                    ConvModule(in_channels=self.embed_channels*2,
                               out_channels=self.text_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=norm_cfg,
                               act_cfg=None),
                    nn.Conv2d(in_channels=self.text_channels,
                              out_channels=self.text_channels,
                              kernel_size=1))
        
        self.query = nn.Sequential(nn.LayerNorm(self.hidden_channel),
                                   Linear(self.hidden_channel, self.hidden_channel))
        self.key = nn.Sequential(nn.LayerNorm(self.hidden_channel),
                                 Linear(self.hidden_channel, self.hidden_channel))
        self.value = nn.Sequential(nn.LayerNorm(self.hidden_channel),
                                   Linear(self.hidden_channel, self.hidden_channel))
        self.proj = Linear(self.hidden_channel, self.text_channels)
    
    def forward(self, img_feat: Tensor, txt_feat: Tensor) -> Tensor:
        B, _, H, W = img_feat.shape
        
        img_feat_tmp = self.img_proj(img_feat)
        img_feat_tmp = img_feat_tmp.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        
        
        q = self.query(txt_feat)
        k = self.key(img_feat_tmp)
        v = self.value(img_feat_tmp)
        
        q = q.reshape(B, -1, self.num_heads, self.head_channels)
        k = k.reshape(B, -1, self.num_heads, self.head_channels)
        v = v.reshape(B, -1, self.num_heads, self.head_channels)
        
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        attn_weight = torch.matmul(q, k)
        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = F.softmax(attn_weight, dim=-1)

        v = v.permute(0, 2, 1, 3)
        aug_v = torch.matmul(attn_weight, v)
        aug_v = aug_v.permute(0, 2, 1, 3).reshape(B, -1, self.hidden_channel)
        
        aug_text_feat = self.proj(aug_v)
        return txt_feat + aug_text_feat

@MODELS.register_module()
class RTDETRFPN(BaseModule):
    """FPN of RTDETR.

    Args:
        in_channels (List[int], optional): The input channels of the
            feature maps. Defaults to [256, 256, 256].
        out_channels (int, optional): The output dimension of the MLP.
            Defaults to 256.
        expansion (float, optional): The expansion of the CSPLayer.
            Defaults to 1.0.
        depth_mult (float, optional): The depth multiplier of the CSPLayer.
            Defaults to 1.0.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (:obj:`ConfigDict` or dict, optional): The config dict for
            normalization layers. Defaults to dict(type='BN').
        act_cfg (:obj:`ConfigDict` or dict, optional): The config dict for
            activation layers. Defaults to dict(type='SiLU', inplace=True).
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization config dict.
    """

    def __init__(
        self,
        in_channels: List[int] = [256, 256, 256],
        out_channels: int = 256,
        expansion: float = 1.0,
        depth_mult: float = 1.0,
        upsample_cfg: ConfigType = dict(scale_factor=2, mode='nearest'),
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='SiLU', inplace=True),
        init_cfg: OptMultiConfig = dict(
            type='Kaiming',
            layer='Conv2d',
            a=math.sqrt(5),
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        num_csp_blocks = round(3 * depth_mult)

        # top-down fpn
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    expand_ratio=expansion,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    expand_ratio=expansion,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None))
            

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: FPN features.
        """
        assert len(inputs) == len(self.in_channels)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_high = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_high)
            inner_outs[0] = feat_high

            upsample_feat = self.upsample(feat_high)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])
        
        return tuple(outs)


@MODELS.register_module()
class RTDETRHybridEncoder(BaseModule):
    """HybridEncoder of RTDETR.

    Args:
        layer_cfg (:obj:`ConfigDict` or dict): The config dict for the encode
            layer.
        in_channels (List[int], optional): The input channels of the
            feature maps. Defaults to [256, 256, 256].
        use_encoder_idx (List[int], optional): The indices of the encoder
            layers to use. Defaults to [2].
        num_encoder_layers (int, optional): The number of encoder layers.
            Defaults to 1.
        pe_temperature (float, optional): The temperature of the positional
            encoding. Defaults to 10000.
        encode_before_fpn (bool, optional): Encoding the features before FPN
            layer. Defaults to True.
        fpn_cfg (:obj:`ConfigDict` or dict): The config dict for the FPN layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 layer_cfg: ConfigType,
                 in_channels: List[int] = [256, 256, 256],
                 use_encoder_idx: List[int] = [2],
                 num_encoder_layers: int = 1,
                 pe_temperature: float = 10000.0,
                 encode_before_fpn: bool = True,
                 fpn_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.encode_before_fpn = encode_before_fpn

        # fpn layer
        self.fpn = MODELS.build(fpn_cfg) \
            if fpn_cfg is not None else nn.Identity()

        # encoder transformer
        self.transformer_blocks = nn.ModuleList([
            DetrTransformerEncoder(num_encoder_layers, layer_cfg)
            for _ in range(len(use_encoder_idx))
        ])

    @staticmethod
    def build_2d_sincos_position_embedding(
        w: int,
        h: int,
        embed_dim: int = 256,
        temperature: float = 10000.,
        device=None,
    ) -> Tensor:
        grid_w = torch.arange(w, dtype=torch.float32, device=device)
        grid_h = torch.arange(h, dtype=torch.float32, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, ('Embed dimension must be divisible by 4 '
                                    'for 2D sin-cos position embedding')
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32, device=device)
        omega = temperature**(omega / -pos_dim)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        pos_embd = [
            torch.sin(out_w),
            torch.cos(out_w),
            torch.sin(out_h),
            torch.cos(out_h)
        ]
        return torch.cat(pos_embd, axis=1)[None, :, :]

    def encode_forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: encoded features.
        """
        assert len(inputs) == len(self.in_channels)
        outs = list(inputs)

        # encoder
        for i, enc_ind in enumerate(self.use_encoder_idx):
            h, w = outs[enc_ind].shape[2:]
            # flatten [B, C, H, W] to [B, HxW, C]
            src_flatten = outs[enc_ind].flatten(2).permute(0, 2,
                                                           1).contiguous()
            pos_embed = self.build_2d_sincos_position_embedding(
                w,
                h,
                embed_dim=self.in_channels[enc_ind],
                temperature=self.pe_temperature,
                device=src_flatten.device)
            memory = self.transformer_blocks[i](
                src_flatten, query_pos=pos_embed, key_padding_mask=None)
            outs[enc_ind] = memory.permute(0, 2, 1).contiguous().reshape(
                -1, self.in_channels[enc_ind], h, w)

        return tuple(outs)

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        if self.encode_before_fpn:
            return self.fpn(self.encode_forward(inputs))
        else:
            return self.encode_forward(self.fpn(inputs))


class RTDETRTransformerDecoder(DinoTransformerDecoder):
    """Transformer decoder of RT-DETR."""
    def __init__(self,
                 txt_dims: int=1024,
                 **kwargs) -> None:
        self.txt_dims = txt_dims
        super().__init__(**kwargs)
        
    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        super()._init_layers()
        self.ref_point_head = MLP(4, self.embed_dims * 2, self.embed_dims, 2)
        self.norm = nn.Identity()  # without norm
        self.Multi_level_TV_Fusion_Decoder = TextGuideVTFusion_Decoder(text_channels=self.txt_dims, embed_channels=self.embed_dims)

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                txt_feats: Tensor, multi_text_feats,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        low_level_text_embedding = multi_text_feats[0]
        medium_level_text_embedding = multi_text_feats[1]
        High_level_text_embedding = multi_text_feats[2]
        
        intermediate = []
        intermediate_reference_points = [reference_points]
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[:, :, None]
            query_pos = self.ref_point_head(reference_points)
            # query = self.TV_Fusion_Decoder(query, txt_feats)
            if lid in [0]:
                query = self.Multi_level_TV_Fusion_Decoder(query, High_level_text_embedding)
            elif lid in [1,2]:
                query = self.Multi_level_TV_Fusion_Decoder(query, medium_level_text_embedding)
            else:
                query = self.Multi_level_TV_Fusion_Decoder(query, low_level_text_embedding)
            # if lid in [0]:
            #     query = self.TV_Fusion_Decoder(query, txt_feats)
            # elif lid in [1]:
            #     query = self.Multi_level_TV_Fusion_Decoder(query, High_level_text_embedding)
            # elif lid in [2]:
            #     query = self.Multi_level_TV_Fusion_Decoder(query, medium_level_text_embedding)
            # else:
            #     query = self.Multi_level_TV_Fusion_Decoder(query, low_level_text_embedding)
            
            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)
            
            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.
            
            
            

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return query, reference_points


class Rtdetr_CdnQueryGenerator(BaseModule):
    """Implement query generator of the Contrastive denoising (CDN) proposed in
    `DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object
    Detection <https://arxiv.org/abs/2203.03605>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        num_classes (int): Number of object classes.
        embed_dims (int): The embedding dimensions of the generated queries.
        num_matching_queries (int): The queries number of the matching part.
            Used for generating dn_mask.
        label_noise_scale (float): The scale of label noise, defaults to 0.5.
        box_noise_scale (float): The scale of box noise, defaults to 1.0.
        group_cfg (:obj:`ConfigDict` or dict, optional): The config of the
            denoising queries grouping, includes `dynamic`, `num_dn_queries`,
            and `num_groups`. Two grouping strategies, 'static dn groups' and
            'dynamic dn groups', are supported. When `dynamic` is `False`,
            the `num_groups` should be set, and the number of denoising query
            groups will always be `num_groups`. When `dynamic` is `True`, the
            `num_dn_queries` should be set, and the group number will be
            dynamic to ensure that the denoising queries number will not exceed
            `num_dn_queries` to prevent large fluctuations of memory. Defaults
            to `None`.
    """

    def __init__(self,
                 num_classes: int,
                 embed_dims: int,
                 num_matching_queries: int,
                 txt_dims: int,
                 label_noise_scale: float = 0.5,
                 box_noise_scale: float = 1.0,
                 group_cfg: OptConfigType = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_matching_queries = num_matching_queries
        self.label_noise_scale = label_noise_scale
        self.box_noise_scale = box_noise_scale
        self.txt_dims = txt_dims
        
        # prepare grouping strategy
        group_cfg = {} if group_cfg is None else group_cfg
        self.dynamic_dn_groups = group_cfg.get('dynamic', True)
        if self.dynamic_dn_groups:
            if 'num_dn_queries' not in group_cfg:
                warnings.warn("'num_dn_queries' should be set when using "
                              'dynamic dn groups, use 100 as default.')
            self.num_dn_queries = group_cfg.get('num_dn_queries', 100)
            assert isinstance(self.num_dn_queries, int), \
                f'Expected the num_dn_queries to have type int, but got ' \
                f'{self.num_dn_queries}({type(self.num_dn_queries)}). '
        else:
            assert 'num_groups' in group_cfg, \
                'num_groups should be set when using static dn groups'
            self.num_groups = group_cfg['num_groups']
            assert isinstance(self.num_groups, int), \
                f'Expected the num_groups to have type int, but got ' \
                f'{self.num_groups}({type(self.num_groups)}). '

        # NOTE The original repo of DINO set the num_embeddings 92 for coco,
        # 91 (0~90) of which represents target classes and the 92 (91)
        # indicates `Unknown` class. However, the embedding of `unknown` class
        # is not used in the original DINO.
        # TODO: num_classes + 1 or num_classes ?
        # self.label_embedding = nn.Embedding(self.txt_dims, self.embed_dims)
        self.label_embedding = nn.Linear(self.txt_dims, embed_dims)

    def __call__(self, txt_feats: Tensor, batch_data_samples: SampleList) -> tuple:
        """Generate contrastive denoising (cdn) queries with ground truth.

        Descriptions of the Number Values in code and comments:
            - num_target_total: the total target number of the input batch
              samples.
            - max_num_target: the max target number of the input batch samples.
            - num_noisy_targets: the total targets number after adding noise,
              i.e., num_target_total * num_groups * 2.
            - num_denoising_queries: the length of the output batched queries,
              i.e., max_num_target * num_groups * 2.

        NOTE The format of input bboxes in batch_data_samples is unnormalized
        (x, y, x, y), and the output bbox queries are embedded by normalized
        (cx, cy, w, h) format bboxes going through inverse_sigmoid.

        Args:
            batch_data_samples (list[:obj:`DetDataSample`]): List of the batch
                data samples, each includes `gt_instance` which has attributes
                `bboxes` and `labels`. The `bboxes` has unnormalized coordinate
                format (x, y, x, y).

        Returns:
            tuple: The outputs of the dn query generator.

            - dn_label_query (Tensor): The output content queries for denoising
              part, has shape (bs, num_denoising_queries, dim), where
              `num_denoising_queries = max_num_target * num_groups * 2`.
            - dn_bbox_query (Tensor): The output reference bboxes as positions
              of queries for denoising part, which are embedded by normalized
              (cx, cy, w, h) format bboxes going through inverse_sigmoid, has
              shape (bs, num_denoising_queries, 4) with the last dimension
              arranged as (cx, cy, w, h).
            - attn_mask (Tensor): The attention mask to prevent information
              leakage from different denoising groups and matching parts,
              will be used as `self_attn_mask` of the `decoder`, has shape
              (num_queries_total, num_queries_total), where `num_queries_total`
              is the sum of `num_denoising_queries` and `num_matching_queries`.
            - dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.
        """
        # normalize bbox and collate ground truth (gt)
        gt_labels_list = []
        gt_bboxes_list = []
        for sample in batch_data_samples:
            img_h, img_w = sample.img_shape
            bboxes = sample.gt_instances.bboxes
            factor = bboxes.new_tensor([img_w, img_h, img_w,
                                        img_h]).unsqueeze(0)
            bboxes_normalized = bboxes / factor
            gt_bboxes_list.append(bboxes_normalized)
            gt_labels_list.append(sample.gt_instances.labels)
        gt_labels = torch.cat(gt_labels_list)  # (num_target_total, 4)
        gt_bboxes = torch.cat(gt_bboxes_list)

        num_target_list = [len(bboxes) for bboxes in gt_bboxes_list]
        max_num_target = max(num_target_list)
        num_groups = self.get_num_groups(max_num_target)

        # The `batch_idx` saves the batch index of the corresponding sample
        # for each target, has shape (num_target_total).
        batch_idx = torch.cat([
            torch.full_like(t.long(), i) for i, t in enumerate(gt_labels_list)
        ])
        
        dn_label_query = self.generate_dn_label_query(txt_feats, batch_idx, len(batch_data_samples), gt_labels, num_groups)
        dn_bbox_query = self.generate_dn_bbox_query(gt_bboxes, num_groups)
        
        dn_label_query, dn_bbox_query = self.collate_dn_queries(
            dn_label_query, dn_bbox_query, batch_idx, len(batch_data_samples),
            num_groups)

        attn_mask = self.generate_dn_mask(
            max_num_target, num_groups, device=dn_label_query.device)

        dn_meta = dict(
            num_denoising_queries=int(max_num_target * 2 * num_groups),
            num_denoising_groups=num_groups)

        return dn_label_query, dn_bbox_query, attn_mask, dn_meta

    def get_num_groups(self, max_num_target: int = None) -> int:
        """Calculate denoising query groups number.

        Two grouping strategies, 'static dn groups' and 'dynamic dn groups',
        are supported. When `self.dynamic_dn_groups` is `False`, the number
        of denoising query groups will always be `self.num_groups`. When
        `self.dynamic_dn_groups` is `True`, the group number will be dynamic,
        ensuring the denoising queries number will not exceed
        `self.num_dn_queries` to prevent large fluctuations of memory.

        NOTE The `num_group` is shared for different samples in a batch. When
        the target numbers in the samples varies, the denoising queries of the
        samples containing fewer targets are padded to the max length.

        Args:
            max_num_target (int, optional): The max target number of the batch
                samples. It will only be used when `self.dynamic_dn_groups` is
                `True`. Defaults to `None`.

        Returns:
            int: The denoising group number of the current batch.
        """
        if self.dynamic_dn_groups:
            assert max_num_target is not None, \
                'group_queries should be provided when using ' \
                'dynamic dn groups'
            if max_num_target == 0:
                num_groups = 1
            else:
                num_groups = self.num_dn_queries // max_num_target
        else:
            num_groups = self.num_groups
        if num_groups < 1:
            num_groups = 1
        return int(num_groups)

    def generate_dn_label_query(self, 
                                txt_feats: Tensor,
                                batch_idx: Tensor,
                                batch_size: int,
                                gt_labels: Tensor,
                                num_groups: int,
                                ) -> Tensor:
        """Generate noisy labels and their query embeddings.

        The strategy for generating noisy labels is: Randomly choose labels of
        `self.label_noise_scale * 0.5` proportion and override each of them
        with a random object category label.

        NOTE Not add noise to all labels. Besides, the `self.label_noise_scale
        * 0.5` arg is the ratio of the chosen positions, which is higher than
        the actual proportion of noisy labels, because the labels to override
        may be correct. And the gap becomes larger as the number of target
        categories decreases. The users should notice this and modify the scale
        arg or the corresponding logic according to specific dataset.

        Args:
            gt_labels (Tensor): The concatenated gt labels of all samples
                in the batch, has shape (num_target_total, ) where
                `num_target_total = sum(num_target_list)`.
            num_groups (int): The number of denoising query groups.

        Returns:
            Tensor: The query embeddings of noisy labels, has shape
            (num_noisy_targets, embed_dims), where `num_noisy_targets =
            num_target_total * num_groups * 2`.
        """
        assert self.label_noise_scale > 0
        gt_labels_expand = gt_labels.repeat(2 * num_groups,
                                            1).view(-1)  # Note `* 2`  # noqa
        p = torch.rand_like(gt_labels_expand.float())
        chosen_indice = torch.nonzero(p < (self.label_noise_scale * 0.5)).view(
            -1)  # Note `* 0.5`
        _, num_cls, _ = txt_feats.shape
        new_labels = torch.randint_like(chosen_indice, 0, num_cls)
        noisy_labels_expand = gt_labels_expand.scatter(0, chosen_indice,
                                                       new_labels)
        
        # txt_feats = self.label_embedding(txt_feats)
        num_target_list = [
            torch.sum(batch_idx == idx) for idx in range(batch_size)
        ]
        
        group_indices = torch.zeros_like(noisy_labels_expand)
        start = 0
        for _ in range(2*num_groups):
            for i, size in enumerate(num_target_list):
                group_indices[start:start + size] = i
                start += size
        # 根据组索引和标签值获取对应的特征
        dn_label_query = txt_feats[group_indices, noisy_labels_expand]
        
        dn_label_query = self.label_embedding(dn_label_query)
        return dn_label_query

    def generate_dn_bbox_query(self, gt_bboxes: Tensor,
                               num_groups: int) -> Tensor:
        """Generate noisy bboxes and their query embeddings.

        The strategy for generating noisy bboxes is as follow:

        .. code:: text

            +--------------------+
            |      negative      |
            |    +----------+    |
            |    | positive |    |
            |    |    +-----|----+------------+
            |    |    |     |    |            |
            |    +----+-----+    |            |
            |         |          |            |
            +---------+----------+            |
                      |                       |
                      |        gt bbox        |
                      |                       |
                      |             +---------+----------+
                      |             |         |          |
                      |             |    +----+-----+    |
                      |             |    |    |     |    |
                      +-------------|--- +----+     |    |
                                    |    | positive |    |
                                    |    +----------+    |
                                    |      negative      |
                                    +--------------------+

         The random noise is added to the top-left and down-right point
         positions, hence, normalized (x, y, x, y) format of bboxes are
         required. The noisy bboxes of positive queries have the points
         both within the inner square, while those of negative queries
         have the points both between the inner and outer squares.

        Besides, the length of outer square is twice as long as that of
        the inner square, i.e., self.box_noise_scale * w_or_h / 2.
        NOTE The noise is added to all the bboxes. Moreover, there is still
        unconsidered case when one point is within the positive square and
        the others is between the inner and outer squares.

        Args:
            gt_bboxes (Tensor): The concatenated gt bboxes of all samples
                in the batch, has shape (num_target_total, 4) with the last
                dimension arranged as (cx, cy, w, h) where
                `num_target_total = sum(num_target_list)`.
            num_groups (int): The number of denoising query groups.

        Returns:
            Tensor: The output noisy bboxes, which are embedded by normalized
            (cx, cy, w, h) format bboxes going through inverse_sigmoid, has
            shape (num_noisy_targets, 4) with the last dimension arranged as
            (cx, cy, w, h), where
            `num_noisy_targets = num_target_total * num_groups * 2`.
        """
        assert self.box_noise_scale > 0
        device = gt_bboxes.device

        # expand gt_bboxes as groups
        gt_bboxes_expand = gt_bboxes.repeat(2 * num_groups, 1)  # xyxy

        # obtain index of negative queries in gt_bboxes_expand
        positive_idx = torch.arange(
            len(gt_bboxes), dtype=torch.long, device=device)
        positive_idx = positive_idx.unsqueeze(0).repeat(num_groups, 1)
        positive_idx += 2 * len(gt_bboxes) * torch.arange(
            num_groups, dtype=torch.long, device=device)[:, None]
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(gt_bboxes)

        # determine the sign of each element in the random part of the added
        # noise to be positive or negative randomly.
        rand_sign = torch.randint_like(
            gt_bboxes_expand, low=0, high=2,
            dtype=torch.float32) * 2.0 - 1.0  # [low, high), 1 or -1, randomly

        # calculate the random part of the added noise
        rand_part = torch.rand_like(gt_bboxes_expand)  # [0, 1)
        rand_part[negative_idx] += 1.0  # pos: [0, 1); neg: [1, 2)
        rand_part *= rand_sign  # pos: (-1, 1); neg: (-2, -1] U [1, 2)

        # add noise to the bboxes
        bboxes_whwh = bbox_xyxy_to_cxcywh(gt_bboxes_expand)[:, 2:].repeat(1, 2)
        noisy_bboxes_expand = gt_bboxes_expand + torch.mul(
            rand_part, bboxes_whwh) * self.box_noise_scale / 2  # xyxy
        noisy_bboxes_expand = noisy_bboxes_expand.clamp(min=0.0, max=1.0)
        noisy_bboxes_expand = bbox_xyxy_to_cxcywh(noisy_bboxes_expand)

        dn_bbox_query = inverse_sigmoid(noisy_bboxes_expand, eps=1e-3)
        return dn_bbox_query

    def collate_dn_queries(self, input_label_query: Tensor,
                           input_bbox_query: Tensor, batch_idx: Tensor,
                           batch_size: int, num_groups: int) -> Tuple[Tensor]:
        """Collate generated queries to obtain batched dn queries.

        The strategy for query collation is as follow:

        .. code:: text

                    input_queries (num_target_total, query_dim)
            P_A1 P_B1 P_B2 N_A1 N_B1 N_B2 P'A1 P'B1 P'B2 N'A1 N'B1 N'B2
              |________ group1 ________|    |________ group2 ________|
                                         |
                                         V
                      P_A1 Pad0 N_A1 Pad0 P'A1 Pad0 N'A1 Pad0
                      P_B1 P_B2 N_B1 N_B2 P'B1 P'B2 N'B1 N'B2
                       |____ group1 ____| |____ group2 ____|
             batched_queries (batch_size, max_num_target, query_dim)

            where query_dim is 4 for bbox and self.embed_dims for label.
            Notation: _-group 1; '-group 2;
                      A-Sample1(has 1 target); B-sample2(has 2 targets)

        Args:
            input_label_query (Tensor): The generated label queries of all
                targets, has shape (num_target_total, embed_dims) where
                `num_target_total = sum(num_target_list)`.
            input_bbox_query (Tensor): The generated bbox queries of all
                targets, has shape (num_target_total, 4) with the last
                dimension arranged as (cx, cy, w, h).
            batch_idx (Tensor): The batch index of the corresponding sample
                for each target, has shape (num_target_total).
            batch_size (int): The size of the input batch.
            num_groups (int): The number of denoising query groups.

        Returns:
            tuple[Tensor]: Output batched label and bbox queries.
            - batched_label_query (Tensor): The output batched label queries,
              has shape (batch_size, max_num_target, embed_dims).
            - batched_bbox_query (Tensor): The output batched bbox queries,
              has shape (batch_size, max_num_target, 4) with the last dimension
              arranged as (cx, cy, w, h).
        """
        device = input_label_query.device
        num_target_list = [
            torch.sum(batch_idx == idx) for idx in range(batch_size)
        ]
        max_num_target = max(num_target_list)
        num_denoising_queries = int(max_num_target * 2 * num_groups)

        map_query_index = torch.cat([
            torch.arange(num_target, device=device)
            for num_target in num_target_list
        ])
        map_query_index = torch.cat([
            map_query_index + max_num_target * i for i in range(2 * num_groups)
        ]).long()
        batch_idx_expand = batch_idx.repeat(2 * num_groups, 1).view(-1)
        mapper = (batch_idx_expand, map_query_index)

        batched_label_query = torch.zeros(
            batch_size, num_denoising_queries, self.embed_dims, device=device)
        batched_bbox_query = torch.zeros(
            batch_size, num_denoising_queries, 4, device=device)

        batched_label_query[mapper] = input_label_query
        batched_bbox_query[mapper] = input_bbox_query
        return batched_label_query, batched_bbox_query

    def generate_dn_mask(self, max_num_target: int, num_groups: int,
                         device: Union[torch.device, str]) -> Tensor:
        """Generate attention mask to prevent information leakage from
        different denoising groups and matching parts.

        .. code:: text

                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
         max_num_target |_|           |_________| num_matching_queries
                        |_____________| num_denoising_queries

               1 -> True  (Masked), means 'can not see'.
               0 -> False (UnMasked), means 'can see'.

        Args:
            max_num_target (int): The max target number of the input batch
                samples.
            num_groups (int): The number of denoising query groups.
            device (obj:`device` or str): The device of generated mask.

        Returns:
            Tensor: The attention mask to prevent information leakage from
            different denoising groups and matching parts, will be used as
            `self_attn_mask` of the `decoder`, has shape (num_queries_total,
            num_queries_total), where `num_queries_total` is the sum of
            `num_denoising_queries` and `num_matching_queries`.
        """
        num_denoising_queries = int(max_num_target * 2 * num_groups)
        num_queries_total = num_denoising_queries + self.num_matching_queries
        attn_mask = torch.zeros(
            num_queries_total,
            num_queries_total,
            device=device,
            dtype=torch.bool)
        # Make the matching part cannot see the denoising groups
        attn_mask[num_denoising_queries:, :num_denoising_queries] = True
        # Make the denoising groups cannot see each other
        for i in range(num_groups):
            # Mask rows of one group per step.
            row_scope = slice(max_num_target * 2 * i,
                              max_num_target * 2 * (i + 1))
            left_scope = slice(max_num_target * 2 * i)
            right_scope = slice(max_num_target * 2 * (i + 1),
                                num_denoising_queries)
            attn_mask[row_scope, right_scope] = True
            attn_mask[row_scope, left_scope] = True
        return attn_mask


@MODELS.register_module()
class RT_detr_ChannelMapper(BaseModule):
    """Channel Mapper to reduce/increase channels of backbone features.

    This is used to reduce/increase channels of backbone features.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        kernel_size (int, optional): kernel_size for reducing channels (used
            at each scale). Default: 3.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Default: None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Default: None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Default: dict(type='ReLU').
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        num_outs (int, optional): Number of output feature maps. There would
            be extra_convs when num_outs larger than the length of in_channels.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or dict],
            optional): Initialization config dict.
    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = ChannelMapper(in_channels, 11, 3).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        kernel_size: int = 3,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = dict(type='ReLU'),
        bias: Union[bool, str] = 'auto',
        num_outs: int = None,
        neck_fusion: bool = True,
        txt_dims: int = 1024,
        init_cfg: OptMultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.extra_convs = None
        if num_outs is None:
            num_outs = len(in_channels)
        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.convs.append(
                ConvModule(
                    in_channel,
                    out_channels,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=bias))
        if num_outs > len(in_channels):
            self.extra_convs = nn.ModuleList()
            for i in range(len(in_channels), num_outs):
                if i == len(in_channels):
                    in_channel = in_channels[-1]
                else:
                    in_channel = out_channels
                self.extra_convs.append(
                    ConvModule(
                        in_channel,
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        bias=bias))
        
        self.neck_fusion = neck_fusion
        self.txt_dims = txt_dims
        self.neck_TV_fusion = TextGuideVTFusion(text_channels=self.txt_dims, embed_channels=out_channels)

    def forward(self, inputs: Tuple[Tensor], txt_feat) -> Tuple[Tensor]:
        """Forward function."""
        assert len(inputs) == len(self.convs)
        outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                if i == 0:
                    outs.append(self.extra_convs[0](inputs[-1]))
                else:
                    outs.append(self.extra_convs[i](outs[-1]))
        if self.neck_fusion:
            for i in range(len(inputs)):
                outs[i] = self.neck_TV_fusion(outs[i], txt_feat)
        return tuple(outs)