import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Parameter, Softmax

# ######################################################################
# ### NEW MODULE 1: Squeeze-and-Excitation (SE) Block
# ######################################################################
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise feature recalibration.
    Improves feature representation by modeling channel interdependencies.
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ######################################################################
# ### NEW MODULE 2: Depthwise Separable Dilated Convolution
# ######################################################################
class DWCDilated(nn.Module):
    '''
    Depthwise Separable Dilated Convolution for efficiency.
    Reduces parameters while maintaining receptive field.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1)/2) * d
        
        # Depthwise Convolution
        self.conv_dw = nn.Conv2d(nIn, nIn, (kSize, kSize), stride=stride, 
                                  padding=(padding, padding), bias=False, 
                                  dilation=d, groups=nIn)
        # Pointwise Convolution
        self.conv_pw = nn.Conv2d(nIn, nOut, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)
    
    def forward(self, input):
        output = self.conv_dw(input)
        output = self.conv_pw(output)
        output = self.bn(output)
        output = self.act(output)
        return output

class PAM_Module(Module):
    """ Position attention module with optimized attention computation """
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
    
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        
        # Use efficient scaled dot product attention
        q = proj_query.unsqueeze(1)
        k = proj_key.permute(0, 2, 1).unsqueeze(1)
        
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        v = proj_value.permute(0, 2, 1).unsqueeze(1)
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.squeeze(1).permute(0, 2, 1)
        out = out.view(m_batchsize, C, height, width)
        
        out = self.gamma * out + x
        return out

class CAM_Module(Module):
    """ Channel attention module with optimized attention computation """
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
    
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        
        # Use efficient scaled dot product attention
        q = proj_query.unsqueeze(1)
        k = proj_key.permute(0, 2, 1).unsqueeze(1)
        v = proj_query.unsqueeze(1)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.squeeze(1).view(m_batchsize, C, height, width)
        
        out = self.gamma * out + x
        return out

class UPx2(nn.Module):
    '''Upsampling with batch normalization and PReLU'''
    def __init__(self, nIn, nOut):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(nIn, nOut, 2, stride=2, padding=0, output_padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)
    
    def forward(self, input):
        output = self.deconv(input)
        output = self.bn(output)
        output = self.act(output)
        return output
    
    def fuseforward(self, input):
        output = self.deconv(input)
        output = self.act(output)
        return output

class CBR(nn.Module):
    '''Convolution + Batch Normalization + PReLU'''
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)
    
    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output
    
    def fuseforward(self, input):
        output = self.conv(input)
        output = self.act(output)
        return output

class CB(nn.Module):
    '''Convolution + Batch Normalization'''
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
    
    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return output

class C(nn.Module):
    '''Convolutional layer'''
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
    
    def forward(self, input):
        output = self.conv(input)
        return output

class CDilated(nn.Module):
    '''Dilated convolution'''
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=d)
    
    def forward(self, input):
        output = self.conv(input)
        return output

class DownSamplerB(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut/5)
        n1 = nOut - 4*n
        self.c1 = C(nIn, n, 3, 2)
        
        # Use efficient Depthwise Separable Dilated Convolutions
        self.d1 = DWCDilated(n, n1, 3, 1, 1)
        self.d2 = DWCDilated(n, n, 3, 1, 2)
        self.d4 = DWCDilated(n, n, 3, 1, 4)
        self.d8 = DWCDilated(n, n, 3, 1, 8)
        self.d16 = DWCDilated(n, n, 3, 1, 16)
        
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)
    
    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        output = self.bn(combine)
        output = self.act(output)
        return output

class BR(nn.Module):
    '''Batch Normalization + PReLU'''
    def __init__(self, nOut):
        super().__init__()
        self.nOut = nOut
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)
    
    def forward(self, input):
        output = self.bn(input)
        output = self.act(output)
        return output

class DilatedParllelResidualBlockB(nn.Module):
    '''ESP block with hierarchical feature fusion'''
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = max(int(nOut/5), 1)
        n1 = max(nOut - 4*n, 1)
        
        self.c1 = C(nIn, n, 1, 1)
        
        # Use efficient Depthwise Separable Dilated Convolutions
        self.d1 = DWCDilated(n, n1, 3, 1, 1)
        self.d2 = DWCDilated(n, n, 3, 1, 2)
        self.d4 = DWCDilated(n, n, 3, 1, 4)
        self.d8 = DWCDilated(n, n, 3, 1, 8)
        self.d16 = DWCDilated(n, n, 3, 1, 16)
        
        self.bn = BR(nOut)
        self.add = add
    
    def forward(self, input):
        output1 = self.c1(input)
        
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        
        if self.add:
            combine = input + combine
        
        output = self.bn(combine)
        return output

class InputProjectionA(nn.Module):
    '''Input projection with pyramid pooling'''
    def __init__(self, samplingTimes):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))
    
    def forward(self, input):
        for pool in self.pool:
            input = pool(input)
        return input

class ESPNet_Encoder(nn.Module):
    '''Enhanced ESPNet encoder with dual attention'''
    def __init__(self, p=5, q=3):
        super().__init__()
        self.level1 = CBR(3, 16, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)
        
        self.b1 = CBR(16 + 3, 19, 3)
        
        self.level2_0 = DownSamplerB(16 + 3, 64)
        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(64, 64))
        
        self.b2 = CBR(128 + 3, 131, 3)
        
        self.level3_0 = DownSamplerB(128 + 3, 128)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(128, 128))
        
        self.b3 = CBR(256, 32, 3)
        
        # Sequential DAM (Channel then Spatial attention - CBAM style)
        self.sc = CAM_Module(32)
        self.conv_sc = CBR(32, 32, 3)
        self.sa = PAM_Module(32)
        self.conv_sa = CBR(32, 32, 3)
        
        self.classifier = CBR(32, 32, 1, 1)
    
    def forward(self, input):
        # Level 1
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        
        # Level 2
        inp2 = self.sample2(input)
        output1_0 = self.level2_0(output0_cat)
        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)
        
        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1))
        
        # Level 3
        output2_0 = self.level3_0(output1_cat)
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        
        cat_ = torch.cat([output2_0, output2], 1)
        output2_cat = self.b3(cat_)
        
        # Sequential Attention (Channel -> Spatial)
        out_sc = self.sc(output2_cat)
        out_sc = self.conv_sc(out_sc)
        out_s = self.sa(out_sc)
        out_s = self.conv_sa(out_s)
        
        x_bottleneck = self.classifier(out_s)
        
        return x_bottleneck, output1_cat, output0_cat

# ######################################################################
# ### TwinLiteNet with Cross-Task Fusion and SE Blocks
# ######################################################################
class TwinLiteNet(nn.Module):
    '''
    Enhanced TwinLiteNet with:
    - SE blocks for feature recalibration
    - Cross-Task Fusion (CFF) between decoders
    - Improved skip connections
    '''
    def __init__(self, p=2, q=3):
        super().__init__()
        self.encoder = ESPNet_Encoder(p, q)
        
        # DECODER 1: Drivable Area (DA)
        self.up_1_1 = UPx2(32, 16)
        self.dec1_cbr1 = CBR(147, 16, 3, 1)
        self.dec1_se1 = SEBlock(16)
        
        self.up_2_1 = UPx2(16, 8)
        self.dec1_cbr2 = CBR(27, 8, 3, 1)
        self.dec1_se2 = SEBlock(8)
        
        self.classifier_1 = UPx2(8, 2)
        
        # DECODER 2: Lane Line (LL)
        self.up_1_2 = UPx2(32, 16)
        self.dec2_cbr1 = CBR(147, 16, 3, 1)
        self.dec2_se1 = SEBlock(16)
        
        # Cross-Task Fusion Block 1
        self.cff_1 = CBR(32, 16, 3, 1)
        
        self.up_2_2 = UPx2(16, 8)
        self.dec2_cbr2 = CBR(27, 8, 3, 1)
        self.dec2_se2 = SEBlock(8)
        
        # Cross-Task Fusion Block 2
        self.cff_2 = CBR(16, 8, 3, 1)
        
        self.classifier_2 = UPx2(8, 2)
    
    def forward(self, input):
        # Encoder
        x_bottleneck, x_skip_l2, x_skip_l1 = self.encoder(input)
        
        # DECODER 1 (Drivable Area)
        x1 = self.up_1_1(x_bottleneck)
        x1_fused = self.dec1_cbr1(torch.cat([x1, x_skip_l2], dim=1))
        x1_fused = self.dec1_se1(x1_fused)
        
        x1_up = self.up_2_1(x1_fused)
        x1_final = self.dec1_cbr2(torch.cat([x1_up, x_skip_l1], dim=1))
        x1_final = self.dec1_se2(x1_final)
        
        classifier1 = self.classifier_1(x1_final)
        
        # DECODER 2 (Lane Line)
        x2 = self.up_1_2(x_bottleneck)
        x2_fused = self.dec2_cbr1(torch.cat([x2, x_skip_l2], dim=1))
        x2_fused = self.dec2_se1(x2_fused)
        
        # Cross-Task Fusion 1
        x2_fused = self.cff_1(torch.cat([x2_fused, x1_fused], dim=1))
        
        x2_up = self.up_2_2(x2_fused)
        x2_final = self.dec2_cbr2(torch.cat([x2_up, x_skip_l1], dim=1))
        x2_final = self.dec2_se2(x2_final)
        
        # Cross-Task Fusion 2
        x2_final = self.cff_2(torch.cat([x2_final, x1_final], dim=1))
        
        classifier2 = self.classifier_2(x2_final)
        
        return (classifier1, classifier2)
