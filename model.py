import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidual(nn.Module):
    """
    MobileNetV2 building block: Inverted Residual with Linear Bottleneck.
    Optimized for high-FPS, lightweight inference.
    """
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # 1x1 Pointwise mapping
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        layers.extend([
            # 3x3 Depthwise (with potential dilation)
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding=dilation, dilation=dilation, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # 1x1 Pointwise-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2Encoder(nn.Module):
    def __init__(self, input_channels=3, output_stride=16):
        super(MobileNetV2Encoder, self).__init__()
        # Architected entirely from scratch (No Pre-trained weights!)
        
        # We enforce an Output Stride of 16 for better boundary detail with ASPP
        if output_stride == 16:
            strides = [2, 2, 2, 1]
            dilations = [1, 1, 1, 2] # Dilate the last block to maintain receptive field
        else:
            strides = [2, 2, 2, 2]
            dilations = [1, 1, 1, 1]
            
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 2, 1, bias=False), # Stride 2
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            InvertedResidual(32, 16, 1, 1) # Stride 2
        )
        self.enc2 = self._make_layer(16, 24, 2, strides[0], 6, dilations[0]) # Stride 4
        
        # enc2 output is our low-level feature hook for DeepLabV3+
        
        self.enc3 = self._make_layer(24, 32, 3, strides[1], 6, dilations[1]) # Stride 8
        self.enc4 = self._make_layer(32, 64, 4, strides[2], 6, dilations[2]) # Stride 16
        self.enc5 = self._make_layer(64, 160, 3, strides[3], 6, dilations[3]) # Stride 16
        
        # Final block for high-level features
        self.enc_final = nn.Sequential(
            nn.Conv2d(160, 320, 1, 1, 0, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU6(inplace=True)
        )

    def _make_layer(self, inp, oup, num_blocks, stride, expand_ratio, dilation):
        layers = [InvertedResidual(inp, oup, stride, expand_ratio, dilation)]
        for _ in range(1, num_blocks):
            layers.append(InvertedResidual(oup, oup, 1, expand_ratio, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        f1 = self.enc1(x)   
        low_level_features = self.enc2(f1)  
        x = self.enc3(low_level_features)  
        x = self.enc4(x)  
        x = self.enc5(x)  
        high_level_features = self.enc_final(x)
        return low_level_features, high_level_features


class ASPPDepthwiseConv(nn.Sequential):
    """Depthwise Separable ASPP component for maximizing FPS"""
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPDepthwiseConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, 3, padding=dilation, dilation=dilation, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        # 1x1 Convolution 
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        ))
        # Atrous Depthwise Convolutions
        for rate in atrous_rates:
            modules.append(ASPPDepthwiseConv(in_channels, out_channels, rate))
            
        # Global Average Pooling
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class RealTimeDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1):
        super(RealTimeDeepLabV3Plus, self).__init__()
        
        # MobileNet Backbone
        self.encoder = MobileNetV2Encoder(input_channels=3, output_stride=16)
        
        # ASPP Module (Catches Boundary Edges)
        self.aspp = ASPP(in_channels=320, out_channels=256, atrous_rates=[3, 6, 9])
        
        # Decoder
        # 1. Project Low-Level Features (Stride 4 map, originally 24 channels)
        self.project_low_level = nn.Sequential(
            nn.Conv2d(24, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU6(inplace=True)
        )
        
        # 2. Final fusion block (combining ASPP and Low-Level features)
        self.decoder_conv = nn.Sequential(
            # Depthwise Separable to keep FPS very high
            nn.Conv2d(304, 304, 3, padding=1, groups=304, bias=False),
            nn.BatchNorm2d(304),
            nn.ReLU6(inplace=True),
            nn.Conv2d(304, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),
            
            # Second Separable block
            nn.Conv2d(256, 256, 3, padding=1, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True)
        )
        
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        input_size = x.shape[-2:]
        
        # 1. Encoder Extract
        low_level_features, high_level_features = self.encoder(x)
        
        # 2. ASPP Application
        aspp_output = self.aspp(high_level_features)
        
        # 3. Upsample ASPP to low-level feature spatial resolution (4x size)
        low_level_projected = self.project_low_level(low_level_features)
        aspp_output_up = F.interpolate(aspp_output, size=low_level_projected.shape[-2:], mode='bilinear', align_corners=False)
        
        # 4. Decoder Fusion Block
        concat_features = torch.cat([aspp_output_up, low_level_projected], dim=1)
        decoder_features = self.decoder_conv(concat_features)
        
        # 5. Classifier and Final Upscale (4x size to hit original resolution)
        logits = self.classifier(decoder_features)
        out = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        
        # Return Raw Logits for BCE+Dice compatibility
        return out
