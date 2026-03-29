import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    def __init__(self, in_planes, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, reduced_dim, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_dim, in_planes, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class MBConv(nn.Module):
    """
    Mobile Inverted Bottleneck Conv heavily optimized for accuracy (EfficientNet core).
    Contains: Expansion -> Depthwise -> Squeeze/Excitation -> Projection -> Skip Connect
    """
    def __init__(self, in_planes, out_planes, expand_ratio, stride, kernel_size=3, dilation=1):
        super(MBConv, self).__init__()
        self.use_res_connect = stride == 1 and in_planes == out_planes
        hidden_planes = int(round(in_planes * expand_ratio))
        
        layers = []
        if expand_ratio != 1:
            # 1. Expand Convolution
            layers.extend([
                nn.Conv2d(in_planes, hidden_planes, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_planes),
                nn.SiLU(inplace=True)
            ])
            
        # 2. Depthwise Convolution with explicit padding alignment for dilations
        padding = ((kernel_size - 1) // 2) * dilation
        layers.extend([
            nn.Conv2d(hidden_planes, hidden_planes, kernel_size, stride, padding=padding,
                      dilation=dilation, groups=hidden_planes, bias=False),
            nn.BatchNorm2d(hidden_planes),
            nn.SiLU(inplace=True)
        ])
        
        # 3. Squeeze and Excitation Algorithmic Attention
        reduced_dim = max(1, int(in_planes / 4))
        layers.append(SqueezeExcitation(hidden_planes, reduced_dim))
        
        # 4. Projection to narrow dimension
        layers.extend([
            nn.Conv2d(hidden_planes, out_planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_planes),
        ]) # No activation after projection!
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class EfficientNetEncoder(nn.Module):
    def __init__(self, input_channels=3):
        super(EfficientNetEncoder, self).__init__()
        # Target: ~5-6M parameters, output_stride=16
        
        # Stage 1: Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Stage 2: Stride 2 map
        self.stage2 = self._make_layer(32, 16, exp=1, k=3, s=1, blocks=1)
        
        # Stage 3: Low-Level Output Hook for DeepLabV3+ (Stride 4)
        self.stage3 = self._make_layer(16, 24, exp=6, k=3, s=2, blocks=2)
        
        # Stage 4: Stride 8 
        self.stage4 = self._make_layer(24, 40, exp=6, k=5, s=2, blocks=2)
        
        # Stage 5: Output Stride 16
        self.stage5 = self._make_layer(40, 80, exp=6, k=3, s=2, blocks=3)
        
        # Stages 6-8: Lock Stride at 16 via Dilation!
        self.stage6 = self._make_layer(80, 112, exp=6, k=5, s=1, blocks=3, dilation=2)
        self.stage7 = self._make_layer(112, 192, exp=6, k=5, s=1, blocks=4, dilation=2)
        self.stage8 = self._make_layer(192, 320, exp=6, k=3, s=1, blocks=1, dilation=2)
        
        # Stage 9: Final Heavy Feature Matrix (1280 channels)
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True)
        )

    def _make_layer(self, inp, out, exp, k, s, blocks, dilation=1):
        layers = [MBConv(inp, out, exp, s, k, dilation)]
        for _ in range(1, blocks):
            layers.append(MBConv(out, out, exp, 1, k, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        
        # Grab Low-Level Feature Map for DeepLab spatial reconstruction
        low_level_features = self.stage3(x)  
        
        x = self.stage4(low_level_features)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.stage8(x)
        
        high_level_features = self.head(x)
        
        return low_level_features, high_level_features

# ==============================================================
# DECODER
# ==============================================================

class ASPPDepthwiseConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPDepthwiseConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, 3, padding=dilation, dilation=dilation, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
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
        # 1x1
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        ))
        
        for rate in atrous_rates:
            modules.append(ASPPDepthwiseConv(in_channels, out_channels, rate))
            
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        return self.project(torch.cat(res, dim=1))

class EfficientDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=3):
        super(EfficientDeepLabV3Plus, self).__init__()
        
        # Heavy Squeeze-And-Excitation Backbone 
        self.encoder = EfficientNetEncoder(input_channels=3)
        
        # ASPP receives the heavy 1280-channel map
        self.aspp = ASPP(in_channels=1280, out_channels=256, atrous_rates=[3, 6, 9])
        
        # Project Low-Level Features (24 channels from Stage 3 -> 48 dims)
        self.project_low_level = nn.Sequential(
            nn.Conv2d(24, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Final Fusion
        self.decoder_conv = nn.Sequential(
            # High Capacity Depthwise Separable Decoder Network!
            nn.Conv2d(304, 304, 3, padding=1, groups=304, bias=False),
            nn.BatchNorm2d(304),
            nn.SiLU(inplace=True),
            nn.Conv2d(304, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(256, 256, 3, padding=1, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )
        
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        input_size = x.shape[-2:]
        
        # 1. Feature Map Extraction
        low_level_features, high_level_features = self.encoder(x)
        
        # 2. ASPP Pyramid Evaluation
        aspp_output = self.aspp(high_level_features)
        
        # 3. ASPP Upscaling to match skip connection geometry
        low_level_projected = self.project_low_level(low_level_features)
        aspp_output_up = F.interpolate(aspp_output, size=low_level_projected.shape[-2:], mode='bilinear', align_corners=False)
        
        # 4. Decoupled Spatial Fusion
        concat_features = torch.cat([aspp_output_up, low_level_projected], dim=1)
        decoder_features = self.decoder_conv(concat_features)
        
        # 5. Output Projection
        logits = self.classifier(decoder_features)
        out = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        
        return out
