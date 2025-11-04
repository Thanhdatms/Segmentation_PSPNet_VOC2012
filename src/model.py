import torch
import torch.nn as nn
import torch.nn.functional as F
class Conv2DBatchNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(Conv2DBatchNormReLU, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu(x)

        return outputs
    
class FeatureMapConvolution(nn.Module):
    def __init__(self):
        super(FeatureMapConvolution, self).__init__()

        #block 1
        in_channels = 3
        out_channels = 64
        kernel_size = 3
        stride = 2
        padding = 1
        dilation = 1
        bias = False

        self.conv2d_batch_norm_relu_1 = Conv2DBatchNormReLU(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        #block 2
        in_channels = 64
        out_channels = 64
        kernel_size = 3
        stride = 1
        padding = 1
        dilation = 1
        bias = False

        self.conv2d_batch_norm_relu_2 = Conv2DBatchNormReLU(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)
        

        #block 3
        in_channels = 64
        out_channels = 128
        kernel_size = 3
        stride = 1
        padding = 1
        dilation = 1
        bias = False

        self.conv2d_batch_norm_relu_3 = Conv2DBatchNormReLU(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        # block 4

        self.maxpooling = nn.MaxPool2d(kernel_size, stride=2 , padding=1)

    def forward(self, x):
        x = self.conv2d_batch_norm_relu_1(x)
        x = self.conv2d_batch_norm_relu_2(x)
        x = self.conv2d_batch_norm_relu_3(x)

        outputs = self.maxpooling(x)

        return outputs
            
class Conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(Conv2DBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        outputs = self.batchnorm(x)

        return outputs
        

class BottleNeckPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super(BottleNeckPSP, self).__init__()
        self.conv2D_BatchNorm_Relu_1 = Conv2DBatchNormReLU(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.conv2D_BatchNorm_Relu_2 = Conv2DBatchNormReLU(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.conv2D_BatchNorm_3 = Conv2DBatchNorm(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        # skip connection
        self.conv_residual = Conv2DBatchNorm(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        main_forward = self.conv2D_BatchNorm_Relu_1(x)
        main_forward = self.conv2D_BatchNorm_Relu_2(main_forward)
        main_forward = self.conv2D_BatchNorm_3(main_forward)

        residual = self.conv_residual(x)

        return self.relu(main_forward + residual)

class BottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super(BottleNeckIdentifyPSP, self).__init__()

        self.conv2D_BatchNorm_Relu_1 = Conv2DBatchNormReLU(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.conv2D_BatchNorm_Relu_2 = Conv2DBatchNormReLU(mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.conv2D_BatchNorm_3 = Conv2DBatchNorm(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        main_forward = self.conv2D_BatchNorm_Relu_1(x)
        main_forward = self.conv2D_BatchNorm_Relu_2(main_forward)
        main_forward = self.conv2D_BatchNorm_3(main_forward)

        residual = x

        return self.relu(main_forward + residual)

class ResidualBlockPSP(nn.Sequential):
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation):
        super(ResidualBlockPSP, self).__init__()

        # bottle Neck PSP
        self.add_module("block1", BottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation))
        for i in range(n_blocks - 1):
            self.add_module(f"block{i+2}", BottleNeckIdentifyPSP(out_channels, mid_channels, out_channels, stride=1, dilation=dilation))


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes = [6,3,2,1], height=60, width=60):
        super(PyramidPooling, self).__init__()
        self.height = height
        self.width = width

        out_channels = int(in_channels/len(pool_sizes))

        # pooling 1
        self.avg_pool_1 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[0])
        self.conv2d_batchnorm_relu_1 = Conv2DBatchNormReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        # pooling 2
        self.avg_pool_2 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[1])
        self.conv2d_batchnorm_relu_2 = Conv2DBatchNormReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        # pooling 3
        self.avg_pool_3 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[2])
        self.conv2d_batchnorm_relu_3 = Conv2DBatchNormReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        # pooling 4
        self.avg_pool_4 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[3])
        self.conv2d_batchnorm_relu_4 = Conv2DBatchNormReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

    def forward(self, x):
        out_1 = self.conv2d_batchnorm_relu_1(self.avg_pool_1(x))
        out_1 = F.interpolate(out_1, size=(self.height, self.width), mode='bilinear', align_corners=True)

        out_2 = self.conv2d_batchnorm_relu_1(self.avg_pool_1(x))
        out_2 = F.interpolate(out_2, size=(self.height, self.width), mode='bilinear', align_corners=True)

        out_3 = self.conv2d_batchnorm_relu_1(self.avg_pool_1(x))
        out_3 = F.interpolate(out_3, size=(self.height, self.width), mode='bilinear', align_corners=True)

        out_4 = self.conv2d_batchnorm_relu_1(self.avg_pool_1(x))
        out_4 = F.interpolate(out_4, size=(self.height, self.width), mode='bilinear', align_corners=True)

        outputs = torch.cat([x, out_1, out_2, out_3, out_4], dim=1)

        return outputs
    
class DecodePSPFeature(nn.Module):
    def __init__(self, height, width, n_classes):
        super(DecodePSPFeature, self).__init__()

        self.height = height
        self.width = width
        self.n_classes = n_classes

        self.conv2d_batchnorm_relu = Conv2DBatchNormReLU(in_channels=4096, out_channels=512, kernel_size=3, padding=1, stride=1, dilation=1, bias=False)
        self.drop_out = nn.Dropout(p=0.1)
        self.classification = nn.Conv2d(in_channels=512, out_channels=21, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv2d_batchnorm_relu(x)
        x = self.drop_out(x)
        x = self.classification(x)
        outputs = F.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=True)

        return outputs
    
class AuxilirayPSPLayers(nn.Module):
    def __init__(self, height, width, n_classes):
        self.height = height
        self.width = width
        self.n_classes = n_classes

        self.conv2d_batchnorm_relu = Conv2DBatchNormReLU(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.classification = nn.Conv2d(in_channels=256, out_channels=21, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.conv2d_batchnorm_relu(x)
        x = self.dropout(x)
        x = self.classification(x)
        outputs = F.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=True)

        return outputs

        


if __name__ == "__main__":
    model = ResidualBlockPSP(3, in_channels=64, mid_channels=64, out_channels=256, stride=1, dilation=1)
    print(model)
    # x = torch.randn(1, 64, 64, 64)
    # y = model(x)
    # print("Input shape:", x.shape)
    # print("Output shape:", y.shape)