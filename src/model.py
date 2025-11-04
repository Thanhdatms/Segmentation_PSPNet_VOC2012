import torch
import torch.nn as nn

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

if __name__ == "__main__":
    model = ResidualBlockPSP(3, in_channels=64, mid_channels=64, out_channels=256, stride=1, dilation=1)
    print(model)
    # x = torch.randn(1, 64, 64, 64)
    # y = model(x)
    # print("Input shape:", x.shape)
    # print("Output shape:", y.shape)