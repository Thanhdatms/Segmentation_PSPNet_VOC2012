import torch
import torch.nn as nn

class Conv2DBatchNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(Conv2DBatchNormReLU, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
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
            
