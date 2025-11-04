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

        self.conv2d_batch_norm_relu_1 = Conv2DBatchNormReLU(in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)

        #block 2
        in_channels = 64
        out_channels = 64
        kernel_size = 3
        stride = 1
        padding = 1
        dilation = 1
        bias = False

        self.conv2d_batch_norm_relu_2 = Conv2DBatchNormReLU(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        

        #block 3
        in_channels = 64
        out_channels = 128
        kernel_size = 3
        stride = 1
        padding = 1
        dilation = 1
        bias = False

        self.conv2d_batch_norm_relu_3 = Conv2DBatchNormReLU(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)

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

        out_2 = self.conv2d_batchnorm_relu_2(self.avg_pool_2(x))
        out_2 = F.interpolate(out_2, size=(self.height, self.width), mode='bilinear', align_corners=True)

        out_3 = self.conv2d_batchnorm_relu_3(self.avg_pool_3(x))
        out_3 = F.interpolate(out_3, size=(self.height, self.width), mode='bilinear', align_corners=True)

        out_4 = self.conv2d_batchnorm_relu_4(self.avg_pool_4(x))
        out_4 = F.interpolate(out_4, size=(self.height, self.width), mode='bilinear', align_corners=True)

        outputs = torch.cat([x, out_1, out_2, out_3, out_4], dim=1)

        return outputs
    
class DecodePSPFeature(nn.Module):
    def __init__(self, height, width, n_classes):
        super(DecodePSPFeature, self).__init__()
        # height: the height output of decoder
        # width: the width output of decoder
        self.height = height
        self.width = width
        self.n_classes = n_classes

        self.conv2d_batchnorm_relu = Conv2DBatchNormReLU(in_channels=4096, out_channels=512, kernel_size=3, padding=1, stride=1, dilation=1, bias=False)
        self.drop_out = nn.Dropout(p=0.1)
        self.classification = nn.Conv2d(in_channels=512, out_channels=self.n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv2d_batchnorm_relu(x)
        x = self.drop_out(x)
        x = self.classification(x)
        outputs = F.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=True) # convert into (475, 475) like actual image

        return outputs
    
class AuxilirayPSPLayers(nn.Module):
    def __init__(self, height, width, n_classes):
        super(AuxilirayPSPLayers, self).__init__()
        # height: height of output 
        # width: width of output
        self.height = height
        self.width = width
        self.n_classes = n_classes

        self.conv2d_batchnorm_relu = Conv2DBatchNormReLU(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.classification = nn.Conv2d(in_channels=256, out_channels=self.n_classes, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.conv2d_batchnorm_relu(x)
        x = self.dropout(x)
        x = self.classification(x)
        outputs = F.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=True) # convert image into actual size (475, 475)

        return outputs


class PSPNet(nn.Module):
    def __init__(self, n_classes):
        super(PSPNet, self).__init__()

        # parameters
        block_config = [3, 4, 6, 3]
        img_size = 475
        img_size_8 = 60

        # feature module
        self.feature_conv = FeatureMapConvolution()
        self.feature_residual_1 = ResidualBlockPSP(n_blocks=block_config[0], in_channels=128, mid_channels=64, out_channels=256, stride=1, dilation=1)
        self.feature_residual_2 = ResidualBlockPSP(n_blocks=block_config[1], in_channels=256, mid_channels=128, out_channels=512, stride=2, dilation=1)

        self.feature_dilated_residual_1 = ResidualBlockPSP(n_blocks=block_config[2], in_channels=512, mid_channels=256, out_channels=1024, stride=1, dilation=2)
        self.feature_dilated_residual_2 = ResidualBlockPSP(n_blocks=block_config[3], in_channels=1024, mid_channels=512, out_channels=2048, stride=1, dilation=4)

        # pyramid module
        self.pyramid_pooling = PyramidPooling(in_channels=2048, pool_sizes=[6,3,2,1], height=img_size_8, width=img_size_8)

        # decoder module
        self.decode_feature = DecodePSPFeature(height=img_size, width=img_size, n_classes=n_classes)

        # aux loss module 
        self.aux_loss = AuxilirayPSPLayers(height=img_size, width=img_size, n_classes=n_classes)

    def forward(self, x):
        x = self.feature_conv(x)
        x = self.feature_residual_1(x)
        x = self.feature_residual_2(x)
        x = self.feature_dilated_residual_1(x)

        output_aux = self.aux_loss(x)

        x = self.feature_dilated_residual_2(x)
        x = self.pyramid_pooling(x)

        outputs = self.decode_feature(x)

        return(outputs, output_aux)


if __name__ == "__main__":
    # model = ResidualBlockPSP(3, in_channels=64, mid_channels=64, out_channels=256, stride=1, dilation=1)
    # print(model)
    # x = torch.randn(1, 64, 64, 64)
    # y = model(x)
    # print("Input shape:", x.shape)
    # print("Output shape:", y.shape)


    # TEST model PSP NET FULL PIPELINE
    dummy_img = torch.rand(2, 3, 475, 475)
    model = PSPNet(n_classes=21)
    # print(model)
    outputs = model(dummy_img)
    print(outputs[0].shape)
    print(outputs[1].shape)