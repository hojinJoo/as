import torch
import torch.nn as nn
import functools



class ResNetBlock(nn.Module):
    def __init__(self, in_channels,out_channels, norm, strides=(1, 1)):
        super(ResNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm
        self.strides = strides

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1, bias=False)
        self.norm1 = norm(num_channels=out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = norm(num_channels=out_channels)
        self.residual_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=self.strides, bias=False)
        
        
        
    def forward(self, x):
        residual = x
    
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)

        if residual.shape != x.shape:
            residual = self.residual_conv(residual)            
            residual = self.norm2(residual)

        x = self.relu(residual + x)
        return x



class BottleNeckResNetBlock(ResNetBlock):
    def __init__(self, in_channels,out_channels, norm, strides=(1, 1)):
        super(BottleNeckResNetBlock, self).__init__(in_channels,out_channels, norm, strides)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm1 = norm(num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=strides, padding=1, bias=False)
        self.norm2 = norm(num_channels=out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, bias=False)
        self.norm3 = norm(num_channels=out_channels * 4)
        self.residual_conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=strides, bias=False)
        
    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)

        if residual.shape != x.shape:
            residual = self.residual_conv(residual)
            residual = self.norm3(residual)

        x = self.relu(residual + x)
        return x


# ResNet model
class ResNet(nn.Module):
    def __init__(self,  block_cls, stage_sizes, norm_type="group",cac=False,channels=2):
        super(ResNet, self).__init__()
        self.block_cls = block_cls
        self.stage_sizes = stage_sizes
        self.norm_type = norm_type
        self.relu = nn.ReLU()
        self.bottle_neck = True if block_cls == BottleNeckResNetBlock else False
        if self.norm_type == "batch":
            self.norm = nn.BatchNorm2d
        elif self.norm_type == "layer":
            self.norm = nn.LayerNorm
        elif self.norm_type == "group":
            self.norm = functools.partial(nn.GroupNorm, num_groups=32) # 논문코드 보니까 32로 고정시켜놓았음
        else:
            raise ValueError(f"Invalid norm_type: {self.norm_type}")

        width = 64

        # Root block
        if cac : 
            self.conv1 = nn.Conv2d(2 * channels ,64, kernel_size=3, stride=1, padding=1, bias=False)
        else : 
            self.conv1 = nn.Conv2d(channels ,64, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = self.norm(num_channels=64)
        
        # Stages
        self.stages = nn.ModuleList()
        in_channels = width
        for i, stage_size in enumerate(self.stage_sizes):
            if i == 0:
                first_block_stride  = (1,1)
            else : 
                first_block_stride  = (2,2)
            if self.bottle_neck:
                stage = self._make_stage(in_channels, width  * 2**i, stage_size, first_block_stride)
                in_channels = width * 4 * 2**i
            else : 
                stage = self._make_stage(in_channels, width * 2**i, stage_size, first_block_stride)
                in_channels = width * 2**i 
            self.stages.append(stage)


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        
        for i,stage in enumerate(self.stages):
            stage= stage.to(x.device)
            x = stage(x)
        return x
 
    def _make_stage(self, in_channels, out_channels, num_blocks, first_block_stride):
        if self.bottle_neck :
            blocks = [self.block_cls(in_channels if i==0 else out_channels * 4 ,out_channels, self.norm, strides= first_block_stride if  i == 0 else (1, 1)) for i in range(num_blocks)]
        else :
            blocks = [self.block_cls(in_channels if i==0 else out_channels  ,out_channels, self.norm, strides= first_block_stride if  i == 0 else (1, 1)) for i in range(num_blocks)]
        # print(f"blocks : {blocks}")
        return nn.Sequential(*blocks)



def get_backbone(resnet,cac=False,channles=2) :
    if resnet =="34" :
        return ResNet(block_cls=ResNetBlock, stage_sizes=[3, 4, 6, 3],cac=cac)
    elif resnet =="50" :
        return ResNet(block_cls=BottleNeckResNetBlock, stage_sizes=[3, 4, 6, 3],cac=cac)
    elif resnet =="101" :
        return ResNet(block_cls=BottleNeckResNetBlock, stage_sizes=[3, 4, 23, 3],cac=cac)
    else : 
        raise ValueError(f"Invalid resnet: {resnet}")



if __name__ == "__main__" :
    sample_wav = torch.randn(8000,device='cpu')
    after_stft = torch.stft(sample_wav, n_fft=512, win_length=512,
                            hop_length=125, return_complex=True)
    after_stft = torch.abs(after_stft)
    print(after_stft.size())
    sample = after_stft.unsqueeze(0).unsqueeze(0).repeat((16,1,1, 1))
    model = get_backbone("101")
    print(model)
    # after_model = model(sample)
    # print(after_model.size())
    # result = sa(after_model,train=True)
    # print(result['slots'].shape)
    # print(result['attn'].shape)
    
# # Creating ResNet models
# ResNet18 = functools.partial(ResNet, block_cls=ResNetBlock, stage_sizes=[2, 2, 2, 2])
# ResNet34 = functools.partial(ResNet, block_cls=ResNetBlock, stage_sizes=[3, 4, 6, 3])
# ResNet50 = functools.partial(ResNet, block_cls=ResNetBlock, stage_sizes=[3, 4, 6, 3])
# ResNet101 = functools.partial(ResNet, block_cls=ResNetBlock, stage_sizes=[3, 4, 23, 3])
# ResNet152 = functools.partial(ResNet, block_cls=ResNetBlock, stage_sizes=[3, 8, 36, 3])
# ResNet200 = functools.partial(ResNet, block_cls=ResNetBlock, stage_sizes=[3, 24, 36, 3])

# test = torch.rand((1,1,257,55))
# print(ResNet34(test).size())