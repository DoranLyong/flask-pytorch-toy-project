# coding=<utf-8>
"""
Author: DoranLyong 

Load the pretrained networks in torchvision project. 
    * [ref] https://github.com/pytorch/vision
    * [ref] https://github.com/deep-learning-with-pytorch/dlwpt-code/blob/master/p1ch2/3_cyclegan.ipynb
"""
import os 
import os.path as osp 

from PIL import Image
import torch 
import torch.nn as nn
from torchvision import transforms

class ResNetBlock(nn.Module): # <1>

    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim)]

        return nn.Sequential(*conv_block)  # conv_block 리스트의 모든 요소를 파라미터로 던짐 

    def forward(self, x):
        out = x + self.conv_block(x) # <2>
        return out


class ResNetGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9): # <3> 

        assert(n_blocks >= 0)
        super(ResNetGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResNetBlock(ngf * mult)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input): # <3>
        return self.model(input)



def main():
    
    """ 1. 모델 초기화 """ 
    net = ResNetGenerator()  # 모델 초기화 

    model_path = osp.join(osp.abspath(""), "params", "horse2zebra_0.4.0.pth")
    model_data = torch.load(model_path)

    net.load_state_dict(model_data)  #모델 아키텍쳐에 학습된 가중치 파라미터를 로드 

    

    """ 2. Data load & transform """
    imgPath = osp.join(osp.abspath(""), "sample_data", "horse_sample.jpg" )
    img = Image.open(imgPath) # 이미지 불러오기 


    preprocess = transforms.Compose([   transforms.Resize(256),
                                        transforms.ToTensor(),
                                    ])

    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)

    
    """ 3. Model Inference """
    net.eval()  # # Set evaluation mode for inference without backward 

    batch_out = net(batch_t)

    out_t = (batch_out.data.squeeze() + 1.0) / 2.0
    out_img = transforms.ToPILImage()(out_t)
    out_img.save(osp.join(osp.abspath(""), "sample_data",'cycleGAN_zebra.jpg'))


if __name__ == "__main__":
    main()