# coding=<utf-8>
"""
Author: DoranLyong 

Load the pretrained networks in torchvision project. 
    * [ref] https://github.com/pytorch/vision
    * [ref] https://github.com/deep-learning-with-pytorch/dlwpt-code/blob/master/p1ch2/2_pre_trained_networks.ipynb
"""
import os 
import os.path as osp 
import sys 
sys.path.insert(0, osp.pardir)


from PIL import Image 
import torch 
from torchvision import models
from torchvision import transforms


""" 1. Loading models from torchvision """ 

print("model lists: ", dir(models) , end="\n\n")


AlexNet = models.AlexNet()   # w/o pretrained parameters 
ResNet = models.resnet101(pretrained=True)

print("ResNet structure: ", ResNet)



""" 2. Data load & transform """

preprocess = transforms.Compose([   transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )
                                ])



imgPath = osp.join(osp.abspath(""), "sample_data", "dog_sample.jpg" )
img = Image.open(imgPath) # 이미지 불러오기 


img_t = preprocess(img)  # Image transformation 
batch_t = torch.unsqueeze(img_t, 0)
print("Data Shape: ", batch_t.shape)



""" 3. Model Inference """

ResNet.eval()  # Set evaluation mode for inference without backward 
out = ResNet(batch_t)      
print("output shape: ", out.shape)   # ImageNet은 1000개의 클래스를 가지고 있음 
                                     # 즉, 1000개의 노드에 각각 score가 출력됨 


with open(osp.join(osp.abspath(""), "imagenet_classes.txt")) as f:
    labels = [line.strip() for line in f.readlines()]

print("# of lables: ", len(labels))


_, index = torch.max(out, 1)   # 1000개의 노드중 score가 높은 노드의 인덱스를 가져옴 
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100  # get confidence score of each outnode

prediction, confidence = labels[index[0]], percentage[index[0]].item()
print("prediction: %s , confidence: %f" %(prediction, confidence ))



# Rank-5 prediction 
_, indices = torch.sort(out, descending=True)     # 값이 높은 상위 5개 가져옴 
rank_5 = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
print(rank_5)