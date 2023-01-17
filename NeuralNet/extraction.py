import numpy as np
from numpy.core.fromnumeric import clip
import torch
from torchvision import transforms
import time
import os
from sklearn import preprocessing
import torchvision
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """Extract feature using Resnet 18, with pretrained weigths on ImageNet. 
    
    Feature are extracted using Resnet without the last two layers (Global Average Pooling and Linear)

    Args:
        avg_pool(:class:`bool`): If True we add a pooling layer with kernel_size=2x2 at the end of the feature extractor.
    """
    def __init__(self, avg_pool=False):
        super().__init__()
        self._avg_pool=avg_pool
        self._model = torchvision.models.resnet18(pretrained=True)
        #self._model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        self._model = torch.nn.Sequential(*(list(self._model.children())[:-2]))
        #print(self._model)
        self.pool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        #self._model.eval()

    def forward(self, data):
        #with torch.no_grad():
        dim_0 = data.shape[0]
        dim_1 = data.shape[1]
        flatt = data.flatten(start_dim=0, end_dim=1)
        
        output = self._model(flatt) # output.shape = [30, 512, 7, 13]
        if self._avg_pool:
            output = self.pool(output)
        print(f"1.output after resnet: {output.shape}")
        output = output.reshape(dim_0, dim_1, output.shape[1], -1)
        #print(f"2.output after reshape: {output.shape}")
        output = output.permute(0, 2, 1, 3)
        #print(f"3.output after permute: {output.shape}")
        output = output.flatten(start_dim=2)
        print(f"4.output after flatten dim 2: {output.shape}")
        return output

    
