



from __future__ import division
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 


def predict_transform(feat_map, input_dim, anchors, num_classes, CUDA=True):
    
    
    batch_size = feat_map.size(0)
    stride =  input_dim // feat_map.size(2)
    grid_size = input_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    feat_map = feat_map.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    feat_map = feat_map.transpose(1,2).contiguous()
    feat_map = feat_map.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    
    feat_map[:,:,0] = torch.sigmoid(feat_map[:,:,0])
    feat_map[:,:,1] = torch.sigmoid(feat_map[:,:,1])
    feat_map[:,:,4] = torch.sigmoid(feat_map[:,:,4])
    
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    feat_map[:,:,:2] += x_y_offset
    
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    feat_map[:,:,2:4] = torch.exp(feat_map[:,:,2:4])*anchors
    
    feat_map[:,:,5: 5 + num_classes] = torch.sigmoid((feat_map[:,:, 5 : 5 + num_classes]))

    feat_map[:,:,:4] *= stride

    return feat_map