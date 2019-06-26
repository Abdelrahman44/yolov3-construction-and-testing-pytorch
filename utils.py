from __future__ import division
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):

    
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    # convert the feature map to a 2d array, each row is the prediction for an anchor for a pixel in the map
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #Sigmoid the centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride
    
    return prediction


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    return unique_tensor

def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b1_y1)
    inter_area = torch.clamp(inter_rect_x2 - b1_x1 +1, min=0) * torch.clamp(inter_rect_y2 - b1_y1, min=0)
    iou = inter_area/(b1_area + b2_area - inter_area)
    
    return iou

def write_results(predictions, confidence_thres, num_classes, IOU_conf=0.4):
    # a mask to disregard the below confidebce threshold bounding boxes
    conf_mask = (predictions[:,:,4] > confidence_thres).float().unsqueeze(2)
    predictions *= conf_mask
    
    # converting from (x,y,width,height) to (x,y of top left and bottom right corners)
    box_corners = predictions.new(*predictions.shape)
    box_corners[:,:,0] = predictions[:,:,0] - predictions[:,:,2]/2
    box_corners[:,:,2] = predictions[:,:,0] + predictions[:,:,2]/2
    box_corners[:,:,1] = predictions[:,:,1] - predictions[:,:,3]/2
    box_corners[:,:,3] = predictions[:,:,1] + predictions[:,:,3]/2
    predictions[:,:,:4] = box_corners[:,:,:4]
    
    batch_size = predictions.size(0)

    write = False
    
    # single image at a time
    for i in range(batch_size):
        preds = predictions[i]
        max_conf, max_class= torch.max((preds[:,5:5+num_classes]), 1)   # class score and class number
        max_class = max_class.float().unsqueeze(1)
        max_conf = max_conf.float().unsqueeze(1)
        preds = torch.cat((preds[:,:5], max_conf, max_class), 1)
        
        nonzero_idx = torch.nonzero(preds[:,4]).squeeze()
        preds_ = preds[nonzero_idx,:].view(-1, 7)
        
        img_classes = unique(preds_[:,-1])      #get the detetcted classes
    
    
        for cls in img_classes:
            
            cls_mask = (preds_[:,-1] == cls).float().unsqueeze(1)
            masked = preds_ * cls_mask
            cls_idx = torch.nonzero(masked[:,-2]).squeeze()
            preds_class = preds_[cls_idx].view(-1,7)
            
            conf_sort_idx = torch.sort(preds_class[:,4], descending=True)[1]
            preds_class = preds_class[conf_sort_idx]
            num_detections = preds_class.size(0)
            
            for d in range(num_detections):
                try:
                    ious = bbox_iou(preds_class[d].unsqueeze(0), preds_class[d+1:])
                except:
                    break
                    
                iou_mask = (ious < IOU_conf).float().unsqueeze(1)
                preds_class[d+1:] *= iou_mask
                
                nonzero_ind = torch.nonzero(preds_class[:,4]).squeeze()
                preds_class = preds_class[nonzero_ind].view(-1, 7)
                
            
            batch_idx = preds_class.new(preds_class.size(0), 1).fill_(i)
            
            if not write:
                output = torch.cat((batch_idx, preds_class), 1)
                write = True
                
            else:
                out = torch.cat((batch_idx, preds_class), 1)
                output = torch.cat((output, out))
                
                
                
    try:
        return output
    except:
        return 0
       
    
