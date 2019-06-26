# # YoloV3 implementation and testing using Pytorch

from __future__ import division
import torch
import torchvision 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import variable
import numpy as np
from utils import *

def parse_cfg(file_path):
    #read the layers and store every block as a dictionary
    block = {}
    blocks = [] 
    with open(file=file_path, mode='r') as file:
        lines = file.read().split('\n')
        lines = [x for x in lines if (len(x)>0)]
        lines = (x for x in lines if x[0] != '#')
        lines = [x.rstrip().lstrip() for x in lines]
    
    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
        
    return blocks


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_modules(blocks):
    net_info = blocks[0] # get the network info as stored in the first block
    module_list = nn.ModuleList()  #network modules
    prev_filters = 3       #stores the filters of the previous layer only
    output_filters = []    #stores the filters of each layer
    
    for index, block in enumerate(blocks[1:]): #making a sequential module for each block containing the layers
        module = nn.Sequential()
        if(block['type'] == 'convolutional'):
            try:
                batch_normalize = int(block["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True                

            filters = int(block["filters"])
            kernel_size = int(block["size"])
            kernel_stride = int(block["stride"])
            kernel_padding = int(block["pad"])
            activation = block["activation"]
            
            if kernel_padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            
            conv = nn.Conv2d(prev_filters, filters, kernel_size, kernel_stride, pad, bias= bias)
            module.add_module("conv{0}".format(index), conv)
            
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm{0}".format(index), bn)
                
            if activation == "leaky":
                act = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky{0}".format(index), act)
                
            
        elif(block['type'] == 'shortcut'):
            shortcut = EmptyLayer()
            module.add_module("emptylayer{0}".format(index), shortcut)
            
            
        elif(block['type'] == 'route'):
            block['layers'] = block['layers'].split(',')
            start = int(block['layers'][0])
            
            try:
                end = int(block["layers"][1])
            except:
                end = 0

            # refer to all layers with negative indices
            if start > 0:
                start -= index
            if end > 0:
                end -= index
                
            route = EmptyLayer()
            module.add_module("route{0}".format(index), route)
            
            #getting the total filters out of this routing layer
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
                
        
        elif(block['type'] == 'upsample'):
            stride = int(block["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module("upsample{0}".format(index), upsample)
            
            
        elif(block['type'] == "yolo"):
            mask = block["mask"].split(',')
            mask = (int(m) for m in mask)
            anchors = block["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
            
            detection = DetectionLayer(anchors)
            module.add_module("detectionlayer{0}".format(index), detection)
            
            
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        
    return (net_info, module_list)

class Darknet(nn.Module):
    def __init__(self, cfg_file):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg_file)
        self.info, self.module_list = create_modules(self.blocks)
        
    def load_weights(self, file_name):
        file = open(file_name, 'rb')
        self.header = torch.from_numpy(np.fromfile(file, np.int32, 5))
        self.seen = self.header[3]
        weights = np.fromfile(file, np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"]
            if module_type == "convolutional":
                model = self.module_list[i]
                
                conv = model[0]
                if 'batch_normalize' in self.blocks[i+1]:
                    bn = model[1]

                    num_bn_biases = bn.bias.numel()   #store number of biases in this batch normalization

                    #load the weights and biases of this batch normalization into variables
                    bn_biases = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_biases = bn_biases.view(*bn.bias.shape)
                    bn_weights = bn_weights.view(*bn.weight.shape)
                    bn_running_mean = bn_running_mean.view(*bn.running_mean.shape)
                    bn_running_var = bn_running_var.view(*bn.running_var.shape)

                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.data.copy_(bn_running_mean)
                    bn.running_var.data.copy_(bn_running_var)

                else:
                    print(self.blocks[i+1])
                    num_conv_biases = conv.bias.numel()
                    conv_biases = torch.from_numpy(weights[ptr:ptr+num_conv_biases])
                    ptr += num_conv_biases

                    conv_biases = conv_biases.view(*conv.bias.shape)

                    conv.bias.data.copy_(conv_biases)


                num_conv_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_conv_weights])
                ptr += num_conv_weights
                
                conv_weights = conv_weights.view(*conv.weight.shape)
                conv.weight.data.copy_(conv_weights)

        
    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}   #stores the output of each layer to perform routing
        
        write = 0
        for i, module in enumerate(modules):
            if (module["type"] == "convolutional" or module["type"] == "upsample"):
                x = self.module_list[i](x)
                
                
            elif (module["type"] == "route"):
                layers = module["layers"]
                layers = [int(l) for l in layers]
                
                if (layers[0] > 0):
                    layers[0] -= i
                    
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                    
                else:
                    if (layers[1]) > 0:
                        layers [1] -= i
                        
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1 )   #concatenate the feature maps of the referenced layers
                
            elif(module["type"] == "shortcut"):
                layer = int(module["from"])
                x = outputs[i-1] + outputs[i + layer] #add the feature maps of the referenced layers
                
            elif(module["type"] == "yolo"):
                anchors = self.module_list[i][0].anchors
                input_dim = int(self.info["height"])
                num_classes = int(module["classes"])
                
                
                x = x.data
                
                x = predict_transform(x, input_dim, anchors, num_classes, CUDA)
                if not write:              
                    detections = x
                    write = 1

                else:       
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x
        
        return detections
    
    
def get_test_input(height, width):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (height,width))          #Resize to the model input dimensions
    img_ =  img[:,:,::-1].transpose((2,0,1))  # convert BGR to RGB
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) and Normalise
    img_ = torch.from_numpy(img_).float()     
    img_ = Variable(img_)                     
    return img_