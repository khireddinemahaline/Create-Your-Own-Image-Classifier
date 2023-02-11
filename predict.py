import numpy as np
import argparse
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict
import matplotlib.pyplot as plt
import json
from PIL import Image
import proceesimage



praser = argparse.ArgumentParser(description = 'how to use preiction file ' )

praser.add_argument('input' , action='store' , default = './flowers/test/1/image_06752.jpg', type = str , nargs = '?')
praser.add_argument('--dir' , action='store' ,dest = 'data_dir' ,default = './flowers/')
praser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
praser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
praser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
praser.add_argument('--gpu', default="gpu", action="store", dest="gpu")



args = praser.parse_args()


path_image = args.input
number_of_classes  = args.top_k
device = args.gpu
classe_name = args.category_names
path = args.checkpoint



    
model = proceesimage.load_checkpoint(path ='checkpoint.pth')
with open(classe_name, 'r') as f:
   classes = json.load(f)
prop = proceesimage.predict(path_image, model, number_of_classes ,device='gpu')
    
    
probability = np.array(prop[0][0])
labels = [classes[str(index + 1)] for index in np.array(prop[1][0])]
    
i = 0
while i < number_of_classes:
    print("{} with a probability of {}".format(labels[i], probability[i]))
    i += 1
print("Finished Predicting!")
    
     
    


    
