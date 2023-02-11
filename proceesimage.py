# import what we need 

import matplotlib.pyplot as plt 
import torch 
from torch import nn, optim 
from torchvision import datasets , transforms  , models 
from collections import OrderedDict
from PIL import Image
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import json 
import argparse
import loaed_model
import datasets 
from PIL import Image




def load_checkpoint(path = 'checkpoint.pth'):
    state = torch.load(path)
    lr = state['learning_rate']
    structure  = state['structure']
    input_size = state['input_size']
    output_size = state['output_size']  
    dropout = state['dropout']
    hidden_units = state['hidden_units']
  
    
    model , criterion , optimizer = loaed_model.loaed_model(hidden_units ,dropout, structure  , lr   )
    
    model.class_to_idx = state['class_to_idx']
    model.load_state_dict(state['state_dict'])
 
    
    
    
    return model



def predict(image_path, model, topk=5, device='gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    if device == 'gpu' and torch.cuda.is_available():
        
        device = torch.device("cuda:0")
    else : 
        device = torch.device("cpu")  

        
    img = process_image(image_path).numpy()  
    img = torch.from_numpy(np.array([img])).float()
   

    model.to(device)
    model.eval()     
    with torch.no_grad():
            logps = model.forward(img.to(device))
            probability = torch.exp(logps).data
   

            
    return  probability.topk(topk)
    
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image_pil = Image.open(image)
    image_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    image = image_transform(image_pil)
    return image 
    