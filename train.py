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

#use argparse to make inputs 
parser = argparse.ArgumentParser( description = ' parse for train model  ')


parser.add_argument('--data_dir' , action='store',  default="./flowers/")
parser.add_argument("--save_dir",action= 'store' , default="./checkpoint.pth")
parser.add_argument("--arch" , action ='store' , default='vgg16' )
parser.add_argument("--learning_rate",action= 'store' , type=float, default='0.001')
parser.add_argument("--hidden_units",action= 'store' ,type= int, default='502')
parser.add_argument("--epochs",action= 'store' , type=int,default='10')
parser.add_argument("--gpu",action= 'store' , default='gpu')
parser.add_argument("--dropout",action= 'store' ,type=float, default='0.4')


args = parser.parse_args()

#define our inputs var

place = args.data_dir
path = args.save_dir
structure = args.arch
lr = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
dropout = args.dropout
types = args.gpu 

# the main function to train our model 
if  torch.cuda.is_available() and types == 'gpu':
        
        device = torch.device("cuda:0")
else : 
        device = torch.device("cpu")

def main():
   # device is to train our model in gpu by default  

   dataloaders , testloader , validationloader ,training_data  = datasets.data(place)
   # call our model from the  loaed_model file
   model , criterion , optimizer  = loaed_model.loaed_model(hidden_units ,dropout, structure , types , lr   )
    
   
   #start tarining 
   model.train()
   print_every = 40
   steps = 0
   for e in range(epochs) :
        running_loss = 0 
        for images , labels in dataloaders :
            if types == 'gpu' and torch.cuda.is_available(): 
                 images , labels = images.to(device) ,  labels.to(device)
                 model = model.to(device)
                    
            steps += 1
            optimizer.zero_grad()
            logs = model.forward(images)
            loss = criterion(logs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0 :
                test_loss = 0 
                accuracy = 0 
                model.eval() # evalute the model and turn off the grad for fast and best result 
                with torch.no_grad():
                    for images  , labels in validationloader:
                        
                        images , labels = images.to(device) ,  labels.to(device)

                        log_los = model(images)
                        test_loss += criterion(log_los,labels).item()


                        # calculate the accurcy

                        ps = torch.exp(log_los)
                        top_p , top_class = ps.topk(1,dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                # print the output 
                print("epoc {}/{}".format(e+1,epochs),
                      "train_loss is {}".format(running_loss/len(dataloaders)),
                      "test_loss is {}".format(test_loss/ len(validationloader)),
                     "accurcy {}%".format(accuracy*100/len(validationloader)))
            
    # save model parameters 
   model.class_to_idx=  training_data.class_to_idx
# save model parameters 
   state = {
    'input_size': 25088,
    'output_size': 102,
    'structure': structure,
    'learning_rate' : lr , 
    'epocs' : epochs ,
    'dropout' : dropout, 
       
    'hidden_units' : hidden_units ,
    'state_dict': model.state_dict(),
    'classifier': model.classifier,
    'optimizer': optimizer.state_dict(),
    'class_to_idx' : model.class_to_idx
}
# save ALL toghether
   torch.save(state,path)
        

if __name__== "__main__":
    main()
