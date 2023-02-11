import torch 
from torch import nn, optim 
from torchvision import datasets , transforms  , models 
from collections import OrderedDict


# this our function orgnize the model in propre way 


def loaed_model(hidden_units=512,dropout=0.4,structure='vgg16', types='gpu', lr=0.001):
    
    if structure == 'vgg16':
         model = models.vgg16(pretrained= True )
    else : 
        model = models.densenet121(pretrained= True )
        
        
    for param in model.parameters():
      param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088,hidden_units)),
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout(p=dropout)),
                            ('fc3', nn.Linear(hidden_units,102)),
                            ('13' ,nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    
    
 
    #define our criterion and optimizer 
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters() , lr = lr )
    
    
    
    
    
    if  torch.cuda.is_available() and types == 'gpu' :
        
        device = torch.device("cuda:0")
    else : 
        device = torch.device("cpu")
        
    model.to(device)
    
    return model , criterion , optimizer 