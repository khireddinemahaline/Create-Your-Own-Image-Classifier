import torch 
from torch import nn, optim 
from torchvision import datasets , transforms  , models 
from collections import OrderedDict
import json 
# our data we going to use in the train and test the model 

def data(root = './flowers/'):
    data_dir = root
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
  

    data_transforms = {
    "traininig" : transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomAffine(degrees=0, translate=(0, 0), scale=(0.9, 1.1), shear=0),
                                     transforms.Resize(224,224),
                                     transforms.RandomHorizontalFlip(30),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225]) ]),
      "testing" : transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomAffine(degrees=0, translate=(0, 0), scale=(0.9, 1.1), shear=0),
                                     transforms.Resize(224,224),
                                     transforms.RandomHorizontalFlip(30),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225]) ]),
      "validation" : transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomAffine(degrees=0, translate=(0, 0), scale=(0.9, 1.1), shear=0),
                                     transforms.Resize(224,224),
                                     transforms.RandomHorizontalFlip(30),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225]) ])
                  }
# TODO: Load the datasets with ImageFolder
    image_datasets ={ 
        "traininig" : datasets.ImageFolder(train_dir , transform = data_transforms["traininig"] ),
         "testing" : datasets.ImageFolder(test_dir , transform = data_transforms["testing"] ), 
        "validation" :datasets.ImageFolder(valid_dir , transform = data_transforms["validation"] )
}
# TODO: Using the image datasets and the trainforms, define the dataloaders

    dataloaders = torch.utils.data.DataLoader(image_datasets["traininig"] , batch_size=64, shuffle=True )
    testloader = torch.utils.data.DataLoader(image_datasets[ "testing"], batch_size=64,shuffle=False)
    validationloader = torch.utils.data.DataLoader(image_datasets["validation"], batch_size=64,shuffle=True)
    
    
    training_data = image_datasets['traininig'] 
    
    
    return dataloaders , testloader , validationloader , training_data


