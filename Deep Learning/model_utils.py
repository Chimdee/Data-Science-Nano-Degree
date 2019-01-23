from torchvision import datasets, transforms, models
import torch
from torch import nn


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.Resize(255),
                                       #transforms.CenterCrop(224),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomRotation(25),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=valid_transforms)

    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle =True)
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle =True)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=1, shuffle =True)
    
    class_to_idx = train_datasets.class_to_idx,
    
    return train_loader, valid_loader, test_loader, class_to_idx


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    model = eval('models.' + arch + '(pretrained = False)')
    #model.classifier = checkpoint['classifier']
    if arch=='vgg16':
        in_features = model.classifier[0].in_features
        ftr = 'features'
        model.classifier = nn.Sequential(nn.Linear(in_features, hidden_units),nn.ReLU(),nn.Dropout(p=0.30),nn.Linear(hidden_units, 102),nn.LogSoftmax(dim=1))
    elif arch=='densenet121':
        in_features = model.classifier.in_features
        ftr = 'features'
        model.classifier = nn.Sequential(nn.Linear(in_features, hidden_units),nn.ReLU(),nn.Dropout(p=0.30),nn.Linear(hidden_units, 102),nn.LogSoftmax(dim=1))
    elif arch =='resnet18':
        in_features = model.fc.in_features
        ftr = 'fc'
        model.fc = nn.Sequential(nn.Linear(in_features, hidden_units),nn.ReLU(),nn.Dropout(p=0.30),nn.Linear(hidden_units, 102),nn.LogSoftmax(dim=1))
    
    model.load_state_dict(checkpoint['state_dict'])
    return model
