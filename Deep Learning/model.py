import torch 
from torch import nn, optim
import torchvision 
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image 

def training(arch, hidden_units, learning_rate, epochs, save_dir, train_loader, valid_loader, class_to_idx):
    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    criterion = nn.NLLLoss()
    
    model = eval('torchvision.models.' + arch + '(pretrained = True)')
    for param in model.parameters():
        param.requires_grad = False
  
    if arch=='vgg16':
        in_features = model.classifier[0].in_features
        ftr = 'features'
        model.classifier = nn.Sequential(nn.Linear(in_features, hidden_units),nn.ReLU(),nn.Dropout(p=0.30),nn.Linear(hidden_units, 102),nn.LogSoftmax(dim=1))
        optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate, weight_decay = 1e-5)
    elif arch=='densenet121':
        in_features = model.classifier.in_features
        ftr = 'features'
        model.classifier = nn.Sequential(nn.Linear(in_features, hidden_units),nn.ReLU(),nn.Dropout(p=0.30),nn.Linear(hidden_units, 102),nn.LogSoftmax(dim=1))
        optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate, weight_decay = 1e-5)
    elif arch =='resnet18':
        in_features = model.fc.in_features
        ftr = 'fc'
        model.fc = nn.Sequential(nn.Linear(in_features, hidden_units),nn.ReLU(),nn.Dropout(p=0.30),nn.Linear(hidden_units, 102),nn.LogSoftmax(dim=1))
        optimizer = optim.Adam(model.fc.parameters(), lr = learning_rate, weight_decay = 1e-5)


    #Training the model
    model.to(device);

    steps = 0
    running_loss = 0 
    print_every = 5
    max_accuracy = 0

    for epoch in range(epochs):
        for images, labels in train_loader:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            #print(labels.shape)
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                val_loss = 0
                accuracy = 0

                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)

                    logps = model(images)
                    loss = criterion(logps, labels)
                    val_loss += loss.item()

                    ps = torch.exp(logps)
                    #print(ps)
                    top_ps, top_class = ps.topk(1, dim =1)
                    equality = top_class==labels.view(*top_class.shape)
                    #print(equality, top_class, labels)
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.."
                      f"Valid loss: {val_loss/len(valid_loader):.3f}.."
                      f"Valid accuracy: {accuracy/len(valid_loader):.3f}")

                if steps == 5:
                    max_accuracy =accuracy/len(valid_loader)
                else: 
                    if accuracy/len(valid_loader) >= max_accuracy:
                        max_accuracy = accuracy/len(valid_loader)
                        checkpoint = {'epoch': epoch + 1,
                                    'arch': arch,
                                    'state_dict': model.state_dict(),
                                    'optimizer' : optimizer.state_dict(),
                                    'hidden_units': hidden_units,
                                    'class_to_idx': class_to_idx,
                                    'optim_dict': optimizer.state_dict(),
                                    'losslogger': running_loss
                                }
                        if arch =='resnet18':
                            checkpoint['classifier'] = model.fc
                        else:
                            checkpoint['classifier'] = model.classifier
                        torch.save(checkpoint, save_dir + str(arch) + '_checkpoint.pth')
                        print(checkpoint.keys())
                        print('.........Checkpoints saved........')

                running_loss = 0
                model.train()
    print("Training is over")
    return model

def predict(image_path, model, topk=5, gpu=False, data_dir ='/home/workspace/aipnd-project/flowers/train/'):
    
    def process_image(image):
        img_loader = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
        image = Image.open(image)
        img = img_loader(image).float()
        np_image = np.array(img)  
        return np_image

    img = process_image(image_path)
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    image_tensor.resize_([1, 3, 224, 224])
    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    if gpu == True:
        device = 'cuda'
        print('Using GPU to predict..', device)
        model.to(device)
        image_tensor = image_tensor.to(device)
    elif gpu is False:
        device = 'cpu'
        print('Using CPU to predict..', device)
       
    model.eval()
    
    ps = torch.exp(model(image_tensor))
    probs, top_class = ps.topk(k = topk, dim =1)
    top_class = list(top_class.cpu().numpy())

    #getting class_to_idx from train_datasets
    train_datasets = datasets.ImageFolder(data_dir, transform= transforms.Compose([transforms.Resize(255)]))
    idx_to_class = dict([[v,k] for k,v in train_datasets.class_to_idx.items()])
    
    classes = []
    for k in np.nditer(top_class):
        for i,v in idx_to_class.items():
            if k == i:
                classes.append(v)
    probs = np.squeeze(probs.detach().cpu().numpy())
    
    return probs, classes 