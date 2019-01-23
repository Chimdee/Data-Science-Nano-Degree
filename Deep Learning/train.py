import argparse
from model_utils import load_data
from model import training
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", default = "/home/workspace/aipnd-project/flowers/", help = "Data directory")
    parser.add_argument("--save_dir", default = "/home/workspace/ImageClassifier/", help = "Saving Directory")
    parser.add_argument("--arch",  default = "densenet121", choices = ['densenet121', 'vgg16', 'resnet18'],
                        type = str, help = "Model Architecture")
    parser.add_argument("--learning_rate", default = 0.005, type= float,  help = "Learning Rate")
    parser.add_argument("--hidden_units", default = 512, type= int, help = "Hidden units")
    parser.add_argument("--epochs", default = 10, type = int,  help = "Epochs")
    parser.add_argument("--gpu", action='store_true', default=False, help = "GPU")
    
    args = parser.parse_args()
    
    print('---------Parameters----------')
    print('gpu              = {!r}'.format(args.gpu))
    print('epoch(s)         = {!r}'.format(args.epochs))
    print('arch             = {!r}'.format(args.arch))
    print('learning_rate    = {!r}'.format(args.learning_rate))
    print('hidden_units     = {!r}'.format(args.hidden_units))
    print('-----------------------------')
    
    train_loader, valid_loader, _, class_to_idx = load_data(args.data_dir)
    best_model = training(args.arch, args.hidden_units, args.learning_rate, args.epochs, args.save_dir, train_loader, valid_loader, class_to_idx)
    
   