import argparse
import json
from model_utils import load_data, load_checkpoint
from model import training, predict
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", default = "/home/workspace/aipnd-project/flowers/test/1/image_06754.jpg", help = "Test image path")
    parser.add_argument("checkpoint_path", default = "/home/workspace/ImageClassifier/dense121_checkpoint.pth", help = "Trained model checkpoint")
    parser.add_argument("--top_k",  default = 5, type = int, help = "Top k probable categories")
    #parser.add_argument("--hidden_units", default = 512, type= int, help = "Hidden units")
    parser.add_argument("--category_names", default = 'cat_to_name.json', type= str, help = "Category to names file")
    parser.add_argument("--gpu", action='store_true', default=False, help = "GPU")
    
    args = parser.parse_args()
    
    print('\n---------Parameters----------')
    print('gpu              = {!r}'.format(args.gpu))
    print('top_k            = {!r}'.format(args.top_k))
    print('arch             = {!r}'.format(args.checkpoint_path.split('/')[-1].split('_')[0]))
    print('Checkpoint       = {!r}'.format(args.checkpoint_path))
    print('-----------------------------\n')
    
    #Prediction
    model = load_checkpoint(args.checkpoint_path)
    probs, classes = predict(args.image_path, model, args.top_k, args.gpu)
   
    with open('/home/workspace/aipnd-project/cat_to_name.json', 'r') as f:
        args.cat_to_name = json.load(f)
    names = []
    for i in classes:
        names.append(args.cat_to_name[str(i)])
    
    print("Probabailities of Top {!r} flowers: ".format(args.top_k),  probs)
    print("Names of Top {!r} flowers: ".format(args.top_k), names)
    print("")