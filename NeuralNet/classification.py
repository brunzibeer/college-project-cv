from param_config import Parameters
import torch
import numpy as np
from torchvision import transforms
import wandb
import sys, getopt, random
from dataset import BigDataset, splitDataset, createDataLoader
from numpy import linalg as LA
from classifier import Classifier

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


if torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'

DEVICE = 'cpu'

DATASETPATH = "dataset"
DESCPATH = f"{DATASETPATH}/descriptors"
#DATASETPATH = ".data/clips"

CLASS_NAMES = ["Corner", "Foul",  "Goal", "Nothing"]



if __name__ == '__main__':

    try:
        opts, args = getopt.getopt(sys.argv[1:], "l:w:p:a:e")
    except getopt.GetoptError:
        print("Usage: classification.py [-l learning_rate(float)  | -w weight_decay(float) | -p patience(int) | -a avgpool (0 or 1) | -e]")
        sys.exit(2)

    params = Parameters()

    extract_descriptors = False
    
    for opt, val in opts:
        if opt == "-l":
            try:
                val = float(val)
                params.learning_rate = val
            except ValueError:
                print("Error!! learning rate must be float")
                sys.exit(1)

        if opt == "-w":
            try:
                val = float(val)
                params.weight_decay = val
            except ValueError:
                print("Error!! weight_decay must be float")
                sys.exit(1)

        if opt == "-p":
            try:
                val = int(val)
                params.patience = val
            except ValueError:
                print("Error!! patience must be int")
                sys.exit(1)
               
        if opt == "-a":
            if val:
                try:
                    val = bool(int(val))
                    params.avg_pool = val
                except ValueError:
                    print("Error!! avgpool must be 0 or 1")
                    sys.exit(1)
    
        if opt == "-e":
            extract_descriptors = True


    print(params)
    
    labels = torch.load(f"{DATASETPATH}/labels.pt")
    print(labels.shape)
    transformations = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])

    dataset = BigDataset(labels, DATASETPATH, transform =transformations)
    
    num_classes = len(labels.unique())

    train, dev, test = splitDataset(dataset)
    
    train_loader, dev_loader, test_loader = createDataLoader(train), createDataLoader(dev), createDataLoader(test)
    print(f"Lunghezza train loader: {len(train_loader.dataset)}")
    print(f"Lunghezza test loader: {len(test_loader.dataset)}")
    
    classifier = Classifier(num_classes, params, DEVICE)
    epochs, best_epoch, max_accuracy = classifier.fit(train_loader, dev_loader)
    
    
    print(f"EPOCHE TRAINING: {epochs}, MIGLIOR EPOCA: {best_epoch}, MAX ACCURACY:{max_accuracy}")

    if not extract_descriptors:
        print("================================EVAL================================")
        _,_,accuracy_eval = classifier.eval(dev_loader, plot_confusion_matrix=True, class_names = CLASS_NAMES, desc="eval")
        wandb.log({"final_accuracy_eval":accuracy_eval})

        print("================================TESTING================================")
        _,_,accuracy_test = classifier.eval(test_loader, plot_confusion_matrix=True, plot_videos = True, class_names = CLASS_NAMES, desc="test")
        wandb.log({"final_accuracy_test":accuracy_test})

    if extract_descriptors:
        print("================================DESCRIPTORS================================")
        # Unione dataset con loader
        classifier.extract_descriptors(test_loader, DESCPATH)#newloader

