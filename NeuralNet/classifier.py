from checkpoint import CheckpointManager
from param_config import JsonManager, Parameters
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torchvision import transforms
from extraction import FeatureExtractor
import wandb
import sys, getopt, random
from dataset import UnNormalize
from numpy import linalg as LA
from models import CNN_1d

MODELS_PATH = f".model"
#DESCPATH = f"{DATASETPATH}/descriptors"
MODELS_PARAMETERS_PATH = ".model/models.json"

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def get_wandb_video(clip, y_pred, y_target, class_names):
    unnorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    clip = clip.cpu().detach()
    clip = (unnorm(clip)*255).byte()
    clip = clip.numpy()
    return wandb.Video(clip, format='gif', caption=f"{class_names[y_pred]}/{class_names[y_target]}")


class Classifier:
    """Classifier composed by feature extractor and classification model.

    Args:
            num_classes (:class:`int`): Total number of classes.

            parameters (:class:`Parameters`): Object containing parameters for the model and training.

            device (:class:`str`): Torch device.
    """
    def __init__(self, num_classes, parameters:Parameters, device):
        print(device)
        self._device = device
        self._avg_pool_extr = parameters.avg_pool
        self._extractor = FeatureExtractor(self._avg_pool_extr).to(device)
        model_class = str_to_class(parameters.model)
        print(type(model_class))
        self._model = model_class(num_classes, avg_pool_extr=self._avg_pool_extr).to(device)
        self._parameters = parameters


    def fit(self, train_loader, dev_loader, log_interval = 1, force_training = False, params_path = MODELS_PARAMETERS_PATH):
        """Fits the model using the given training loader.
        
        If :param:`force_training` is False, checks if exists a checkpoint with the current parameters and loads it.

        Log the train and validation results using wandb. 

        Args:
            train_loader (:class:`DataLoader`): DataLoader with the training data
            dev_loader (:class:`DataLoader`): DataLoader with the validation data
            log_interval (:class:`int`, optional): Specifies how many iterations you need to log the loss function. Defaults to 1.
            force_training (:class:`bool`, optional): If True force the training. Defaults to False.
            params_path (:class:`str`, optional): Path of json file containing informations about the runs configurations and checkpoints. Defaults to MODELS_PARAMETERS_PATH.

        Returns:
        current_epoch, ck_manager.best_epoch, ck_manager.max_accuracy
            :class:`int`: current epoch, 
            :class:`int`: best epoch, 
            :class:`float`: best accuracy
        """
        j_manager = JsonManager(params_path)
        resume = False
        if j_manager.exists(self._parameters):
            self._parameters = j_manager.find(self._parameters)
            resume = True
        else:
            j_manager.save(self._parameters)
        
        code = str(self._parameters.get_code())
        
        print(self._parameters)
        path = f"{MODELS_PATH}/{code}.pth"
        print(path)
        optimizer = optim.Adam(self._model.parameters(), lr=self._parameters.learning_rate, weight_decay=self._parameters.weight_decay)
        
        criterion = nn.CrossEntropyLoss()
        current_epoch = 0
        ended = False

        ck_manager = CheckpointManager(path)
        

        if not force_training and resume:
            ck_manager.load()
            if ck_manager.load():
                print("checkpoint loaded")
                current_epoch = ck_manager.current_epoch
                ended = ck_manager.is_ended
                print(f"ended: {ended}")
            else:
                print("load failed")
                resume = False
        else:
            resume = False
        
        print(f"resume: {resume}")
        wandb.init(project='ProjectCVCS', entity='project-cvcs', id=code, resume=resume)
        wandb.config = self._parameters.to_dict()
        if resume and not ended:
            self._model.load_state_dict(ck_manager.current_model_state_dict)
            optimizer.load_state_dict(ck_manager.current_optimizer_state_dict)
        else:
            wandb.run.name = self._parameters.short_str()

        wandb.run.save()


        if ended:
            print("TRAINING ENDED")
            self._model.load_state_dict(ck_manager.best_model_state_dict)
            return current_epoch, ck_manager.best_epoch, ck_manager.max_accuracy
        
        print('Training')
        self._model.train()
        self._extractor.eval()
        wandb.watch(self._model)

        for epoch in range(current_epoch + 1, self._parameters.max_epochs + 1):
            if epoch - ck_manager.best_epoch > self._parameters.patience:
                break 
            for batch_idx, (data, target) in enumerate(train_loader):                
                data, target = data.to(self._device), target.to(self._device)
                with torch.no_grad():
                    features = self._extractor(data)
                optimizer.zero_grad()  # zero the gradient buffers
                outputs = self._model(features)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()  # Does the update
                if batch_idx % log_interval == 0:
                    #print(f'Train Epoch: {epoch+1}, idx:{batch_idx} \tLoss: {loss.item()}')
                    wandb.log({"loss": loss, "epoch": epoch})

            with torch.no_grad():
                print("Train ", end="")
                _,_, accuracy_train = self.eval(train_loader)
                print("Eval ", end="")
                _,_, accuracy_eval = self.eval(dev_loader)
                
                ck_manager.save(self._model, optimizer, accuracy_eval, epoch)
                wandb.log({f"Accuracy train" : accuracy_train, "epoch": epoch})
                wandb.log({f"Accuracy eval" : accuracy_eval, "epoch": epoch})
        
        ck_manager.ended()
        self._model.load_state_dict(ck_manager.best_model_state_dict)
        return epoch - 1, ck_manager.best_epoch, ck_manager.max_accuracy


    def eval(self, test_loader, plot_videos = False, plot_confusion_matrix = False, class_names=None, desc=""):
        """Evals a batch of data.

        Args:
            test_loader (:class:`DataLoader`): DataLoader containing the data to eval
            plot_videos (:class:`bool`, optional): If True plots some videos in wandb console. Defaults to False.
            plot_confusion_matrix (:class:`bool`, optional): If True plots confusion matrix in wandb console. Defaults to False.
            class_names (list of strings | None, optional): List containing the class names. Defaults to None.
            desc (str, optional): String to display in some messages/log. Defaults to "".

        Returns:
            :class:`int`: number of correct predictions,
            :class:`int`: predicted classes,
            :class:`float`: accuracy

        """
        print("eval")
        self._model.eval()
        self._extractor.eval()
        correct = 0
        predictions = []
        y_true = []
        clips = []
        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self._device), target.to(self._device)
                features = self._extractor(data)
                output = self._model(features)    
                pred = output.data.max(1, keepdim=True)[1]
                pred_unsq = torch.squeeze(pred, dim = 1)
                predictions.append(pred_unsq)
                target_unsq = target.data.view_as(pred_unsq)
                y_true.append(target_unsq)
                correct += pred.eq(target.data.view_as(pred)).sum()
                if plot_videos and idx <= 10:
                    for clip, y_pred, y_target in zip(data, pred_unsq, target_unsq):
                        video = get_wandb_video(clip, y_pred, y_target, class_names)
                        clips.append(video)
       
        predictions = torch.cat(predictions)
        y_true = torch.cat(y_true)

        accuracy =   100. * correct / len(test_loader.dataset)
        print('{}: {}/{} ({:.0f}%)\n'.format("accuracy", correct, len(test_loader.dataset),accuracy))
        if plot_confusion_matrix:
            wandb.log({"conf_mat_{desc}" : wandb.plot.confusion_matrix(probs=None, y_true=y_true.cpu().detach().numpy(), preds=predictions.cpu().detach().numpy(), class_names=class_names)})

        if plot_videos:
            wandb.log({f"videos_{desc}":clips})
        
        return correct, predictions.int(), accuracy
    
    def extract_descriptors(self, loader, descpath):
        """Extract descriptors from the last but one fully connected layer.

        Args:
            loader (:class:`DataLoader`:): DataLoader containing the data to extract
            descpath (:class:`str`): Path where we save extracted descriptor
        """
        print("Descriptors")
        self._model.eval()
        self._extractor.eval()
        descriptors = list()
        labels = list()

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self._device), target.to(self._device)
                features = self._extractor(data)
                desc = self._model(features, returnDesc = True)
                desc = desc.cpu().detach().numpy()
                descriptors.append(desc)
                labels.append(target)
           
           
            descriptors = np.array(descriptors)
            descriptors = np.concatenate(descriptors, axis=0)
            np.save(f"{descpath}/descriptors.npy", descriptors)
            print(descriptors.shape)
            labels = np.array(labels)
            np.save(f"{descpath}/labels.npy", labels) 
                
