import torch
import torch.nn as nn
import torch.nn.functional as F

FC_INPUT = 21728
FC_INPUT_AVGPOOL = 4224

class CNN_1d(nn.Module):

    def __init__(self, num_classes, avg_pool_extr=False):
        super(CNN_1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5)
        #4224 avg pool
        #21728 no pool
        if avg_pool_extr:
            self.fc1 = nn.Linear(FC_INPUT_AVGPOOL, 2048)
        else:
            self.fc1 = nn.Linear(FC_INPUT, 2048)
        
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.pool = nn.AvgPool1d(4, stride = 4)

        
    def forward(self, x, returnDesc = False):
        """Classic forward operations but with the option of extracting the descriptors. 

        Args:
            x (:class:`torch.Tensor`): Tensor data
            returnDesc (bool, optional): If True extract the descriptors before the last fully connected layer. Defaults to False.

        Returns:
            :class:`torch.Tensor`: Tensor of shape (n_batch, num_classes) if returnDesc = False, otherwise tensor of shape (n_batch, 512) 
        """
        x = F.relu(self.conv1(x))
        print(f"Shape after conv1: {x.shape}")
        x = F.relu(self.conv2(x))
        print(f"Shape after conv2: {x.shape}")
        x = F.relu(self.conv3(x))
        print(f"Shape after conv3: {x.shape}")
        
        bs, _, _ = x.shape
        x = self.pool(x).reshape(bs, -1)
        print(f"5.Shape before fully connected:{x.shape}")
        
        x = F.relu(self.fc1(x))
        y = F.relu(self.fc2(x))
        if returnDesc:
            return y
        x = self.fc3(y)
        return x

