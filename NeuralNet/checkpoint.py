import torch


class CheckpointManager:
    """Manager of models checkpoints.

    Args:
        path (:class:`str`): The path of the checkpoint file
    """

    def __init__(self, path):
        self.path = path

        self._max_accuracy = 0
        self._best_epoch = 0
        self._best_model_state = None
        self._best_optimizer_state = None
        self._current_model_state = None
        self._current_optimizer_state = None
        self._current_epoch = 0
        self._ended = False

    def _copy_dict(self, dict_to_copy):
        d = dict_to_copy.copy()
        return d

    def __save(self):
        torch.save({
            'epoch': self._current_epoch ,
            'model_state_dict': self._best_model_state,
            'optimizer_state_dict': self._best_optimizer_state,
            'current_model_state': self._current_model_state,
            'current_optimizer_state': self._current_optimizer_state,
            'max_accuracy': self._max_accuracy,
            'best_epoch': self._best_epoch,
            'ended': self._ended,
            }, self.path)


    def save(self, model, optimizer, accuracy, epoch):
        """Saves checkpoint information of the running model on the filesystem.
        """
        self._current_epoch = epoch
        self._current_model_state = self._copy_dict(model.state_dict())
        self._current_optimizer_state = self._copy_dict(optimizer.state_dict())
        if self._max_accuracy <= accuracy:
            self._best_model_state = self._copy_dict(model.state_dict())
            self._best_optimizer_state = self._copy_dict(optimizer.state_dict())
            self._max_accuracy = accuracy
            self._best_epoch = epoch
            self._ended = False

        self.__save()


    def ended(self):
        """Set ended state to True and save it on the filesystem.
        """
        self._ended = True
        self.__save()

    def load(self) -> bool:
        """Loads checkpoint from the filesystem at :params:`path`

        Returns:
            bool: Checkpoint loaded successfully.
        """
        try:
            checkpoint = torch.load(self.path)
            self._max_accuracy =  checkpoint['max_accuracy']
            self._best_model_state =  checkpoint['model_state_dict']
            self._best_optimizer_state =  checkpoint['optimizer_state_dict']
            self._best_epoch = checkpoint['best_epoch']
            self._current_epoch = checkpoint['epoch']
            self._ended = checkpoint['ended']
            self._current_model_state = checkpoint['current_model_state']
            self._current_optimizer_state = checkpoint['current_optimizer_state']
            return True
        except FileNotFoundError as e:
            return False

    @property
    def max_accuracy(self):
        return self._max_accuracy

    @property
    def best_epoch(self):
        return self._best_epoch

    @property
    def best_model_state_dict(self):
        return self._best_model_state

    @property
    def best_optimizer_state_dict(self):
        return self._best_optimizer_state

    @property
    def current_model_state_dict(self):
        return self._current_model_state

    @property
    def current_optimizer_state_dict(self):
        return self._current_optimizer_state

    @property
    def is_ended(self):
        return self._ended
    
    @property
    def current_epoch(self):
        return self._current_epoch
