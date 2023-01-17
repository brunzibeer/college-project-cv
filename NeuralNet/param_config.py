import json
import uuid

MODEL = "CNN_1d"
LR = 0.00001
DROPOUT = 0
WEIGHT_DECAY = 0
PATIENCE = 5
AVGPOOL = False
DATA_AUGMENTATION = False
GOAL_WEIGHTS = 1
MAX_EPOCHS = 40


class Parameters:
    """Class that saves all the possbile parameters of the models.

    Provide :method:`__eq__` method to check if model with given parameters is already trained.
    """

    def __init__(self, p : dict = {} ):
        self.model = p.get("model", MODEL)
        self.learning_rate = p.get("learning_rate", LR)
        self.patience = p.get("patience", PATIENCE)
        self.max_epochs = p.get("max_epochs", MAX_EPOCHS)
        self.weight_decay =  p.get("weight_decay", WEIGHT_DECAY)
        self.dropout = p.get("dropout", DROPOUT)
        self.avg_pool = p.get("avg_pool", AVGPOOL)
        self.data_augmentation = p.get("data_augmentation", DATA_AUGMENTATION)
        self.goal_weigths = p.get("goal_weigths", GOAL_WEIGHTS)
        self.__code = p.get("_Parameters__code", str(uuid.uuid4()).replace("-", ""))

    def __eq__(self, other):
        if not isinstance(other, Parameters):
            return False
        return all([other.__dict__[k]==v for k, v in self.__dict__.items() if k!="_Parameters__code"])

    def to_dict(self):
        d = self.__dict__.copy()

    def get_code(self):
        return self.__code

    def __str__(self):
        return str(self.__dict__)

    def short_str(self):
        avgpool = "avgpool2" if self.avg_pool else "noavgpool"
        return f"aug_{avgpool}_lr={self.learning_rate}_dr={self.dropout}_wd={self.weight_decay}_pat={self.patience}"


class JsonManager:
    """Utils class that provides base functionality to work with Json files.
    """
    def __init__(self, path):
        self._path = path
        try: 
            with open(self._path) as f:
                self._json = json.load(f)
                assert isinstance(self._json, list)
        except IOError:
            self._json = []

    def exists(self, params:Parameters) -> bool:
        return self.find(params) is not None

    def find(self, params:Parameters)->Parameters:
        for el in self._json:
            par = Parameters(el)
            if par == params:
                return par 
        return None

    def save(self, params:Parameters):
        if not self.exists(params):
            old_json = self._json.copy()
            self._json.append(params.__dict__)

        with open(self._path, 'w') as fp:
            try:
                json.dump(self._json, fp)
            except TypeError:
                self._json = old_json
                json.dump(self._json)
