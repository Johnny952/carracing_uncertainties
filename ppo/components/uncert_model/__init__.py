from .basic_model import BaseTrainerModel
from .dropout_model import DropoutTrainerModel
from .bootstrap_model import BootstrapTrainerModel
from .sensitivity_model import SensitivityTrainerModel
from .bnn_model import BNNTrainerModel
from .aleatoric_model import AleatoricTrainerModel
from .custom_model import CustomTrainerModel

def make_model(nb_nets, lr, img_stack, gamma, batch_size, buffer_capacity, device='cpu', model='base'):
    switcher = {
        'base': BaseTrainerModel,
        'dropout': DropoutTrainerModel,
        'bootstrap': BootstrapTrainerModel,
        'sensitivity': SensitivityTrainerModel,
        'bnn': BNNTrainerModel,
        'aleatoric': AleatoricTrainerModel,
        'custom': CustomTrainerModel,
    }
    return switcher.get(model, BaseTrainerModel)(nb_nets, lr, img_stack, gamma, batch_size, buffer_capacity, device=device)
