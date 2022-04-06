from .basic_model import BaseTrainerModel
from .dropout_model import DropoutTrainerModel
from .bootstrap_model import BootstrapTrainerModel
from .sensitivity_model import SensitivityTrainerModel
from .bnn_model import BNNTrainerModel
from .aleatoric_model import AleatoricTrainerModel
from .vae_model import VAETrainerModel
from .bnn_model2 import BNNTrainerModel2
from .dropout_model2 import DropoutTrainerModel2
from .bootstrap_model2 import BootstrapTrainerModel2

def make_model(nb_nets, lr, img_stack, gamma, batch_size, buffer_capacity, device='cpu', model='base'):
    switcher = {
        'base': BaseTrainerModel,
        'dropout': DropoutTrainerModel,
        'dropout2': DropoutTrainerModel2,
        'bootstrap': BootstrapTrainerModel,
        'bootstrap2': BootstrapTrainerModel2,
        'sensitivity': SensitivityTrainerModel,
        'bnn': BNNTrainerModel,
        'bnn2': BNNTrainerModel2,
        'aleatoric': AleatoricTrainerModel,
        'vae': VAETrainerModel,
    }
    return switcher.get(model, BaseTrainerModel)(nb_nets, lr, img_stack, gamma, batch_size, buffer_capacity, device=device)
