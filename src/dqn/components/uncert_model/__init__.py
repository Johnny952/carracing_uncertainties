from .basic_model import BaseTrainerModel
# from .dropout_model import DropoutTrainerModel
# from .bootstrap_model import BootstrapTrainerModel
# from .sensitivity_model import SensitivityTrainerModel
# from .bnn_model import BNNTrainerModel
# from .aleatoric_model import AleatoricTrainerModel
# from .vae_model import VAETrainerModel

def make_model(model='base', *args, **kwargs):
    switcher = {
        'base': BaseTrainerModel,
        # 'dropout': DropoutTrainerModel,
        # 'bootstrap': BootstrapTrainerModel,
        # 'sensitivity': SensitivityTrainerModel,
        # 'bnn': BNNTrainerModel,
        # 'aleatoric': AleatoricTrainerModel,
        # 'vae': VAETrainerModel,
    }
    return switcher.get(model, BaseTrainerModel)(*args, **kwargs)
