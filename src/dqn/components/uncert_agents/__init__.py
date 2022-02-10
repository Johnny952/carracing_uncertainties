from .base import BaseModel
from .aleatoric import AleatoricModel
# from .dropout_model import DropoutTrainerModel
# from .bootstrap_model import BootstrapTrainerModel
# from .sensitivity_model import SensitivityTrainerModel
# from .bnn_model import BNNTrainerModel
# from .aleatoric_model import AleatoricTrainerModel
# from .vae_model import VAETrainerModel

def make_agent(
        model,
        nb_nets,
        img_stack,
        actions,
        learning_rate,
        gamma,
        buffer,
        epsilon,
        batch_size,
        device="cpu",
        clip_grad=False
    ):
    switcher = {
        'base': BaseModel,
        # 'dropout': DropoutTrainerModel,
        # 'bootstrap': BootstrapTrainerModel,
        # 'sensitivity': SensitivityTrainerModel,
        # 'bnn': BNNTrainerModel,
        'aleatoric': AleatoricModel,
        # 'vae': VAETrainerModel,
    }
    return switcher.get(model, BaseModel)(
        nb_nets,
        img_stack,
        actions,
        learning_rate,
        gamma,
        buffer,
        epsilon,
        batch_size,
        device=device,
        clip_grad=clip_grad,
    )
