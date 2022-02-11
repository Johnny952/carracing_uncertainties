from .base import BaseAgent
from .aleatoric import AleatoricAgent
from .sensitivity import SensitivityAgent
from .vae import VaeAgent

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
        'base': BaseAgent,
        # 'dropout': DropoutTrainerModel,
        # 'bootstrap': BootstrapTrainerModel,
        'sensitivity': SensitivityAgent,
        # 'bnn': BNNTrainerModel,
        'aleatoric': AleatoricAgent,
        'vae': VaeAgent,
    }
    return switcher.get(model, BaseAgent)(
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
