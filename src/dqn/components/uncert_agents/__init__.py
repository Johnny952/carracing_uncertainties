from .base import BaseAgent
from .aleatoric import AleatoricAgent
from .sensitivity import SensitivityAgent
from .vae import VaeAgent
from .bnn import BnnAgent
from .bootstrap import BootstrapAgent
from .dropout import DropoutAgent
from .bnn2 import BnnAgent2

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
        'dropout': DropoutAgent,
        'bootstrap': BootstrapAgent,
        'sensitivity': SensitivityAgent,
        'bnn': BnnAgent,
        'aleatoric': AleatoricAgent,
        'vae': VaeAgent,
        'bnn2': BnnAgent2,
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
