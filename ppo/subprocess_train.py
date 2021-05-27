from components import Agent, SubprocessEnv

from pyvirtualdisplay import Display
import torch


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    # Virtual display
    display = Display(visible=0, size=(1400, 900))
    display.start()

    # Whether to use cuda or cpu
    torch.cuda.empty_cache()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(0)
    if use_cuda:
        torch.cuda.manual_seed(0)

    agent = Agent(
        10, 
        4,
        0.99,
        model='base',
        device=device)

    subproc = SubprocessEnv(3, agent, 4, 8)

    scores, uncertainties = subproc.main()
    print(scores)