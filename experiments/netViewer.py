from torchviz import make_dot
import sys
import argparse
import torch
import hiddenlayer as hl


sys.path.append('../models')
from dropout import DropoutModel
from model import Net


parser = argparse.ArgumentParser(description='Visualize models')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
args = parser.parse_args()


net = DropoutModel(args)

x = torch.zeros(1, 4, 96, 96, dtype=torch.float, requires_grad=False)

transforms = [ hl.transforms.Prune('Constant') ] # Removes Constant nodes from graph.
graph = hl.build_graph(net, x, transforms=transforms)
graph.theme = hl.graph.THEMES['blue'].copy()
graph.save('rnn_hiddenlayer', format='png')




def show_torchviz():
    parser = argparse.ArgumentParser(description='Visualize models')
    parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
    args = parser.parse_args()


    net = Net(args)

    x = torch.zeros(1, 4, 96, 96, dtype=torch.float, requires_grad=False)
    out = net(x)
    out = [out[0][0], out[0][1], out[1]]
    out = torch.cat(out, dim=1)
    make_dot(out, params=dict(list(net.named_parameters()))).render("rnn_torchviz", format="png")