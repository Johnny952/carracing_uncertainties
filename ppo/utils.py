import visdom
import numpy as np


class DrawLine():

    def __init__(self, env, title, xlabel=None, ylabel=None):
        self.vis = visdom.Visdom()
        self.update_flag = False
        self.env = env
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

    def __call__(
            self,
            xdata,
            ydata,
    ):
        if not self.update_flag:
            self.win = self.vis.line(
                X=np.array([xdata]),
                Y=np.array([ydata]),
                opts=dict(
                    xlabel=self.xlabel,
                    ylabel=self.ylabel,
                    title=self.title,
                ),
                env=self.env,
            )
            self.update_flag = True
        else:
            self.vis.line(
                X=np.array([xdata]),
                Y=np.array([ydata]),
                win=self.win,
                env=self.env,
                update='append',
            )

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_uncert(epoch, val_episode, score, uncert, file='uncertainties/train.txt', sigma=None):
    with open(file, 'a+') as f:
        if sigma is None:
            np.savetxt(f, np.concatenate(([epoch], [val_episode], [score], uncert.T.reshape(-1))).reshape(1, -1), delimiter=',')
        else:
            np.savetxt(f, np.concatenate(([epoch], [val_episode], [score], [sigma], uncert.T.reshape(-1))).reshape(1, -1), delimiter=',')