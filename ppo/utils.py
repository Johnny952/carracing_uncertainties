import numpy as np

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