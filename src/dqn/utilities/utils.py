import numpy as np
import argparse


def imgstackRGB2graystack(imgstack, R=0.299, G=0.587, B=0.114):
    imgstack = np.array(imgstack)
    return (
        imgstack[:, :, :, 0] * R + imgstack[:, :, :, 1] * G + imgstack[:, :, :, 2] * B
    )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def save_uncert(
    epoch, val_episode, score, uncert, file="uncertainties/train.txt", sigma=None
):
    with open(file, "a+") as f:
        if sigma is None:
            np.savetxt(
                f,
                np.concatenate(
                    ([epoch], [val_episode], [score], uncert.T.reshape(-1))
                ).reshape(1, -1),
                delimiter=",",
            )
        else:
            np.savetxt(
                f,
                np.concatenate(
                    ([epoch], [val_episode], [score], [sigma], uncert.T.reshape(-1))
                ).reshape(1, -1),
                delimiter=",",
            )


def init_uncert_file(file="uncertainties/train.txt"):
    with open(file, "w+") as f:
        pass
