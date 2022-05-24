import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.express as px
from scipy.ndimage import gaussian_filter1d
from plot_uncertainties import _NAN_, read_uncert, plot_uncert_train, plot_uncert_test, plot_eval, plot_vs_time

def plot_comparative(train_paths, test0_paths, test_paths, names, linewidths=None, imgs_path='images/'):
    assert len(train_paths) == len(names)
    assert len(test_paths) == len(names)
    assert len(test0_paths) == len(names)

    for idx, (train_path, test0_path, test_path, name) in enumerate(zip(train_paths, test0_paths, test_paths, names)):
        linewidth = linewidths[idx] if linewidths is not None else None
        file_name = f"comparative_{name}"

        fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)
        fig.set_figheight(7)
        fig.set_figwidth(20)
        fig.suptitle(f"Uncertainties comparative during train and test model: {name}", fontsize=18)
        
        ax[0].set_ylabel("Epistemic", fontsize=16)
        ax[0].set_xlabel("Episode", fontsize=16)
        ax[0].set_title("Epistemic Uncertainty in evaluation", fontsize=16)
        ax[1].set_xlabel("Test Number", fontsize=16)
        ax[1].set_title("Epistemic Uncertainty in test without noise", fontsize=16)
        ax[2].set_xlabel("Noise Variance", fontsize=16)
        ax[2].set_title("Epistemic Uncertainty in test with noise", fontsize=16)

        (
            _,
            (unique_ep, _, mean_epist, _),
            (_, std_epist, std_aleat),
            _,
        ) = read_uncert(train_path)[0]
        mean_epist = np.nan_to_num(mean_epist, nan=_NAN_)
        ax[0].plot(
            unique_ep, mean_epist, linewidth=linewidth
        )


        (
            epochs,
            (unique_ep, mean_reward, mean_epist, mean_aleat),
            (std_reward, std_epist, std_aleat),
            (epist, aleat),
        ) = read_uncert(test0_path)[0]
        epist = [np.mean(e) for e in epist]
        ax[1].plot(
            epist, linewidth=linewidth
        )

        (
            (
                _,
                (_, _, mean_epist, mean_aleat),
                (_, std_epist, std_aleat),
                _,
            ),
            sigma,
        ) = read_uncert(test_path)
        ax[2].plot(
            sigma, mean_epist, linewidth=linewidth
        )

        fig.savefig(f"{imgs_path}{file_name}")



if __name__ == "__main__":
    smooth = 2
    plot_variance = False
    train_paths = [
        # "uncertainties/eval/base.txt",
        #"uncertainties/eval/bnn2.txt",
        #"uncertainties/eval/bootstrap.txt",
        "uncertainties/eval/dropout.txt",
        "uncertainties/eval/sensitivity.txt",
        "uncertainties/eval/vae.txt",
        #"uncertainties/eval/aleatoric.txt",
        "uncertainties/eval/bootstrap2.txt",
        #"uncertainties/eval/dropout2.txt",
    ]
    names = [
        # "Base",
        #"Bayesian NN",
        #"Bootstrap",
        "Dropout",
        "Sensitivity",
        "VAE",
        #"Aleatoric",
        "Bootstrap 2",
        #"Dropout 2",
    ]
    multipliers = [1] * 10
    colors_px = px.colors.qualitative.Plotly
    linewidths = [2] * 10

    if not os.path.exists("images"):
        os.makedirs("images")

    plot_uncert_train(
        train_paths, names, colors=colors_px, linewidths=linewidths, smooth=smooth, plot_variance=plot_variance, multipliers=multipliers
    )

    test_paths = [
        # "uncertainties/test/base.txt",
        #"uncertainties/test/bnn2.txt",
        #"uncertainties/test/bootstrap.txt",
        "uncertainties/test/dropout.txt",
        "uncertainties/test/sensitivity.txt",
        "uncertainties/test/vae.txt",
        #"uncertainties/test/aleatoric.txt",
        "uncertainties/test/bootstrap2.txt",
        #"uncertainties/test/dropout2.txt",
    ]

    plot_uncert_test(test_paths, names, colors=colors_px, linewidths=linewidths, smooth=smooth, plot_variance=plot_variance, multipliers=multipliers)

    test0_paths = [
        # "uncertainties/test0/base.txt",
        #"uncertainties/test0/bnn2.txt",
        #"uncertainties/test0/bootstrap.txt",
        "uncertainties/test0/dropout.txt",
        "uncertainties/test0/sensitivity.txt",
        "uncertainties/test0/vae.txt",
        #"uncertainties/test0/aleatoric.txt",
        "uncertainties/test0/bootstrap2.txt",
        #"uncertainties/test0/dropout2.txt",
    ]
    plot_comparative(train_paths,test0_paths, test_paths, names, linewidths)

    eval_paths = [
        "uncertainties/customeval/dropout.txt",
        "uncertainties/customeval/sensitivity.txt",
        "uncertainties/customeval/vae.txt",
        "uncertainties/customeval/bootstrap2.txt",
    ]

    plot_eval(eval_paths, names, colors=colors_px, linewidths=linewidths, smooth=smooth, multipliers=multipliers)

    log_scales = [
        # False,
        # False,
        # False,
        False,
        False,
        True,
        # False,
        False,
        # False,
    ]
    plot_vs_time(eval_paths, names, log_scales)