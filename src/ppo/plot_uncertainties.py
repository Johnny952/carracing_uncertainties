import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from scipy.ndimage import gaussian_filter1d

_NAN_ = -1
def scale01(array):
    max_ = np.max(array)
    min_ = np.min(array)
    if max_ == min_:
        if max_ == 0:
            return array, 0
        else:
            return array / max_, 0
    return (array - min_) / (max_ - min_), (max_ - min_)

def read_uncert(path):
    epochs = []
    val_idx = []
    reward = []
    sigma = []
    epist = []
    aleat = []
    with open(path, "r") as f:
        for row in f:
            data = np.array(row[:-1].split(",")).astype(np.float32)
            epochs.append(data[0])
            val_idx.append(data[1])
            reward.append(data[2])
            sigma.append(data[3])
            l = len(data) - 4
            epist.append(data[4 : l // 2 + 4])
            aleat.append(data[l // 2 + 4 :])
    return process(np.array(epochs), np.array(reward), epist, aleat), np.unique(sigma)


def process(epochs, reward, epist, aleat):
    unique_ep = np.unique(epochs)
    mean_reward = np.zeros(unique_ep.shape, dtype=np.float32)
    mean_epist = np.zeros(unique_ep.shape, dtype=np.float32)
    mean_aleat = np.zeros(unique_ep.shape, dtype=np.float32)
    std_reward = np.zeros(unique_ep.shape, dtype=np.float32)
    std_epist = np.zeros(unique_ep.shape, dtype=np.float32)
    std_aleat = np.zeros(unique_ep.shape, dtype=np.float32)
    for idx, ep in enumerate(unique_ep):
        indexes = np.argwhere(ep == epochs).astype(int)
        mean_reward[idx] = np.mean(reward[indexes])
        std_reward[idx] = np.std(reward[indexes])
        for i in range(indexes.shape[0]):
            mean_epist[idx] += np.mean(epist[indexes[i][0]]) / indexes.shape[0]
            std_epist[idx] += np.std(epist[indexes[i][0]]) / indexes.shape[0]
            mean_aleat[idx] += np.mean(aleat[indexes[i][0]]) / indexes.shape[0]
            std_aleat[idx] += np.std(aleat[indexes[i][0]]) / indexes.shape[0]
    return (
        epochs,
        (unique_ep, mean_reward, mean_epist, mean_aleat),
        (std_reward, std_epist, std_aleat),
        (epist, aleat),
    )


def plot_uncert_train(
    paths,
    names,
    colors=None,
    linewidths=None,
    unc_path="images/train.png",
    smooth=None,
    plot_variance=False,
    multipliers=None,
):
    assert len(paths) == len(names)
    if colors is not None:
        assert len(colors) > len(paths)
    if linewidths is not None:
        assert len(linewidths) > len(paths)
    if multipliers is None:
        multipliers = [1 for _ in range(len(paths))]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    fig.suptitle("Rewards and Uncertainties during training", fontsize=18)
    ax.set_ylabel("Reward", fontsize=16)
    # ax[1].set_ylabel("Epistemic Uncertainty", fontsize=16)
    ax.set_xlabel("Episode", fontsize=16)
    
    for idx, (path, name, multiplier) in enumerate(zip(paths, names, multipliers)):
        color = colors[idx] if colors is not None else None
        linewidth = linewidths[idx] if linewidths is not None else None
        (
            _,
            (unique_ep, mean_reward, mean_epist, mean_aleat),
            (std_reward, std_epist, std_aleat),
            _,
        ) = read_uncert(path)[0]

        mean_epist = np.nan_to_num(mean_epist, nan=_NAN_)
        mean_aleat = np.nan_to_num(mean_aleat, nan=_NAN_)

        if smooth is not None:
            mean_reward = gaussian_filter1d(mean_reward, smooth)
            mean_epist[mean_epist != _NAN_] = gaussian_filter1d(mean_epist[mean_epist != _NAN_], smooth)
            mean_aleat[mean_aleat != _NAN_] = gaussian_filter1d(mean_aleat[mean_aleat != _NAN_], smooth)

        mean_epist[mean_epist != _NAN_], magnitude_epist = scale01(mean_epist[mean_epist != _NAN_]*multiplier)
        mean_aleat[mean_aleat != _NAN_], magnitude_aleat = scale01(mean_aleat[mean_aleat != _NAN_]*multiplier)

        # Plot uncertainties
        # ax[1].plot(
        #     unique_ep, mean_epist, color, label=f"Mean {name} ({magnitude_epist:.3f})", linewidth=linewidth
        # )
        
        # Plot rewards
        ax.plot(
            unique_ep, mean_reward, color, label="Mean " + name, linewidth=linewidth
        )
        if plot_variance:
            ax.fill_between(
                unique_ep,
                (mean_reward - std_reward),
                (mean_reward + std_reward),
                color=color,
                alpha=0.2,
                label="Std " + name,
            )

    ax.legend()
    fig.savefig(unc_path)


def plotly_train(
    paths, names, colors=None, save_fig="images/uncertainties_train.html", smooth=None, plot_variance=False
):
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        shared_xaxes="all",
        # subplot_titles=("Epistemic uncertainty","Aleatoric uncertainty", "Rewards"),
    )

    for idx, (path, name) in enumerate(zip(paths, names)):
        color = colors[idx] if colors is not None else None
        (
            _,
            (unique_ep, mean_reward, mean_epist, mean_aleat),
            (std_reward, std_epist, std_aleat),
            _,
        ) = read_uncert(path)[0]

        mean_epist = np.nan_to_num(mean_epist, nan=_NAN_)
        mean_aleat = np.nan_to_num(mean_aleat, nan=_NAN_)

        if smooth is not None:
            mean_reward = gaussian_filter1d(mean_reward, smooth)
            mean_epist[mean_epist != _NAN_] = gaussian_filter1d(mean_epist[mean_epist != _NAN_], smooth)
            mean_aleat[mean_aleat != _NAN_] = gaussian_filter1d(mean_aleat[mean_aleat != _NAN_], smooth)

        mean_epist[mean_epist != _NAN_], magnitude_epist = scale01(mean_epist[mean_epist != _NAN_])
        mean_aleat[mean_aleat != _NAN_], magnitude_aleat = scale01(mean_aleat[mean_aleat != _NAN_])

        rwd_upper, rwd_lower = mean_reward + std_reward, (mean_reward - std_reward)

        aux = color.lstrip("#")
        rgb_color = [str(int(aux[i : i + 2], 16)) for i in (0, 2, 4)] + ["0.2"]
        str_color = "rgba({})".format(",".join(rgb_color))

        if plot_variance:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([unique_ep, unique_ep[::-1]]),
                    y=np.concatenate([rwd_upper, rwd_lower[::-1]]),
                    fill="toself",
                    line_color="rgba(255,255,255,0)",
                    fillcolor=str_color,
                    showlegend=False,
                    legendgroup=name,
                    name=name,
                ),
                row=2,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=unique_ep,
                y=mean_reward,
                line_color=color,
                legendgroup=name,
                name=name,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=unique_ep,
                y=mean_epist,
                line_color=color,
                showlegend=False,
                legendgroup=name,
                name=name,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=unique_ep,
                y=mean_aleat,
                line_color=color,
                showlegend=False,
                legendgroup=name,
                name=name,
            ),
            row=1,
            col=2,
        )

    fig.update_yaxes(
        {"title": {"text": "Epistemic Uncertainty", "font": {"size": 20}}}, row=1, col=1
    )
    fig.update_yaxes(
        {"title": {"text": "Aleatoric Uncertainty", "font": {"size": 20}}}, row=1, col=2
    )
    fig.update_yaxes({"title": {"text": "Reward", "font": {"size": 20}}}, row=2, col=1)

    fig.update_xaxes({"title": {"text": "Episode", "font": {"size": 20}}}, row=1, col=1)
    fig.update_xaxes({"title": {"text": "Episode", "font": {"size": 20}}}, row=1, col=2)
    fig.update_xaxes({"title": {"text": "Episode", "font": {"size": 20}}}, row=2, col=1)

    fig.update_layout({"title": {"text": "Traning", "font": {"size": 26}}})
    fig.write_html(save_fig)


def plot_uncert_test(
    paths,
    names,
    colors=None,
    linewidths=None,
    unc_path="images/test.png",
    smooth=None,
    plot_variance=False,
    multipliers=None
):
    assert len(paths) == len(names)
    if colors is not None:
        assert len(colors) > len(paths)
    if linewidths is not None:
        assert len(linewidths) > len(paths)
    if multipliers is None:
        multipliers = [1 for _ in range(len(paths))]

    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    fig.suptitle("Rewards and Uncertainties during test", fontsize=18)
    # ax[0].set_xlabel("Noise Variance", fontsize=16)
    ax[0].set_ylabel("Reward", fontsize=16)
    # ax[1].set_ylabel("Epistemic Uncertainty", fontsize=16)
    # ax[1].set_xlabel("Noise Variance")
    ax[1].set_ylabel("Aleatoric Uncertainty", fontsize=16)
    ax[1].set_xlabel("Noise Variance", fontsize=16)
    

    for idx, (path, name, multiplier) in enumerate(zip(paths, names, multipliers)):
        color = colors[idx] if colors is not None else None
        linewidth = linewidths[idx] if linewidths is not None else None
        (
            (
                _,
                (_, mean_reward, mean_epist, mean_aleat),
                (std_reward, std_epist, std_aleat),
                _,
            ),
            sigma,
        ) = read_uncert(path)

        mean_epist = np.nan_to_num(mean_epist, nan=_NAN_)
        mean_aleat = np.nan_to_num(mean_aleat, nan=_NAN_)

        if smooth is not None:
            mean_reward = gaussian_filter1d(mean_reward, smooth)
            mean_epist[mean_epist != _NAN_] = gaussian_filter1d(mean_epist[mean_epist != _NAN_], smooth)
            mean_aleat[mean_aleat != _NAN_] = gaussian_filter1d(mean_aleat[mean_aleat != _NAN_], smooth)

        # if 'mix' in name.lower():
        #    mean_epist = 1 - np.exp(mean_epist)
        mean_epist[mean_epist != _NAN_], magnitude_epist = scale01(mean_epist[mean_epist != _NAN_]*multiplier)
        mean_aleat[mean_aleat != _NAN_], magnitude_aleat = scale01(mean_aleat[mean_aleat != _NAN_]*multiplier)

        # Plot uncertainties
        # ax[1].plot(
        #     sigma, mean_epist, color, label=f"Mean {name} ({magnitude_epist:.3f})", linewidth=linewidth
        # )
        # ax_unc[0].fill_between(sigma, (mean_epist/np.max(mean_epist) - std_epist/np.max(std_epist)), (mean_epist/np.max(mean_epist) + std_epist/np.max(std_epist)), color=color, alpha=0.2, label="Std " + name)
        ax[1].plot(
            sigma, mean_aleat, color, label=f"Mean {name} ({magnitude_aleat:.3f})", linewidth=linewidth
        )
        # ax_unc[1].fill_between(sigma, (mean_aleat/np.max(mean_aleat) - std_aleat/np.max(std_aleat)), (mean_aleat/np.max(mean_aleat) + std_aleat/np.max(std_aleat)), color=color, alpha=0.2, label="Std " + name)

        # Plot rewards
        ax[0].plot(
            sigma, mean_reward, color, label="Mean " + name, linewidth=linewidth
        )
        if plot_variance:
            ax[0].fill_between(
                sigma,
                (mean_reward - std_reward),
                (mean_reward + std_reward),
                color=color,
                alpha=0.2,
                label="Std " + name,
            )

    ax[0].legend()
    # ax[1].legend()
    # ax[2].legend()
    fig.savefig(unc_path)


def plotly_test(
    paths, names, colors=None, save_fig="images/uncertainties_test.html", smooth=None, plot_variance=False
):
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        shared_xaxes="all",
        # subplot_titles=("Epistemic uncertainty","Aleatoric uncertainty", "Rewards"),
    )

    for idx, (path, name) in enumerate(zip(paths, names)):
        color = colors[idx] if colors is not None else None
        (
            (
                _,
                (_, mean_reward, mean_epist, mean_aleat),
                (std_reward, std_epist, std_aleat),
                _,
            ),
            sigma,
        ) = read_uncert(path)

        mean_epist = np.nan_to_num(mean_epist, nan=_NAN_)
        mean_aleat = np.nan_to_num(mean_aleat, nan=_NAN_)

        if smooth is not None:
            mean_reward = gaussian_filter1d(mean_reward, smooth)
            mean_epist[mean_epist != _NAN_] = gaussian_filter1d(mean_epist[mean_epist != _NAN_], smooth)
            mean_aleat[mean_aleat != _NAN_] = gaussian_filter1d(mean_aleat[mean_aleat != _NAN_], smooth)

        mean_epist[mean_epist != _NAN_], magnitude_epist = scale01(mean_epist[mean_epist != _NAN_])
        mean_aleat[mean_aleat != _NAN_], magnitude_aleat = scale01(mean_aleat[mean_aleat != _NAN_])

        rwd_upper, rwd_lower = mean_reward + std_reward, (mean_reward - std_reward)

        aux = color.lstrip("#")
        rgb_color = [str(int(aux[i : i + 2], 16)) for i in (0, 2, 4)] + ["0.2"]
        str_color = "rgba({})".format(",".join(rgb_color))

        if plot_variance:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([sigma, sigma[::-1]]),
                    y=np.concatenate([rwd_upper, rwd_lower[::-1]]),
                    fill="toself",
                    line_color="rgba(255,255,255,0)",
                    fillcolor=str_color,
                    showlegend=False,
                    legendgroup=name,
                    name=name,
                ),
                row=2,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=sigma, y=mean_reward, line_color=color, legendgroup=name, name=name,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=sigma,
                y=mean_epist,
                line_color=color,
                showlegend=False,
                legendgroup=name,
                name=name,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sigma,
                y=mean_aleat,
                line_color=color,
                showlegend=False,
                legendgroup=name,
                name=name,
            ),
            row=1,
            col=2,
        )

    fig.update_yaxes(
        {"title": {"text": "Epistemic Uncertainty", "font": {"size": 20}}}, row=1, col=1
    )
    fig.update_yaxes(
        {"title": {"text": "Aleatoric Uncertainty", "font": {"size": 20}}}, row=1, col=2
    )
    fig.update_yaxes({"title": {"text": "Reward", "font": {"size": 20}}}, row=2, col=1)

    fig.update_xaxes(
        {"title": {"text": "Noise Variance", "font": {"size": 20}}}, row=1, col=1
    )
    fig.update_xaxes(
        {"title": {"text": "Noise Variance", "font": {"size": 20}}}, row=1, col=2
    )
    fig.update_xaxes(
        {"title": {"text": "Noise Variance", "font": {"size": 20}}}, row=2, col=1
    )

    fig.update_layout({"title": {"text": "Test", "font": {"size": 26}}})
    fig.write_html(save_fig)

def plot_uncert_comparative(train_paths, test_paths, names, linewidths=None, smooth=None, imgs_path='images/', log_scales=[]):
    assert len(train_paths) == len(names)
    assert len(test_paths) == len(names)
    if linewidths is not None:
        assert len(linewidths) > len(train_paths)

    for idx, (train_path, test_path, name) in enumerate(zip(train_paths, test_paths, names)):
        linewidth = linewidths[idx] if linewidths is not None else None
        file_name = f"comparative_{name}"

        (
            _,
            (unique_ep, _, mean_epist, mean_aleat),
            (_, std_epist, std_aleat),
            _,
        ) = read_uncert(train_path)[0]

        ncols = 0
        plot_epist = False
        plot_aleat = False
        if np.mean(mean_epist) != 0 and np.std(mean_epist) != 0:
            ncols += 1
            plot_epist = True
        if np.mean(mean_aleat) != 0 and np.std(mean_aleat) != 0:
            ncols += 1
            plot_aleat = True
        if ncols == 0:
            ncols = 1

        mean_epist = np.nan_to_num(mean_epist, nan=_NAN_)
        mean_aleat = np.nan_to_num(mean_aleat, nan=_NAN_)
        
        fig, ax = plt.subplots(nrows=1, ncols=ncols)
        fig.set_figheight(7)
        fig.set_figwidth(20)
        fig.suptitle(f"Uncertainties comparative during train and test model: {name}", fontsize=18)
        if ncols == 2:
            ax[0].set_ylabel("Epistemic", fontsize=16)
            ax[0].set_xlabel("Episode", fontsize=16)
            ax2 = ax[0].twiny()
            ax2.set_xlabel("Noise Variance", fontsize=16, color='r')
            ax2.tick_params(axis='x', labelcolor='r')
            ax[1].set_ylabel("Aleatoric", fontsize=16)
            ax[1].set_xlabel("Episode", fontsize=16)
            ax3 = ax[1].twiny()
            ax3.set_xlabel("Noise Variance", fontsize=16, color='r')
            ax3.tick_params(axis='x', labelcolor='r')
        elif plot_epist:
            ax.set_ylabel("Epistemic", fontsize=16)
            ax.set_xlabel("Episode", fontsize=16)
            ax2 = ax.twiny()
            ax2.set_xlabel("Noise Variance", fontsize=16, color='r')
            ax2.tick_params(axis='x', labelcolor='r')
        elif plot_aleat:
            ax.set_ylabel("Aleatoric", fontsize=16)
            ax.set_xlabel("Episode", fontsize=16)
            ax2 = ax.twiny()
            ax2.set_xlabel("Noise Variance", fontsize=16, color='r')
            ax2.tick_params(axis='x', labelcolor='r')
        else:
            print(f"Nothing to plot {name}")

        if ncols == 2:
            lns1 = ax[0].plot(
                unique_ep, mean_epist, label="train", linewidth=linewidth
            )
            lns2 = ax[1].plot(
                unique_ep, mean_aleat, label="train", linewidth=linewidth
            )
        elif plot_epist:
            lns1 = ax.plot(
                unique_ep, mean_epist, label="train", linewidth=linewidth
            )
        elif plot_aleat:
            lns1 = ax.plot(
                unique_ep, mean_aleat, label="train", linewidth=linewidth
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

        if ncols == 2:
            lns3 = ax2.plot(
                sigma, mean_epist, color="r", label="test", linewidth=linewidth
            )
            lns4 = ax3.plot(
                sigma, mean_aleat, color="r", label="test", linewidth=linewidth
            )

            lns = lns1+lns3
            labs = [l.get_label() for l in lns]
            ax[0].legend(lns, labs)

            lns = lns2+lns4
            labs = [l.get_label() for l in lns]
            ax[1].legend(lns, labs)
        elif plot_epist:
            lns2 = ax2.plot(
                sigma, mean_epist, color="r", label="test", linewidth=linewidth
            )

            lns = lns1+lns2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs)
        elif plot_aleat:
            lns2 = ax2.plot(
                sigma, mean_aleat, color="r", label="test", linewidth=linewidth
            )

            lns = lns1+lns2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs)
        
        if idx in log_scales and ncols == 2:
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')
        elif idx in log_scales:
            ax.set_yscale('log')



        fig.savefig(f"{imgs_path}{file_name}")

def plot_eval(
    paths,
    names,
    colors=None,
    linewidths=None,
    unc_path="images/eval.png",
    smooth=None,
    multipliers=None,
):
    assert len(paths) == len(names)
    if colors is not None:
        assert len(colors) > len(paths)
    if linewidths is not None:
        assert len(linewidths) > len(paths)
    if multipliers is None:
        multipliers = [1 for _ in range(len(paths))]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    fig.suptitle("Mean epistemic uncertainty during evaluation", fontsize=18)
    ax.set_ylabel("Mean Epistemic Uncertainty", fontsize=16)
    ax.set_xlabel("Episode", fontsize=16)
    
    for idx, (path, name, multiplier) in enumerate(zip(paths, names, multipliers)):
        color = colors[idx] if colors is not None else None
        linewidth = linewidths[idx] if linewidths is not None else None
        (
            _,
            (unique_ep, mean_reward, mean_epist, mean_aleat),
            (std_reward, std_epist, std_aleat),
            _,
        ) = read_uncert(path)[0]

        mean_epist = np.nan_to_num(mean_epist, nan=_NAN_)

        if smooth is not None:
            mean_reward = gaussian_filter1d(mean_reward, smooth)
            mean_epist[mean_epist != _NAN_] = gaussian_filter1d(mean_epist[mean_epist != _NAN_], smooth)

        mean_epist[mean_epist != _NAN_], magnitude_epist = scale01(mean_epist[mean_epist != _NAN_]*multiplier)

        # Plot uncertainties
        ax.plot(
            unique_ep, mean_epist, color, label=name, linewidth=linewidth
        )

    ax.legend()
    fig.savefig(unc_path)

def plot_vs_time(paths, names, figure='images/time_*.png', nb_eval=-1):
    for idx, (path, name) in enumerate(zip(paths, names)):
        (
            _,
            (unique_ep, mean_reward, mean_epist, mean_aleat),
            (std_reward, std_epist, std_aleat),
            (epist, aleat),
        ) = read_uncert(path)[0]
        
        plt.figure(figsize=(20, 10))
        plt.plot(epist[nb_eval])
        plt.plot([70, 70], [0, max(epist[nb_eval])], 'r')
        plt.xlabel('Step', fontsize=16)
        plt.ylabel('Epistemic Uncertainty', fontsize=16)
        plt.title(f"Epistemic Uncertainty of {name} during evaluation", fontsize=18)
        plt.savefig(figure.replace('*', name))


if __name__ == "__main__":
    smooth = 2
    plot_variance = False
    train_paths = [
        # "uncertainties/train/base.txt",
        #"uncertainties/train/bnn2.txt",
        "uncertainties/train/bootstrap.txt",
        "uncertainties/train/dropout.txt",
        "uncertainties/train/sensitivity.txt",
        #"uncertainties/train/vae.txt",
        #"uncertainties/train/aleatoric.txt",
        #"uncertainties/train/bootstrap2.txt",
        #"uncertainties/train/dropout2.txt",
    ]
    names = [
        # "Base",
        #"Bayesian NN",
        "Bootstrap",
        "Dropout",
        "Sensitivity",
        #"VAE",
        #"Aleatoric",
        #"Bootstrap 2",
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
    plotly_train(train_paths, names, colors=colors_px, smooth=smooth, plot_variance=plot_variance)

    test_paths = [
        # "uncertainties/test/base.txt",
        #"uncertainties/test/bnn2.txt",
        "uncertainties/test/bootstrap.txt",
        "uncertainties/test/dropout.txt",
        "uncertainties/test/sensitivity.txt",
        #"uncertainties/test/vae.txt",
        #"uncertainties/test/aleatoric.txt",
        #"uncertainties/test/bootstrap2.txt",
        #"uncertainties/test/dropout2.txt",
    ]

    plot_uncert_test(test_paths, names, colors=colors_px, linewidths=linewidths, smooth=smooth, plot_variance=plot_variance, multipliers=multipliers)
    plotly_test(test_paths, names, colors=colors_px, smooth=smooth, plot_variance=plot_variance)

    plot_uncert_comparative(train_paths, test_paths, names, linewidths)

    eval_paths = [
        "uncertainties/custom_eval/bootstrap.txt",
        "uncertainties/custom_eval/dropout.txt",
        "uncertainties/custom_eval/sensitivity.txt",
    ]

    plot_eval(eval_paths, names, colors=colors_px, linewidths=linewidths, smooth=smooth, multipliers=multipliers)

    plot_vs_time(eval_paths, names)

    plot_vs_time(eval_paths, names, nb_eval=50, figure='images/time100_*.png')