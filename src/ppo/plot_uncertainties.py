import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from scipy.ndimage import gaussian_filter1d

def scale01(array):
    max_ = np.max(array)
    min_ = np.min(array)
    if max_ == min_:
        if max_ == 0:
            return array
        else:
            return array / max_
    return (array - min_) / (max_ - min_)


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
        indexes = np.argwhere(ep == epochs).astype(np.int)
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
    unc_path="images/uncertainties_train.png",
    rwd_path="images/rewards_train.png",
    smooth=None,
):
    assert len(paths) == len(names)
    if colors is not None:
        assert len(colors) > len(paths)
    if linewidths is not None:
        assert len(linewidths) > len(paths)

    fig_rwd, ax_rwd = plt.subplots(nrows=1, ncols=1)
    fig_rwd.set_figheight(7)
    fig_rwd.set_figwidth(20)
    ax_rwd.set_xlabel("Episode", fontsize=16)
    ax_rwd.set_ylabel("Reward", fontsize=16)

    fig_unc, ax_unc = plt.subplots(nrows=1, ncols=2)
    fig_unc.set_figheight(7)
    fig_unc.set_figwidth(20)
    fig_unc.suptitle("Uncertainties during training", fontsize=18)
    ax_unc[0].set_title("Epistemic Uncertainty", fontsize=16)
    ax_unc[0].set_xlabel("Episode")
    ax_unc[1].set_title("Aleatoric Uncertainty", fontsize=16)
    ax_unc[1].set_xlabel("Episode")

    for idx, (path, name) in enumerate(zip(paths, names)):
        color = colors[idx] if colors is not None else None
        linewidth = linewidths[idx] if linewidths is not None else None
        (
            _,
            (unique_ep, mean_reward, mean_epist, mean_aleat),
            (std_reward, std_epist, std_aleat),
            _,
        ) = read_uncert(path)[0]
        if smooth is not None:
            mean_reward = gaussian_filter1d(mean_reward, smooth)
            mean_epist = gaussian_filter1d(mean_epist, smooth)
            mean_aleat = gaussian_filter1d(mean_aleat, smooth)

        mean_epist = scale01(mean_epist)
        mean_aleat = scale01(mean_aleat)

        # Plot uncertainties
        ax_unc[0].plot(
            unique_ep, mean_epist, color, label="Mean " + name, linewidth=linewidth
        )
        # ax_unc[0].fill_between(unique_ep, (mean_epist/np.max(mean_epist) - std_epist/np.max(std_epist)), (mean_epist/np.max(mean_epist) + std_epist/np.max(std_epist)), color=color, alpha=0.2, label="Std " + name)
        ax_unc[1].plot(
            unique_ep, mean_aleat, color, label="Mean " + name, linewidth=linewidth
        )
        # ax_unc[1].fill_between(unique_ep, (mean_aleat/np.max(mean_aleat) - std_aleat/np.max(std_aleat)), (mean_aleat/np.max(mean_aleat) + std_aleat/np.max(std_aleat)), color=color, alpha=0.2, label="Std " + name)

        # Plot rewards
        ax_rwd.plot(
            unique_ep, mean_reward, color, label="Mean " + name, linewidth=linewidth
        )
        ax_rwd.fill_between(
            unique_ep,
            (mean_reward - std_reward),
            (mean_reward + std_reward),
            color=color,
            alpha=0.2,
            label="Std " + name,
        )

    ax_unc[0].legend()
    ax_unc[1].legend()
    fig_unc.savefig(unc_path)

    ax_rwd.legend()
    fig_rwd.savefig(rwd_path)


def plotly_train(
    paths, names, colors=None, save_fig="images/uncertainties_train.html", smooth=None
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
        if smooth is not None:
            mean_reward = gaussian_filter1d(mean_reward, smooth)
            mean_epist = gaussian_filter1d(mean_epist, smooth)
            mean_aleat = gaussian_filter1d(mean_aleat, smooth)

        mean_epist = scale01(mean_epist)
        mean_aleat = scale01(mean_aleat)

        rwd_upper, rwd_lower = mean_reward + std_reward, (mean_reward - std_reward)

        aux = color.lstrip("#")
        rgb_color = [str(int(aux[i : i + 2], 16)) for i in (0, 2, 4)] + ["0.2"]
        str_color = "rgba({})".format(",".join(rgb_color))

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
    unc_path="images/uncertainties_test.png",
    rwd_path="images/rewards_test.png",
    smooth=None,
):
    assert len(paths) == len(names)
    if colors is not None:
        assert len(colors) > len(paths)
    if linewidths is not None:
        assert len(linewidths) > len(paths)

    fig_rwd, ax_rwd = plt.subplots(nrows=1, ncols=1)
    fig_rwd.set_figheight(7)
    fig_rwd.set_figwidth(20)
    ax_rwd.set_xlabel("Noise Variance", fontsize=16)
    ax_rwd.set_ylabel("Reward", fontsize=16)

    fig_unc, ax_unc = plt.subplots(nrows=1, ncols=2)
    fig_unc.set_figheight(7)
    fig_unc.set_figwidth(20)
    fig_unc.suptitle("Uncertainties during test", fontsize=18)
    ax_unc[0].set_title("Epistemic Uncertainty", fontsize=16)
    ax_unc[0].set_xlabel("Noise Variance")
    ax_unc[1].set_title("Aleatoric Uncertainty", fontsize=16)
    ax_unc[1].set_xlabel("Noise Variance")

    for idx, (path, name) in enumerate(zip(paths, names)):
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
        if smooth is not None:
            mean_reward = gaussian_filter1d(mean_reward, smooth)
            mean_epist = gaussian_filter1d(mean_epist, smooth)
            mean_aleat = gaussian_filter1d(mean_aleat, smooth)

        # if 'mix' in name.lower():
        #    mean_epist = 1 - np.exp(mean_epist)

        mean_epist = scale01(mean_epist)
        mean_aleat = scale01(mean_aleat)

        # Plot uncertainties
        ax_unc[0].plot(
            sigma, mean_epist, color, label="Mean " + name, linewidth=linewidth
        )
        # ax_unc[0].fill_between(sigma, (mean_epist/np.max(mean_epist) - std_epist/np.max(std_epist)), (mean_epist/np.max(mean_epist) + std_epist/np.max(std_epist)), color=color, alpha=0.2, label="Std " + name)
        ax_unc[1].plot(
            sigma, mean_aleat, color, label="Mean " + name, linewidth=linewidth
        )
        # ax_unc[1].fill_between(sigma, (mean_aleat/np.max(mean_aleat) - std_aleat/np.max(std_aleat)), (mean_aleat/np.max(mean_aleat) + std_aleat/np.max(std_aleat)), color=color, alpha=0.2, label="Std " + name)

        # Plot rewards
        ax_rwd.plot(
            sigma, mean_reward, color, label="Mean " + name, linewidth=linewidth
        )
        ax_rwd.fill_between(
            sigma,
            (mean_reward - std_reward),
            (mean_reward + std_reward),
            color=color,
            alpha=0.2,
            label="Std " + name,
        )

    ax_unc[0].legend()
    ax_unc[1].legend()
    fig_unc.savefig(unc_path)

    ax_rwd.legend()
    fig_rwd.savefig(rwd_path)


def plotly_test(
    paths, names, colors=None, save_fig="images/uncertainties_test.html", smooth=None
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
        if smooth is not None:
            mean_reward = gaussian_filter1d(mean_reward, smooth)
            mean_epist = gaussian_filter1d(mean_epist, smooth)
            mean_aleat = gaussian_filter1d(mean_aleat, smooth)

        mean_epist = scale01(mean_epist)
        mean_aleat = scale01(mean_aleat)

        rwd_upper, rwd_lower = mean_reward + std_reward, (mean_reward - std_reward)

        aux = color.lstrip("#")
        rgb_color = [str(int(aux[i : i + 2], 16)) for i in (0, 2, 4)] + ["0.2"]
        str_color = "rgba({})".format(",".join(rgb_color))

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


if __name__ == "__main__":
    train_paths = [
        "uncertainties/train/base.txt",
        "uncertainties/train/sens.txt",
        "uncertainties/train/drop.txt",
        "uncertainties/train/custom.txt",
    ]
    names = [
        "Base",
        "Sensitivity",
        "Dropout",
        "Normalizing Flows",
    ]
    colors = ["k", "r", "y", "g", "b", "c", "m", "tab:cyan"]
    colors_px = px.colors.qualitative.Plotly
    linewidths = [2, 2, 2, 2, 2, 2, 2, 2]

    if not os.path.exists("images"):
        os.makedirs("images")

    plot_uncert_train(
        train_paths, names, colors=colors, linewidths=linewidths, smooth=2
    )
    plotly_train(train_paths, names, colors=colors_px, smooth=2)

    # test_paths = [
    #     "uncertainties/test/base.txt",
    #     "uncertainties/test/drop10.txt",
    #     "uncertainties/test/mix50.txt",
    # ]
    # names = ["Base", "Dropout 10", "Mixture 50"]

    # plot_uncert_test(test_paths, names, colors=colors, linewidths=linewidths, smooth=2)
    # plotly_test(test_paths, names, colors=colors_px, smooth=2)
