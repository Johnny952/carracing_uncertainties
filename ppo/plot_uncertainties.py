import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d


def read_uncert(path):
    epochs = []
    val_idx = []
    reward = []
    epist = []
    aleat = []
    mean_epist = []
    mean_aleat = []
    with open(path, 'r') as f:
        for row in f:
            data = np.array(row[:-1].split(',')).astype(np.float32)
            epochs.append(data[0])
            val_idx.append(data[1])
            reward.append(data[2])
            l = len(data) - 3
            epist.append(data[3: l//2 + 3])
            aleat.append(data[l//2 + 3:])
    return process(np.array(epochs), np.array(reward), epist, aleat)

def read_uncert_test(path):
    epochs = []
    val_idx = []
    reward = []
    sigma = []
    epist = []
    aleat = []
    with open(path, 'r') as f:
        for row in f:
            data = np.array(row[:-1].split(',')).astype(np.float32)
            epochs.append(data[0])
            val_idx.append(data[1])
            reward.append(data[2])
            sigma.append(data[3])
            l = len(data) - 4
            epist.append(data[4: l//2 + 4])
            aleat.append(data[l//2 + 4:])
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
    return epochs, (unique_ep, mean_reward, mean_epist, mean_aleat), (std_reward, std_epist, std_aleat), (epist, aleat)


def plot_uncert_train(paths, names, colors=None, linewidths=None, unc_path='images/uncertainties.png', rwd_path='images/rewards.png', smooth=None):
    assert len(paths) == len(names)
    if colors is not None:
        assert len(colors) > len(paths)
    if linewidths is not None:
        assert len(linewidths) > len(paths)
    

    fig_rwd, ax_rwd = plt.subplots(nrows=1, ncols=1)
    fig_rwd.set_figheight(7)
    fig_rwd.set_figwidth(20)
    ax_rwd.set_xlabel("Épocas", fontsize=16)
    ax_rwd.set_ylabel("Recompensa", fontsize=16)



    fig_unc, ax_unc = plt.subplots(nrows=1, ncols=2)
    fig_unc.set_figheight(7)
    fig_unc.set_figwidth(20)
    fig_unc.suptitle("Incertezas durante el entrenamiento", fontsize=18)
    ax_unc[0].set_title('Incerteza epistemica', fontsize=16)
    ax_unc[0].set_xlabel('Epoca')
    ax_unc[1].set_title('Incerteza aleatoria', fontsize=16)
    ax_unc[1].set_xlabel('Epoca')

    for idx, (path, name) in enumerate(zip(paths, names)):
        color = colors[idx] if colors is not None else None
        linewidth = linewidths[idx] if linewidths is not None else None
        epochs, (unique_ep, mean_reward, mean_epist, mean_aleat), (std_reward, std_epist, std_aleat), (epist, aleat) = read_uncert(path)
        if smooth is not None:
            mean_reward = gaussian_filter1d(mean_reward, smooth)
            mean_epist = gaussian_filter1d(mean_epist, smooth)
            mean_aleat = gaussian_filter1d(mean_aleat, smooth)

        # Plot uncertainties
        if name.lower() != 'base':
            ax_unc[0].plot(unique_ep, mean_epist/np.max(mean_epist), color, label="Mean " + name, linewidth=linewidth)
            #ax_unc[0].fill_between(unique_ep, (mean_epist/np.max(mean_epist) - std_epist/np.max(std_epist)), (mean_epist/np.max(mean_epist) + std_epist/np.max(std_epist)), color=color, alpha=0.2, label="Std " + name)
            ax_unc[1].plot(unique_ep, mean_aleat/np.max(mean_aleat), color, label="Mean " + name, linewidth=linewidth)
            #ax_unc[1].fill_between(unique_ep, (mean_aleat/np.max(mean_aleat) - std_aleat/np.max(std_aleat)), (mean_aleat/np.max(mean_aleat) + std_aleat/np.max(std_aleat)), color=color, alpha=0.2, label="Std " + name)
        
        # Plot rewards
        ax_rwd.plot(unique_ep, mean_reward, color, label="Mean " + name, linewidth=linewidth)
        ax_rwd.fill_between(unique_ep, (mean_reward - std_reward), (mean_reward + std_reward), color=color, alpha=0.2, label="Std " + name)

    ax_unc[0].legend()
    ax_unc[1].legend()
    fig_unc.savefig(unc_path)


    ax_rwd.legend()
    fig_rwd.savefig(rwd_path)




def plotly_plot(paths, names, colors=None, save_fig='images/uncertainties.html', smooth=None):
    columns = [
        'Epoch',
        'Model',
        'Mean Reward',
        'Std Reward',
        'Mean Epist',
        'Std Epist',
        'Mean Aleat',
        'Std Aleat'
    ]

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{}, {}],
            [{"colspan": 2}, None]],
        shared_xaxes='all',
        #subplot_titles=("Epistemic uncertainty","Aleatoric uncertainty", "Rewards"),
    )

    for idx, (path, name) in enumerate(zip(paths, names)):
        color = colors[idx] if colors is not None else None
        epochs, (unique_ep, mean_reward, mean_epist, mean_aleat), (std_reward, std_epist, std_aleat), (epist, aleat) = read_uncert(path)
        if smooth is not None:
            mean_reward = gaussian_filter1d(mean_reward, smooth)
            mean_epist = gaussian_filter1d(mean_epist, smooth)
            mean_aleat = gaussian_filter1d(mean_aleat, smooth)
        
        rwd_upper, rwd_lower = mean_reward + std_reward, (mean_reward - std_reward)

        aux = color.lstrip('#')
        rgb_color = [str(int(aux[i:i+2], 16)) for i in (0, 2, 4)] + ["0.2"]
        str_color = "rgba({})".format(",".join(rgb_color))

        fig.add_trace(go.Scatter(
            x=np.concatenate([unique_ep, unique_ep[::-1]]),
            y=np.concatenate([rwd_upper, rwd_lower[::-1]]),
            fill='toself',
            line_color='rgba(255,255,255,0)',
            fillcolor=str_color,
            showlegend=False,
            legendgroup=name,
            name=name
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=unique_ep, y=mean_reward,
            line_color=color,
            legendgroup=name,
            name=name,
        ), row=2, col=1)


        if name.lower() != 'base':
            fig.add_trace(go.Scatter(
                x=unique_ep, y=mean_epist/np.max(mean_epist),
                line_color=color,
                showlegend=False,
                legendgroup=name,
                name=name,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=unique_ep, y=mean_aleat/np.max(mean_aleat),
                line_color=color,
                showlegend=False,
                legendgroup=name,
                name=name,
            ), row=1, col=2)



    fig.update_yaxes({"title": {"text": "Epistemic Uncertainty", "font": {"size": 20}}}, row=1, col=1)
    fig.update_yaxes({"title": {"text": "Aleatoric Uncertainty", "font": {"size": 20}}}, row=1, col=2)
    fig.update_yaxes({"title": {"text": "Reward", "font": {"size": 20}}}, row=2, col=1)

    fig.update_xaxes({"title": {"text": "Episode", "font": {"size": 20}}}, row=1, col=1)
    fig.update_xaxes({"title": {"text": "Episode", "font": {"size": 20}}}, row=1, col=2)
    fig.update_xaxes({"title": {"text": "Episode", "font": {"size": 20}}}, row=2, col=1)

    fig.update_layout({"title": {"text": "Traning", "font": {"size": 26}}})
    fig.write_html(save_fig)

if __name__ == "__main__":
    train_paths = [
        'uncertainties/train/base.txt',
        'uncertainties/train/sens.txt',
        'uncertainties/train/drop100.txt',
        'uncertainties/train/boot10.txt',
    ]

    names = [
        'Base',
        'Sensitivity',
        'Dropout 100',
        'Bootstrap 10',
    ]

    colors = [
        'k', 'r', 'y', 'g', 'b'
    ]

    linewidths = [
        2, 2, 2, 2, 2
    ]

    if not os.path.exists('images'):
        os.makedirs('images')
    
    plot_uncert_train(train_paths, names, colors=colors, linewidths=linewidths, smooth=2)


    colors = px.colors.qualitative.Plotly
    plotly_plot(train_paths, names, save_fig='images/uncertainties.html', colors=colors, smooth=2)