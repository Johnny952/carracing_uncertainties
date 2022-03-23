import plotly.express as px
import os
import sys
sys.path.append('..')
from ppo.plot_uncertainties import plot_uncert_train, plot_uncert_test, plotly_train, plotly_test


if __name__ == "__main__":
    plot_variance = False
    train_paths = [
        "uncertainties/train/fix_ddqn_base_c077a8fa-b895-4aeb-85f6-0396baaf46c7.txt",
        "uncertainties/train/ddqn_bootstrap_8b9714f5-5aee-4427-9e62-383c0b2d0ccc.txt",
        "uncertainties/train/fix_ddqn_dropout_481dd29d-3ecc-48aa-9249-31896c98036f.txt",
        "uncertainties/train/fix_ddqn_sensitivity_03ec9544-d59e-4a64-9dc3-7997dbd74d4d.txt",
        "uncertainties/train/ddqn_vae_c3e688f6-e9d6-4dc9-94ad-27b66938ea3c.txt",
        "uncertainties/train/ddqn_aleatoric_9d9459b1-999d-49b3-87d4-06b4dcb86f6c.txt",
        "uncertainties/train/fix_ddqn_bnn_a0653354-b837-4782-8b3b-81f4de95be1e.txt",
    ]
    names = [
        "Base",
        "Bootstrap",
        "Dropout",
        "Sensitivity",
        "VAE",
        "Aleatoric",
        "Bayesian NN",
    ]
    multipliers = [
        10,
        1/100,
        1e-14,
        1/10,
        1/100,
        1,
        1,
    ]
    colors = ["k", "r", "y", "g", "b", "c", "m", "tab:cyan"]
    colors_px = px.colors.qualitative.Plotly
    linewidths = [2, 2, 2, 2, 2, 2, 2, 2]

    if not os.path.exists("images"):
        os.makedirs("images")

    plot_uncert_train(
        train_paths, names, colors=colors, linewidths=linewidths, smooth=2, plot_variance=plot_variance, multipliers=multipliers
    )
    plotly_train(train_paths, names, colors=colors_px, smooth=2, plot_variance=plot_variance)

    test_paths = [
        "uncertainties/test/base.txt",
        "uncertainties/test/bootstrap.txt",
        "uncertainties/test/dropout.txt",
        "uncertainties/test/sensitivity.txt",
        "uncertainties/test/vae.txt",
        # "uncertainties/test/aleatoric.txt",
    ]
    names = [
        "Base",
        "Bootstrap",
        "Dropout",
        "Sensitivity",
        "VAE",
        # "Aleatoric",
    ]

    plot_uncert_test(test_paths, names, colors=colors, linewidths=linewidths, smooth=2, plot_variance=plot_variance, multipliers=multipliers)
    plotly_test(test_paths, names, colors=colors_px, smooth=2, plot_variance=plot_variance)

