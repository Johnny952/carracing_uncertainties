import plotly.express as px
import os
import sys
sys.path.append('..')
from ppo.plot_uncertainties import plot_uncert_train, plot_uncert_test, plotly_train, plotly_test, plot_uncert_comparative


if __name__ == "__main__":
    smooth = 2
    plot_variance = False
    log_scales = [0, 6, 2, 7, 4, 1]
    train_paths = [
        # "uncertainties/train/ddqn_base_c077a8fa-b895-4aeb-85f6-0396baaf46c7.txt",
        "uncertainties/train/ddqn_bnn2_7cbd4bbe-c12a-4639-92f6-161a6a038c8b.txt",
        "uncertainties/train/ddqn_bootstrap_8b9714f5-5aee-4427-9e62-383c0b2d0ccc.txt",
        "uncertainties/train/ddqn_dropout_03d6df46-3085-49fe-9523-b2a13b776303.txt",
        "uncertainties/train/ddqn_sensitivity_03ec9544-d59e-4a64-9dc3-7997dbd74d4d.txt",
        "uncertainties/train/ddqn_vae_c3e688f6-e9d6-4dc9-94ad-27b66938ea3c.txt",
        "uncertainties/train/ddqn_aleatoric_556e588d-b6e2-4040-944d-1dec942e03f5.txt",
        "uncertainties/train/ddqn_bootstrap2_89dcc4a7-d0c2-4a2e-b219-55cde3438bd2.txt",
        "uncertainties/train/ddqn_dropout2_956283eb-5946-408d-867e-17f414bcf6e1.txt",
    ]
    names = [
        # "Base",
        "Bayesian NN",
        "Bootstrap",
        "Dropout",
        "Sensitivity",
        "VAE",
        "Aleatoric",
        "Bootstrap 2",
        "Dropout 2",
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
        "uncertainties/test/bnn2.txt",
        "uncertainties/test/bootstrap.txt",
        "uncertainties/test/dropout.txt",
        "uncertainties/test/sensitivity.txt",
        "uncertainties/test/vae.txt",
        "uncertainties/test/aleatoric.txt",
        "uncertainties/test/bootstrap2.txt",
        "uncertainties/test/dropout2.txt",
    ]

    plot_uncert_test(test_paths, names, colors=colors_px, linewidths=linewidths, smooth=smooth, plot_variance=plot_variance, multipliers=multipliers)
    plotly_test(test_paths, names, colors=colors_px, smooth=smooth, plot_variance=plot_variance)

    plot_uncert_comparative(train_paths, test_paths, names, linewidths, log_scales=log_scales)