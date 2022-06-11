import os
import plotly.express as px
import sys
sys.path.append('..')
from shared.utils.plot import _NAN_, read_uncert, plot_uncert_train, plot_uncert_test, plot_eval, plot_vs_time, plot_comparative

if __name__ == "__main__":
    smooth = 2
    plot_variance = False
    train_paths = [
        # "uncertainties/eval/base.txt",
        "uncertainties/eval/bnn2.txt",
        "uncertainties/eval/bootstrap.txt",
        "uncertainties/eval/dropout.txt",
        "uncertainties/eval/sensitivity.txt",
        "uncertainties/eval/vae.txt",
        "uncertainties/eval/aleatoric.txt",
        "uncertainties/eval/bootstrap2.txt",
        "uncertainties/eval/dropout2.txt",
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
    uncertainties = [
        # 1,
        1,
        1,
        1,
        1,
        1,
        2,
        1,
        1,
    ]
    multipliers = [1] * 10
    colors_px = px.colors.qualitative.Plotly
    linewidths = [2] * 10

    if not os.path.exists("images"):
        os.makedirs("images")

    plot_uncert_train(
        train_paths, names, colors=colors_px, linewidths=linewidths, smooth=smooth, plot_variance=plot_variance, multipliers=multipliers, max_=150
    )

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

    test0_paths = [
        # "uncertainties/test0/base.txt",
        "uncertainties/test0/bnn2.txt",
        "uncertainties/test0/bootstrap.txt",
        "uncertainties/test0/dropout.txt",
        "uncertainties/test0/sensitivity.txt",
        "uncertainties/test0/vae.txt",
        "uncertainties/test0/aleatoric.txt",
        "uncertainties/test0/bootstrap2.txt",
        "uncertainties/test0/dropout2.txt",
    ]
    plot_comparative(
        train_paths=train_paths,
        test0_paths=test0_paths,
        test_paths=test_paths,
        names=names,
        linewidths=linewidths,
        uncertainties=uncertainties
    )

    eval_paths = [
        # "uncertainties/customtest1/base.txt",
        "uncertainties/customtest1/bnn2.txt",
        "uncertainties/customtest1/bootstrap.txt",
        "uncertainties/customtest1/dropout.txt",
        "uncertainties/customtest1/sensitivity.txt",
        "uncertainties/customtest1/vae.txt",
        "uncertainties/customtest1/aleatoric.txt",
        "uncertainties/customtest1/bootstrap2.txt",
        "uncertainties/customtest1/dropout2.txt",
    ]

    # plot_eval(eval_paths, names, colors=colors_px, linewidths=linewidths, smooth=smooth, multipliers=multipliers)

    log_scales = [False] * 10
    plot_vs_time(
        paths=eval_paths,
        names=names,
        log_scales=log_scales,
        uncertainties=uncertainties,
    )

    eval2_paths = [
        # "uncertainties/customtest2/base.txt",
        "uncertainties/customtest2/bnn2.txt",
        "uncertainties/customtest2/bootstrap.txt",
        "uncertainties/customtest2/dropout.txt",
        "uncertainties/customtest2/sensitivity.txt",
        "uncertainties/customtest2/vae.txt",
        "uncertainties/customtest2/aleatoric.txt",
        "uncertainties/customtest2/bootstrap2.txt",
        "uncertainties/customtest2/dropout2.txt",
    ]
    plot_vs_time(
        paths=eval2_paths,
        names=names,
        log_scales=log_scales,
        figure='images/time2_*.png',
        red_lines=[25, 100],
        uncertainties=uncertainties,
    )

    