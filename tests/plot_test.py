from functools import wraps
import inspect

import interplot as ip

import pytest

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd


# prepare some 2d data
np.random.seed(10)
data2d = np.random.normal(1, 1, (5, 5))
df2d = pd.DataFrame(
    data=data2d,
    columns=("A", "B", "C", "D", "E"),
    index=(5, 10, 15, 20, 25),
)


@pytest.mark.skip(reason="just a decorator, for no error assertion")
def test_for_errors(f=lambda: None):

    @wraps(f)
    @pytest.mark.parametrize(
            "interactive",
            (True, False),
        )
    def inner(interactive, *args, **kwargs):
        no_error = True

        try:
            f(*args, **kwargs, interactive=interactive)
            if not interactive:
                plt.close("all")
        except:  # noqa: E722
            no_error = False

        assert no_error

    return inner


@test_for_errors
def test_line_basic(interactive):
    ip.line([1, 2, 3], [4, 5, 6], interactive=interactive)


@test_for_errors
def test_line_adv(interactive):
    ip.line(
        interactive=interactive,
        x=np.array(((1, 2), (3, 4))),
        x_error=((0.1, 0.2), (0.3, 0.4)),
        y_error=2,
        mode="lines+markers",
        line_style="dash",
        marker=13,
        marker_size=12,
        marker_line_color="red",
        marker_line_width=2,
        label="test",
        show_legend=True,
        color="blue",
        opacity=0.5,
        rows=2,
        cols=2,
        row=1,
        col=1,
    )


@pytest.mark.parametrize(
    "line_style",
    ("solid", "dashed", "dash", "dotted", "dot", "dashdot",
     "-", "-.", ":", "--"),
)
@test_for_errors
def test_line_line_style(line_style, interactive):
    ip.line(
        [1, 2, 3],
        [4, 5, 6],
        line_style=line_style,
        interactive=interactive,
    )


@pytest.mark.parametrize(
    "marker",
    np.arange(55),
)
@test_for_errors
def test_line_markers(marker, interactive):
    ip.linescatter(
        [1, 2, 3], [4, 5, 6],
        marker=marker,
        interactive=interactive,
    )


@pytest.mark.parametrize(
    "color",
    ("red", "c0", "C1", "C123", "#FF00FF", None),
)
@test_for_errors
def test_line_colors(color, interactive):
    ip.line([1, 2, 3], [4, 5, 6], color=color, interactive=interactive)


@test_for_errors
def test_bar_basic(interactive):
    ip.bar([1, 2, 3], [4, 5, 6], interactive=interactive)


@test_for_errors
def test_bar_adv(interactive):
    ip.bar(
        interactive=interactive,
        x=np.array(((1, 2), (3, 4))),
        horizontal=True,
        width=0.2,
        label="test",
        show_legend=True,
        color="blue",
        opacity=0.5,
        line_width=2,
        line_color="red",
        rows=2,
        cols=2,
        row=1,
        col=1,
    )


@test_for_errors
def test_2d_data(interactive):
    fig = ip.Plot(rows=2, cols=2)

    fig.add_line(data2d)
    fig.add_line(df2d, col=1)

    fig.add_bar(data2d, row=1)
    fig.add_bar(df2d, col=1, row=1)

    fig.post_process()


@pytest.mark.parametrize(
    "opacity",
    (0, 0.5, 1),
)
@test_for_errors
def test_line_opacity(opacity, interactive):
    ip.line([1, 2, 3], [4, 5, 6], opacity=opacity, interactive=interactive)


def test_default_funcs():
    def mpl_custom_func_1(fig, ax):
        fig.customvalue = 1
        return fig, ax

    def mpl_custom_func_2(fig, ax):
        fig.customvalue = 2
        return fig, ax

    def mpl_custom_func_3(fig, ax):
        fig.customvalue = 3
        return fig, ax

    ip.conf.MPL_CUSTOM_FUNC = mpl_custom_func_1

    fig = ip.Plot(
        interactive=False,
    )
    fig.add_line((1, 4))
    fig.post_process()
    assert fig.fig.customvalue == 1

    fig = ip.Plot(
        interactive=False,
        mpl_custom_func=mpl_custom_func_2,
    )
    fig.add_line((1, 4))
    fig.post_process()
    assert fig.fig.customvalue == 2

    fig = ip.Plot(
        interactive=False,
        mpl_custom_func=mpl_custom_func_2,
    )
    fig.add_line((1, 4))
    fig.post_process(mpl_custom_func=mpl_custom_func_3)
    assert fig.fig.customvalue == 3

    plt.close("all")


@test_for_errors
def test_complex_plot(interactive):
    np.random.seed(0)
    fig = ip.Plot(
        interactive=interactive,
        title="Get Fancy!",
        xlabel="X",
        ylabel=("Y1", "Y2"),
        xlim=(0, 100),
        shared_xaxes="all",
        shared_yaxes=False,
        xlog=False,
        ylog=(False, True),
        rows=2,
        cols=2,
        fig_size=(800, 600),
        row_heights=(1, 2),
        legend_loc=(
            ("center right", "best"),
            ("best", False),
        ),
        legend_togglegroup=True,
        save_fig="tests/temp_exports/fancy_{}.png".format(
            "interactive" if interactive else "static"),
    )

    fig.add_line((20, 50, 80), (30, 20, 10), y_error=((4, 2, 6), (3, 12, 9)),
                 line_style="dashdot", label="line 1")
    fig.add_linescatter((20, 50, 80), (0, 20, 40), marker="*", label="line 2")
    fig.add_text(50, 10, "some\nannotation", color="red")

    fig.add_bar(
        ("f", "g", "h", "a"), (1, 2, 5, 2), horizontal=True, col=1,
        label="secondary", color="black")
    fig.add_bar(
        ("f", "g", "h", "a"), (1, 2, 5, 2), label="styling", color=None,
        line_width=10, line_color="blue")

    fig.add_hist(y=np.random.normal(0, 10, 1000), row=0, col=1,
                 label="hist 1, horizontal", bins=40)

    fig.add_boxplot([np.random.normal(30, 6, 1000),
                     np.random.normal(70, 5, 1000)],
                    horizontal=True, row=1, col=1,
                    label=("boxplot 1", "boxplot 2"))

    fig.add_fill((0, 100), (30, 60), (70, 100), row=1, col=0, color="blue",
                 label="fill")
    x = np.random.normal(50, 15, 50)
    y = -x + np.random.normal(120, 5, 50)
    fig.add_regression(x, y, row=1, col=0, color="purple", )

    fig.post_process()
    # fig.show()

@test_for_errors
def test_labeling(interactive):
    label = ip.LabelGroup(
        "group_id",
        group_title="GROUP",
        default_label="default",
    )

    fig = ip.Plot(
        interactive=interactive,
    )
    fig.add_line(
        (1,2,4,3),
        label=label.element("line visible")
    )
    fig.add_line(
        (1,2,4,3),
        label=label.element("line invisible", show=False)
    )
    fig.add_line(
        (1,2,4,3),
        label=label.element("line legendonly", legend_only=False)
    )
    fig.add_fill(
        (0,2,3,5),
        (2,3,5,4),
        (1,2,4,3),
        label=label,
    )
    fig.add_regression(
        np.array([0, 1, 3, 4]),
        np.array([0, 1, 2, 3]),
        label="Regression Group",
        label_data="DATA",
        label_ci="CONFIDENCE INTERVAL",
        label_pi="PREDICTION INTERVAL",
        label_reg="REGRESSION",
    )
    fig.post_process()

def test_magic_plot_kwargs():
    Plot_args = inspect.getfullargspec(ip.Plot).args[1:]
    magic_plot_args = inspect.getfullargspec(
        ip.magic_plot(lambda fig: fig)
    ).kwonlyargs

    for arg in Plot_args:
        assert arg in magic_plot_args, (
            f"the keyword `{arg}` should also be added to ip.magic_plot"
        )
