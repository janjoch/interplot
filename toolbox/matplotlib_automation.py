"""
Deprecated. Use toolbox.plot instead.

However, some code snippets might still be valuable to copy&paste.
"""


import os

import matplotlib.pyplot as plt


class GenericClass:
    def _generate_plot_prepare(self, plotFormat={}):
        """
        sets display options for plots
        """

        plotFormatInt = self._plot_format_int.copy()
        plotFormatInt.update(plotFormat)

        plt.rcParams["axes.facecolor"] = plotFormatInt["faceColorPlot"]
        plt.figure(
            facecolor=plotFormatInt["faceColor"],
            figsize=plotFormatInt["figsize"],
        )

        for line in plotFormatInt["xLines"]:
            preset = {
                "pos": 0,
                "c": "grey",
                "ls": "-",
                "lw": 2,
                "range": (0, 1),
            }
            preset.update(line)

            plt.axvline(
                x=preset["pos"],
                c=preset["c"],
                ls=preset["ls"],
                lw=preset["lw"],
                ymin=preset["range"][0],
                ymax=preset["range"][1],
            )

        for line in plotFormatInt["yLines"]:
            preset = {
                "pos": 0,
                "c": "grey",
                "ls": "-",
                "lw": 2,
                "range": (0, 1),
            }
            preset.update(line)

            plt.axhline(
                y=preset["pos"],
                c=preset["c"],
                ls=preset["ls"],
                lw=preset["lw"],
                xmin=preset["range"][0],
                xmax=preset["range"][1],
            )

    def _generate_plot_line(
        self,
        x,
        y=None,
        lineStyle="x-",
        c=None,
        label=None,
    ):
        """
        Draws a line into the plot
        """

        if y is None:
            plt.plot(x, lineStyle, c=c, label=label)
        else:
            plt.plot(x, y, lineStyle, c=c, label=label)

    def _generate_plot_finish(
        self, title, xlab=None, ylab=None, cycles=None, plotFormat={}
    ):
        """
        finishes the plot layout and displays/exports it
        """

        plotFormatInt = self._plot_format_int.copy()
        plotFormatInt.update(plotFormat)

        if plotFormatInt["showGrid"]:
            plt.grid(linestyle=plotFormatInt["gridStyle"])

        # compose plot title
        title_annex = ""
        if plotFormatInt["titleDUT"] and self.DUT is not None:
            title_annex += ", " + str(self.DUT)

        if plotFormatInt["titleSN"] and self.SN is not None:
            title_annex += ", SN " + str(self.SN)

        if plotFormatInt["title_annex"] != "":
            title_annex += ", " + plotFormatInt["title_annex"]
        plt.title(title + title_annex, fontsize=14)
        # plt.suptitle("Suptitle", fontsize=18)

        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.xlim(plotFormatInt["xlim"])
        plt.ylim(plotFormatInt["ylim"])
        plt.xticks(plotFormatInt["xTicks"])
        plt.yticks(plotFormatInt["yTicks"])

        for kwargs in plotFormatInt["text"]:
            plt.text(**kwargs)

        if plotFormatInt["legend"]:
            plt.legend(loc=plotFormatInt["legendLoc"])

        # save plot
        if plotFormatInt["exportImg"]:

            if plotFormatInt["exportName"] == "" or plotFormatInt["exportAdd"]:
                plotFormatInt["exportName"] += title + "_SN" + str(self.SN)
                if plotFormatInt["figsize"] is not None:
                    plotFormatInt["exportName"] += (
                        "_"
                        + str(plotFormatInt["figsize"][0])
                        + "x"
                        + str(plotFormatInt["figsize"][1])
                    )

            plotFormatInt["exportName"] += "." + plotFormatInt["exportType"]

            plotFormatInt["exportName"] = (
                plotFormatInt["exportName"]
                .replace("/", "_").replace("\\", "_")
            )

            path = os.path.join(
                plotFormatInt["exportPath"], plotFormatInt["exportName"]
            )
            plt.savefig(
                path,
                facecolor=plotFormatInt["faceColor"],
                bbox_inches="tight",
            )

        if plotFormatInt["showPlot"]:
            plt.show()
