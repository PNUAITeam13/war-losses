import seaborn as sns
import matplotlib.pyplot as plt
import calplot
from sklearn.preprocessing import OrdinalEncoder

import pandas as pd
import numpy as np

plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)

class VizualLib:

    def __init__(self, X: pd.DataFrame, target_column: str, transform: bool = True):
        self.__data: pd.DataFrame = X.copy()
        self.__transform = transform

        self.__transform_data()
        self.__target = self.__data[target_column]

    def __transform_data(self):
        if not self.__transform:
            return

        object_columns = [col for col in self.__data if self.__data[col].dtype == 'object']

        ordinal_encoder = OrdinalEncoder()

        transformed_X = self.__data.copy()
        transformed_X[object_columns] = ordinal_encoder.fit_transform(transformed_X[object_columns])

        self.__data = transformed_X
        
    def show_data(self):
        print(self.__data)
        return self.__data

    def heat_map(self, figsize: tuple = (16, 12), fmt: str = '.2f', annot_kws: dict = {"size": 8},
                 linewidths: float = 1.9, cmap: str = 'BuGn',
                 linecolor: str = 'w', square=True, **kwargs):
        mask = np.zeros_like(self.__data.corr(), dtype=np.float32)
        mask[np.triu_indices_from(mask)] = True

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(self.__data.corr(), linewidths=linewidths, square=square, cmap="BuGn",
                        linecolor=linecolor, annot=True, annot_kws=annot_kws, mask=mask, cbar_kws={"shrink": .5}, ax=ax,
                        fmt=fmt, **kwargs)

        plt.show()

    def barplot(self, col: str | list[str], figsize: tuple = (10, 6)):
        plt.figure(figsize=figsize)
        sns.barplot(x=self.__data[col], y=self.__target)

    def lineplot(self, col: str | list[str], figsize: tuple = (20, 6)):
        plt.figure(figsize=figsize)
        sns.lineplot(x=self.__data[col], y=self.__target)

    def hist(self, bins=40, figsize=(18, 14)):
        self.__data.hist(bins=bins, figsize=figsize)

    def plt_hist(self, col: str, bins=50, figsize: tuple = (10, 6)):
        plt.figure(figsize=figsize)
        plt.hist(self.__data[col], bins=bins, ec='black', color='#2196f3')
        plt.xlabel(col)
        plt.ylabel(self.__target.name)
        plt.show()

    @staticmethod
    def plot_periodogram(ts, detrend='linear', ax=None):
        from scipy.signal import periodogram
        fs = pd.Timedelta("365D") / pd.Timedelta("1D")
        freqencies, spectrum = periodogram(
            ts,
            fs=fs,
            detrend=detrend,
            window="boxcar",
            scaling='spectrum',
        )
        if ax is None:
            _, ax = plt.subplots()
        ax.step(freqencies, spectrum, color="purple")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
        ax.set_xticklabels(
            [
                "Annual (1)",
                "Semiannual (2)",
                "Quarterly (4)",
                "Bimonthly (6)",
                "Monthly (12)",
                "Biweekly (26)",
                "Weekly (52)",
                "Semiweekly (104)",
            ],
            rotation=30,
        )
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_ylabel("Variance")
        ax.set_title("Periodogram")
        return ax

    @staticmethod
    def moving_average(data, rolling, min_periods, color):
        moving_average = data.rolling(window=rolling, min_periods=min_periods).mean()
        ax = data.plot(style='.', color=color)
        moving_average.plot(ax=ax, linewidth=2, title='Losses', legend=False, color='m')

    @staticmethod
    def figure(data):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        plt.subplot(1, 1, 1)
        sns.lineplot(x=data.index, y='Losses', data=data)
        plt.title('Losses')
        plt.show()
        fig.tight_layout()

    @staticmethod
    def show_calendar(df_values, index):
        values = pd.Series(df_values, index=index)
        calplot.calplot(values,
                cmap = 'RdBu',
                figsize=(20, 13),
                suptitle = 'Calendar',
                suptitle_kws = {'x': 0.5, 'y': 1.0})
        plt.show()

    @staticmethod
    def lagplot(x, y=None, lag=1, standardize=False, ax=None, **kwargs):
        from matplotlib.offsetbox import AnchoredText
        x_ = x.shift(lag)
        if standardize:
            x_ = (x_ - x_.mean()) / x_.std()
        if y is not None:
            y_ = (y - y.mean()) / y.std() if standardize else y
        else:
            y_ = x
        corr = y_.corr(x_)
        if ax is None:
            fig, ax = plt.subplots()
        scatter_kws = dict(
            alpha=0.75,
            s=3,
        )
        line_kws = dict(color='C3', )
        ax = sns.regplot(x=x_,
                         y=y_,
                         scatter_kws=scatter_kws,
                         line_kws=line_kws,
                         lowess=True,
                         ax=ax,
                         **kwargs)
        at = AnchoredText(
            f"{corr:.2f}",
            prop=dict(size="large"),
            frameon=True,
            loc="upper left",
        )
        at.patch.set_boxstyle("square, pad=0.0")
        ax.add_artist(at)
        ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
        return ax
    
    @staticmethod
    def plot_lags(x, y=None, lags=6, nrows=1, lagplot_kwargs={}, **kwargs):
        import math
        kwargs.setdefault('nrows', nrows)
        kwargs.setdefault('ncols', math.ceil(lags / nrows))
        kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
        fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
        for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
            if k + 1 <= lags:
                ax = VizualLib.lagplot(x, y, lag=k + 1, ax=ax, **lagplot_kwargs)
                ax.set_title(f"Lag {k + 1}", fontdict=dict(fontsize=14))
                ax.set(xlabel="", ylabel="")
            else:
                ax.axis('off')
        plt.setp(axs[-1, :], xlabel=x.name)
        plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
        fig.tight_layout(w_pad=0.1, h_pad=0.1)
        return fig
    
    @staticmethod
    def scatter(X, y, result, line_color='pink', dots_color='gray'):
        plt.scatter(X, y, color=line_color)
        plt.plot(X.to_numpy(), result, color=dots_color)
        plt.show()
