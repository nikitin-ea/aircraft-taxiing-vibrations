# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 16:52:27 2023

@author: devoi
"""
import numpy as np


def set_xmargin(ax, left=0.0, right=0.3):
    ax.set_xmargin(0)
    ax.autoscale_view()
    lim = ax.get_xlim()
    delta = np.diff(lim)
    left = lim[0] - delta*left
    right = lim[1] + delta*right
    ax.set_xlim(left,right)

def draw_subplot(ax, xdata, ydata, style=None, text=None, limits=None):
    if style is None:
        ax.plot(xdata, ydata)
    else:
        ax.plot(xdata, ydata, style)
    ax.grid(True)

    if text is not None:
        ax.set_title(f"{text['title']}")
        ax.set_xlabel(f"{text['xlabel']}")
        ax.set_ylabel(f"{text['ylabel']}")

    if limits is not None:
        print("lims is not none")
        try:
            ax.set_xlim(limits["xlim"])
        except KeyError:
            pass
        try:
            ax.set_ylim(limits["ylim"])
        except KeyError:
            pass
