# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 14:53:08 2023

@author: devoi

Performing a virtual drop tests of aircraft landing gear.
"""
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from drop_test_sim import DropTestModel
from text_utils import print_msg
from plot_utils import draw_subplot, set_xmargin

def plot_results(drop_test):
    tic = time.perf_counter()
    print_msg("Creating plots...")

    try:
        t_plot_end = drop_test.events_dict["tyre_fall"]["t"][1]
    except IndexError:
        t_plot_end = drop_test.result.t[-1]

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3,
                                                             ncols=2,
                                                             figsize=(4,5),
                                                             dpi=400)
    y_text = {"title": r"\textit{а}",
              "xlabel": r"$t$, c",
              "ylabel": r"$y$, мм"}
    u_text = {"title": r"\textit{б}",
              "xlabel": r"$t$, c",
              "ylabel": r"$u$, мм"}
    v_text = {"title": r"\textit{в}",
              "xlabel": r"$t$, c",
              "ylabel": r"$v$, мм"}
    Fyt_text = {"title": r"\textit{г}",
                "xlabel": r"$t$, c",
                "ylabel": r"$F_y$, кН"}
    Fyu_text = {"title": r"\textit{д}",
                "xlabel": r"$u$, мм",
                "ylabel": r"$F_y$, кН"}
    phi_text = {"title": r"\textit{е}",
                "xlabel": r"$t$, с",
                "ylabel": r"$\dot{\varphi}$, рад/с"}

    Fyy_lims = {"xlim": [0.0, np.max(drop_test.result.y[0][0] -
                                     drop_test.result.y[0])]}

    draw_subplot(ax1, drop_test.result.t, drop_test.result.y[0],
                 text=y_text)
    draw_subplot(ax2, drop_test.result.t, drop_test.result.y[1],
                 text=u_text)
    draw_subplot(ax3, drop_test.result.t, drop_test.result.y[2],
                 text=v_text)
    draw_subplot(ax4, drop_test.result.t,
                 1e-3 * drop_test.result.vertical_force,
                 text=Fyt_text)
    draw_subplot(ax5,
                 drop_test.result.y[1][drop_test.result.t<t_plot_end],
                 1e-3 * drop_test.result.vertical_force[
                     drop_test.result.t<t_plot_end
                     ],
                 text=Fyu_text)
    draw_subplot(ax6, drop_test.result.t,
                 drop_test.result.y[7],
                 text=phi_text)

    u_max = np.max(drop_test.result.y[1][drop_test.result.t<t_plot_end])
    drop_test.landing_gear.strut.deflection = np.linspace(0.0, u_max, 100)
    draw_subplot(ax5, drop_test.landing_gear.strut.deflection,
                 1e-3 * drop_test.landing_gear.strut.force_gas,
                 style="--")

    [set_xmargin(ax, left=0.0, right=0.05) for ax in (ax1, ax2, ax3,
                                                      ax4, ax5, ax6)]

    fig.tight_layout()
    toc = time.perf_counter()
    print_msg(f"Plots creating took {toc-tic:3.2f} s. Rendering...")
    tic = time.perf_counter()
    plt.show()
    toc = time.perf_counter()
    print_msg(f"Image rendering took {toc-tic:3.2f} s. Saving...")
    return fig

if __name__ == "__main__":
    figures_path = os.getcwd() + r"\results\results_fig"
    results_path = os.getcwd() + r"\results\results_data"
    num_pts = 10000
    t_end = 0.05

    test_cond = {"path_to_params":
                 r"C:/Users/devoi/Thesis/dev/aircraft-taxiing-vibrations/parameters_data",
                "lg_type": "MLG",
                "impact_energy_kJ": 161,
                 "cage_mass_t": 33.9,
                 "angle_deg": 5.5}


    drop_test = DropTestModel(test_cond, t_end)
    result = drop_test.get_result()
    drop_test.process_events()
    print_msg("During integration happened:")
    print(f"                    {drop_test.events_dict['tyre_fall']['t'].shape[0]} impacts on surface;")
    print(f"                    {drop_test.events_dict['tyre_rise']['t'].shape[0]} jumps after first impact;")
    print(f"                    {drop_test.events_dict['piston_fall']['t'].shape[0]} direction changes of piston.")

    fig = plot_results(drop_test)
    fig.savefig(figures_path + "\\" + drop_test.filename + ".svg",
                transparent=True)
    print_msg("Done. Exit...")
