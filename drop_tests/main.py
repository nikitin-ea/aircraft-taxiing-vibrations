# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 14:53:08 2023

@author: devoi

Performing a virtual drop tests of aircraft landing gear.
"""
import os
import drop_test_sim as dts

if __name__ == "__main__":
    figures_path = os.getcwd() + r"\results_fig"
    results_path = os.getcwd() + r"\results_data"
    num_pts = 10000
    t_end = 7

    test_cond = {"path_to_params": r"C:/Users/devoi/Thesis/Git/aircraft-taxiing-vibrations/parameters_data",
                "lg_type": "NLG",
                "impact_energy_kJ": 163,
                 "cage_mass_t": 13.2,
                 "angle_deg": 0.0}
    

    landing_gear = dts.load_landing_gear_model(test_cond)
    tt, result, events, tyre_deflection, vertical_force = dts.perform_virtual_drop_test(
        landing_gear, test_cond, num_pts, t_end)
    events_dict = dts.process_events(events)
    dts.print_msg("During integration happened:")
    print(f"                    {events_dict['tyre_fall']['t'].shape[0]} impacts on surface;")
    print(f"                    {events_dict['tyre_rise']['t'].shape[0]} jumps after first impact;")
    print(f"                    {events_dict['piston_fall']['t'].shape[0]} direction changes of piston.")
    fig = dts.plot_drop_test_results(tt, result, events_dict, tyre_deflection, 
                                     vertical_force, landing_gear)
    dts.save_figure(fig, figures_path, test_cond, file_format="svg")
    dts.save_data(results_path, tt, result, test_cond)
    dts.print_msg("Done. Exit...")