import json

NLG_parameters = {
    'p0':1.1,
    'S':11590.0,
    'V0':3.63e6,
    'chi':1.1,
    'p0_hat':7.7,
    'S_hat':5030.0,
    'V0_hat':1.7e6,
    'chi_hat':1.3,
    'rho_f':0.8e-9,
    'zeta1':2.1,
    'zeta2':1.7,
    'Sp':13270.0,
    'Sr':2300.0,
    'uf':[0.0,410.0],
    'f':[320.0,320.0],
    'fr':19.0,
    'a':753.0,
    'b':423.0,
    'E':110.0e3,
    'd1':130.0,
    'd2':115.0,
    'mu_s':0.7,
    'mu_p':0.7,
    'kappa':1.0e4,
    'd_bend':20.0}

MLG_parameters = {
    'p0':4.4,
    'S':20106.0,
    'V0':1.70e7,
    'chi':1.1,
    'rho_f':0.8e-9,
    'zeta1':2.6,
    'zeta2':1.9,
    'Sp':17300.0,
    'Sr':3700.0,
    'uf':[0.0,235.0,237.0,500.0],
    'f':[470.0,470.0,320.0,320.0],
    'fr':22.0,
    'a':900.0,
    'b':200.0,
    'E':110.0e3,
    'd1':190.0,
    'd2':160.0,
    'mu_s':0.7,
    'mu_p':0.7,
    'kappa':1.0e4,
    'd_bend':20.0}

NLG_tyre_parameters = {
    'C1':9.590e-3,
    'C2':9.285e-4,
    'pt':1.05,
    'R_t':375.0,
    'R_w':115.0,
    'mass':0.130,
    'damp_ratio':0.15}

MLG_tyre_parameters = {
    'C1':8.469e-3,
    'C2':5.346e-4,
    'pt':1.15,
    'R_t':610.0,
    'R_w':200.0,
    'mass':0.570,
    'damp_ratio':0.15}

with open('NLG_properties.json', 'w') as file:
    json.dump(NLG_parameters, file, indent=4)

with open('MLG_properties.json', 'w') as file:
    json.dump(MLG_parameters, file, indent=4)
    
with open('NLG_tyre_properties.json', 'w') as file:
    json.dump(NLG_tyre_parameters, file, indent=4)
    
with open('MLG_tyre_properties.json', 'w') as file:
    json.dump(MLG_tyre_parameters, file, indent=4)