import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
from slab.instruments import *
from slab import AttrDict
import qutip as qt
import time 
# import matplotlib.pyplot as plt

# from TomoAnalysis import TomoAnalysis
from TomoILC import TomoILC 

experiment_path = "s:\Connie\experiments\qram_tprocv1_expts"
config_file = 'config_q3diamond_full688and638_reset.yml'
config_path = os.path.join(experiment_path, "configs", config_file)
with open(config_path, 'r') as cfg_file:
    yaml_cfg = yaml.safe_load(cfg_file)
yaml_cfg = AttrDict(yaml_cfg)

waveforms_path = "S:\QRAM\qram_4QR2\optctrl_pulses"
pulse_filename = yaml_cfg.device.qubit.pulses.pulse_pp.filename
pulse_filepath = os.path.join(waveforms_path, pulse_filename + '.npz')
pulse_IQ = dict() # open file
with np.load(pulse_filepath) as npzfile:
    for key in npzfile.keys():
        pulse_IQ.update({key:npzfile[key]})

psi_str = pulse_filename.split('_')[-1] 
q0 = psi_str[0]
q1 = psi_str[1]

# define the psi_ideal 

if q0 =='0':
    psi0 = qt.basis(2,0)
elif q0 == '1':
    psi0 = qt.basis(2,1)
elif q0 == '+':
    psi0 = (qt.basis(2,0) + qt.basis(2,1)).unit()

if q1 =='0':
    psi1 = qt.basis(2,0)
elif q1 == '1':
    psi1 = qt.basis(2,1)
elif q1 == '+':
    psi1 = (qt.basis(2,0) + qt.basis(2,1)).unit()
    
psi_ideal = qt.tensor(psi0, psi1)
rho_ideal = psi_ideal * psi_ideal.dag()

print(rho_ideal)

def get_baseline_pulse():
    pulse_filename = yaml_cfg.device.qubit.pulses.pulse_pp.filename
    pulse_filepath = os.path.join(os.getcwd(), pulse_filename + '.npz')
    pulse_IQ = dict() # open file
    with np.load(pulse_filepath) as npzfile:
        for key in npzfile.keys():
            pulse_IQ.update({key:npzfile[key]})
    pulse = np.array([pulse_IQ['I_0'], pulse_IQ['Q_0'], pulse_IQ['I_1'], pulse_IQ['Q_1']])
    pulse *= 1e-9
    times = pulse_IQ['times']
    times *= 1e9
    return pulse, times


def get_result(pulse, times, shot_factor=1):

    pulse *= 1e9
    times *= 1e-9
    plt.figure()
    plt.plot(times, pulse[0])
    plt.plot(times, pulse[1])
    plt.plot(times, pulse[2])
    plt.plot(times, pulse[3])
    plt.show()

    # create pulse dictionary
    pulse_IQ = dict()
    pulse_IQ['I_0'] = pulse[0]
    pulse_IQ['Q_0'] = pulse[1]
    pulse_IQ['I_1'] = pulse[2]
    pulse_IQ['Q_1'] = pulse[3]
    pulse_IQ['times'] = times

    tomo_ILC = TomoILC(
        IQ_pulse_seed=pulse_IQ, 
        gains_filename=pulse_filename, 
        nb_qubits=2, 
        n_shot_calib=shot_factor * 4000, 
        n_shot_tomo=shot_factor * 4000, 
        qubit_drive=[0, 1],
        debug=False
    )

    rho = tomo_ILC.get_tomo_results(pulse_IQ)

    print("    fidelity: ", qt.fidelity(rho_ideal, qt.Qobj(rho, dims=[[2, 2], [2, 2]])))

    return rho

    