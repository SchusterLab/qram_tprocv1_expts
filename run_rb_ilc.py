import sys
sys.path.append('C:\\_Lib\\python\\rfsoc\\qram_4QR2_BF5')


from experiments.two_qubit.randbench import *
from slab.instruments import InstrumentManager

import qick
# import json
import os 
import yaml
import h5py

def gaussian(x, sigma):
    return np.exp(-x**2/2/sigma**2)
t_pulse_sigma = 15
t_max = 4/2*t_pulse_sigma # point of max in gaussian

times = np.linspace(0, 4*t_pulse_sigma, 1000)
controls = [gaussian(times - t_max, t_pulse_sigma), np.zeros(len(times))]

controls = 0.98 * np.array(controls).T

expt_path=os.getcwd()+'\data\data_230717'
config_file = 'config_q3diamond_full688and638_reset.yml'
config_path = os.getcwd() + '\\' + config_file

def connect_to_instruments():
    with open("C:\\_Lib\\python\\rfsoc\\qram_4QR2_BF5\\config_q3diamond_full688and638_reset.yml", 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)
    im = InstrumentManager(ns_address='192.168.137.1') # SLAC lab
    soc = QickConfig(im[yaml_cfg['aliases']['soc']].get_cfg())
    return soc

def take_controls_and_measure(soc, times, controls, rb=True):
    
    # print('Config will be', config_path)

    qubit_i = 1
    controls = np.array(controls).T
    #get Is and Qs from controls 
    Is = 1e3 * controls[0]
    Qs = 1e3 * controls[1]


    us_times = 1e-3 * times

    # a-adag is Q and a + adag is I
    if rb:


        experiment = SimultaneousRBExperiment(
            soccfg=soc,
            path=expt_path,
            prefix=f"rb1Q_qubit{qubit_i}",
            config_file="C:\\_Lib\\python\\rfsoc\\qram_4QR2_BF5\\config_q3diamond_full688and638_reset.yml",
        )
        

        cfg_dict = dict(
            start=1, # rb depth start
            step=8, # step rb depth
            expts=17, # number steps
            reps=3500, # number averages per unique sequence
            variations=40, # number different sequences per depth
            gate_char=None, # single qubit clifford gate (str) to characterize. if not None, runs interleaved RB instead of regular RB
            use_EgGf_subspace=False, # specifies whether to run RB treating EgGf as the TLS subspace
            # qubits=[qubit_i], # the qubits to perform simultaneous RB on (if eg-gf, q should be qA != 1)
            qubits=[qubit_i],
            singleshot_reps=10000, # reps per state for singleshot calibration
            post_process='scale', # 'threshold' (uses single shot binning), 'scale' (scale by ge_avgs), or None
        )
        experiment.cfg.expt = cfg_dict

        # run another experiment that is not interleaved and then divide the 2

        experiment.acquire(special=True, 
                           Is = Is, 
                           Qs = Qs, 
                           us = us_times,
                           progress=False)
        
        data = experiment.analyze()

        error_rb = data["error"][0]
        p_rb = data["fit"][0][0]
        cov_rb = data['fit_err'][0][0][0]

        print("RB error is", error_rb)
        print("RB Cov is", cov_rb)
        d=2**(len(experiment.cfg.expt.qubits))

        cfg_dict = dict(
            start=1, # rb depth start
            step=8, # step rb depth
            expts=17, # number steps
            reps=3500, # number averages per unique sequence
            variations=40, # number different sequences per depth
            gate_char='X', # single qubit clifford gate (str) to characterize. if not None, runs interleaved RB instead of regular RB
            use_EgGf_subspace=False, # specifies whether to run RB treating EgGf as the TLS subspace
            # qubits=[qubit_i], # the qubits to perform simultaneous RB on (if eg-gf, q should be qA != 1)
            qubits=[qubit_i],
            singleshot_reps=10000, # reps per state for singleshot calibration
            post_process='scale', # 'threshold' (uses single shot binning), 'scale' (scale by ge_avgs), or None
        )
        
        #difference between scale and threshold?

        experiment.cfg.expt = cfg_dict

        # run another experiment that is not interleaved and then divide the 2

        experiment.acquire(special=True, 
                           Is = Is, 
                           Qs = Qs, 
                           us = us_times,
                           progress=False)
        
        data = experiment.analyze()

        error_irb = data["error"][0]
        p_irb = data["fit"][0][0]
        cov_irb = data['fit_err'][0][0][0]

        err = (d-1)*(1 - p_irb/p_rb)/d

    # play controls using handle_IQ_pulse?
        print("IRB error is", error_irb)
        print("IRB Cov is", cov_irb)

        print("err is", err)
        return [err, error_irb, cov_irb, error_rb, cov_rb]

# soc = connect_to_instruments()
# errs_list = []
# for i in range(30):
#     with h5py.File('errs_list.h5', 'w') as f:
#         f.create_dataset('errs_list', data=errs_list)
#     res = take_controls_and_measure(soc, times, controls, rb=True)   
#     errs_list.append(res)


# print(errs_list)

with h5py.File('errs_list.h5', 'r') as f:
    errs_list = f['errs_list'][()]
    print(errs_list[0:8][:,0])
