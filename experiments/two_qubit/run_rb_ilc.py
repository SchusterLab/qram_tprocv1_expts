
from experiments.two_qubit.randbench import *

import json
import os 

controls = np.array

def take_controls_and_measure(times, controls, rb=True):
    

    #get Is and Qs from controls 
    Is = 1e3 * controls[0]
    Qs = 1e3 * controls[1]


    us_times = 1e-3 * times

    # a-adag is Q and a + adag is I
    if rb:


        experiment = SimultaneousRBExperiment()
        
        cfg_dict = dict(
            start=1, # rb depth start
            step=8, # step rb depth
            expts=10, # number steps
            reps=1500, # number averages per unique sequence
            variations=30, # number different sequences per depth
            gate_char='X', # single qubit clifford gate (str) to characterize. if not None, runs interleaved RB instead of regular RB
            use_EgGf_subspace=False, # specifies whether to run RB treating EgGf as the TLS subspace
            # qubits=[qubit_i], # the qubits to perform simultaneous RB on (if eg-gf, q should be qA != 1)
            qubits=[1],
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
                           progress=True)
        
        data = experiment.analyze()

        error = data["error"][0]

    # play controls using handle_IQ_pulse?

        return [error]
