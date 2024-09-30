import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from copy import deepcopy
import json
import math
import os

from qick import *
from qick.helpers import gauss
from slab import Experiment, NpEncoder, AttrDict

import experiments.fitting as fitter

from experiments.single_qubit.single_shot import hist
from experiments.two_qubit.twoQ_state_tomography import ErrorMitigationStateTomo1QProgram
from experiments.qram_protocol_timestepped import QramProtocol1QTomoProgram

from TomoAnalysis import TomoAnalysis



class EgGfPhaseExperiment(Experiment):
    """
    Measure 1 qubit in the X basis to determine the phase after n eg-gf pulses.
    Experimental Config:
    expt = dict(
        start_phase: phase sweep to apply to selected swap
        step_phase
        expts_phase
        reps: number averages per measurement basis iteration
        singleshot_reps: number averages in single shot calibration
        swap_qubit: 2 or 3, driving this qubit to swap with q1
        test_pi_half: True/False
        n_pulses: number of test pulses to repeat
        qubit: qubit to measure
    )
    """

    def __init__(self, soccfg=None, path='', prefix='EgGfPhaseExperiment', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)
        self.cfg.all_qubits = [0, 1, 2, 3]

    def acquire(self, progress=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.readout.frequency)
        self.qubit = self.cfg.expt.qubit

        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update({key3: [value3]*num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})
        
        if 'meas_order' not in self.cfg.expt or self.cfg.expt.meas_order is None:
            self.meas_order = ['X']
        else: self.meas_order = self.cfg.expt.meas_order
        self.calib_order = ['g', 'e'] # should match with order of counts for each tomography measurement 
        data={'counts_tomo':np.zeros((self.cfg.expt.expts_phase, len(self.meas_order), len(self.calib_order))), 'counts_calib':[]}
        self.pulse_dict = dict()

        # ================= #
        # Get single shot calibration for qubits
        # ================= #

        if 'angles' in self.cfg.expt and 'thresholds' in self.cfg.expt and 'ge_avgs' in self.cfg.expt and 'counts_calib' in self.cfg.expt and None not in (self.cfg.expt.angles, self.cfg.expt.thresholds, self.cfg.expt.ge_avgs, self.cfg.expt.counts_calib):
            angles_q = self.cfg.expt.angles
            thresholds_q = self.cfg.expt.thresholds
            ge_avgs_q = self.cfg.expt.ge_avgs
            for q in range(num_qubits_sample):
                if ge_avgs_q[q] is None:
                    ge_avgs_q[q] = np.zeros_like(ge_avgs_q[self.cfg.expt.tomo_qubits[0]]) # just get the shape of the arrays correct by picking the old ge_avgs_q of a q that was definitely measured
            ge_avgs_q = np.array(ge_avgs_q)
            data['counts_calib'] = self.cfg.expt.counts_calib
            print('Re-using provided angles, thresholds, ge_avgs, counts_calib')

        else:
            # Error mitigation measurements: prep in g, e to recalibrate measurement angle and measure confusion matrix
            calib_prog_dict = dict()
            for prep_state in tqdm(self.calib_order):
                # print(prep_state)
                cfg = AttrDict(deepcopy(self.cfg))
                cfg.expt.reps = self.cfg.expt.singleshot_reps
                cfg.expt.state_prep_kwargs = dict(prep_state=prep_state)
                err_tomo = ErrorMitigationStateTomo1QProgram(soccfg=self.soccfg, cfg=cfg)
                err_tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                calib_prog_dict.update({prep_state:err_tomo})

            g_prog = calib_prog_dict['g']
            Ig, Qg = g_prog.get_shots(verbose=False)
            thresholds_q = [0]*num_qubits_sample
            angles_q = [0]*num_qubits_sample
            ge_avgs_q = [[0]*4]*num_qubits_sample

            # Get readout angle + threshold for qubit
            e_prog = calib_prog_dict['e']
            Ie, Qe = e_prog.get_shots(verbose=False)
            shot_data = dict(Ig=Ig[self.qubit], Qg=Qg[self.qubit], Ie=Ie[self.qubit], Qe=Qe[self.qubit])
            fid, threshold, angle = hist(data=shot_data, plot=progress, verbose=False)
            thresholds_q[self.qubit] = threshold[0]
            angles_q[self.qubit] = angle
            ge_avgs_q[self.qubit] = [np.average(Ig[self.qubit]), np.average(Qg[self.qubit]), np.average(Ie[self.qubit]), np.average(Qe[self.qubit])]

            if progress:
                print(f'thresholds={thresholds_q},')
                print(f'angles={angles_q},')
                print(f'ge_avgs={ge_avgs_q}',',')

            # Process the shots taken for the confusion matrix with the calibration angles
            for prep_state in self.calib_order:
                counts = calib_prog_dict[prep_state].collect_counts(angle=angles_q, threshold=thresholds_q)
                data['counts_calib'].append(counts)
            if progress: print(f"counts_calib={np.array(data['counts_calib']).tolist()}")

        data.update(dict(thresholds=thresholds_q, angles=angles_q, ge_avgs=ge_avgs_q)) 

        # ================= #
        # Begin experiment
        # ================= #

        swap_qubit = self.cfg.expt.swap_qubit
        n_pulses = self.cfg.expt.n_pulses
        test_pi_half = self.cfg.expt.test_pi_half
        qubit = self.cfg.expt.qubit

        if test_pi_half and n_pulses % 2 != 0:
            print(f'WARNING: you are doing {n_pulses} pi/2 swaps, which will not give you an integer number of swaps!')
        
        pi_half_swaps = [False, False]
        if swap_qubit == 2:
            init_state = '|0>|0+1>' 
            play_pulses = [1]*self.cfg.expt.n_pulses
            pi_half_swaps[0] = test_pi_half
            ZZ_qubit = None
        elif swap_qubit == 3:
            init_state = '|1>|0+1>' 
            play_pulses = [2]*self.cfg.expt.n_pulses
            pi_half_swaps[1] = test_pi_half
            ZZ_qubit = 0 # measure the 1q tomo using tomography pulses with ZZ from Q0
        else: assert False, f'swap_qubit {swap_qubit} is not allowed'

        assert qubit == 1, 'this experiment is meant to measure qubit 1'
        swap_mod = (n_pulses % 4) if not test_pi_half else ((n_pulses/2) % 4)
        assert swap_mod in [0, 2], f'if you want to measure in the X basis on qubit {qubit}, you need num pi pulses to be even'

        phases = self.cfg.expt["start_phase"] + self.cfg.expt["step_phase"] * np.arange(self.cfg.expt["expts_phase"])

        for i_phase, swap_phase in enumerate(tqdm(phases)):
            # Tomography measurements
            for i_basis, basis in enumerate(self.meas_order):
                # print(basis)
                cfg = AttrDict(deepcopy(self.cfg))
                cfg.expt.timestep = np.inf
                cfg.expt.start = np.inf
                cfg.expt.step = 0
                cfg.expt.expts = 1
                cfg.expt.qubit = qubit

                cfg.expt.basis = basis
                cfg.expt.init_state = init_state
                cfg.expt.play_pulses = play_pulses
                cfg.expt.pi_half_swaps = pi_half_swaps
                cfg.expt.ZZ_qubit = ZZ_qubit
                cfg.expt.add_phase = True

                # print('swap phase', swap_phase)
                if not test_pi_half: cfg.device.qubit.pulses.pi_EgGf_Q.phase[swap_qubit] = swap_phase
                else: cfg.device.qubit.pulses.pi_EgGf_Q.half_phase[swap_qubit] = swap_phase

                tomo = QramProtocol1QTomoProgram(soccfg=self.soccfg, cfg=cfg)
                # print(tomo)
                tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                counts = tomo.collect_counts(angle=angles_q, threshold=thresholds_q)
                data['counts_tomo'][i_phase, i_basis, :] = counts
                self.pulse_dict.update({basis:tomo.pulse_dict})

        data['xpts'] = phases
        self.data=data
        return data

    def analyze(self, data=None):
        if data is None: data = self.data
        counts_temp = np.copy(data['counts_tomo'])
        tomo_analysis = TomoAnalysis(nb_qubits=1, tomo_qubits=None)
        fix_neg_counts = tomo_analysis.fix_neg_counts
        correct_readout_err = tomo_analysis.correct_readout_err
        popln_X = np.zeros(self.cfg.expt.expts_phase)
        for i_phase in range(self.cfg.expt.expts_phase):
            counts = counts_temp[i_phase, 0]
            counts = fix_neg_counts(correct_readout_err([counts], data['counts_calib']))[0]
            # print('adjust', counts)
            popln_X[i_phase] = counts[1]/sum(counts)
        data['popln_X'] = popln_X
        return data

    def display(self, data=None):
        if data is None: data=self.data 

        n_pulses = self.cfg.expt.n_pulses
        test_pi_half = self.cfg.expt.test_pi_half
        swap_mod = (n_pulses % 4) if not test_pi_half else ((n_pulses/2) % 4)
        if swap_mod == 0: self.cfg.expt.final_X_pop = 0 # + state measures as g
        else: self.cfg.expt.final_X_pop = 1

        final_X_pop = self.cfg.expt.final_X_pop # target population
        print('target X pop', final_X_pop)
        best_ind = np.argmin(np.abs(data['popln_X']-final_X_pop))
        best_phase = data['xpts'][best_ind]
        print(f'phase adjust for 1 {"pi/2" if test_pi_half else "pi"} swap:', best_phase, 'deg')
        
        plt.figure()
        plt.plot(data['xpts'], data['popln_X'], '.-')
        plt.xlabel('Swap Phase (deg)')
        plt.ylabel('X axis population')
        plt.title(f'Swap Q{self.cfg.expt.swap_qubit}/Q1, {self.cfg.expt.n_pulses} {"pi/2" if self.cfg.expt.test_pi_half else "pi"} pulses')
        plt.ylim(-0.01, 1.01)
        plt.show()

    
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        # print(self.pulse_dict)
        with self.datafile() as f:
            f.attrs['pulse_dict'] = json.dumps(self.pulse_dict, cls=NpEncoder)
            f.attrs['meas_order'] = json.dumps(self.meas_order, cls=NpEncoder)
            f.attrs['calib_order'] = json.dumps(self.calib_order, cls=NpEncoder)
        return self.fname


# ===================================================================== #