import matplotlib.pyplot as plt
import numpy as np
from qick import *
import json
from copy import deepcopy

from slab import Experiment, NpEncoder, AttrDict
from tqdm import tqdm_notebook as tqdm

from experiments.clifford_averager_program import QutritAveragerProgram, CliffordAveragerProgram
from experiments.single_qubit.single_shot import hist

def sort_counts(shotsA, shotsB):
    # data is returned as n00, n01, n10, n11 measured for the two qubits
    n00 = np.sum(np.logical_and(np.logical_not(shotsA), np.logical_not(shotsB)))
    n01 = np.sum(np.logical_and(np.logical_not(shotsA), shotsB))
    n10 = np.sum(np.logical_and(shotsA, np.logical_not(shotsB)))
    n11 = np.sum(np.logical_and(shotsA, shotsB))
    return np.array([n00, n01, n10, n11])


def correct_readout_err(n, n_conf):
    conf_mat = np.array(n_conf, dtype=float)
    for r, row in enumerate(conf_mat):
        conf_mat[r] /= sum(row) # normalize so counts for each state prep sum to 1
    conf_mat = np.transpose(conf_mat) # want counts for each state prep on columns
    conf_mat_inv = np.linalg.inv(conf_mat)
    # C_id = invM . C_noisy
    n = np.array(n, dtype=float)
    for r in range(len(n)):
        n[r] = (conf_mat_inv @ n[r].T).T
    return np.around(n, decimals=5)


def fix_neg_counts(counts):
    counts = np.array(counts)
    assert len(counts.shape) == 2 # 2d array

    for i_n, n in enumerate(counts):
        while len(n[n<0]) > 0: # repeat while still has neg counts
            assert sum(n) > 0, 'Negative sum of counts'
            neg_indices = np.where(n<0)[0]
            neg_vals = n[n<0]
            len_pos = len(n) - len(neg_vals)
            n += sum(neg_vals)/len_pos # redistribute the negative counts over all the positive elements evenly
            for i in neg_indices:
                n[i] = 0

        counts[i_n] = np.round(n)
    return counts

class AbstractStateTomo2QProgram(QutritAveragerProgram):
    """
    Performs a state_prep_pulse (abstract method) on two qubits, then measures in a desired basis.
    Repeat this program multiple times in the experiment to loop over all the bases necessary for tomography.
    Experimental Config:
    expt = dict(
        reps: number averages per measurement basis iteration
        qubits: the qubits to perform the two qubit tomography on (drive applied to the second qubit)
        basis: 'ZZ', 'ZX', 'ZY', 'XZ', 'XX', 'XY', 'YZ', 'YX', 'YY' the measurement bases for the 2 qubits
        state_prep_kwargs: dictionary containing kwargs for the state_prep_pulse function
    )
    """

    def setup_measure(self, qubit, basis:str, play=False, flag='ZZcorrection'):
        """
        Convert string indicating the measurement basis into the appropriate single qubit pulse (pre-measurement pulse)
        """
        assert basis in 'IXYZ'
        assert len(basis) == 1
        if basis == 'X':
            self.Y_pulse(qubit, pihalf=True, play=play, neg=True, flag=flag) # -Y/2 pulse to get from +X to +Z
            # print('x pulse dict', self.pulse_dict)
        elif basis == 'Y': self.X_pulse(qubit, pihalf=True, neg=False, play=play, flag=flag) # X/2 pulse to get from +Y to +Z
        else: pass # measure in I/Z basis
        self.sync_all(15)

    def state_prep_pulse(self, qubits, **kwargs):
        """
        Plays the pulses to prepare the state we want to do tomography on.
        Pass in kwargs to state_prep_pulse through cfg.expt.state_prep_kwargs
        """
        raise NotImplementedError('Inherit this class and implement the state prep method!')

    def initialize(self):
        super().initialize()
        self.sync_all(200)
    
    def body(self):
        # Collect single shots and measure throughout pulses
        qubits = self.cfg.expt.tomo_qubits
        self.basis = self.cfg.expt.basis

        # Phase reset all channels
        for ch in self.gen_chs.keys():
            if self.gen_chs[ch]['mux_freqs'] is None: # doesn't work for the mux channels
                # print('resetting', ch)
                self.setup_and_pulse(ch=ch, style='const', freq=100, phase=0, gain=100, length=10, phrst=1)
            self.sync_all()
        self.sync_all(10)

        # Prep state to characterize
        kwargs = self.cfg.expt.state_prep_kwargs
        if kwargs is None: kwargs = dict()
        self.state_prep_pulse(qubits, **kwargs)
        self.sync_all(10)

        # Go to the basis for the tomography measurement
        self.setup_measure(qubit=qubits[0], basis=self.basis[0], play=True)
        # self.sync_all(10) # necessary for ZZ?
        self.setup_measure(qubit=qubits[1], basis=self.basis[1], play=True)
        # self.sync_all(10)

        # self.sync_all(50)
        # Simultaneous measurement
        syncdelay = self.us2cycles(max(self.cfg.device.readout.relax_delay))
        measure_chs = self.res_chs
        if self.res_ch_types[0] == 'mux4': measure_chs = self.res_chs[0]
        self.measure(pulse_ch=measure_chs, adcs=self.adc_chs, adc_trig_offset=self.cfg.device.readout.trig_offset[0], wait=True, syncdelay=syncdelay) 

    def collect_counts(self, angle=None, threshold=None):
        shots, avgq = self.get_shots(angle=angle, threshold=threshold)
        # collect shots for all adcs, then sorts into e, g based on >/< threshold and angle rotation
        # shots = np.array([np.heaviside(avgi[i] - threshold[i], 0) for i in range(len(self.adc_chs))])

        qubits = self.cfg.expt.tomo_qubits
        # get the shots for the qubits we care about
        shots = np.array([shots[self.adc_chs[q]] for q in qubits])

        return sort_counts(shots[0], shots[1])
    
    # def acquire(self, soc, angle=None, threshold=None, shot_avg=1, load_pulses=True, progress=False, debug=False):
    #     avgi, avgq = super().acquire(soc, load_pulses=load_pulses, progress=progress, debug=debug)
    #     # print()
    #     # print(avgi)
    #     return self.collect_counts(angle=angle, threshold=threshold, shot_avg=shot_avg)

# ===================================================================== #

class ErrorMitigationStateTomo2QProgram(AbstractStateTomo2QProgram):
    """
    Prep the error mitigation matrix state and then perform 2Q state tomography.
    Experimental Config:
    expt = dict(
        reps: number averages per measurement basis iteration
        qubits: the qubits to perform the two qubit tomography on (drive applied to the second qubit)
        state_prep_kwargs.prep_state: gg, ge, eg, ee - the state to prepare in before measuring
        state_prep_kwargs.apply_q1_pi2: whether to initialize Q1 in 0+1 prior to measuring
    )
    """
    def state_prep_pulse(self, qubits, **kwargs):
        # pass in kwargs via cfg.expt.state_prep_kwargs
        prep_state = kwargs['prep_state'] # should be gg, ge, eg, or ee

        # Do all the calibrations with Q1 in 0+1
        if 'apply_q1_pi2' in kwargs: apply_q1_pi2 = kwargs['apply_q1_pi2']
        else: apply_q1_pi2 = False
        if apply_q1_pi2:
            assert 1 not in qubits
            
        if prep_state == 'gg':
            if apply_q1_pi2:
                self.Y_pulse(q=1, pihalf=True, play=True)
                self.sync_all(10)

        elif prep_state == 'ge':
            self.X_pulse(q=qubits[1], play=True)
            self.sync_all(10)
            if apply_q1_pi2:
                self.setup_and_pulse(ch=self.qubit_chs[1], style='arb', freq=self.f_Q1_ZZ_regs[qubits[1]], phase=self.deg2reg(-90, gen_ch=self.qubit_chs[1]), gain=self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[qubits[1]] // 2, waveform=f'qubit1_ZZ{qubits[1]}')
                self.sync_all(10)

        elif prep_state == 'eg':
            self.X_pulse(q=qubits[0], play=True)
            self.sync_all(10)
            if apply_q1_pi2:
                self.setup_and_pulse(ch=self.qubit_chs[1], style='arb', freq=self.f_Q1_ZZ_regs[qubits[0]], phase=self.deg2reg(-90, gen_ch=self.qubit_chs[1]), gain=self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[qubits[0]] // 2, waveform=f'qubit1_ZZ{qubits[0]}')
                self.sync_all(10)

        else: # ee
            self.X_pulse(q=qubits[0], play=True)
            self.sync_all(10)

            ZZs = np.reshape(self.cfg.device.qubit.ZZs, (4,4))
            freq = self.freq2reg(self.cfg.device.qubit.f_ge[qubits[1]] + ZZs[qubits[1], qubits[0]], gen_ch=self.qubit_chs[qubits[1]])
            waveform = f'qubit{qubits[1]}_ZZ{qubits[0]}'
            if waveform not in self.pulses:
                sigma_cycles = self.us2cycles(self.pi_sigmas_us[qubits[1]], gen_ch=self.qubit_chs[qubits[1]])
                self.add_gauss(ch=self.qubit_chs[qubits[1]], name=waveform, sigma=sigma_cycles, length=4*sigma_cycles)
                gain = self.cfg.device.qubit.pulses.pi_ge.gain[qubits[1]]
            else: gain = self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[qubits[0]]
            self.setup_and_pulse(ch=self.qubit_chs[qubits[1]], style='arb', freq=freq, phase=0, gain=gain, waveform=waveform)
            self.sync_all(10)

            if apply_q1_pi2:
                freq = (self.cfg.device.qubit.f_Q1_ZZ[qubits[0]] + self.cfg.device.qubit.f_Q1_ZZ[qubits[1]]) / 2
                freq = self.freq2reg(freq, gen_ch=self.qubit_chs[1])
                waveform = f'qubit1_ZZ{qubits[0]}_ZZ{qubits[1]}'
                sigma_cycles = self.us2cycles(self.pi_sigmas_us[1], gen_ch=self.qubit_chs[1])
                self.add_gauss(ch=self.qubit_chs[1], name=waveform, sigma=sigma_cycles, length=4*sigma_cycles)
                gain = self.cfg.device.qubit.pulses.pi_ge.gain[1] // 2
                self.setup_and_pulse(ch=self.qubit_chs[1], style='arb', freq=freq, phase=self.deg2reg(-90, gen_ch=self.qubit_chs[1]), gain=gain, waveform=waveform)
                self.sync_all(10)

    def initialize(self):
        self.cfg.expt.basis = 'ZZ'
        super().initialize()
        self.sync_all(200)

# ===================================================================== #

class EgGfStateTomo2QProgram(AbstractStateTomo2QProgram):
    """
    Perform the EgGf swap and then perform 2Q state tomography.
    Experimental Config:
    expt = dict(
        reps: number averages per measurement basis iteration
        qubits: the qubits to perform the two qubit tomography on (drive applied to the second qubit)
    )
    """
    def state_prep_pulse(self, qubits, **kwargs):
        qA, qB = self.cfg.expt.tomo_qubits
        
        # self.X_pulse(q=qA, pihalf=False, play=True)
        # self.Xef_pulse(q=qA, pihalf=False, play=True)
        # self.sync_all()
        # assert qB == 1

        # self.Y_pulse(q=0, play=True, pihalf=True) # -> 0+1
        # self.sync_all()

        # freq_reg = int(np.average([self.f_Q1_ZZ_regs[0], self.f_ge_regs[1]]))
        # gain = int(np.average([self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[0], self.cfg.device.qubit.pulses.pi_ge.gain[1]]))
        # sigma_us = np.average([self.pi_Q1_ZZ_sigmas_us[0]/2, self.pi_sigmas_us[1]/2])
        # pi2_sigma_cycles = self.us2cycles(sigma_us, gen_ch=self.qubit_chs[1])
        # phase = self.deg2reg(-90, gen_ch=self.qubit_chs[1]) # +Y/2 -> 0+1
        # self.add_gauss(ch=self.qubit_chs[1], name='qubit1_semiZZ0_half', sigma=pi2_sigma_cycles, length=4*pi2_sigma_cycles)
        # self.setup_and_pulse(ch=self.qubit_chs[1], style='arb', freq=freq_reg, phase=phase, gain=gain, waveform='qubit1_semiZZ0_half')
        # self.sync_all()

        # self.Y_pulse(q=qA, pihalf=True, play=True)
        # self.sync_all()

        # self.X_pulse(q=qA, pihalf=False, play=True, phrst=False)
        # # self.Y_pulse(q=qA, pihalf=True, play=True, divide_len=False)
        # self.sync_all(15)

        # # self.Y_pulse(q=qB, pihalf=True, neg=False, play=True, phrst=False)
        # ZZs = np.reshape(self.cfg.device.qubit.ZZs, (4,4))
        # waveform = f'qubit{qB}_ZZ{qA}'
        # if qB == 1:
        #     freq = self.f_Q1_ZZ_regs[qA]
        #     gain = self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[qA] // 2
        #     sigma_cycles = self.us2cycles(self.pi_Q1_ZZ_sigmas_us[qA], gen_ch=self.qubit_chs[qB])
        # else:
        #     freq = self.freq2reg(self.cfg.device.qubit.f_ge[qB] + ZZs[qB, qA], gen_ch=self.qubit_chs[qB])
        #     gain = self.cfg.device.qubit.pulses.pi_ge.gain[qB] // 2
        #     sigma_cycles = self.us2cycles(self.pi_sigmas_us[qB], gen_ch=self.qubit_chs[qB])
        # self.add_gauss(ch=self.qubit_chs[qB], name=waveform, sigma=sigma_cycles, length=4*sigma_cycles)
        # self.setup_and_pulse(ch=self.qubit_chs[qB], style='arb', freq=freq, phase=self.deg2reg(0, gen_ch=self.qubit_chs[qB]), gain=gain, waveform=waveform)
        # self.sync_all(15)

        # # self.setup_and_pulse(ch=self.qubit_chs[qB], style='arb', freq=freq, phase=self.deg2reg(0, gen_ch=self.qubit_chs[qB]), gain=gain, waveform=waveform)

        # print('freq', freq, self.freq2reg(3458.5946219528296, gen_ch=self.qubit_chs[qB]))
        # print('freq2', self.cfg.device.qubit.f_ge[qB], self.freq2reg(self.cfg.device.qubit.f_ge[qB] - (3463.808810074577 -3458.5946219528296), gen_ch=self.qubit_chs[qB]))
        # self.set_pulse_registers(
        #     ch=self.qubit_chs[qB],
        #     style="arb",
        #     freq=self.freq2reg(self.cfg.device.qubit.f_ge[qB], gen_ch=self.qubit_chs[qB]),
        #     phase=0,
        #     gain=0, # gain set by update
        #     waveform=f'X_q{qB}')
        # self.mathi(self.q_rps[qB], self.r_gain, self.r_gain2, "+", 0)
        # self.pulse(ch=self.qubit_chs[qB])
        # self.setup_and_pulse(ch=self.qubit_chs[qB], style='arb', freq=self.freq2reg(self.cfg.device.qubit.f_ge[qB], gen_ch=self.qubit_chs[qB]), phase=self.deg2reg(0, gen_ch=self.qubit_chs[qB]), gain=gain, waveform=f'qubit{qB}')

        # self.X_pulse(q=qB, pihalf=True, play=True, divide_len=True)
        # self.sync_all()
        # self.X_pulse(q=qB, pihalf=True, play=True, divide_len=False)
        # self.sync_all(15)

        # # pass in kwargs via cfg.expt.state_prep_kwargs
        # self.Y_pulse(q=0, play=True, pihalf=True)

        # # self.X_pulse(q=qubits[0], play=True, pihalf=False)
        # self.sync_all()
        # self.pi_Q1_ZZ_sigmas_us = self.cfg.device.qubit.pulses.pi_Q1_ZZ.sigma
        # pi_Q1_ZZ_sigma_cycles = self.us2cycles(self.pi_Q1_ZZ_sigmas_us[0], gen_ch=self.qubit_chs[1]) // 2
        # self.add_gauss(ch=self.qubit_chs[1], name='qubit1_ZZ0_half', sigma=pi_Q1_ZZ_sigma_cycles, length=4*pi_Q1_ZZ_sigma_cycles)
        # self.setup_and_pulse(ch=self.qubit_chs[1], style='arb', freq=self.f_Q1_ZZ_regs[0], phase=0, gain=self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[0], waveform='qubit1_ZZ0_half')


        # self.X_pulse(q=qubits[0], pihalf=False, play=True)
        # self.sync_all()
        # self.X_pulse(q=qubits[1], pihalf=False, play=True)

        # initialize to Eg
        self.X_pulse(q=1, play=True, pihalf=True)
        # self.Xef_pulse(q=1, play=True, pihalf=False)

        # # apply Eg -> Gf pulse on B: expect to end in Gf
        # type = self.cfg.device.qubit.pulses.pi_EgGf.type[qA]
        # pulse = self.handle_const_pulse
        # if type == 'gauss': pulse = self.handle_gauss_pulse
        # elif type == 'flat_top': pulse = self.handle_flat_top_pulse
        # pulse(name=f'pi_EgGf_{qubits[0]}{qubits[1]}', reload=False, play=True)

        # type = self.cfg.device.qubit.pulses.pi_EgGf.type[2]
        # pulse = self.handle_const_pulse
        # if type == 'gauss': pulse = self.handle_gauss_pulse
        # elif type == 'flat_top': pulse = self.handle_flat_top_pulse
        # pulse(name=f'pi_EgGf_31', reload=False, play=True)

        # self.setup_and_pulse(ch=self.swap_chs[qA], style='flat_top', freq=self.freq2reg(self.cfg.device.qubit.f_EgGf[qA], gen_ch=self.swap_chs[qA]), phase=self.deg2reg(0, gen_ch=self.swap_chs[qA]), gain=self.cfg.device.qubit.pulses.pi_EgGf.gain[qA], length=self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf.sigma[qA], gen_ch=self.swap_chs[qA]))
        # self.sync_all()
        # self.setup_and_pulse(ch=self.swap_chs[qA], style='const', freq=self.freq2reg(self.cfg.device.qubit.f_EgGf[qA], gen_ch=self.swap_chs[qA]), phase=self.deg2reg(-90, gen_ch=self.swap_chs[qA]), gain=self.cfg.device.qubit.pulses.pi_EgGf.gain[qA], length=self.us2cycles(0.80*self.cfg.device.qubit.pulses.pi_EgGf.sigma[qA], gen_ch=self.swap_chs[qA]))
        # self.sync_all()
        # self.setup_and_pulse(ch=self.swap_chs[qA], style='const', freq=self.freq2reg(self.cfg.device.qubit.f_EgGf[qA], gen_ch=self.swap_chs[qA]), phase=self.deg2reg(-90, gen_ch=self.swap_chs[qA]), gain=self.cfg.device.qubit.pulses.pi_EgGf.gain[qA], length=self.us2cycles(0.80*self.cfg.device.qubit.pulses.pi_EgGf.sigma[qA], gen_ch=self.swap_chs[qA]))
        # self.sync_all()

        # take qubit B f->e: expect to end in Ge (or Eg if incomplete Eg-Gf)
        # self.X_pulse(q=qubits[0], play=True) # G->E on qubits[0]
        # self.sync_all() # necessary for ZZ?
        # self.handle_gauss_pulse(name=f'ef_qubit{qB}', reload=False, play=True) # f->e on qubits[1]
        # self.sync_all()
    
    # def update(self):
    #     qA, qB = self.cfg.expt.tomo_qubits
    #     step = self.cfg.expt.step
    #     if self.qubit_ch_types[qB] == 'int4': step = step << 16
    #     self.mathi(self.q_rps[qB], self.r_gain2, self.r_gain2, '+', step) # update test gain

    def initialize(self):

        super().initialize()
        qubits = self.cfg.expt.tomo_qubits
        qA, qB = qubits
        self.cfg.expt.state_prep_kwargs = None

        # if self.qubit_ch_types[qB] == 'int4':
        #     self.r_gain = self.sreg(self.qubit_chs[qB], "addr") # get gain register for qubit_ch    
        # else: self.r_gain = self.sreg(self.qubit_chs[qB], "gain") # get gain register for qubit_ch    
        # self.r_gain2 = 4
        # self.safe_regwi(self.q_rps[qB], self.r_gain2, self.cfg.expt.start)

        self.swap_chs = self.cfg.hw.soc.dacs.swap.ch
        self.swap_ch_types = self.cfg.hw.soc.dacs.swap.type

        # initialize ef pulse on qB
        qA, qB = qubits
        # self.handle_gauss_pulse(ch=self.qubit_chs[qB], name=f"ef_qubit{qB}", sigma=self.us2cycles(self.cfg.device.qubit.pulses.pi_ef.sigma[qB], gen_ch=self.qubit_chs[qB]), freq_MHz=self.cfg.device.qubit.f_ef[qB], phase_deg=0, gain=self.cfg.device.qubit.pulses.pi_ef.gain[qB], play=False)

        # initialize EgGf pulse
        # apply the sideband drive on qB, indexed by qA
        if qA != 1 and qB == 1:
            type = self.cfg.device.qubit.pulses.pi_EgGf.type[qA]
            freq_MHz = self.cfg.device.qubit.f_EgGf[qA]
            gain = self.cfg.device.qubit.pulses.pi_EgGf.gain[qA]
            if type == 'const':
                sigma = self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf.sigma[qA], gen_ch=self.swap_chs[qA])
                self.handle_const_pulse(name=f'pi_EgGf_{qA}{qB}', ch=self.swap_chs[qA], length=sigma, freq_MHz=freq_MHz, phase_deg=0, gain=gain, play=False) 
            elif type == 'gauss':
                sigma = self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf.sigma[qA], gen_ch=self.swap_chs[qA])
                self.handle_gauss_pulse(name=f'pi_EgGf_{qA}{qB}', ch=self.swap_chs[qA], sigma=sigma, freq_MHz=freq_MHz, phase_deg=0, gain=gain, play=False)
            elif type == 'flat_top':
                flat_length = self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf.sigma[qA], gen_ch=self.swap_chs[qA]) - 3*4
                self.handle_flat_top_pulse(name=f'pi_EgGf_{qA}{qB}', ch=self.swap_chs[qA], flat_length=flat_length, freq_MHz=freq_MHz, phase_deg=0, gain=gain, play=False) 
            else: assert False, f'Pulse type {type} not supported.'
        self.sync_all(200)

# ===================================================================== #

class EgGfStateTomographyExperiment(Experiment):
# outer loop over measurement bases
# set the state prep pulse to be preparing the gg, ge, eg, ee states for confusion matrix
    """
    Perform state tomography on the EgGf state with error mitigation.
    Experimental Config:
    expt = dict(
        reps: number averages per measurement basis iteration
        singleshot_reps: number averages in single shot calibration
        calib_apply_q1_pi2: initialize Q1 in 0+1 for all calibrations
        tomo_qubits: the qubits to perform the two qubit tomography on (drive applied to the second qubit)
    )
    """

    def __init__(self, soccfg=None, path='', prefix='EgGfStateTomography', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        qA, qB = self.cfg.expt.tomo_qubits

        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update({key3: [value3]*num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})
        
        self.meas_order = ['ZZ', 'ZX', 'ZY', 'XZ', 'XX', 'XY', 'YZ', 'YX', 'YY']
        self.calib_order = ['gg', 'ge', 'eg', 'ee'] # should match with order of counts for each tomography measurement 
        data={'counts_tomo':[], 'counts_calib':[]}
        self.pulse_dict = dict()

        # Error mitigation measurements: prep in gg, ge, eg, ee to recalibrate measurement angle and measure confusion matrix
        calib_prog_dict = dict()
        for prep_state in tqdm(self.calib_order):
            # print(prep_state)
            cfg = AttrDict(deepcopy(self.cfg))
            cfg.expt.reps = self.cfg.expt.singleshot_reps
            cfg.expt.state_prep_kwargs = dict(prep_state=prep_state, apply_q1_pi2=cfg.expt.calib_apply_q1_pi2)
            err_tomo = ErrorMitigationStateTomo2QProgram(soccfg=self.soccfg, cfg=cfg)
            err_tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False, debug=debug)
            calib_prog_dict.update({prep_state:err_tomo})

        g_prog = calib_prog_dict['gg']
        Ig, Qg = g_prog.get_shots(verbose=False)
        threshold = [0]*num_qubits_sample
        angle = [0]*num_qubits_sample

        # Get readout angle + threshold for qubit A
        e_prog = calib_prog_dict['eg']
        Ie, Qe = e_prog.get_shots(verbose=False)
        shot_data = dict(Ig=Ig[qA], Qg=Qg[qA], Ie=Ie[qA], Qe=Qe[qA])
        print(f'Qubit  ({qA})')
        fid, thresholdA, angleA = hist(data=shot_data, plot=True, verbose=False)
        threshold[qA] = thresholdA[0]
        angle[qA] = angleA

        # Get readout angle + threshold for qubit B
        e_prog = calib_prog_dict['ge']
        Ie, Qe = e_prog.get_shots(verbose=False)
        shot_data = dict(Ig=Ig[qB], Qg=Qg[qB], Ie=Ie[qB], Qe=Qe[qB])
        print(f'Qubit  ({qB})')
        fid, thresholdB, angleB = hist(data=shot_data, plot=True, verbose=False)
        threshold[qB] = thresholdB[0]
        angle[qB] = angleB

        print('thresholds', threshold)
        print('angles', angle)

        # Process the shots taken for the confusion matrix with the calibration angles
        for prep_state in self.calib_order:
            counts = calib_prog_dict[prep_state].collect_counts(angle=angle, threshold=threshold)
            data['counts_calib'].append(counts)

        # Tomography measurements
        for basis in tqdm(self.meas_order):
            # print(basis)
            cfg = AttrDict(deepcopy(self.cfg))
            cfg.expt.basis = basis

            # initialize registers
            # cfg.expt.update(dict(
            #     start=0,
            #     step=int(32000/101),
            #     expts=101,
            #     rounds=10,
            # ))
            tomo = EgGfStateTomo2QProgram(soccfg=self.soccfg, cfg=cfg)
            # print(tomo)
            # from qick.helpers import progs2json
            # print(progs2json([tomo.dump_prog()]))
            # xpts, avgi, avgq = tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False, debug=debug)
            avgi, avgq = tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False, debug=debug)

            # print(basis)
            adc_chs = self.cfg.hw.soc.adcs.readout.ch
            # avgi, avgq = tomo.get_shots(angle=None, avg_shots=True)
            # for q in self.cfg.expt.tomo_qubits:
            #     print('q', q, 'avgi', avgi[adc_chs[q]])
            #     print('q', q, 'avgq', avgq[adc_chs[q]])
            #     print('q', q, 'amps', np.abs(avgi[adc_chs[q]]+1j*avgi[adc_chs[q]]))

            counts = tomo.collect_counts(angle=angle, threshold=threshold)
            data['counts_tomo'].append(counts)
            self.pulse_dict.update({basis:tomo.pulse_dict})

        self.data=data
        return data

    def analyze(self, data=None, **kwargs):
        if data is None: data = self.data
        print('Analyze function does nothing, use the analysis notebook.')
        return data

    def display(self, qubit, data=None, fit=True, **kwargs):
        if data is None: data=self.data 
        print('Display function does nothing, use the analysis notebook.')
    
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
# 1 QUBIT TOMOGRAPHY CLASSES #
# ===================================================================== #

class AbstractStateTomo1QProgram(AbstractStateTomo2QProgram):
    """
    Performs a state_prep_pulse (abstract method) on 1 qubit, then measures in a desired basis.
    Repeat this program multiple times in the experiment to loop over all the bases necessary for tomography.
    Experimental Config:
    expt = dict(
        reps: number averages per measurement basis iteration
        qubits: the qubits to perform the two qubit tomography on (drive applied to the second qubit)
        basis: 'Z', 'X', 'Y' the measurement bases for the qubit
        state_prep_kwargs: dictionary containing kwargs for the state_prep_pulse function
    )
    """

    def setup_measure(self, basis:str, play=False):
        """
        Convert string indicating the measurement basis into the appropriate single qubit pulse (pre-measurement pulse)
        """
        super().setup_measure(qubit=self.qubit, basis=basis, play=play)

    def state_prep_pulse(self, **kwargs):
        """
        Plays the pulses to prepare the state we want to do tomography on.
        Pass in kwargs to state_prep_pulse through cfg.expt.state_prep_kwargs
        """
        raise NotImplementedError('Inherit this class and implement the state prep method!')

    def initialize(self):
        super().initialize()
        assert len(np.shape(self.cfg.expt.qubit)) == 0
        self.qubit = self.cfg.expt.qubit
    
    def body(self):
        # Collect single shots and measure throughout pulses
        self.basis = self.cfg.expt.basis

        # Prep state to characterize
        kwargs = self.cfg.expt.state_prep_kwargs
        if kwargs is None: kwargs = dict()
        self.state_prep_pulse(**kwargs)
        self.sync_all(5)

        # Go to the basis for the tomography measurement
        self.setup_measure(basis=self.basis[0], play=True)
        self.sync_all()

        # Simultaneous measurement
        syncdelay = self.us2cycles(max(self.cfg.device.readout.relax_delay))
        measure_chs = self.res_chs
        if self.res_ch_types[0] == 'mux4': measure_chs = self.res_chs[0]
        self.measure(pulse_ch=measure_chs, adcs=self.adc_chs, adc_trig_offset=self.cfg.device.readout.trig_offset[0], wait=True, syncdelay=syncdelay) 

    def collect_counts(self, angle=None, threshold=None):
        avgi, avgq = self.get_shots(angle=angle)
        # collect shots for all adcs, then sorts into e, g based on >/< threshold and angle rotation
        shots = np.array([np.heaviside(avgi[i] - threshold[i], 0) for i in range(len(self.adc_chs))])

        # get the shots for the qubits we care about
        shots = np.array(shots[self.adc_chs[self.qubit]])

        # print(self.adc_chs[qubits[0]], angle, self.ro_chs) 
        # print('shots 0', shots[0])
        # print('shots 1', shots[1])
        # print()

        # data is returned as n00, n01, n10, n11 measured for the two qubits
        n0 = np.sum(np.logical_not(shots))
        n1 = np.sum(shots)
        return np.array([n0, n1])

    # def acquire(self, soc, angle=None, threshold=None, shot_avg=1, load_pulses=True, progress=False, debug=False):
    #     avgi, avgq = super().acquire(soc, load_pulses=load_pulses, progress=progress, debug=debug)
    #     # print()
    #     # print(avgi)
    #     return self.collect_counts(angle=angle, threshold=threshold, shot_avg=shot_avg)

# ===================================================================== #

class ErrorMitigationStateTomo1QProgram(AbstractStateTomo1QProgram):
    """
    Prep the error mitigation matrix state and then perform 2Q state tomography.
    Experimental Config:
    expt = dict(
        reps: number averages per measurement basis iteration
        qubits: the qubits to perform the two qubit tomography on (drive applied to the second qubit)
        state_prep_kwargs.prep_state: gg, ge, eg, ee - the state to prepare in before measuring
    )
    """
    def state_prep_pulse(self, **kwargs):
        # pass in kwargs via cfg.expt.state_prep_kwargs
        prep_state = kwargs['prep_state'] # should be gg, ge, eg, or ee
        if prep_state == 'e':
            # print('q0: e')
            self.X_pulse(q=self.qubit, play=True)
            self.sync_all() # necessary for ZZ?
        else:
            # print('q0: g')
            assert prep_state == 'g'
            
    def initialize(self):
        super().initialize()
        self.cfg.expt.basis = 'Z'
        self.sync_all(200)

# ===================================================================== #

class StateTomo1QProgram(AbstractStateTomo1QProgram):
    """
    Setup a state and then perform 1Q state tomography.
    Experimental Config:
    expt = dict(
        reps: number averages per measurement basis iteration
    )
    """
    def state_prep_pulse(self, **kwargs):
        # pass in kwargs via cfg.expt.state_prep_kwargs
        # self.X_pulse(q=self.qubit, play=True, pihalf=True, neg=False)
        self.Y_pulse(q=self.qubit, play=True, pihalf=True, neg=False)

        # self.Y_pulse(q=0, play=True)
        # self.sync_all()

        # pi2_Q1_ZZ_sigma_cycles = self.us2cycles(self.pi_Q1_ZZ_sigmas_us[0], gen_ch=self.qubit_chs[1]) // 2
        # phase = self.deg2reg(-90, gen_ch=self.qubit_chs[1]) # +Y/2 -> 0+1
        # self.add_gauss(ch=self.qubit_chs[1], name='qubit1_ZZ0_half', sigma=pi2_Q1_ZZ_sigma_cycles, length=4*pi2_Q1_ZZ_sigma_cycles)
        # self.setup_and_pulse(ch=self.qubit_chs[1], style='arb', freq=self.f_Q1_ZZ_regs[0], phase=phase, gain=self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[0], waveform='qubit1_ZZ0_half')
        # self.sync_all()

    def initialize(self):
        super().initialize()
        self.cfg.expt.state_prep_kwargs = None

# ===================================================================== #

class StateTomography1QExperiment(Experiment):
# outer loop over measurement bases
# set the state prep pulse to be preparing the gg, ge, eg, ee states for confusion matrix
    """
    Perform state tomography on 1Q state with error mitigation.
    Experimental Config:
    expt = dict(
        reps: number averages per measurement basis iteration
    )
    """

    def __init__(self, soccfg=None, path='', prefix='StateTomography1Q', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        q = self.cfg.expt.qubit

        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update({key3: [value3]*num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})
        
        self.meas_order = ['Z', 'X', 'Y']
        self.calib_order = ['g', 'e'] # should match with order of counts for each tomography measurement 
        data={'counts_tomo':[], 'counts_calib':[]}
        self.pulse_dict = dict()

        # Error mitigation measurements: prep in g, e to recalibrate measurement angle and measure confusion matrix
        calib_prog_dict = dict()
        for prep_state in tqdm(self.calib_order):
            # print(prep_state)
            cfg = AttrDict(deepcopy(self.cfg))
            cfg.expt.state_prep_kwargs = dict(prep_state=prep_state)
            err_tomo = ErrorMitigationStateTomo1QProgram(soccfg=self.soccfg, cfg=cfg)
            err_tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False, debug=debug)
            calib_prog_dict.update({prep_state:err_tomo})

        g_prog = calib_prog_dict['g']
        Ig, Qg = g_prog.get_shots(verbose=False)
        threshold = [0]*num_qubits_sample
        angle = [0]*num_qubits_sample

        # Get readout angle + threshold for qubit
        e_prog = calib_prog_dict['e']
        Ie, Qe = e_prog.get_shots(verbose=False)
        shot_data = dict(Ig=Ig[q], Qg=Qg[q], Ie=Ie[q], Qe=Qe[q])
        fid, thresholdq, angleq = hist(data=shot_data, plot=True, verbose=False)
        threshold[q] = thresholdq
        angle[q] = angleq

        print('thresholds', threshold)
        print('angles', angle)

        # Process the shots taken for the confusion matrix with the calibration angles
        for prep_state in self.calib_order:
            counts = calib_prog_dict[prep_state].collect_counts(angle=angle, threshold=threshold)
            data['counts_calib'].append(counts)

        # Tomography measurements
        for basis in tqdm(self.meas_order):
            # print(basis)
            cfg = AttrDict(deepcopy(self.cfg))
            cfg.expt.basis = basis
            tomo = StateTomo1QProgram(soccfg=self.soccfg, cfg=cfg)
            tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False, debug=debug)
            counts = tomo.collect_counts(angle=angle, threshold=threshold)
            data['counts_tomo'].append(counts)
            self.pulse_dict.update({basis:tomo.pulse_dict})

        self.data=data
        return data

    def analyze(self, data=None, **kwargs):
        if data is None: data = self.data
        print('Analyze function does nothing, use the analysis notebook.')
        return data

    def display(self, qubit, data=None, fit=True, **kwargs):
        if data is None: data=self.data 
        print('Display function does nothing, use the analysis notebook.')
    
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        # print(self.pulse_dict)
        with self.datafile() as f:
            f.attrs['pulse_dict'] = json.dumps(self.pulse_dict, cls=NpEncoder)
            f.attrs['meas_order'] = json.dumps(self.meas_order, cls=NpEncoder)
            f.attrs['calib_order'] = json.dumps(self.calib_order, cls=NpEncoder)
        return self.fname