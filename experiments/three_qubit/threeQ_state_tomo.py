import matplotlib.pyplot as plt
import numpy as np
from qick import *
import json
from copy import deepcopy

from slab import Experiment, NpEncoder, AttrDict
from tqdm import tqdm_notebook as tqdm

from experiments.clifford_averager_program import QutritAveragerProgram, CliffordAveragerProgram
from experiments.single_qubit.single_shot import hist
from experiments.two_qubit.twoQ_state_tomography import correct_readout_err, fix_neg_counts

def sort_counts_3q(shotsA, shotsB, shotsC):
    # data is returned as n000, n001, ... measured for the 3 qubits
    counts = []
    qcounts_e = [shotsA, shotsB, shotsC]
    qcounts_g = np.logical_not(qcounts_e)
    qcounts = [qcounts_g, qcounts_e]
    for q1_state in (0, 1):
        for q2_state in (0, 1):
            for q3_state in (0, 1):
                and_counts = np.logical_and(qcounts[q1_state][0], qcounts[q2_state][1])
                and_counts = np.logical_and(and_counts, qcounts[q3_state][2])
                counts.append(np.sum(and_counts))
    return np.array(counts)

def make_3q_meas_order():
    meas_order = []
    for pauli1 in ('Z', 'X', 'Y'):
        for pauli2 in ('Z', 'X', 'Y'):
            for pauli3 in ('Z', 'X', 'Y'):
                meas_order.append(pauli1 + pauli2 + pauli3)
    return np.array(meas_order)

def make_3q_calib_order():
    calib_order = []
    for state0 in ('g', 'e'):
        for state1 in ('g', 'e'):
            for state2 in ('g', 'e'):
                calib_order.append(state0 + state1 + state2)
    return np.array(calib_order)

class AbstractStateTomo3QProgram(QutritAveragerProgram):
    """
    Performs a state_prep_pulse (abstract method) on 3 qubits, then measures in a desired basis.
    Repeat this program multiple times in the experiment to loop over all the bases necessary for tomography.
    Experimental Config:
    expt = dict(
        reps: number averages per measurement basis iteration
        qubits: the qubits to perform the 3 qubit tomography on
        basis: see make_3q_meas_order: measurement bases for the 3 qubits
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
        self.setup_measure(qubit=qubits[1], basis=self.basis[1], play=True)
        self.setup_measure(qubit=qubits[2], basis=self.basis[2], play=True)

        # self.sync_all(50)
        # Simultaneous measurement
        syncdelay = self.us2cycles(max(self.cfg.device.readout.relax_delay))
        measure_chs = self.res_chs
        if self.res_ch_types[0] == 'mux4': measure_chs = self.res_chs[0]
        self.measure(pulse_ch=measure_chs, adcs=self.adc_chs, adc_trig_offset=self.cfg.device.readout.trig_offset[0], wait=True, syncdelay=syncdelay) 

    def collect_counts(self, angle=None, threshold=None):
        shots, avgq = self.get_shots(angle=angle, threshold=threshold)
        # collect shots for all adcs, then sorts into e, g based on >/< threshold and angle rotation

        qubits = self.cfg.expt.tomo_qubits
        # get the shots for the qubits we care about
        shots = np.array([shots[self.adc_chs[q]] for q in qubits])

        return sort_counts_3q(shots[0], shots[1], shots[2])

# ===================================================================== #

class ErrorMitigationStateTomo3QProgram(AbstractStateTomo3QProgram):
    """
    Prep the error mitigation matrix state and then perform 3Q state tomography.
    Experimental Config:
    expt = dict(
        reps: number averages per measurement basis iteration
        qubits: the qubits to perform the 3 qubit tomography on
        state_prep_kwargs.prep_state: ggg, gge, geg, gee, ... - the state to prepare in before measuring
    )
    """
    def state_prep_pulse(self, qubits, **kwargs):
        # pass in kwargs via cfg.expt.state_prep_kwargs
        prep_state = kwargs['prep_state'] # should be gg, ge, eg, or ee

        # Do all the calibrations with Q1 in 0+1
        num_in_e = np.sum([char == 'e' for char in prep_state])
        if num_in_e == 0: return

        first_e = prep_state.index('e')
        if num_in_e >= 1:
            self.X_pulse(q=qubits[first_e], play=True)
            self.sync_all(10)

        if num_in_e >= 2: # ee
            second_e = prep_state[first_e+1:].index('e') + first_e + 1
            ZZs = np.reshape(self.cfg.device.qubit.ZZs, (4,4))
            freq = self.freq2reg(self.cfg.device.qubit.f_ge[qubits[second_e]] + ZZs[qubits[second_e], qubits[first_e]], gen_ch=self.qubit_chs[qubits[second_e]])
            waveform = f'qubit{qubits[second_e]}_ZZ{qubits[first_e]}'
            if waveform not in self.pulses:
                sigma_cycles = self.us2cycles(self.pi_sigmas_us[qubits[second_e]], gen_ch=self.qubit_chs[qubits[second_e]])
                self.add_gauss(ch=self.qubit_chs[qubits[second_e]], name=waveform, sigma=sigma_cycles, length=4*sigma_cycles)
                gain = self.cfg.device.qubit.pulses.pi_ge.gain[qubits[second_e]]
            else: gain = self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[qubits[first_e]]
            self.setup_and_pulse(ch=self.qubit_chs[qubits[second_e]], style='arb', freq=freq, phase=0, gain=gain, waveform=waveform)
            self.sync_all(10)

        if num_in_e == 3: # eee    
            third_e = 2
            freq = self.freq2reg( # not totally sure... do ZZ shifts add?
                self.cfg.device.qubit.f_ge[qubits[third_e]] + ZZs[qubits[third_e], qubits[first_e]] + ZZs[qubits[third_e], qubits[second_e]], 
                gen_ch=self.qubit_chs[qubits[third_e]])
            waveform = f'pi_ge_q{qubits[third_e]}'
            gain = self.cfg.device.qubit.pulses.pi_ge.gain[qubits[third_e]]
            self.setup_and_pulse(ch=self.qubit_chs[qubits[third_e]], style='arb', freq=freq, phase=0, gain=gain, waveform=waveform)
            self.sync_all(10)

    def initialize(self):
        self.cfg.expt.basis = 'ZZZ'
        super().initialize()
        self.sync_all(200)

# ===================================================================== #

class TestStateTomo3QProgram(AbstractStateTomo3QProgram):
    def state_prep_pulse(self, qubits, **kwargs):
        qA, qB, qC = self.cfg.expt.tomo_qubits
        
        self.Y_pulse(q=qA, play=True, pihalf=False)
        self.sync_all()
        self.Y_pulse(q=qC, play=True, pihalf=False)
        self.sync_all()

    def initialize(self):

        super().initialize()
        qubits = self.cfg.expt.tomo_qubits
        qA, qB, qC = qubits
        self.cfg.expt.state_prep_kwargs = None

        self.swap_chs = self.cfg.hw.soc.dacs.swap.ch
        self.swap_ch_types = self.cfg.hw.soc.dacs.swap.type

        # initialize ef pulse on qB
        qA, qB, qC = qubits
        # self.handle_gauss_pulse(ch=self.qubit_chs[qB], name=f"ef_qubit{qB}", sigma=self.us2cycles(self.cfg.device.qubit.pulses.pi_ef.sigma[qB], gen_ch=self.qubit_chs[qB]), freq_MHz=self.cfg.device.qubit.f_ef[qB], phase_deg=0, gain=self.cfg.device.qubit.pulses.pi_ef.gain[qB], play=False)

        # initialize EgGf pulse
        # apply the sideband drive on qB, indexed by qA
        for q in range(3):
            if q == 1: continue
            qA = q
            qB = 1
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

class TestStateTomo3QExperiment(Experiment):
    
    def __init__(self, soccfg=None, path='', prefix='TestStateTomography3Q', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        qubits = self.cfg.expt.tomo_qubits
        qA, qB, qC = qubits

        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update({key3: [value3]*num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})
        
        self.meas_order = make_3q_meas_order()
        self.calib_order = make_3q_calib_order() # should match with order of counts for each tomography measurement 
        data={'counts_tomo':[], 'counts_calib':[]}
        self.pulse_dict = dict()

        # Error mitigation measurements: prep in gg, ge, eg, ee to recalibrate measurement angle and measure confusion matrix
        calib_prog_dict = dict()
        for prep_state in tqdm(self.calib_order):
            # print(prep_state)
            cfg = AttrDict(deepcopy(self.cfg))
            cfg.expt.reps = self.cfg.expt.singleshot_reps
            cfg.expt.state_prep_kwargs = dict(prep_state=prep_state)
            err_tomo = ErrorMitigationStateTomo3QProgram(soccfg=self.soccfg, cfg=cfg)
            err_tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False, debug=debug)
            calib_prog_dict.update({prep_state:err_tomo})

        g_prog = calib_prog_dict['ggg']
        Ig, Qg = g_prog.get_shots(verbose=False)
        thresholds_q = [0]*num_qubits_sample
        angles_q = [0]*num_qubits_sample
        for iq, q in enumerate(qubits):
            state = 'ggg'
            state = state[:iq] + 'e' + state[iq+1:]
            e_prog = calib_prog_dict[state]
            Ie, Qe = e_prog.get_shots(verbose=False)
            shot_data = dict(Ig=Ig[q], Qg=Qg[q], Ie=Ie[q], Qe=Qe[q])
            print(f'Qubit  ({q})')
            fid, threshold, angle = hist(data=shot_data, plot=progress, verbose=False)
            thresholds_q[q] = threshold[0]
            angles_q[q] = angle

        print('thresholds', thresholds_q)
        print('angles', angles_q)

        # Process the shots taken for the confusion matrix with the calibration angles
        for prep_state in self.calib_order:
            counts = calib_prog_dict[prep_state].collect_counts(angle=angles_q, threshold=thresholds_q)
            data['counts_calib'].append(counts)

        # Tomography measurements
        for basis in tqdm(self.meas_order):
            # print(basis)
            cfg = AttrDict(deepcopy(self.cfg))
            cfg.expt.basis = basis

            tomo = TestStateTomo3QProgram(soccfg=self.soccfg, cfg=cfg)
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

            counts = tomo.collect_counts(angle=angles_q, threshold=thresholds_q)
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