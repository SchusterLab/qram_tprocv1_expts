import matplotlib.pyplot as plt
import numpy as np
from qick import *
import json

from slab import Experiment, NpEncoder, AttrDict
from tqdm import tqdm_notebook as tqdm

from experiments.clifford_averager_program import CliffordAveragerProgram

class AbstractStateTomo2QProgram(CliffordAveragerProgram):
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

    def setup_basis_measure(self, qubit, basis:str, play=False):
        """
        Convert string indicating the measurement basis into the appropriate single qubit pulse (pre-measurement pulse)
        """
        assert basis in 'IXYZ'
        assert len(basis) == 1
        if basis == 'X': self.Y_pulse(qubit, pihalf=True, play=play) # Y/2 pulse
        elif basis == 'Y': self.X_pulse(qubit, pihalf=True, neg=True, play=play) # -X/2 pulse
        else: return # measure in I/Z basis

    def state_prep_pulse(self, qubits, **kwargs):
        """
        Plays the pulses to prepare the state we want to do tomography on.
        Pass in kwargs to state_prep_pulse through cfg.expt.state_prep_kwargs
        """
        raise NotImplementedError('Inherit this class and implement the state prep method!')

    def initialize(self):
        super().initialize()
        self.sync_all(self.us2cycles(0.2))
    
    def body(self):
        # Collect single shots and measure throughout pulses
        qubits = self.cfg.expt.qubits
        self.basis = self.cfg.expt.basis

        syncdelay = self.us2cycles(max(self.cfg.device.readout.relax_delay))

        # Prep state to characterize
        kwargs = self.cfg.expt.state_prep_kwargs
        if kwargs is None: kwargs = dict()
        self.state_prep_pulse(qubits, **kwargs)
        self.sync_all(self.us2cycles(0.01)) # align channels and wait 10ns

        # Go to the basis for the tomography measurement
        self.setup_basis_measure(qubit=qubits[0], basis=self.basis[0], play=True)
        self.sync_all() # necessary for ZZ?
        self.setup_basis_measure(qubit=qubits[1], basis=self.basis[1], play=True)
        self.sync_all(self.us2cycles(0.01)) # align channels and wait 10ns

        # Simultaneous measurement
        self.measure(pulse_ch=self.res_chs, adcs=[0,1], adc_trig_offset=self.cfg.device.readout.trig_offset, wait=True, syncdelay=syncdelay) 

    def collect_counts(self, threshold=None, shot_avg=1):
        # collect shots for 2 adcs (indexed by qubit order) in the I channel, then sorts into e, g based on >/< threshold - assumes readout phase already takes into account IQ plane rotation
        bufi = np.array([self.di_buf[i] for i, ch in enumerate(self.ro_chs)])
        avgi = []
        for bufi_ch in bufi:
            # drop extra shots that aren't divisible into averages
            new_bufi_ch = bufi_ch[:len(bufi_ch) - (len(bufi_ch) % shot_avg)]
            # average over shots_avg number of consecutive shots
            new_bufi_ch = np.reshape(new_bufi_ch, (len(new_bufi_ch)//shot_avg, shot_avg))
            new_bufi_ch = np.average(new_bufi_ch, axis=1)
            avgi.append(new_bufi_ch)
        avgi = np.array(avgi)
        shots = np.array([np.heaviside(avgi[i]/self.ro_chs[ch].length-threshold[i], 0) for i, ch in enumerate(self.ro_chs)])

        qubits = self.cfg.expt.qubits
        # get the shots for the qubits we care about
        shots = np.array([shots[self.adc_chs[q]] for q in qubits])

        # data is returned as n00, n01, n10, n11 measured for the two qubits
        n00 = np.sum(np.logical_and(np.logical_not(shots[0]), np.logical_not(shots[1])))
        n01 = np.sum(np.logical_and(np.logical_not(shots[0]), shots[1]))
        n10 = np.sum(np.logical_and(shots[0], np.logical_not(shots[1])))
        n11 = np.sum(np.logical_and(shots[0], shots[1]))
        return np.array([n00, n01, n10, n11])

    def acquire(self, soc, threshold, shot_avg=1, load_pulses=True, progress=False, debug=False):
        super().acquire(soc, threshold=threshold, load_pulses=load_pulses, progress=progress, debug=debug)
        return self.collect_counts(threshold=threshold, shot_avg=shot_avg)

# ===================================================================== #

class ErrorMitigationStateTomo2QProgram(AbstractStateTomo2QProgram):
    """
    Prep the error mitigation matrix state and then perform 2Q state tomography.
    Experimental Config:
    expt = dict(
        reps: number averages per measurement basis iteration
        qubits: the qubits to perform the two qubit tomography on (drive applied to the second qubit)
        state_prep_kwargs.prep_state: gg, ge, eg, ee - the state to prepare in before measuring
    )
    """
    def state_prep_pulse(self, qubits, **kwargs):
        # pass in kwargs via cfg.expt.state_prep_kwargs
        prep_state = kwargs['prep_state'] # should be gg, ge, eg, or ee
        if prep_state[0] == 'e':
            # print('q0: e')
            self.X_pulse(q=qubits[0], play=True)
            self.sync_all() # necessary for ZZ?
        else:
            # print('q0: g')
            assert prep_state[0] == 'g'
        if prep_state[1] == 'e':
            # print('q1: e')
            self.X_pulse(q=qubits[1], play=True)
        else:
            # print('q1: g')
            assert prep_state[1] == 'g'
            
    def initialize(self):
        self.cfg.expt.basis = 'ZZ'
        super().initialize()
        self.sync_all(self.us2cycles(0.2))

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
        # pass in kwargs via cfg.expt.state_prep_kwargs

        # # initialize to Eg
        # self.X_pulse(q=qubits[0], play=True)

        self.X_pulse(q=qubits[0], pihalf=False, play=True)
        self.sync_all()
        self.X_pulse(q=qubits[1], pihalf=False, play=True)
        # # apply Eg -> Gf pulse on B: expect to end in Gf
        # type = self.cfg.device.qubit.pulses.pi_EgGf.type
        # pulse = self.handle_const_pulse
        # if type == 'gauss': pulse = self.handle_gauss_pulse
        # elif type == 'flat_top': pulse = self.flat_top
        # pulse(name=f'pi_EgGf_{qubits[0]}{qubits[1]}', play=True)
        # self.sync_all()

        # # measure as Ee
        # self.X_pulse(q=qubits[0], play=True) # G->E on qubits[0]
        # self.sync_all() # necessary for ZZ?
        # self.handle_gauss_pulse(f'ef_qubit{qubits[1]}', play=True) # f->e on qubits[1]
    
    def initialize(self):
        super().initialize()
        qubits = self.cfg.expt.qubits
        self.cfg.expt.state_prep_kwargs = None

        # initialize ef pulse on qubits[1]
        qB = qubits[1]
        self.handle_gauss_pulse(ch=self.qubit_chs[qubits[1]], name=f"ef_qubit{qB}", sigma=self.us2cycles(self.cfg.device.qubit.pulses.pi_ef.sigma[qB]), freq=self.freq2reg(self.cfg.device.qubit.f_ef[qB], gen_ch=self.qubit_chs[qB]), phase=0, gain=self.cfg.device.qubit.pulses.pi_ef.gain[qB], play=False)

        # initialize EgGf pulse
        # apply the sideband drive on qubits[1]
        type = self.cfg.device.qubit.pulses.pi_EgGf.type
        freq = self.freq2reg(self.cfg.device.qubit.f_EgGf, gen_ch=self.qubit_chs[qubits[1]])
        gain = self.cfg.device.qubit.pulses.pi_EgGf.gain
        sigma = self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf.sigma)
        if type == 'const':
            self.handle_const_pulse(name=f'pi_EgGf_{qubits[0]}{qubits[1]}', ch=self.qubit_chs[1], length=sigma, freq=freq, phase=0, gain=gain, play=False) 
        elif type == 'gauss':
            self.handle_gauss_pulse(name=f'pi_EgGf_{qubits[0]}{qubits[1]}', ch=self.qubit_chs[1], sigma=sigma, freq=freq, phase=0, gain=gain, play=False)
        elif type == 'flat_top':
            flat_length = self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf.flat_length)
            self.handle_flat_top_pulse(name=f'pi_EgGf_{qubits[0]}{qubits[1]}', ch=self.qubit_chs[1], sigma=sigma, flat_length=flat_length, freq=freq, phase=0, gain=gain, play=False) 
        else: assert False, f'Pulse type {type} not supported.'
        self.sync_all(self.us2cycles(0.2))

# ===================================================================== #

class EgGfStateTomographyExperiment(Experiment):
# outer loop over measurement bases
# set the state prep pulse to be preparing the gg, ge, eg, ee states for confusion matrix
    """
    Perform state tomography on the EgGf state with error mitigation.
    Experimental Config:
    expt = dict(
        reps: number averages per measurement basis iteration
        shot_avg: number of shots to average over before sorting via threshold
        qubits: the qubits to perform the two qubit tomography on (drive applied to the second qubit)
    )
    """

    def __init__(self, soccfg=None, path='', prefix='EgGfStateTomography', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        qubits = self.cfg.expt.qubits
        adc_chs = [self.cfg.hw.soc.adcs.readout.ch[q] for q in qubits]
        
        self.meas_order = ['ZZ', 'ZX', 'ZY', 'XZ', 'XX', 'XY', 'YZ', 'YX', 'YY']
        self.calib_order = ['gg', 'ge', 'eg', 'ee'] # should match with order of counts for each tomography measurement 
        data={'counts_tomo':[], 'counts_calib':[]}
        self.pulse_dict = dict()
        
        threshold = self.cfg.device.readout.threshold
        # phase = self.cfg.device.readout.phase

        # Tomography measurements
        for basis in tqdm(self.meas_order):
            # print(basis)
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.basis = basis
            tomo = EgGfStateTomo2QProgram(soccfg=self.soccfg, cfg=cfg)
            counts = tomo.acquire(self.im[self.cfg.aliases.soc], shot_avg=self.cfg.expt.shot_avg, threshold=threshold, load_pulses=True, progress=False, debug=debug)
            data['counts_tomo'].append(counts)
            self.pulse_dict.update({basis:tomo.pulse_dict})
        
        # Error mitigation measurements: prep in gg, ge, eg, ee and measure confusion matrix
        for prep_state in tqdm(self.calib_order):
            # print(prep_state)
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.state_prep_kwargs = dict(prep_state=prep_state)
            err_tomo = ErrorMitigationStateTomo2QProgram(soccfg=self.soccfg, cfg=cfg)
            counts = err_tomo.acquire(self.im[self.cfg.aliases.soc], shot_avg=self.cfg.expt.shot_avg, threshold=threshold, load_pulses=True, progress=False, debug=debug)
            data['counts_calib'].append(counts)
        
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
        with self.datafile() as f:
            f.attrs['pulse_dict'] = json.dumps(self.pulse_dict, cls=NpEncoder)
            f.attrs['meas_order'] = json.dumps(self.meas_order, cls=NpEncoder)
            f.attrs['calib_order'] = json.dumps(self.calib_order, cls=NpEncoder)