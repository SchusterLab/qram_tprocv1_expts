import matplotlib.pyplot as plt
import numpy as np
from qick import *

from slab import Experiment, dsfit, AttrDict
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

    def setup_measure(self, qubit, basis:str, play=False):
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

        # Execute tomography measurement
        self.setup_measure(qubit=qubits[0], basis=self.basis[0], play=True)
        self.setup_measure(qubit=qubits[1], basis=self.basis[1], play=True)
        self.sync_all()
        # need sho to update measure so it can do simultaneous readout!
        self.measure(pulse_ch=self.res_chs, adcs=[0,1], adc_trig_offset=self.cfg.device.readout.trig_offset, wait=True, syncdelay=syncdelay) # trigger simultaneous measurement

    def collect_counts(self):
        # collect shots for 2 adcs (indexed by qubit order) and I and Q channels
        # data is returned as n00, n01, n10, n11 measured for the two qubits
        qubits = self.cfg.expt.qubits
        shots = np.array([self.shots[[self.adc_chs[q]]] for q in qubits])
        n00 = np.sum(np.logical_and(np.logical_not(shots[0]), np.logical_not(shots[1])))
        n01 = np.sum(np.logical_and(np.logical_not(shots[0]), shots[1]))
        n10 = np.sum(np.logical_and(shots[0], np.logical_not(shots[1])))
        n11 = np.sum(np.logical_and(shots[0], shots[1]))
        return np.array([n00, n01, n10, n11])

    def acquire(self, soc, threshold, angle, load_pulses=True, progress=False, debug=False):
        super().acquire(soc, threshold=threshold, angle=angle, load_pulses=load_pulses, progress=progress, debug=debug)
        return self.collect_counts()

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
        if prep_state[0] == 'e': self.X_pulse(q=qubits[0], play=True)
        else: assert prep_state[0] == 'g'
        if prep_state[1] == 'e': self.X_pulse(q=qubits[1], play=True)
        else: assert prep_state[1] == 'g'
            
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
        self.X_pulse(q=qubits[0], play=True) # initialize to Eg
        type = self.cfg.device.qubit.pulses.pi_EgGf.type
        pulse = self.handle_const_pulse
        if type == 'gauss': pulse = self.handle_gauss_pulse
        elif type == 'flat_top': pulse = self.flat_top
        pulse(name=f'pi_EgGf_{qubits[0]}{qubits[1]}', play=True)
        self.X_pulse(q=qubits[1], play=True) # measure as gE
    
    def initialize(self):
        super().initialize()
        qubits = self.cfg.expt.qubits
        self.cfg.expt.state_prep_kwargs = None

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
        qubits: the qubits to perform the two qubit tomography on (drive applied to the second qubit)
    )
    """

    def __init__(self, soccfg=None, path='', prefix='EgGfStateTomography', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        qubits = self.cfg.expt.qubits
        adc_chs = [self.cfg.hw.soc.adcs.readout.ch[q] for q in qubits]
        
        meas_order = np.array(['ZZ', 'ZX', 'ZY', 'XZ', 'XX', 'XY', 'YZ', 'YX', 'YY'])
        calib_order = np.array(['gg', 'ge', 'eg', 'ee']) # should match with order of counts for each tomography measurement 
        data={'counts_tomo':[], 'counts_calib':[], 'meas_order':meas_order, 'calib_order':calib_order}
        
        threshold = self.cfg.device.readout.threshold
        phase = self.cfg.device.readout.phase

        # Tomography measurements
        for basis in tqdm(meas_order):
            self.cfg.expt.basis = basis
            tomo = EgGfStateTomo2QProgram(soccfg=self.soccfg, cfg=self.cfg)
            counts = tomo.acquire(self.im[self.cfg.aliases.soc], threshold=threshold, angle=phase, load_pulses=True, progress=False, debug=debug)
            data['counts_tomo'].append(counts)
        
        # Error mitigation measurements: prep in gg, ge, eg, ee and measure confusion matrix
        for prep_state in tqdm(calib_order):
            self.cfg.expt.state_prep_kwargs = dict(prep_state=prep_state)
            err_tomo = ErrorMitigationStateTomo2QProgram(soccfg=self.soccfg, cfg=self.cfg)
            counts = err_tomo.acquire(self.im[self.cfg.aliases.soc], threshold=threshold, angle=phase, load_pulses=True, progress=False, debug=debug)
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