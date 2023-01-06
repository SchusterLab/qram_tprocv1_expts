import matplotlib.pyplot as plt
import numpy as np
from qick import *
import json

from slab import Experiment, NpEncoder, AttrDict
from tqdm import tqdm_notebook as tqdm

from experiments.clifford_averager_program import QutritAveragerProgram
from experiments.two_qubit.twoQ_state_tomography import AbstractStateTomo2QProgram

class AbstractStateTomo2qutritProgram(QutritAveragerProgram):
    """
    Performs a state_prep_pulse (abstract method) on two qutrits, then measures in a desired basis.
    Repeat this program multiple times in the experiment to loop over all the bases necessary for tomography.
    Experimental Config:
    expt = dict(
        reps: number averages per measurement prep iteration
        qubits: the qubits to perform the two qubit tomography on (drive applied to the second qubit)
        prep: 2 element tuple where each element comes from:
            'I', 'Xge', 'Yge', 'Xef', 'Yef', 'Pge', 'PgeXef', 'PgeYef'
            which determine the pre-measurement operations for the 2 qubits
        state_prep_kwargs: dictionary containing kwargs for the state_prep_pulse function
    )
    """
    meas_order_1Q = ['I', 'Xge', 'Yge', 'Xef', 'Yef', 'Pge', 'PgeXef', 'PgeYef']

    def setup_measure(self, qubit, prep:str, play=False):
        """
        Convert string indicating the measurement prep into the appropriate single qubit pulse (pre-measurement pulse)
        """
        assert prep in self.meas_order_1Q
        if prep == 'Xge': self.X_pulse(qubit, pihalf=True, play=play) # X/2 pulse on ge
        elif prep == 'Yge': self.Y_pulse(qubit, pihalf=True, play=play) # Y/2 pulse on ge
        elif prep == 'Xef': self.Xef_pulse(qubit, pihalf=True, play=play) # X/2 pulse on ef 
        elif prep == 'Yef': self.Yef_pulse(qubit, pihalf=True, play=play) # Y/2 pulse on ef
        elif prep == 'Pge': self.X_pulse(qubit, pihalf=False, play=play) # X pulse on ge
        elif prep == 'PgeXef':
            self.X_pulse(qubit, pihalf=False, play=play) # X pulse on ge
            self.Xef_pulse(qubit, pihalf=True, play=play) # X pulse on ge, X/2 pulse on ef
        elif prep == 'PgeYef':
            self.X_pulse(qubit, pihalf=False, play=play) # X pulse on ge
            self.Yef_pulse(qubit, pihalf=True, play=play) # X pulse on ge, Y/2 pulse on ef
        else: return # measure in I/Z prep

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
        qubits = self.cfg.expt.qubits
        self.prep = self.cfg.expt.prep

        # Prep state to characterize
        kwargs = self.cfg.expt.state_prep_kwargs
        if kwargs is None: kwargs = dict()
        self.state_prep_pulse(qubits, **kwargs)
        self.sync_all(5)

        # Prep for the tomography measurement
        self.setup_measure(qubit=qubits[0], prep=self.prep[0], play=True)
        self.sync_all() # necessary for ZZ?
        self.setup_measure(qubit=qubits[1], prep=self.prep[1], play=True)
        self.sync_all(5)

        # Simultaneous measurement
        syncdelay = self.us2cycles(max(self.cfg.device.readout.relax_delay))
        measure_chs = self.res_chs
        if self.res_ch_types[0] == 'mux4': measure_chs = self.res_chs[0]
        self.measure(pulse_ch=measure_chs, adcs=self.adc_chs, adc_trig_offset=self.cfg.device.readout.trig_offset[0], wait=True, syncdelay=syncdelay) 

    def collect_counts(self, angle=None, threshold_ge=None, threshold_ef=None, shot_avg=1):
        # collect shots for 2 adcs (indexed by qubit order) in the I channel, then sorts into e, g based on >/< threshold and angle rotation
        bufi = np.array([
            self.di_buf[i]*np.cos(np.pi/180*(angle[i]) - self.dq_buf[i]*np.sin(angle[i]))
            for i, ch in enumerate(self.ro_chs)])
        avgi = []
        for bufi_ch in bufi:
            # drop extra shots that aren't divisible into averages
            new_bufi_ch = bufi_ch[:len(bufi_ch) - (len(bufi_ch) % shot_avg)]
            # average over shots_avg number of consecutive shots
            new_bufi_ch = np.reshape(new_bufi_ch, (len(new_bufi_ch)//shot_avg, shot_avg))
            new_bufi_ch = np.average(new_bufi_ch, axis=1)
            avgi.append(new_bufi_ch)
        avgi = np.array(avgi)
        assert threshold_ef > threshold_ge
        shots_cut_ge = np.array([np.heaviside(avgi[i]/self.ro_chs[ch].length-threshold_ge[i], 0) for i, ch in enumerate(self.ro_chs)])
        shots_cut_ef = np.array([np.heaviside(avgi[i]/self.ro_chs[ch].length-threshold_ef[i], 0) for i, ch in enumerate(self.ro_chs)])
        shots = shots_cut_ge + shots_cut_ef # 0, 1, or 2 depending on the measured state

        qubits = self.cfg.expt.qubits
        # get the shots for the qubits we care about
        shots = np.array([shots[self.adc_chs[q]] for q in qubits])

        # data is returned as n00, n01, n02, n10, n11, n12, n20, n21, n22 measured for the two qubits
        shots_q0 = shots[0]
        shots_q1 = shots[1]
        nij = np.zeros(9)
        ind = 0
        for i in range(3):
            for j in range(3):
                nij[ind] = np.sum(np.logical_and(shots_q0 == i, shots_q1 == j))
                ind += 1
        return nij

    def acquire(self, soc, angle=None, threshold_ge=None, threshold_ef=None, shot_avg=1, load_pulses=True, progress=False, debug=False):
        super().acquire(soc, load_pulses=load_pulses, progress=progress, debug=debug)
        return self.collect_counts(angle=angle, threshold_ge=threshold_ge, threshold_ef=threshold_ef, shot_avg=shot_avg)

    def initialize(self):
        super().initialize() # calls super in the order of the inheritance
        self.sync_all(200) # this is probably unnecessary but just to be safe

# ===================================================================== #

class ErrorMitigationStateTomo2qutritProgram(AbstractStateTomo2qutritProgram):
    """
    Prep the error mitigation matrix state and then perform 2Q state tomography.
    Experimental Config:
    expt = dict(
        reps: number averages per measurement prep iteration
        qubits: the qubits to perform the two qubit tomography on (drive applied to the second qubit)
        state_prep_kwargs.prep_state: gg, ge, gf, eg, ee, ef, fg, fe, ff - the state to prepare in before measuring
    )
    """
    def state_prep_pulse(self, qubits, **kwargs):
        # pass in kwargs via cfg.expt.state_prep_kwargs
        prep_state = kwargs['prep_state'] # should be gg, ge, gf, eg, ee, ef, fg, fe, or ff
        for q in range(2):
            if prep_state[q] == 'e':
                # print(f'q{q}: e')
                self.X_pulse(q=qubits[q], play=True)
                self.sync_all()
            elif prep_state[q] == 'f':
                # print(f'q{q}: f')
                self.X_pulse(q=qubits[q], play=True)
                self.Xef_pulse(q=qubits[q], play=True)
                self.sync_all()
            else:
                # print(f'q{q}: g')
                assert prep_state[q] == 'g'
            
    def initialize(self):
        self.cfg.expt.prep = ('I', 'I')
        super().initialize()
        self.sync_all(self.us2cycles(0.2))

# ===================================================================== #

class EgGfStateTomo2qutritProgram(AbstractStateTomo2qutritProgram):
    """
    Perform the EgGf swap and then perform 2Q state tomography.
    Experimental Config:
    expt = dict(
        reps: number averages per measurement prep iteration
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

        self.swap_chs = self.cfg.hw.soc.dacs.swap.ch
        self.swap_ch_types = self.cfg.hw.soc.dacs.swap.type

        # initialize ef pulse on qB
        qA, qB = qubits
        self.handle_gauss_pulse(ch=self.qubit_chs[qB], name=f"ef_qubit{qB}", sigma=self.us2cycles(self.cfg.device.qubit.pulses.pi_ef.sigma[qB], gen_ch=self.qubit_chs[qB]), freq=self.freq2reg(self.cfg.device.qubit.f_ef[qB], gen_ch=self.qubit_chs[qB]), phase=0, gain=self.cfg.device.qubit.pulses.pi_ef.gain[qB], play=False)

        # initialize EgGf pulse
        # apply the sideband drive on qB, indexed by qA
        type = self.cfg.device.qubit.pulses.pi_EgGf.type[qA]
        freq = self.freq2reg(self.cfg.device.qubit.f_EgGf[qA], gen_ch=self.swap_chs[qA])
        gain = self.cfg.device.qubit.pulses.pi_EgGf.gain[qA]
        sigma = self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf.sigma[qA], gen_ch=self.swap_chs[qA])
        if type == 'const':
            self.handle_const_pulse(name=f'pi_EgGf_{qA}{qB}', ch=self.swap_chs[qA], length=sigma, freq=freq, phase=0, gain=gain, play=False) 
        elif type == 'gauss':
            self.handle_gauss_pulse(name=f'pi_EgGf_{qA}{qB}', ch=self.swap_chs[qA], sigma=sigma, freq=freq, phase=0, gain=gain, play=False)
        elif type == 'flat_top':
            flat_length = self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf.flat_length[qA], gen_ch=self.swap_chs[qA])
            self.handle_flat_top_pulse(name=f'pi_EgGf_{qA}{qB}', ch=self.swap_chs[qA], sigma=sigma, flat_length=flat_length, freq=freq, phase=0, gain=gain, play=False) 
        else: assert False, f'Pulse type {type} not supported.'
        self.sync_all(200)

# ===================================================================== #

class EgGfStateTomographyQutritExperiment(Experiment):
# outer loop over measurement bases
# set the state prep pulse to be preparing the states for confusion matrix
    """
    Perform state tomography on the EgGf state with error mitigation.
    Experimental Config:
    expt = dict(
        reps: number averages per measurement prep iteration
        shot_avg: number of shots to average over before sorting via threshold
        qubits: the qubits to perform the two qubit tomography on (drive applied to the second qubit)
    )
    """

    def __init__(self, soccfg=None, path='', prefix='EgGfStateTomography', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update({key3: [value3]*num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})
        
        self.meas_order_1Q = ['I', 'Xge', 'Yge', 'Xef', 'Yef', 'Pge', 'PgeXef', 'PgeYef']
        self.calib_order = ['gg', 'ge', 'gf', 'eg', 'ee', 'ef', 'fg', 'fe', 'ff'] # should match with order of counts for each tomography measurement 
        data={'counts_tomo':[], 'counts_calib':[]}
        self.pulse_dict = dict()
        
        angle = self.cfg.device.readout.phase
        threshold_ge = self.cfg.device.readout.threshold_ge
        threshold_ef = self.cfg.device.readout.threshold_ef
        # phase = self.cfg.device.readout.phase

        # Tomography measurements
        for prep0 in tqdm(self.meas_order_1Q):
            for prep1 in tqdm(self.meas_order_1Q):
                prep = (prep0, prep1)
                # print(prep)
                cfg = AttrDict(self.cfg.copy())
                cfg.expt.prep = prep
                tomo = EgGfStateTomo2qutritProgram(soccfg=self.soccfg, cfg=cfg)
                counts = tomo.acquire(self.im[self.cfg.aliases.soc], shot_avg=self.cfg.expt.shot_avg, angle=angle, threshold_ge=threshold_ge, threshold_ef=threshold_ef, load_pulses=True, progress=False, debug=debug)
                data['counts_tomo'].append(counts)
                self.pulse_dict.update({prep0+'-'+prep1:tomo.pulse_dict})

        # Error mitigation measurements: prep in gg, ge, gf, eg, ee, ef, fg, fe, ff and measure confusion matrix
        for prep_state in tqdm(self.calib_order):
            # print(prep_state)
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.state_prep_kwargs = dict(prep_state=prep_state)
            err_tomo = ErrorMitigationStateTomo2qutritProgram(soccfg=self.soccfg, cfg=cfg)
            counts = err_tomo.acquire(self.im[self.cfg.aliases.soc], shot_avg=self.cfg.expt.shot_avg, angle=angle, threshold_ge=threshold_ge, threshold_ef=threshold_ef, load_pulses=True, progress=False, debug=debug)
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
            f.attrs['meas_order_1Q'] = json.dumps(self.meas_order_1Q, cls=NpEncoder)
            f.attrs['calib_order'] = json.dumps(self.calib_order, cls=NpEncoder)
        return self.fname