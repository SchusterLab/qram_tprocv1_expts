import matplotlib.pyplot as plt
import numpy as np
from qick import *

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm

"""
Averager program that takes care of the standard pulse loading for basic X, Y, Z +/- pi and pi/2 pulses.
"""
class CliffordAveragerProgram(AveragerProgram):

    """
    Wrappers to load and play pulses.
    If play is false, must specify all parameters and all params will be saved (load params).

    If play is true, uses the default values saved from the load call, temporarily overriding freq, phase, or gain if specified to not be None. Sets the pulse registers with these settings and plays the pulse. If you want to set freq, phase, or gain via registers/update,
    be sure to set the default value to be None at loading time.

    If play is True, registers will automatically be set regardless of set_reg flag.
    If play is False, registers will be set based on value of set_reg flag, but pulse will not be played.
    """
    def handle_const_pulse(self, name, ch=None, length=None, freq=None, phase=None, gain=None, play=False, set_reg=False):
        """
        Load/play a constant pulse of given length.
        """
        if name is not None and name not in self.pulse_dict.keys():
            assert ch is not None
            self.pulse_dict.update({name:dict(ch=ch, name=name, type='const', length=length, freq=freq, phase=phase, gain=gain)})
        if play or set_reg:
            assert name in self.pulse_dict.keys()
            # if not (ch == None):
            #     print('Warning: you have specified a pulse parameter that can only be changed when loading.')
            params = self.pulse_dict[name].copy()
            if freq is not None: params['freq'] = freq
            if phase is not None: params['phase'] = phase
            if gain is not None: params['gain'] = gain
            self.set_pulse_registers(ch=params['ch'], style='const', freq=params['freq'], phase=params['phase'], gain=params['gain'], length=params['length'])
            if play: self.pulse(ch=params['ch'])

    def handle_gauss_pulse(self, name, ch=None, sigma=None, freq=None, phase=None, gain=None, play=False, set_reg=False):
        """
        Load/play a gaussian pulse of length 4 sigma on channel ch
        """
        if name is not None and name not in self.pulse_dict.keys():
            assert ch is not None
            assert sigma is not None
            self.pulse_dict.update({name:dict(ch=ch, name=name, type='gauss', sigma=sigma, freq=freq, phase=phase, gain=gain)})
            self.add_gauss(ch=ch, name=name, sigma=sigma, length=sigma*4)
            # print('added gauss pulse', name, 'on ch', ch)
        if play or set_reg:
            assert name in self.pulse_dict.keys()
            # if not (ch == sigma == None):
            #     print('Warning: you have specified a pulse parameter that can only be changed when loading.')
            params = self.pulse_dict[name].copy()
            if freq is not None: params['freq'] = freq
            if phase is not None: params['phase'] = phase
            if gain is not None: params['gain'] = gain
            self.set_pulse_registers(ch=params['ch'], style='arb', freq=params['freq'], phase=params['phase'], gain=params['gain'], waveform=params['name'])
            if play:
                # print('playing gauss pulse', params['name'], 'on ch', params['ch'])
                self.pulse(ch=params['ch'])
            
    def handle_flat_top_pulse(self, ch=None, name=None, sigma=None, flat_length=None, freq=None, phase=None, gain=None, play=False, set_reg=False):
        """
        Plays a gaussian ramp up (2.5*sigma), a constant pulse of length flat_length,
        gaussian ramp down (2.5*sigma) on channel ch
        """
        if name is not None and name not in self.pulse_dict.keys():
            assert None not in [ch, sigma, flat_length]
            self.pulse_dict.update({name:dict(ch=ch, name=name, type='flat_top', sigma=sigma, flat_length=flat_length, freq=freq, phase=phase, gain=gain)})
            self.add_gauss(ch=ch, name=name, sigma=sigma, length=sigma*5)
        if play or set_reg:
            # if not (ch == name == sigma == flat_length == None):
            #     print('Warning: you have specified a pulse parameter that can only be changed when loading.')
            assert name in self.pulse_dict.keys()
            params = self.pulse_dict[name].copy()
            if freq is not None: params['freq'] = freq
            if phase is not None: params['phase'] = phase
            if gain is not None: params['gain'] = gain
            self.set_pulse_registers(ch=params['ch'], style='flat_top', freq=params['freq'], phase=params['phase'], gain=params['gain'], waveform=params['name'], length=params['flat_length'])
            if play: self.pulse(ch=params['ch'])

    """
    Clifford pulse defns
    If play=False, just loads pulse.
    """
    # General drive: Omega cos((wt+phi)X) -> Delta/2 Z + Omega/2 (cos(phi) X + sin(phi) Y)
    def X_pulse(self, q, pihalf=False, neg=False, extra_phase=0, play=False):
        # q: qubit number in config
        f_ge = self.freq2reg(self.cfg.device.qubit.f_ge[q], gen_ch=self.qubit_chs[q])
        gain = self.cfg.device.qubit.pulses.pi_ge.gain[q]
        phase = self.overall_phase[q] + extra_phase
        sigma = self.us2cycles(self.cfg.device.qubit.pulses.pi_ge.sigma[q], gen_ch=self.qubit_chs[q])
        if pihalf: gain = gain//2
        if neg: phase += 180
        type = self.cfg.device.qubit.pulses.pi_ge.type
        if type == 'const':
            self.handle_const_pulse(name=f'qubit{q}', ch=self.qubit_chs[q], length=sigma, freq=f_ge, phase=phase, gain=gain, play=play) 
        elif type == 'gauss':
            self.handle_gauss_pulse(name=f'qubit{q}', ch=self.qubit_chs[q], sigma=sigma, freq=f_ge, phase=phase, gain=gain, play=play) 
        elif type == 'flat_top':
            flat_length = self.us2cycles(self.cfg.device.qubit.pulses.pi_ge.flat_length[q], gen_ch=self.qubit_chs[q])
            self.handle_flat_top_pulse(name=f'qubit{q}', ch=self.qubit_chs[q], sigma=sigma, flat_length=flat_length, freq=f_ge, phase=phase, gain=gain, play=play) 
        else: assert False, f'Pulse type {type} not supported.'
    
    def Y_pulse(self, q, pihalf=False, neg=False, extra_phase=0, play=False):
        self.X_pulse(q, pihalf=pihalf, neg=neg, extra_phase=90+extra_phase, play=play)

    def Z_pulse(self, q, pihalf=False, neg=False, extra_phase=0, play=False):
        phase_adjust = 180 # NOT SURE ABOUT SIGN
        if pihalf: phase_adjust = 90
        if neg: phase_adjust *= -1
        if play: self.overall_phase[q] += phase_adjust + extra_phase
 
    def initialize(self):
        self.cfg = AttrDict(self.cfg)
        self.cfg.update(self.cfg.expt)
        qubits = self.cfg.expt.qubits
        self.pulse_dict = dict()

        # all of these saved self.whatever instance variables should be indexed by the actual qubit number. this means that more values are saved as instance variables than is strictly necessary, but this is overall less confusing
        self.res_chs = self.cfg.hw.soc.dacs.readout.ch
        self.qubit_chs = self.cfg.hw.soc.dacs.qubit.ch
        self.adc_chs = self.cfg.hw.soc.adcs.readout.ch

        self.overall_phase = [0]*len(qubits)

        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs] # get register page for qubit_ch
        self.f_res = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(self.cfg.device.readout.frequency, self.res_chs)]

        self.readout_length = [self.us2cycles(len) for len in self.cfg.device.readout.readout_length]

        # copy over parameters for the acquire method
        self.cfg.reps = self.cfg.expt.reps

        for q in qubits:
            self.declare_gen(ch=self.res_chs[q], nqz=self.cfg.hw.soc.dacs.readout.nyquist[q])
            self.declare_gen(ch=self.qubit_chs[q], nqz=self.cfg.hw.soc.dacs.qubit.nyquist[q])
            self.declare_readout(ch=self.adc_chs[q], length=self.readout_length[q], freq=self.cfg.device.readout.frequency[q], gen_ch=self.res_chs[q])
            self.X_pulse(q=q, play=False)
            self.handle_const_pulse(name=f'measure{q}', ch=self.res_chs[q], length=self.readout_length[q], freq=self.f_res[q], phase=self.deg2reg(self.cfg.device.readout.phase[q]), gain=self.cfg.device.readout.gain[q], play=False, set_reg=True)