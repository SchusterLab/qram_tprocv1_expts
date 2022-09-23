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
    
    def handle_mux4_pulse(self, name, ch=None, mask=None, length=None, play=False, set_reg=False):
        """
        Load/play a constant pulse of given length on the mux4 channel.
        """
        if name is not None and name not in self.pulse_dict.keys():
            assert ch is not None
            assert ch == 6, 'Only ch 6 on q3diamond supports mux4 currently!'
            self.pulse_dict.update({name:dict(ch=ch, name=name, type='mux4', mask=mask, length=length)})
        if play or set_reg:
            assert name in self.pulse_dict.keys()
            params = self.pulse_dict[name].copy()
            if mask is not None: params['mask'] = mask
            if length is not None: params['length'] = length
            self.set_pulse_registers(ch=params['ch'], style='const', length=length, mask=mask)
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
        type = self.cfg.device.qubit.pulses.pi_ge.type[q]
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
        dac_type = self.qubit_ch_types[q]
        assert not dac_type == 'mux4', "Currently cannot set phase for mux4!"
        phase_adjust = 180 # NOT SURE ABOUT SIGN
        if pihalf: phase_adjust = 90
        if neg: phase_adjust *= -1
        if play: self.overall_phase[q] += phase_adjust + extra_phase
 
    def initialize(self):
        self.cfg = AttrDict(self.cfg)
        self.cfg.update(self.cfg.expt)
        if 'qubits' in self.cfg.expt: self.qubits = self.cfg.expt.qubits
        else: self.qubits = range(4)
        self.pulse_dict = dict()
        self.num_qubits_sample = len(self.cfg.device.qubit.f_ge)

        # all of these saved self.whatever instance variables should be indexed by the actual qubit number as opposed to qubits_i. this means that more values are saved as instance variables than is strictly necessary, but this is overall less confusing
        self.adc_chs = self.cfg.hw.soc.adcs.readout.ch
        self.res_chs = self.cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = self.cfg.hw.soc.dacs.readout.type
        self.qubit_chs = self.cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = self.cfg.hw.soc.dacs.qubit.type

        self.overall_phase = [0]*self.num_qubits_sample

        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs] # get register page for qubit_ch
        self.f_res_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(self.cfg.device.readout.frequency, self.res_chs)]

        self.f_ge_regs = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(self.cfg.device.qubit.f_ge, self.qubit_chs)]
        self.f_ef_regs = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(self.cfg.device.qubit.f_ef, self.qubit_chs)]
        self.f_res_regs = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(self.cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        self.f_Q1_ZZ_regs = [self.freq2reg(f, gen_ch=self.qubit_chs[1]) for f in self.cfg.device.qubit.f_Q1_ZZ]

        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [1+self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)]

        # copy over parameters for the acquire method
        self.cfg.reps = self.cfg.expt.reps

        self.prog_gen_chs = []

        # declare res dacs, add readout pulses
        if self.res_ch_types[0] == 'mux4': # only supports having all resonators be on mux, or none
            assert np.all([ch == 6 for ch in self.res_chs])
            mask = [0, 1, 2, 3] # indices of mux_freqs, mux_gains list to play
            mux_freqs = self.cfg.device.readout.frequency
            mux_gains = self.cfg.device.readout.gain
            self.declare_gen(ch=6, nqz=self.cfg.hw.soc.dacs.readout.nyquist[0], mixer_freq=self.cfg.hw.soc.dacs.readout.mixer_freq[0], mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=0)
            self.handle_mux4_pulse(name=f'measure', ch=6, length=max(self.readout_lengths_dac), mask=mask, play=False, set_reg=True)
            self.prog_gen_chs.append(6)
        else:
            for q in self.qubits:
                mixer_freq = 0
                if self.res_ch_types[q] == 'int4':
                    mixer_freq = self.cfg.hw.soc.dacs.readout.mixer_freq[q]
                self.declare_gen(ch=self.res_chs[q], nqz=self.cfg.hw.soc.dacs.readout.nyquist[q], mixer_freq=mixer_freq)
                self.handle_const_pulse(name=f'measure{q}', ch=self.res_chs[q], length=self.readout_length[q], freq=self.f_res_reg[q], phase=0, gain=self.cfg.device.readout.gain[q], play=False, set_reg=True)
                self.prog_gen_chs.append(self.res_chs[q])

        # declare qubit dacs, add qubit pi_ge pulses
        for q in self.qubits:
            mixer_freq = 0
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = self.cfg.hw.soc.dacs.qubit.mixer_freq[q]
            self.declare_gen(ch=self.qubit_chs[q], nqz=self.cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)
            self.X_pulse(q=q, play=False)
            self.prog_gen_chs.append(self.qubit_chs[q])

        # declare adcs - readout for all qubits everytime, defines number of buffers returned regardless of number of adcs triggered
        for q in range(self.num_qubits_sample):
            self.declare_readout(ch=self.adc_chs[q], length=self.readout_lengths_adc[q], freq=self.cfg.device.readout.frequency[q], gen_ch=self.res_chs[q])

    """
    Collect shots for all adcs, rotates by given angle (degrees), and averages over shot_avg adjacent shots
    Returns avgi, avgq, avgi_err, avgq_err which avgi/q are avg over shot_avg and avgi/q_err is (std dev of each group of shots)/sqrt(shot_avg)
    """
    def get_shots(self, angle=None, shot_avg=1, verbose=True, return_err=False):
        if angle == None: angle = [0]*len(self.cfg.device.qubit.f_ge)
        bufi = np.array([
            self.di_buf[i]*np.cos(np.pi/180*angle[i]) - self.dq_buf[i]*np.sin(np.pi/180*angle[i])
            for i, ch in enumerate(self.ro_chs)])
        avgi = []
        avgi_err = []
        for bufi_ch in bufi:
            # drop extra shots that aren't divisible into averages
            new_bufi_ch = np.copy(bufi_ch[:len(bufi_ch) - (len(bufi_ch) % shot_avg)])
            # average over shots_avg number of consecutive shots
            new_bufi_ch = np.reshape(new_bufi_ch, (len(new_bufi_ch)//shot_avg, shot_avg))
            new_bufi_ch_err = np.std(new_bufi_ch, axis=1) / np.sqrt(shot_avg)
            new_bufi_ch = np.average(new_bufi_ch, axis=1)
            avgi_err.append(new_bufi_ch_err)
            avgi.append(new_bufi_ch)
        avgi = np.array(avgi)
        avgi = np.array([avgi[i]/ro.length for i, (ch, ro) in enumerate(self.ro_chs.items())])
        avgi_err = np.array([avgi[i]/ro.length for i, (ch, ro) in enumerate(self.ro_chs.items())])
        # print(avgi[self.cfg.expt.qubits[1]])
        if verbose: print([np.median(avgi[i]) for i in range(4)])

        bufq = np.array([
            self.di_buf[i]*np.sin(np.pi/180*angle[i]) + self.dq_buf[i]*np.cos(np.pi/180*angle[i])
            for i, ch in enumerate(self.ro_chs)])
        avgq = []
        avgq_err = []
        for bufq_ch in bufq:
            # drop extra shots that aren't divisible into averages
            new_bufq_ch = np.copy(bufq_ch[:len(bufq_ch) - (len(bufq_ch) % shot_avg)])
            # average over shots_avg number of consecutive shots
            new_bufq_ch = np.reshape(new_bufq_ch, (len(new_bufq_ch)//shot_avg, shot_avg))
            new_bufq_ch_err = np.std(new_bufq_ch, axis=1) / np.sqrt(shot_avg)
            new_bufq_ch = np.average(new_bufq_ch, axis=1)
            assert(np.shape(new_bufq_ch_err) == np.shape(new_bufq_ch))
            avgq_err.append(new_bufq_ch_err)
            avgq.append(new_bufq_ch)
        avgq = np.array(avgq)
        avgq = np.array([avgq[i]/ro.length for i, (ch, ro) in enumerate(self.ro_chs.items())])
        avgq_err = np.array([avgq[i]/ro.length for i, (ch, ro) in enumerate(self.ro_chs.items())])

        if return_err: return avgi, avgq, avgi_err, avgq_err
        else: return avgi, avgq

    def acquire_threshold(self, soc, progress, angle=None, threshold=None, shot_avg=1, verbose=False):
        avgi, avgq = self.acquire(soc, load_pulses=True, progress=progress, debug=False)
        # if angle == None or threshold == None: return avgi[:,0], avgq[:,0]
        if angle == None or threshold == None: shot_avg = len(self.di_buf[0])
        avgi_rot, avgq_rot, avgi_err, avgq_err = self.get_shots(angle=angle, shot_avg=shot_avg, verbose=verbose, return_err=True)
        # print(np.shape(avgi_rot), np.average(avgi_rot[0])) #, avgi_rot[0])
        if angle == None or threshold == None: 
            return avgi_rot, avgq_rot, avgi_err, avgq_err
        else:
            e_counts = np.array([avgi_rot_ch > threshold_ch for avgi_rot_ch, threshold_ch in zip(avgi_rot, threshold)]) # 1 for e, 0 for g
            popln = np.average(e_counts, axis=1)
            popln_err = np.std(e_counts, axis=1) / np.sqrt(np.shape(e_counts)[1])
            avgq = np.zeros(np.shape(popln))
            avgq_err = np.zeros(np.shape(popln_err))
            return popln, avgq, popln_err, avgq_err
        
"""
Take care of extra clifford pulses for qutrits.
"""
class QutritAveragerProgram(CliffordAveragerProgram):
    def Xef_pulse(self, q, pihalf=False, neg=False, extra_phase=0, play=False):
        # q: qubit number in config
        f_ef = self.freq2reg(self.cfg.device.qubit.f_ef[q], gen_ch=self.qubit_chs[q])
        gain = self.cfg.device.qubit.pulses.pi_ef.gain[q]
        phase = self.overall_phase[q] + extra_phase
        sigma = self.us2cycles(self.cfg.device.qubit.pulses.pi_ef.sigma[q], gen_ch=self.qubit_chs[q])
        if pihalf: gain = gain//2
        if neg: phase += 180
        type = self.cfg.device.qubit.pulses.pi_ef.type[q]
        if type == 'const':
            self.handle_const_pulse(name=f'qubit{q}', ch=self.qubit_chs[q], length=sigma, freq=f_ef, phase=phase, gain=gain, play=play) 
        elif type == 'gauss':
            self.handle_gauss_pulse(name=f'qubit{q}', ch=self.qubit_chs[q], sigma=sigma, freq=f_ef, phase=phase, gain=gain, play=play) 
        elif type == 'flat_top':
            flat_length = self.us2cycles(self.cfg.device.qubit.pulses.pi_ef.flat_length[q], gen_ch=self.qubit_chs[q])
            self.handle_flat_top_pulse(name=f'qubit{q}', ch=self.qubit_chs[q], sigma=sigma, flat_length=flat_length, freq=f_ef, phase=phase, gain=gain, play=play) 
        else: assert False, f'Pulse type {type} not supported.'
    
    def Yef_pulse(self, q, pihalf=False, neg=False, extra_phase=0, play=False):
        self.Xef_pulse(q, pihalf=pihalf, neg=neg, extra_phase=90+extra_phase, play=play)

    def initialize(self):
        super().initialize()
        # declare qubit ef pulses 
        for q in self.qubits:
            self.Xef_pulse(q=q, play=False)
