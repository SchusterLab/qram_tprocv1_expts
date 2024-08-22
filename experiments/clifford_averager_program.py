import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import ch2list

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

import scipy as sp

import experiments.fitting as fitter

import logging
logger = logging.getLogger('qick.qick_asm')
logger.setLevel(logging.ERROR)

"""
Averager program that takes care of the standard pulse loading for basic X, Y, Z +/- pi and pi/2
"""
class CliffordAveragerProgram(AveragerProgram):
    # def update(self):
    #     pass

    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        self.gen_delays = [0]*len(soccfg['gens']) # need to calibrate via oscilloscope

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        
        super().__init__(soccfg, self.cfg)

    """
    Wrappers to load and play pulses.
    If play is false, must specify all parameters and all params will be saved (load params).

    If play is true, uses the default values saved from the load call, temporarily overriding freq, phase, or gain if specified to not be None. Sets the pulse registers with these settings and plays the pulse. If you want to set freq, phase, or gain via registers/update,
    be sure to set the default value to be None at loading time.

    If play is True, registers will automatically be set regardless of set_reg flag.
    If play is False, registers will be set based on value of set_reg flag, but pulse will not be played.
    """
    def handle_const_pulse(self, name, waveformname=None, ch=None, length=None, freq_MHz=None, phase_deg=None, gain=None, reload=False, play=False, set_reg=False, ro_ch=None, flag=None, phrst=0, sync_after=True):
        """
        Load/play a constant pulse of given length.
        """
        if name is not None and (name not in self.pulse_dict.keys() or reload):
            assert ch is not None
            self.pulse_dict.update({name:dict(ch=ch, name=name, type='const', length=length, freq_MHz=freq_MHz, phase_deg=phase_deg, gain=gain, ro_ch=ro_ch, flag=flag)})
        if play or set_reg:
            assert name in self.pulse_dict.keys()
            # if not (ch == None):
            #     print('Warning: you have specified a pulse parameter that can only be changed when loading.')
            params = self.pulse_dict[name].copy()
            if freq_MHz is not None: params['freq_MHz'] = freq_MHz
            if phase_deg is not None: params['phase_deg'] = phase_deg
            if gain is not None: params['gain'] = gain
            if ro_ch is not None: params['ro_ch'] = ro_ch
            self.set_pulse_registers(ch=params['ch'], style='const', freq=self.freq2reg(params['freq_MHz'], gen_ch=params['ch'], ro_ch=params['ro_ch']), phase=self.deg2reg(params['phase_deg'], gen_ch=params['ch']), gain=params['gain'], length=params['length'], phrst=phrst)
            if play:
                self.pulse(ch=params['ch'])
                if sync_after: self.sync_all()

    def handle_gauss_pulse(self, name, waveformname=None, ch=None, sigma=None, freq_MHz=None, phase_deg=None, gain=None, reload=False, play=False, set_reg=False, flag=None, phrst=0, sync_after=True):
        """
        Load/play a gaussian pulse of length 4 sigma on channel ch
        """
        if name not in self.pulse_dict.keys() or reload:
            assert None not in [ch, sigma]
            if waveformname is None: waveformname = name
            self.pulse_dict.update({name:dict(ch=ch, name=name, waveformname=waveformname, type='gauss', sigma=sigma, freq_MHz=freq_MHz, phase_deg=phase_deg, gain=gain, flag=flag)})
            if reload or waveformname not in self.envelopes[ch].keys():
                self.add_gauss(ch=ch, name=waveformname, sigma=sigma, length=sigma*4)
        if play or set_reg:
            # if not (ch == sigma == None):
            #     print('Warning: you have specified a pulse parameter that can only be changed when loading.')
            params = self.pulse_dict[name].copy()
            if freq_MHz is not None: params['freq_MHz'] = freq_MHz
            if phase_deg is not None: params['phase_deg'] = phase_deg
            if gain is not None: params['gain'] = gain
            self.set_pulse_registers(ch=params['ch'], style='arb', freq=self.freq2reg(params['freq_MHz'], gen_ch=params['ch']), phase=self.deg2reg(params['phase_deg'], gen_ch=params['ch']), gain=params['gain'], waveform=params['waveformname'], phrst=phrst)
            if play:
                # print('playing gauss pulse', params['name'], 'on ch', params['ch'])
                self.pulse(ch=params['ch'])
                if sync_after: self.sync_all()

    def handle_flat_top_pulse(self, name, waveformname=None, ch=None, sigma=3, flat_length=None, freq_MHz=None, phase_deg=None, gain=None, reload=False, play=False, set_reg=False, flag=None, phrst=0, sync_after=True):
        """
        Plays a gaussian ramp up (2*sigma), a constant pulse of length flat_length+4*sigma,
        plus a gaussian ramp down (2*sigma) on channel ch.
        By default: sigma=3 clock cycles
        """
        if name not in self.pulse_dict.keys() or reload:
            assert None not in [ch, sigma, flat_length]
            if waveformname is None: waveformname = name
            self.pulse_dict.update({name:dict(ch=ch, name=name, waveformname=waveformname, type='flat_top', sigma=sigma, flat_length=flat_length, freq_MHz=freq_MHz, phase_deg=phase_deg, gain=gain, flag=flag)})
            if reload or waveformname not in self.envelopes[ch].keys():
                # print('all waveforms')
                # for i_ch in range(len(self.envelopes)):
                #     print(self.envelopes[i_ch].keys())
                self.add_gauss(ch=ch, name=waveformname, sigma=sigma, length=sigma*4)
                # print('added', waveformname, 'ch', ch)
                # print(self.gen_chs.keys())
        if play or set_reg:
            # if not (ch == name == sigma == length == None):
            #     print('Warning: you have specified a pulse parameter that can only be changed when loading.')
            assert name in self.pulse_dict.keys()
            params = self.pulse_dict[name].copy()
            if freq_MHz is not None: params['freq_MHz'] = freq_MHz
            if phase_deg is not None: params['phase_deg'] = phase_deg
            if gain is not None: params['gain'] = gain
            self.set_pulse_registers(ch=params['ch'], style='flat_top', freq=self.freq2reg(params['freq_MHz'], gen_ch=params['ch']), phase=self.deg2reg(params['phase_deg'], gen_ch=params['ch']), gain=params['gain'], waveform=params['waveformname'], length=params['flat_length'], phrst=phrst)
            if play:
                self.pulse(ch=params['ch'])
                if sync_after: self.sync_all()

    def handle_mux4_pulse(self, name, ch=None, mask=None, length=None, reload=False, play=False, set_reg=False, flag=None):
        """
        Load/play a constant pulse of given length on the mux4 channel.
        """
        # if name is not None or reload: # and name not in self.pulse_dict.keys():
        if name not in self.pulse_dict.keys() or reload:
            assert ch is not None
            assert ch == 6, 'Only ch 6 on q3diamond supports mux4 currently!'
            self.pulse_dict.update({name:dict(ch=ch, name=name, type='mux4', mask=mask, length=length, flag=flag)})
        if play or set_reg:
            assert name in self.pulse_dict.keys()
            params = self.pulse_dict[name].copy()
            if mask is not None: params['mask'] = mask
            if length is not None: params['length'] = length
            self.set_pulse_registers(ch=params['ch'], style='const', length=params['length'], mask=params['mask'])
            if play:
                self.pulse(ch=params['ch'])
                self.sync_all()

    # mu, beta are dimensionless
    def add_adiabatic(self, ch, name, mu, beta, period_us):
        period = self.us2cycles(period_us, gen_ch=ch)
        gencfg = self.soccfg['gens'][ch]
        maxv = gencfg['maxv']*gencfg['maxv_scale']
        samps_per_clk = gencfg['samps_per_clk']
        length = np.round(period) * samps_per_clk
        period *= samps_per_clk
        t = np.arange(0, length)
        iamp, qamp = fitter.adiabatic_iqamp(t, amp_max=1, mu=mu, beta=beta, period=period)
        self.add_pulse(ch=ch, name=name, idata=maxv*iamp, qdata=maxv*qamp)

    def handle_adiabatic_pulse(self, name, waveformname=None, ch=None, mu=None, beta=None, period_us=None, freq_MHz=None, phase_deg=None, gain=None, reload=False, play=False, set_reg=False, flag=None, phrst=0):
        """
        Load/play an adiabatic pi pulse on channel ch
        """
        if name not in self.pulse_dict.keys() or reload:
            assert None not in [ch, mu, beta, period_us]
            if waveformname is None: waveformname = name
            self.pulse_dict.update({name:dict(ch=ch, name=name, waveformname=waveformname, type='adiabatic', mu=mu, beta=beta, period_us=period_us, freq_MHz=freq_MHz, phase_deg=phase_deg, gain=gain, flag=flag)})
            if reload or waveformname not in self.envelopes[ch].keys():
                self.add_adiabatic(ch=ch, name=waveformname, mu=mu, beta=beta, period_us=period_us)
                # print('added gauss pulse', name, 'on ch', ch)
        if play or set_reg:
            # if not (ch == sigma == None):
            #     print('Warning: you have specified a pulse parameter that can only be changed when loading.')
            params = self.pulse_dict[name].copy()
            if freq_MHz is not None: params['freq_MHz'] = freq_MHz
            if phase_deg is not None: params['phase_deg'] = phase_deg
            if gain is not None: params['gain'] = gain
            self.set_pulse_registers(ch=params['ch'], style='arb', freq=self.freq2reg(params['freq_MHz'], gen_ch=params['ch']), phase=self.deg2reg(params['phase_deg'], gen_ch=params['ch']), gain=params['gain'], waveform=params['waveformname'], phrst=phrst)
            if play:
                # print('playing gauss pulse', params['name'], 'on ch', params['ch'])
                self.pulse(ch=params['ch'])
                self.sync_all()

    # I_mhz_vs_us, Q_mhz_vs_us = functions of time in us, in units of MHz
    # times_us = times at which I_mhz_vs_us and Q_mhz_vs_us are defined
    def add_IQ(self, ch, name, I_mhz_vs_us, Q_mhz_vs_us, times_us, plot_IQ=True):
        gencfg = self.soccfg['gens'][ch]
        maxv = gencfg['maxv']*gencfg['maxv_scale'] - 1
        samps_per_clk = gencfg['samps_per_clk']
        times_cycles = np.linspace(0, self.us2cycles(times_us[-1], gen_ch=ch), len(times_us))
        times_samps = samps_per_clk * times_cycles
        IQ_scale = max((np.max(np.abs(I_mhz_vs_us)), np.max(np.abs(Q_mhz_vs_us))))
        I_func = sp.interpolate.interp1d(times_samps, I_mhz_vs_us/IQ_scale, kind='linear', fill_value=0)
        Q_func = sp.interpolate.interp1d(times_samps, -Q_mhz_vs_us/IQ_scale, kind='linear', fill_value=0)
        t = np.arange(0, np.round(times_samps[-1]))
        iamps = I_func(t)
        qamps = Q_func(t)
        
        if plot_IQ:
            plt.figure()
            plt.title(f"Pulse on ch{ch}, waveform {name}")
            # plt.plot(iamps, '.-')
            plt.plot(times_samps, I_func(times_samps), '.-', label='I')
            # plt.plot(qamps, '.-')
            plt.plot(times_samps, Q_func(times_samps), '.-', label='Q')
            plt.ylabel('Amplitude [a.u.]')
            plt.xlabel('Sample Index')
            plt.legend()
            plt.show()

        self.add_pulse(ch=ch, name=name, idata=maxv*iamps, qdata=maxv*qamps)     
        
    def add_IQ_ILC(self, ch, name, I_mhz_vs_us, Q_mhz_vs_us, times_us, plot_IQ=True):
        
        gencfg = self.soccfg['gens'][ch]
        maxv = gencfg['maxv']*gencfg['maxv_scale'] - 1
        samps_per_clk = gencfg['samps_per_clk']
        times_cycles = np.linspace(0, self.us2cycles(times_us[-1], gen_ch=ch), len(times_us))
        times_samps = samps_per_clk * times_cycles
        I_func = sp.interpolate.interp1d(times_samps, I_mhz_vs_us, kind='linear', fill_value=0)
        Q_func = sp.interpolate.interp1d(times_samps, -Q_mhz_vs_us, kind='linear', fill_value=0)
        t = np.arange(0, np.round(times_samps[-1]))
        iamps = I_func(t)
        qamps = Q_func(t)
        
        if plot_IQ:
            plt.figure()
            plt.title(f"Pulse on ch{ch}, waveform {name}")
            # plt.plot(iamps, '.-')
            plt.plot(times_samps, I_func(times_samps), '.-', label='I')
            # plt.plot(qamps, '.-')
            plt.plot(times_samps, Q_func(times_samps), '.-', label='Q')
            plt.ylabel('Amplitude [a.u.]')
            plt.xlabel('Sample Index')
            plt.legend()
            plt.show()
            
        # rescale with the voltage 
        v_scale = maxv//2
        _i = v_scale*iamps
        _q = v_scale*qamps
        
        # check that the IQ values are not larger than the max 
        assert np.all(np.abs(_i) <= maxv)
        assert np.all(np.abs(_i) <= maxv)

        self.add_pulse(ch=ch, name=name, idata=_i, qdata=_q)     

    def handle_IQ_pulse(self, name, waveformname=None, ch=None, I_mhz_vs_us=None, Q_mhz_vs_us=None, times_us=None, freq_MHz=None, phase_deg=None, gain=None, reload=False, play=False, set_reg=False, flag=None, phrst=0, sync_after=True, plot_IQ=True, ILC=False):
        """
        Load/play an arbitrary IQ pulse on channel ch
        """
        if name not in self.pulse_dict.keys() or reload:
            assert ch is not None and I_mhz_vs_us is not None and Q_mhz_vs_us is not None and times_us is not None
            if waveformname is None: waveformname = name
            self.pulse_dict.update({name:dict(ch=ch, name=name, waveformname=waveformname, type='IQpulse', I_mhz_vs_us=I_mhz_vs_us, Q_mhz_vs_us=Q_mhz_vs_us, times_us=times_us, freq_MHz=freq_MHz, phase_deg=phase_deg, gain=gain, flag=flag)})
            if reload or waveformname not in self.envelopes[ch].keys():
                if ILC:
                    self.add_IQ_ILC(ch=ch, name=waveformname, I_mhz_vs_us=I_mhz_vs_us, Q_mhz_vs_us=Q_mhz_vs_us, times_us=times_us, plot_IQ=plot_IQ)
                else: 
                    self.add_IQ(ch=ch, name=waveformname, I_mhz_vs_us=I_mhz_vs_us, Q_mhz_vs_us=Q_mhz_vs_us, times_us=times_us, plot_IQ=plot_IQ)
        if play or set_reg:
            # if not (ch == sigma == None):
            #     print('Warning: you have specified a pulse parameter that can only be changed when loading.')
            params = self.pulse_dict[name].copy()
            if freq_MHz is not None: params['freq_MHz'] = freq_MHz
            if phase_deg is not None: params['phase_deg'] = phase_deg
            if gain is not None: params['gain'] = gain
            self.set_pulse_registers(ch=params['ch'], style='arb', freq=self.freq2reg(params['freq_MHz'], gen_ch=params['ch']), phase=self.deg2reg(params['phase_deg'], gen_ch=params['ch']), gain=params['gain'], waveform=params['waveformname'], phrst=phrst)
            if play:
                self.pulse(ch=params['ch'])
                # print('played iq pulse on ch', ch, params)
                if sync_after: self.sync_all()


    """
    Clifford pulse defns. extra_phase is given in deg. flag can be used to identify certain pulses.
    If play=False, just loads pulse.
    special: adiabatic, pulseiq
    """
    # General drive: Omega cos((wt+phi)X) -> Delta/2 Z + Omega/2 (cos(phi) X + sin(phi) Y)
    def X_pulse(self, q, pihalf=False, divide_len=True, ZZ_qubit=None, neg=False, extra_phase=0, play=False, name='X', flag=None, special=None, phrst=0, reload=False, **kwargs):
        # q: qubit number in config
        if ZZ_qubit is None: ZZ_qubit = q
        f_ge_MHz = self.f_ges[q, ZZ_qubit]
        gain = self.pi_ge_gains[q, ZZ_qubit]
        phase_deg = self.overall_phase[q] + extra_phase
        sigma_cycles = self.us2cycles(self.pi_ge_sigmas[q, ZZ_qubit], gen_ch=self.qubit_chs[q])
        type = self.cfg.device.qubit.pulses.pi_ge.type[q]
        waveformname = 'pi_ge'
        if ZZ_qubit != q:
            waveformname += f'_ZZ{ZZ_qubit}'
            name += f'_ZZ{ZZ_qubit}'
        if special:
            if special == 'adiabatic':
                gain = self.cfg.device.qubit.pulses.pi_ge_adiabatic.gain[q]
                period_us = self.cfg.device.qubit.pulses.pi_ge_adiabatic.period[q]
                mu = self.cfg.device.qubit.pulses.pi_ge_adiabatic.mu[q]
                beta = self.cfg.device.qubit.pulses.pi_ge_adiabatic.beta[q]
                if 'adiabatic' not in name : name = name + '_adiabatic'
                waveformname = 'pi_ge_adiabatic'
                type = 'adiabatic'
            elif special == 'pulseiq':
                type = 'pulseiq'
                gain = self.cfg.device.qubit.pulses.pi_ge_IQ.gain[q]
                waveformname = 'pi_ge_IQ'
                assert 'I_mhz_vs_us' in kwargs.keys() and 'Q_mhz_vs_us' in kwargs.keys() and 'times_us' in kwargs.keys()
                I_mhz_vs_us = kwargs['I_mhz_vs_us']
                Q_mhz_vs_us = kwargs['Q_mhz_vs_us']
                times_us = kwargs['times_us']
        if pihalf:
            if divide_len:
                sigma_cycles = sigma_cycles // 2
                waveformname += '_half'
                gain = self.pi_ge_half_gains[q, ZZ_qubit]
            else: gain = self.pi_ge_half_gain_pi_sigmas[q, ZZ_qubit]
            name += '_half'
        assert f_ge_MHz > 0, f'pulse on {q} {"ZZ "+str(ZZ_qubit) if ZZ_qubit != q else ""}may not be calibrated'
        assert gain > 0, f'pulse on {q} {"ZZ "+str(ZZ_qubit) if ZZ_qubit != q else ""}may not be calibrated'
        assert sigma_cycles > 0, f'pulse on {q} {"ZZ "+str(ZZ_qubit) if ZZ_qubit != q else ""}may not be calibrated'
        if neg: phase_deg -= 180
        if type == 'const':
            self.handle_const_pulse(name=f'{name}_q{q}', ch=self.qubit_chs[q], waveformname=f'{waveformname}_q{q}', length=sigma_cycles, freq_MHz=f_ge_MHz, phase_deg=phase_deg, gain=gain, play=play, flag=flag, phrst=phrst, reload=reload) 
        elif type == 'gauss':
            self.handle_gauss_pulse(name=f'{name}_q{q}', ch=self.qubit_chs[q], waveformname=f'{waveformname}_q{q}', sigma=sigma_cycles, freq_MHz=f_ge_MHz, phase_deg=phase_deg, gain=gain, play=play, flag=flag, phrst=phrst, reload=reload) 
        elif type == 'adiabatic':
            assert not pihalf, 'Cannot do pihalf pulse with adiabatic'
            self.handle_adiabatic_pulse(name=f'{name}_q{q}', ch=self.qubit_chs[q], waveformname=f'{waveformname}_q{q}', mu=mu, beta=beta, period_us=period_us, freq_MHz=f_ge_MHz, phase_deg=phase_deg, gain=gain, play=play, flag=flag, phrst=phrst, reload=reload)
        elif type == 'pulseiq':
            assert not pihalf, 'Cannot do pihalf pulse with pulseiq'
            self.handle_IQ_pulse(name=f'{name}_q{q}', ch=self.qubit_chs[q], waveformname=f'{waveformname}_q{q}', I_mhz_vs_us=I_mhz_vs_us, Q_mhz_vs_us=Q_mhz_vs_us, times_us=times_us, freq_MHz=f_ge_MHz, phase_deg=phase_deg, gain=gain, play=play, flag=flag, phrst=phrst, reload=reload)
        elif type == 'flat_top':
            assert False, 'flat top not checked yet'
            flat_length = self.us2cycles(self.cfg.device.qubit.pulses.pi_ge.length[q], gen_ch=self.qubit_chs[q]) - 3*4
            self.handle_flat_top_pulse(name=f'{name}_q{q}', ch=self.qubit_chs[q], waveformname=f'{waveformname}_q{q}', sigma=sigma_cycles, flat_length=flat_length, freq_MHz=f_ge_MHz, phase_deg=phase_deg, gain=gain, play=play, flag=flag, phrst=phrst, reload=reload) 
        else: assert False, f'Pulse type {type} not supported.'

    def Y_pulse(self, q, pihalf=False, divide_len=True, ZZ_qubit=None, neg=False, extra_phase=0, adiabatic=False, play=False, flag=None, phrst=0, reload=False):
        # the sign of the 180 does not matter, but the sign of the pihalf does!
        self.X_pulse(q, pihalf=pihalf, divide_len=divide_len, ZZ_qubit=ZZ_qubit, neg=not neg, extra_phase=90+extra_phase, play=play, name='Y', flag=flag, adiabatic=adiabatic, phrst=phrst, reload=reload)

    def Z_pulse(self, q, pihalf=False, neg=False, extra_phase=0, play=False, **kwargs):
        dac_type = self.qubit_ch_types[q]
        assert not dac_type == 'mux4', "Currently cannot set phase for mux4!"
        phase_adjust = 180
        if pihalf: phase_adjust = 90 # the sign of the 180 does not matter, but the sign of the pihalf does!
        if neg: phase_adjust *= -1
        if play: self.overall_phase[q] += phase_adjust + extra_phase

    def reset_and_sync(self):
        # Phase reset all channels except readout DACs (since mux ADCs can't be phase reset)
        for ch in self.gen_chs.keys():
            if ch not in self.measure_chs: # doesn't work for the mux ADCs
                # print('resetting', ch)
                self.setup_and_pulse(ch=ch, style='const', freq=100, phase=0, gain=100, length=10, phrst=1)
        self.sync_all(10)

    def set_gen_delays(self):
        for ch in self.gen_chs:
            delay_ns = self.cfg.hw.soc.dacs.delay_chs.delay_ns[np.argwhere(np.array(self.cfg.hw.soc.dacs.delay_chs.ch) == ch)[0][0]]
            delay_cycles = self.us2cycles(delay_ns*1e-3, gen_ch=ch)
            self.gen_delays[ch] = delay_cycles

    def sync_all(self, t=0):
        super().sync_all(t=t, gen_t0=self.gen_delays)
    

    def initialize(self):
        self.cfg = AttrDict(self.cfg)
        self.cfg.update(self.cfg.expt)
        if 'qubits' in self.cfg.expt: self.qubits = self.cfg.expt.qubits
        else: self.qubits = range(4)
        self.pulse_dict = dict()
        self.num_qubits_sample = len(self.cfg.device.readout.frequency)

        # all of these saved self.whatever instance variables should be indexed by the actual qubit number as opposed to qubits_i. this means that more values are saved as instance variables than is strictly necessary, but this is overall less confusing
        self.adc_chs = self.cfg.hw.soc.adcs.readout.ch
        self.res_chs = self.cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = self.cfg.hw.soc.dacs.readout.type
        self.qubit_chs = self.cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = self.cfg.hw.soc.dacs.qubit.type

        if 'cool_qubits' in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            self.swap_f0g1_chs = self.cfg.hw.soc.dacs.swap_f0g1.ch
            self.swap_f0g1_ch_types = self.cfg.hw.soc.dacs.swap_f0g1.type
            mixer_freqs = self.cfg.hw.soc.dacs.swap_f0g1.mixer_freq

        self.overall_phase = [0]*self.num_qubits_sample

        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs] # get register page for qubit_ch

        self.f_ges = np.reshape(self.cfg.device.qubit.f_ge, (4,4))
        self.f_efs = np.reshape(self.cfg.device.qubit.f_ef, (4,4))
        self.pi_ge_gains = np.reshape(self.cfg.device.qubit.pulses.pi_ge.gain, (4,4))
        self.pi_ge_sigmas = np.reshape(self.cfg.device.qubit.pulses.pi_ge.sigma, (4,4))
        self.pi_ge_half_gains = np.reshape(self.cfg.device.qubit.pulses.pi_ge.half_gain, (4,4))
        self.pi_ge_half_gain_pi_sigmas = np.reshape(self.cfg.device.qubit.pulses.pi_ge.half_gain_pi_sigma, (4,4))
        self.pi_ef_gains = np.reshape(self.cfg.device.qubit.pulses.pi_ef.gain, (4,4))
        self.pi_ef_sigmas = np.reshape(self.cfg.device.qubit.pulses.pi_ef.sigma, (4,4))
        self.pi_ef_half_gains = np.reshape(self.cfg.device.qubit.pulses.pi_ef.half_gain, (4,4))
        self.pi_ef_half_gain_pi_sigmas = np.reshape(self.cfg.device.qubit.pulses.pi_ef.half_gain_pi_sigma, (4,4))
        self.pi_ge_types = self.cfg.device.qubit.pulses.pi_ge.type
        self.pi_ef_types = self.cfg.device.qubit.pulses.pi_ef.type

        self.f_res_regs = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(self.cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        if 'cool_qubits' in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            self.f_f0g1_regs = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(self.cfg.device.qubit.f_f0g1, self.qubit_chs)]

        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)]

        # declare res dacs, add readout pulses
        self.measure_chs = []
        self.meas_ch_types = []
        self.meas_ch_qs = []
        self.mask = [] # indices of mux_freqs, mux_gains list to play
        mux_mixer_freq = None
        mux_freqs = [0]*4 # MHz
        mux_gains = [0]*4
        mux_ro_ch = None
        mux_nqz = None
        for q in range(self.num_qubits_sample):
            assert self.res_ch_types[q] in ['full', 'mux4']
            if self.res_ch_types[q] == 'full':
                if self.res_chs[q] not in self.measure_chs:
                    self.declare_gen(ch=self.res_chs[q], nqz=self.cfg.hw.soc.dacs.readout.nyquist[q]) #, ro_ch=self.adc_chs[q])

                    if self.cfg.device.readout.gain[q] < 1:
                        gain = int(self.cfg.device.readout.gain[q] * 2**15)
                    self.handle_const_pulse(name=f'measure{q}', ch=self.res_chs[q], ro_ch=self.adc_chs[q], length=max(self.readout_lengths_dac), freq_MHz=self.cfg.device.readout.frequency[q], phase_deg=0, gain=gain, play=False, set_reg=True)
                    self.measure_chs.append(self.res_chs[q])
                    self.meas_ch_types.append(self.res_ch_types[q])
                    self.meas_ch_qs.append(q)
                
            elif self.res_ch_types[q] == 'mux4':
                assert self.res_chs[q] == 6
                self.mask.append(q)
                if mux_mixer_freq is None: mux_mixer_freq = self.cfg.hw.soc.dacs.readout.mixer_freq[q]
                else: assert mux_mixer_freq == self.cfg.hw.soc.dacs.readout.mixer_freq[q] # ensure all mux channels have specified the same mixer freq
                mux_freqs[q] = self.cfg.device.readout.frequency[q]
                mux_gains[q] = self.cfg.device.readout.gain[q]
                mux_ro_ch = self.adc_chs[q]
                mux_nqz = self.cfg.hw.soc.dacs.readout.nyquist[q]
                if self.res_chs[q] not in self.measure_chs:
                    self.measure_chs.append(self.res_chs[q])
                    self.meas_ch_types.append('mux4')
                    self.meas_ch_qs.append(-1)
        if 'mux4' in self.res_ch_types: # declare mux4 channel
            self.declare_gen(ch=6, nqz=mux_nqz, mixer_freq=mux_mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=mux_ro_ch)
            self.handle_mux4_pulse(name=f'measure', ch=6, length=max(self.readout_lengths_dac), mask=self.mask, play=False, set_reg=True)

        # declare qubit dacs, add qubit pi_ge pulses
        for q in range(self.num_qubits_sample):
            mixer_freq = None
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = self.cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in self.gen_chs:
                self.declare_gen(ch=self.qubit_chs[q], nqz=self.cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)
            self.X_pulse(q=q, play=False, reload=True)

        if 'cool_qubits' in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            mixer_freq = None
            for q in self.cfg.expt.cool_qubits:
                if self.swap_f0g1_ch_types[q] == 'int4':
                    mixer_freq = mixer_freqs[q]
                if self.swap_f0g1_chs[q] not in self.gen_chs: 
                    self.declare_gen(ch=self.swap_f0g1_chs[q], nqz=self.cfg.hw.soc.dacs.swap_f0g1.nyquist[q], mixer_freq=mixer_freq)

                self.pisigma_ef = self.us2cycles(self.pi_ef_sigmas[q, q], gen_ch=self.qubit_chs[q]) # default pi_ef value
                self.add_gauss(ch=self.qubit_chs[q], name=f"pi_ef_qubit{q}", sigma=self.pisigma_ef, length=self.pisigma_ef*4)
                if self.cfg.device.qubit.pulses.pi_f0g1.type[q] == 'flat_top':
                    self.add_gauss(ch=self.swap_f0g1_chs[q], name=f"pi_f0g1_{q}", sigma=3, length=3*4)
                else: assert False, 'not implemented'


        # declare adcs - readout for all qubits everytime, defines number of buffers returned regardless of number of adcs triggered
        for q in range(self.num_qubits_sample):
            if self.adc_chs[q] not in self.ro_chs:
                self.declare_readout(ch=self.adc_chs[q], length=self.readout_lengths_adc[q], freq=self.cfg.device.readout.frequency[q], gen_ch=self.res_chs[q])

        self.set_gen_delays()
        self.sync_all(200)

    """ Collect shots for all adcs, rotates by given angle (degrees), separate based on threshold (if not None), and averages over all shots (i.e. returns data[num_chs, 1] as opposed to data[num_chs, num_shots]) if requested.
    Returns avgi, avgq, avgi_err, avgq_err which avgi/q are avg over shot_avg and avgi/q_err is (std dev of each group of shots)/sqrt(shot_avg)
    """
    def get_shots(self, angle=None, threshold=None, avg_shots=False, verbose=False, return_err=False):
        buf_len = len(self.di_buf[0])

        self.num_qubits_sample = len(self.cfg.device.readout.frequency)
        if angle is None: angle = [0]*self.num_qubits_sample
        bufi = np.array([
            self.di_buf[i]*np.cos(np.pi/180*angle[i]) - self.dq_buf[i]*np.sin(np.pi/180*angle[i])
            for i, ch in enumerate(self.ro_chs)])
        bufi = np.array([bufi[i]/ro['length'] for i, (ch, ro) in enumerate(self.ro_chs.items())])
        if threshold is not None: # categorize single shots
            bufi = np.array([np.heaviside(bufi[ch] - threshold[ch], 0) for ch in range(len(self.adc_chs))])
        avgi = np.average(bufi, axis=1) # [num_chs]
        bufi_err = np.std(bufi, axis=1) / np.sqrt(buf_len) # [num_chs]
        if verbose: print([np.median(bufi[i]) for i in range(4)])

        bufq = np.array([
            self.di_buf[i]*np.sin(np.pi/180*angle[i]) + self.dq_buf[i]*np.cos(np.pi/180*angle[i])
            for i, ch in enumerate(self.ro_chs)])
        bufq = np.array([bufq[i]/ro['length'] for i, (ch, ro) in enumerate(self.ro_chs.items())])
        avgq = np.average(bufq, axis=1) # [num_chs]
        bufq_err = np.std(bufq, axis=1) / np.sqrt(buf_len) # [num_chs]
        if verbose: print([np.median(bufq[i]) for i in range(4)])

        if avg_shots:
            idata = avgi
            qdata = avgq
        else:
            idata = bufi
            qdata = bufq

        if return_err: return idata, qdata, bufi_err, bufq_err
        else: return idata, qdata 

    """
    If post_process == 'threshold': uses angle + threshold to categorize shots into 0 or 1 and calculate the population
    If post_process == 'scale': uses angle + ge_avgs to scale the average of all shots on a scale of 0 to 1. ge_avgs should be of shape (num_total_qubits, 4) and should represent the pre-rotation Ig, Qg, Ie, Qe
    If post_process == None: uses angle to rotate the i and q and then returns the avg i and q
    """
    def acquire_rotated(self, soc, progress, angle=None, threshold=None, ge_avgs=None, post_process=None, verbose=False):
        avgi, avgq = self.acquire(soc, load_pulses=True, progress=progress)
        if post_process == None: 
            avgi_rot, avgq_rot, avgi_err, avgq_err = self.get_shots(angle=angle, avg_shots=True, verbose=verbose, return_err=True)
            if angle is None: return avgi_rot, avgq_rot
            else: return avgi_rot, avgi_err
        elif post_process == 'threshold':
            assert threshold is not None
            popln, avgq_rot, popln_err, avgq_err = self.get_shots(angle=angle, threshold=threshold, avg_shots=True, verbose=verbose, return_err=True)
            return popln, popln_err
        elif post_process == 'scale':
            assert ge_avgs is not None
            avgi_rot, avgq_rot, avgi_err, avgq_err = self.get_shots(angle=angle, avg_shots=True, verbose=verbose, return_err=True)

            ge_avgs_rot = [None]*4
            for q, angle_q in enumerate(angle):
                if not isinstance(ge_avgs[q], (list, np.ndarray)): continue # this qubit was not calibrated
                Ig_q, Qg_q, Ie_q, Qe_q = ge_avgs[q]
                ge_avgs_rot[q] = [
                    Ig_q*np.cos(np.pi/180*angle_q) - Qg_q*np.sin(np.pi/180*angle_q),
                    Ie_q*np.cos(np.pi/180*angle_q) - Qe_q*np.sin(np.pi/180*angle_q)
                ]
            shape = None
            for q in range(4):
                if ge_avgs_rot[q] is not None:
                    shape = np.shape(ge_avgs_rot[q])
                    break
            for q in range(4):
                if ge_avgs_rot[q] is None: ge_avgs_rot[q] = np.zeros(shape=shape)
                
            ge_avgs_rot = np.asarray(ge_avgs_rot)
            avgi_rot -= ge_avgs_rot[:,0]
            avgi_rot /= ge_avgs_rot[:,1] - ge_avgs_rot[:,0]
            avgi_err /= ge_avgs_rot[:,1] - ge_avgs_rot[:,0]
            return avgi_rot, avgi_err
        else:
            assert False, 'Undefined post processing flag, options are None, threshold, scale'

# ===================================================================== #

"""
Take care of extra clifford pulses for qutrits.
"""
class QutritAveragerProgram(CliffordAveragerProgram):
    def Xef_pulse(self, q, pihalf=False, divide_len=True, name='X_ef', ZZ_qubit=None, neg=False, extra_phase=0, play=False, flag=None, phrst=0, reload=True):
        ch = self.qubit_chs[q]
        if ZZ_qubit is None: ZZ_qubit = q
        f_ef_MHz = self.f_efs[q, ZZ_qubit]
        gain = self.pi_ef_gains[q, ZZ_qubit]
        phase_deg = self.overall_phase_ef[q] + extra_phase
        sigma_cycles = self.us2cycles(self.pi_ef_sigmas[q, ZZ_qubit], gen_ch=ch)
        type = self.cfg.device.qubit.pulses.pi_ef.type[q]
        waveformname = 'pi_ef'
        if ZZ_qubit != q:
            waveformname += f'_ZZ{ZZ_qubit}'
            name += f'_ZZ{ZZ_qubit}'
        if pihalf:
            if divide_len:
                sigma_cycles = sigma_cycles // 2
                waveformname += '_half'
                gain = self.pi_ef_half_gains[q, ZZ_qubit]
            else: gain = self.pi_ef_half_gain_pi_sigmas[q, ZZ_qubit]
            name += '_half'
        assert f_ef_MHz > 0, f'EF pulse on {q} {"ZZ "+str(ZZ_qubit) if ZZ_qubit != q else ""}may not be calibrated'
        assert gain > 0, f'EF pulse on {q} {"ZZ "+str(ZZ_qubit) if ZZ_qubit != q else ""}may not be calibrated'
        assert sigma_cycles > 0, f'EF pulse on {q} {"ZZ "+str(ZZ_qubit) if ZZ_qubit != q else ""}may not be calibrated'
        if neg: phase_deg -= 180
        if type == 'const':
            self.handle_const_pulse(name=f'{name}_q{q}', ch=ch, waveformname=f'{waveformname}_q{q}', length=sigma_cycles, freq_MHz=f_ef_MHz, phase_deg=phase_deg, gain=gain, play=play, flag=flag, phrst=phrst, reload=reload)
        elif type == 'gauss':
            self.handle_gauss_pulse(name=f'{name}_q{q}', ch=ch, waveformname=f'{waveformname}_q{q}', sigma=sigma_cycles, freq_MHz=f_ef_MHz, phase_deg=phase_deg, gain=gain, play=play, flag=flag, phrst=phrst, reload=reload)
        elif type == 'flat_top':
            sigma_ramp_cycles = 3
            flat_length_cycles = sigma_cycles - sigma_ramp_cycles*4
            self.handle_flat_top_pulse(name=f'{name}_q{q}', ch=ch, waveformname=f'{waveformname}_q{q}', sigma=sigma_ramp_cycles, flat_length=flat_length_cycles, freq_MHz=f_ef_MHz, phase_deg=phase_deg, gain=gain, play=play, flag=flag, phrst=phrst, reload=reload)
        else: assert False, f'Pulse type {type} not supported.'
    
    def Yef_pulse(self, q, pihalf=False, divide_len=True, ZZ_qubit=None, neg=False, extra_phase=0, play=False, flag=None, phrst=0, reload=True):
        # the sign of the 180 does not matter, but the sign of the pihalf does!
        self.Xef_pulse(q, pihalf=pihalf, divide_len=divide_len, ZZ_qubit=ZZ_qubit, neg=not neg, extra_phase=90+extra_phase, play=play, name='Y_ef', flag=flag, phrst=phrst, reload=reload)

    def Zef_pulse(self, q, pihalf=False, neg=False, extra_phase=0, play=False, **kwargs):
        dac_type = self.qubit_ch_types[q]
        assert not dac_type == 'mux4', "Currently cannot set phase for mux4!"
        phase_adjust = 180
        if pihalf: phase_adjust = 90 # the sign of the 180 does not matter, but the sign of the pihalf does!
        if neg: phase_adjust *= -1
        if play: self.overall_phase_ef[q] += phase_adjust + extra_phase


    """
    cool_idle should be the same length as cool_qubits
    """
    def active_cool(self, cool_qubits, cool_idle):
        # print('cooling qubits', cool_qubits, 'with idle times', cool_idle)
        assert len(cool_idle) == len(cool_qubits)

        sorted_indices = np.argsort(cool_idle)[::-1] # sort cooling times longest first
        cool_qubits = np.array(cool_qubits)
        cool_idle = np.array(cool_idle)
        sorted_cool_qubits = cool_qubits[sorted_indices]
        sorted_cool_idle = cool_idle[sorted_indices]
        # print('sorted cool_qubits', sorted_cool_qubits)
        max_idle = sorted_cool_idle[0]
        
        last_pulse_len = 0
        remaining_idle = max_idle
        for q, idle in zip(sorted_cool_qubits, sorted_cool_idle):
            remaining_idle -= last_pulse_len

            last_pulse_len = 0
            self.Xef_pulse(q=q, play=True)
            last_pulse_len += self.pi_ef_sigmas[q, q]*4

            pulse_type = self.cfg.device.qubit.pulses.pi_f0g1.type[q]
            pisigma_f0g1 = self.us2cycles(self.cfg.device.qubit.pulses.pi_f0g1.sigma[q], gen_ch=self.swap_f0g1_chs[q])
            if pulse_type == 'flat_top':
                sigma_ramp_cycles = 3
                flat_length_cycles = pisigma_f0g1 - sigma_ramp_cycles*4
                self.setup_and_pulse(ch=self.swap_f0g1_chs[q], style="flat_top", freq=self.f_f0g1_regs[q], phase=0, gain=self.cfg.device.qubit.pulses.pi_f0g1.gain[q], length=flat_length_cycles, waveform=f"pi_f0g1_{q}")
            else: assert False, 'not implemented'
            self.sync_all()
            last_pulse_len += self.cfg.device.qubit.pulses.pi_f0g1.sigma[q]
            # print(f'last pulse len q{q}', last_pulse_len)

        remaining_idle -= last_pulse_len
        last_idle = max((remaining_idle, sorted_cool_idle[-1]))
        # print('last idle', last_idle)
        self.sync_all(self.us2cycles(last_idle))
        

    def initialize(self):
        super().initialize()
        self.overall_phase_ef = [0]*self.num_qubits_sample
        # declare qubit ef pulses 
        # print(self.gen_chs)
        for q in range(self.num_qubits_sample):
            self.Xef_pulse(q=q, play=False)
        self.sync_all(200)


# ===================================================================== #
"""
Multiple inheritence testing
"""
# class Clifford():
#     def xpulse(self):
#         print('normal clifford')

#     def ypulse(self):
#         print('y')
#         self.xpulse()

# class CliffordEF(Clifford):
#     def xefpulse(self):
#         print('ef')

# class CliffordEgGf(CliffordEF):
#     def xpulse(self):
#         super().xpulse()
#         print('EgGf')

# class SimRB(Clifford):
#     def clifford(self, flag=None):
#         if flag == 'X': self.xpulse()
#         elif flag == 'Y': self.ypulse()
    
# class RBEgGf(CliffordEgGf, SimRB):
#     pass

# rbeggf = RBEgGf()
# print(RBEgGf.__mro__)
# rbeggf.clifford(flag='X')
# rbeggf.clifford(flag='Y')

"""
Replace the X/Y/Z pulses with an effective TLS represented by the Eg-Gf pulse.
add_virtual_Z is a flag to determine whether is a virtual Z gate applied on the drive qubit *after* the swap.
extra_phase is applied to the swap itself.
"""
class CliffordEgGfAveragerProgram(QutritAveragerProgram):
    # self.overall_phase keeps track of the EgGf phase insetad of the e-g pulse phase

    def XEgGf_pulse(self, qDrive, qNotDrive, pihalf=False, divide_len=True, name='X_EgGf', neg=False, extra_phase=0, add_virtual_Z=True, play=False, flag=None, phrst=0, reload=True):
        # convention is waveformname is pi_EgGf_qNotDriveqDrive
        if qDrive == 1:
            ch = self.swap_chs[qNotDrive]
            f_EgGf_MHz = self.cfg.device.qubit.f_EgGf[qNotDrive]
            gain = self.cfg.device.qubit.pulses.pi_EgGf.gain[qNotDrive]
            phase_deg = self.overall_phase[qNotDrive] + extra_phase
            sigma_cycles = self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf.sigma[qNotDrive], gen_ch=ch)
            type = self.cfg.device.qubit.pulses.pi_EgGf.type[qNotDrive]
            waveformname = 'pi_EgGf'
            if add_virtual_Z:
                virtual_Z = self.cfg.device.qubit.pulses.pi_EgGf.phase[qNotDrive]
        else:
            ch = self.swap_Q_chs[qDrive]
            f_EgGf_MHz = self.cfg.device.qubit.f_EgGf_Q[qDrive]
            gain = self.cfg.device.qubit.pulses.pi_EgGf_Q.gain[qDrive]
            phase_deg = self.overall_phase[qDrive] + extra_phase
            sigma_cycles = self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf_Q.sigma[qDrive], gen_ch=ch)
            type = self.cfg.device.qubit.pulses.pi_EgGf_Q.type[qDrive]
            waveformname = 'pi_EgGf'
            if add_virtual_Z:
                virtual_Z = self.cfg.device.qubit.pulses.pi_EgGf_Q.phase[qDrive]
        if pihalf:
            if divide_len:
                # sigma_cycles = sigma_cycles // 2
                if qDrive == 1:
                    sigma_cycles = self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf.half_sigma[qNotDrive], gen_ch=ch)
                    virtual_Z = self.cfg.device.qubit.pulses.pi_EgGf.half_phase[qNotDrive]
                else:
                    sigma_cycles = self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf_Q.half_sigma[qDrive], gen_ch=ch)
                    virtual_Z = self.cfg.device.qubit.pulses.pi_EgGf_Q.half_phase[qDrive]
                waveformname += 'half'
            else:
                assert False, 'dividing gain for an eg-gf pi/2 pulse is a bad idea!'
            name += 'half'
        if neg: phase_deg -= 180
        if type == 'const':
            self.handle_const_pulse(name=f'{name}_{qNotDrive}{qDrive}', ch=ch, waveformname=f'{waveformname}_{qNotDrive}{qDrive}', length=sigma_cycles, freq_MHz=f_EgGf_MHz, phase_deg=phase_deg, gain=gain, play=play, flag=flag, phrst=phrst, reload=reload)
        elif type == 'gauss':
            self.handle_gauss_pulse(name=f'{name}_{qNotDrive}{qDrive}', ch=ch, waveformname=f'{waveformname}_{qNotDrive}{qDrive}', sigma=sigma_cycles, freq_MHz=f_EgGf_MHz, phase_deg=phase_deg, gain=gain, play=play, flag=flag, phrst=phrst, reload=reload)
        elif type == 'flat_top':
            sigma_ramp_cycles = 3
            flat_length_cycles = sigma_cycles - sigma_ramp_cycles*4
            self.handle_flat_top_pulse(name=f'{name}_{qNotDrive}{qDrive}', ch=ch, waveformname=f'{waveformname}_{qNotDrive}{qDrive}', sigma=sigma_ramp_cycles, flat_length=flat_length_cycles, freq_MHz=f_EgGf_MHz, phase_deg=phase_deg, gain=gain, play=play, flag=flag, phrst=phrst, reload=reload)
        else: assert False, f'Pulse type {type} not supported.'

        if add_virtual_Z: self.overall_phase[qDrive] += virtual_Z
        # print('ch keys', self.gen_chs.keys())

    def YEgGf_pulse(self, qDrive, qNotDrive, pihalf=False, neg=False, extra_phase=0, add_virtual_Z=False, play=False, flag=None, phrst=0, reload=True):
        # the sign of the 180 does not matter, but the sign of the pihalf does!
        self.XEgGf_pulse(qDrive, qNotDrive, pihalf=pihalf, neg=not neg, extra_phase=90+extra_phase, add_virtual_Z=add_virtual_Z, play=play, name='Y_EgGf', flag=flag, phrst=phrst, reload=reload)

    def ZEgGf_pulse(self, qDrive, qNotDrive, pihalf=False, divide_len=True, neg=False, extra_phase=0,  add_virtual_Z=False, play=False, reload=None):
        if qDrive == 1:
            sigma_us = self.cfg.device.qubit.pulses.pi_EgGf.sigma[qNotDrive]
            dac_type = self.swap_ch_types[qNotDrive]
        else:
            sigma_us = self.cfg.device.qubit.pulses.pi_EgGf_Q.sigma[qDrive]
            dac_type = self.swap_Q_ch_types[qDrive]
        if pihalf:
            if divide_len:
                sigma_us /= 2
        assert not dac_type == 'mux4', "Currently cannot set phase for mux4!"
        phase_adjust = 180
        if pihalf: phase_adjust = 90 # the sign of the 180 does not matter, but the sign of the pihalf does!
        if neg: phase_adjust *= -1
        if play:
            self.overall_phase[qDrive] += phase_adjust + extra_phase
            # self.sync_all(self.us2cycles(sigma_us))

    def initialize(self):
        self.swap_chs = self.cfg.hw.soc.dacs.swap.ch
        self.swap_ch_types = self.cfg.hw.soc.dacs.swap.type
        self.swap_Q_chs = self.cfg.hw.soc.dacs.swap_Q.ch
        self.swap_Q_ch_types = self.cfg.hw.soc.dacs.swap_Q.type
        super().initialize()
        for q in self.qubits:
            if q==1: continue
            mixer_freq = 0
            if self.swap_ch_types[q] == 'int4':
                mixer_freq = self.cfg.hw.soc.dacs.swap.mixer_freq[q]
            if self.swap_chs[q] not in self.gen_chs:
                self.declare_gen(ch=self.swap_chs[q], nqz=self.cfg.hw.soc.dacs.swap.nyquist[q], mixer_freq=mixer_freq)
            # else: print('nqz', self.gen_chs[self.swap_chs[q]]['nqz'])
            mixer_freq=0
            if self.swap_Q_ch_types[q] == 'int4':
                mixer_freq = self.cfg.hw.soc.dacs.swap_Q.mixer_freq[q]
            if self.swap_Q_chs[q] not in self.gen_chs: 
                self.declare_gen(ch=self.swap_Q_chs[q], nqz=self.cfg.hw.soc.dacs.swap_Q.nyquist[q], mixer_freq=mixer_freq)
        self.sync_all(100)