import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from copy import deepcopy
import json

from slab import Experiment, NpEncoder, AttrDict
from tqdm import tqdm_notebook as tqdm

import scipy as sp
import matplotlib.pyplot as plt

import experiments.fitting as fitter
from experiments.single_qubit.single_shot import hist
from experiments.two_qubit.twoQ_state_tomography import ErrorMitigationStateTomo2QProgram, AbstractStateTomo2QProgram, ErrorMitigationStateTomo1QProgram

# ====================================================== #

class AmplitudeRabiOptimalCtrlProgram(RAveragerProgram):
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
        Q_func = sp.interpolate.interp1d(times_samps, Q_mhz_vs_us/IQ_scale, kind='linear', fill_value=0)
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

    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        self.gen_delays = [0]*len(soccfg['gens']) # need to calibrate via oscilloscope

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
        
        super().__init__(soccfg, self.cfg)

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
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)

        self.num_qubits_sample = len(self.cfg.device.readout.frequency)

        assert 'Icontrols' in self.cfg.expt and 'Qcontrols' in self.cfg.expt and 'times_us' in self.cfg.expt
        assert 'IQ_qubits' in self.cfg.expt
        assert 'gains' in self.cfg.expt
        self.Icontrols = self.cfg.expt.Icontrols
        self.Qcontrols = self.cfg.expt.Qcontrols
        self.IQ_qubits = self.cfg.expt.IQ_qubits
        self.times_us = self.cfg.expt.times_us

        self.adc_chs = cfg.hw.soc.adcs.readout.ch
        self.res_chs = cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = cfg.hw.soc.dacs.readout.type
        self.qubit_chs = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = cfg.hw.soc.dacs.qubit.type

        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs] # get register page for qubit_chs

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

        self.f_res_regs = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)]

        
        # declare all res dacs
        self.measure_chs = []
        mask = [] # indices of mux_freqs, mux_gains list to play
        mux_mixer_freq = None
        mux_freqs = [0]*4 # MHz
        mux_gains = [0]*4
        mux_ro_ch = None
        mux_nqz = None
        for q in range(self.num_qubits_sample):
            assert self.res_ch_types[q] in ['full', 'mux4']
            if self.res_ch_types[q] == 'full':
                if self.res_chs[q] not in self.measure_chs:
                    self.declare_gen(ch=self.res_chs[q], nqz=cfg.hw.soc.dacs.readout.nyquist[q], mixer_freq=cfg.hw.soc.dacs.readout.mixer_freq[q], ro_ch=self.adc_chs[q])
                    self.measure_chs.append(self.res_chs[q])
                
            elif self.res_ch_types[q] == 'mux4':
                assert self.res_chs[q] == 6
                mask.append(q)
                if mux_mixer_freq is None: mux_mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[q]
                else: assert mux_mixer_freq == cfg.hw.soc.dacs.readout.mixer_freq[q] # ensure all mux channels have specified the same mixer freq
                mux_freqs[q] = cfg.device.readout.frequency[q]
                mux_gains[q] = cfg.device.readout.gain[q]
                mux_ro_ch = self.adc_chs[q]
                mux_nqz = cfg.hw.soc.dacs.readout.nyquist[q]
                if self.res_chs[q] not in self.measure_chs:
                    self.measure_chs.append(self.res_chs[q])
        if 'mux4' in self.res_ch_types: # declare mux4 channel
            self.declare_gen(ch=6, nqz=mux_nqz, mixer_freq=mux_mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=mux_ro_ch)


        # declare adcs - readout for all qubits everytime, defines number of buffers returned regardless of number of adcs triggered
        for q in range(self.num_qubits_sample):
            if self.adc_chs[q] not in self.ro_chs:
                self.declare_readout(ch=self.adc_chs[q], length=self.readout_lengths_adc[q], freq=self.cfg.device.readout.frequency[q], gen_ch=self.res_chs[q])

        # declare qubit dacs
        for q in self.IQ_qubits:
            mixer_freq = None
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in self.gen_chs:
                self.declare_gen(ch=self.qubit_chs[q], nqz=cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)

        # add qubit and readout pulses to respective channels
        if 'plot_IQ' not in self.cfg.expt or self.cfg.expt.plot_IQ == None: self.cfg.expt.plot_IQ = False
        for iq, q in enumerate(self.IQ_qubits):
            self.add_IQ(ch=self.qubit_chs[q], name=f'pulse_Q{q}', I_mhz_vs_us=self.Icontrols[iq], Q_mhz_vs_us=self.Qcontrols[iq], times_us=self.cfg.expt.times_us, plot_IQ=self.cfg.expt.plot_IQ)

        # add readout pulses to respective channels
        if 'mux4' in self.res_ch_types:
            self.set_pulse_registers(ch=6, style="const", length=max(self.readout_lengths_dac), mask=mask)
        for q in range(self.num_qubits_sample):
            if self.res_ch_types[q] != 'mux4':
                if cfg.device.readout.gain[q] < 1:
                    gain = int(cfg.device.readout.gain[q] * 2**15)
                self.set_pulse_registers(ch=self.res_chs[q], style="const", freq=self.f_res_regs[q], phase=0, gain=gain, length=max(self.readout_lengths_dac))

        # initialize registers for first qubit in IQ_qubits. To loop over gain on other qubits, do this in an outer loop.
        if self.qubit_ch_types[q] == 'int4':
            self.r_gain = self.sreg(self.qubit_chs[self.IQ_qubits[0]], "addr") # get gain register for qubit_ch    
        else: self.r_gain = self.sreg(self.qubit_chs[self.IQ_qubits[0]], "gain") # get gain register for qubit_ch    
        self.r_gain2 = 4
        self.safe_regwi(self.q_rps[self.IQ_qubits[0]], self.r_gain2, self.cfg.expt.start)

        self.set_gen_delays()
        self.sync_all(200)

    def body(self):
        cfg=AttrDict(self.cfg)

        self.reset_and_sync()

        for q in self.IQ_qubits:
            if q == 0: gain = 0
            else: gain = self.cfg.expt.gains[q]
            self.set_pulse_registers(
                ch=self.qubit_chs[q],
                style="arb",
                freq=self.freq2reg(self.f_ges[q,q], gen_ch=self.qubit_chs[q]),
                phase=0,
                gain=gain, # gain set by update for 0th Q in IQ_qubits only
                waveform=f"pulse_Q{q}")
        self.mathi(self.q_rps[self.IQ_qubits[0]], self.r_gain, self.r_gain2, "+", 0)

        for q in self.IQ_qubits:
            self.pulse(ch=self.qubit_chs[q])
            # NO SYNC ALL HERE

        # align channels and measure
        self.sync_all()
        syncdelay = self.us2cycles(max(self.cfg.device.readout.relax_delay))
        self.measure(pulse_ch=self.measure_chs, adcs=self.adc_chs, adc_trig_offset=self.cfg.device.readout.trig_offset[0], wait=True, syncdelay=syncdelay) 

    def update(self):
        step = self.cfg.expt.step
        if self.qubit_ch_types[self.IQ_qubits[0]] == 'int4': step = step << 16
        self.mathi(self.q_rps[self.IQ_qubits[0]], self.r_gain2, self.r_gain2, '+', step) # update test gain

    """ Collect shots for all adcs, rotates by given angle (degrees), separate based on threshold (if not None), and averages over all shots (i.e. returns data[num_chs, 1] as opposed to data[num_chs, num_shots]) if requested.
    Returns avgi, avgq, avgi_err, avgq_err which avgi/q are avg over shot_avg and avgi/q_err is (std dev of each group of shots)/sqrt(shot_avg)
    """
    def get_shots(self, angle=None, threshold=None, avg_shots=False, verbose=False, return_err=False):
        buf_len = len(self.di_buf[0])
        # for raverager program, bufi is length expts x reps, with each expt repeated reps times before moving to the next parameter value (so if you want to look at separate experiments, reshape to (chs, expts, reps) and average over axis 2)

        if angle is None: angle = [0]*len(self.cfg.device.qubit.f_ge)
        bufi = np.array([
            self.di_buf[i]*np.cos(np.pi/180*angle[i]) - self.dq_buf[i]*np.sin(np.pi/180*angle[i])
            for i, ch in enumerate(self.ro_chs)])
        bufi = np.array([bufi[i]/ro['length'] for i, (ch, ro) in enumerate(self.ro_chs.items())])
        if threshold is not None: # categorize single shots
            bufi = np.array([np.heaviside(bufi[ch] - threshold[ch], 0) for ch in range(len(self.adc_chs))])
        avgi = np.average(np.reshape(bufi, (len(self.ro_chs.items()), self.cfg.expt.expts, self.cfg.expt.reps)), axis=2)
        bufi_err = np.std(np.reshape(bufi, (len(self.ro_chs.items()), self.cfg.expt.expts, self.cfg.expt.reps)), axis=2) / np.sqrt(self.cfg.expt.reps)
        if verbose: print([np.median(bufi[i]) for i in range(4)])

        bufq = np.array([
            self.di_buf[i]*np.sin(np.pi/180*angle[i]) + self.dq_buf[i]*np.cos(np.pi/180*angle[i])
            for i, ch in enumerate(self.ro_chs)])
        bufq = np.array([bufq[i]/ro['length'] for i, (ch, ro) in enumerate(self.ro_chs.items())])
        avgq = np.average(np.reshape(bufq, (len(self.ro_chs.items()), self.cfg.expt.expts, self.cfg.expt.reps)), axis=2)
        bufq_err = np.std(np.reshape(bufq, (len(self.ro_chs.items()), self.cfg.expt.expts, self.cfg.expt.reps)), axis=2) / np.sqrt(self.cfg.expt.reps)
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
        xpts, avgi, avgq = self.acquire(soc, load_pulses=True, progress=progress)
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
            for i_ch in range(ge_avgs_rot.shape[0]):
                avgi_rot[i_ch] -= ge_avgs_rot[i_ch,0]
                avgi_rot[i_ch] /= ge_avgs_rot[i_ch,1] - ge_avgs_rot[i_ch,0]
                avgi_err[i_ch] /= ge_avgs_rot[i_ch,1] - ge_avgs_rot[i_ch,0]
            return avgi_rot, avgi_err
        else:
            assert False, 'Undefined post processing flag, options are None, threshold, scale'

# ====================================================== #

class AmplitudeRabiOptimalCtrlExperiment(Experiment):
    """
    Amplitude Rabi Experiment
    Experimental Config:
    expt = dict(
        start: qubit gain [dac level] for 0th qubit in IQ_qubits
        step: gain step [dac level]
        expts: number steps
        gains: gains to play on each qubit in IQ_qubits (only gains on index > 0 are used, 0th gain is set by start/sweep)
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        sigma_test: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        Icontrols: 2D array with array of Icontrols for each of IQ_qubits
        Qcontrols: 2D array with array of Qcontrols for each of IQ_qubits
        times_us: same length as each array in Icontrols, Qcontrols
        IQ_qubits: qubits to play Icontrols, Qcontrols on
        plot_IQ: whether to plot the sampled IQ functions
    )
    """

    def __init__(self, soccfg=None, path='', prefix='AmpRabiOptCtrl', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.readout.frequency)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update({key3: [value3]*num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})

        
        prog = AmplitudeRabiOptimalCtrlProgram(soccfg=self.soccfg, cfg=self.cfg)
        # print(amprabi)
        # from qick.helpers import progs2json
        # print(progs2json([amprabi.dump_prog()]))
        
        xpts, avgi, avgq = prog.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress)
        # print(amprabi)

        # shots_i = amprabi.di_buf[adc_ch].reshape((self.cfg.expt.expts, self.cfg.expt.reps)) / amprabi.readout_length_adc
        # shots_i = np.average(shots_i, axis=1)
        # print(len(shots_i), self.cfg.expt.expts)
        # shots_q = amprabi.dq_buf[adc_ch] / amprabi.readout_length_adc
        # print(np.std(shots_i), np.std(shots_q))
        
        data={'xpts': xpts, 'avgi':[], 'avgq':[], 'amps':[], 'phases':[]}
        
        for q in self.cfg.expt.IQ_qubits:
            data['avgi'].append(avgi[q][0])
            data['avgq'].append(avgq[q][0])
            data['amps'].append(np.abs(avgi[q][0] + 1j*avgq[q][0]))
            data['phases'].append(np.angle(avgi[q][0] + 1j*avgq[q][0])) # Calculating the phase        
        
        self.data=data
        return data

    def analyze(self, data=None, fit=True, fitparams=None):
        if data is None:
            data=self.data
        
        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset]
            # fitparams=[yscale, freq, phase_deg, y0]
            # Remove the first and last point from fit in case weird edge measurements
            xdata = data['xpts']
            if fitparams is None:
                fitparams=[None]*4
                n_pulses = 1
                if 'n_pulses' in self.cfg.expt: n_pulses = self.cfg.expt.n_pulses
                fitparams[1]=n_pulses/xdata[-1]
                # print(fitparams[1])

            ydata = data["amps"]
            # print(abs(xdata[np.argwhere(ydata==max(ydata))[0,0]] - xdata[np.argwhere(ydata==min(ydata))[0,0]]))
            # fitparams=[max(ydata)-min(ydata), 1/2 / abs(xdata[np.argwhere(ydata==max(ydata))[0,0]] - xdata[np.argwhere(ydata==min(ydata))[0,0]]), None, None, None]
            # fitparams=[max(ydata)-min(ydata), 1/2 / (max(xdata) - min(xdata)), 0, None, None]

            p_avgi, pCov_avgi = fitter.fitsin(data['xpts'][:-1], data["avgi"][:-1], fitparams=fitparams)
            p_avgq, pCov_avgq = fitter.fitsin(data['xpts'][:-1], data["avgq"][:-1], fitparams=fitparams)
            p_amps, pCov_amps = fitter.fitsin(data['xpts'][:-1], data["amps"][:-1], fitparams=fitparams)
            data['fit_avgi'] = p_avgi   
            data['fit_avgq'] = p_avgq
            data['fit_amps'] = p_amps
            data['fit_err_avgi'] = pCov_avgi   
            data['fit_err_avgq'] = pCov_avgq
            data['fit_err_amps'] = pCov_amps
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 

        plt.figure(figsize=(10, 6))
        n_pulses = 1
        if 'n_pulses' in self.cfg.expt: n_pulses = self.cfg.expt.n_pulses
        title = f"Amplitude Rabi {'EF ' if self.cfg.expt.checkEF else ''}on Q{qTest} (Pulse Length {self.cfg.expt.sigma_test}{(', ZZ Q'+str(qZZ)) if self.checkZZ else ''}, {n_pulses} pulse)"
        plt.subplot(111, title=title, xlabel="Gain [DAC units]", ylabel="Amplitude [ADC units]")
        plt.plot(data["xpts"][1:-1], data["amps"][1:-1],'.-')
        if fit:
            p = data['fit_amps']
            plt.plot(data["xpts"][1:-1], fitter.sinfunc(data["xpts"][1:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
            else: pi_gain = (3/2 - p[2]/180)/2/p[1]
            pi_gain += (n_pulses-1)*1/2/p[1]
            pi2_gain = pi_gain/2
            print(f'Pi gain from amps data [dac units]: {int(pi_gain)}')
            print(f'\tPi/2 gain from amps data [dac units]: {int(pi2_gain)}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi2_gain, color='0.2', linestyle='--')

        plt.figure(figsize=(10,10))
        plt.subplot(211, title=title, ylabel="I [ADC units]")
        plt.plot(data["xpts"][1:-1], data["avgi"][1:-1],'.-')
        # plt.axhline(390)
        # plt.axhline(473)
        # plt.axvline(2114)
        # plt.axvline(3150)
        if fit:
            p = data['fit_avgi']
            plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
            else: pi_gain= (3/2 - p[2]/180)/2/p[1]
            pi_gain += (n_pulses-1)*1/2/p[1]
            pi2_gain = pi_gain/2
            print(f'Pi gain from avgi data [dac units]: {int(pi_gain)}')
            print(f'\tPi/2 gain from avgi data [dac units]: {int(pi2_gain)}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi2_gain, color='0.2', linestyle='--')
        plt.subplot(212, xlabel="Gain [DAC units]", ylabel="Q [ADC units]")
        plt.plot(data["xpts"][1:-1], data["avgq"][1:-1],'.-')
        if fit:
            p = data['fit_avgq']
            plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
            else: pi_gain= (3/2 - p[2]/180)/2/p[1]
            pi_gain += (n_pulses-1)*1/2/p[1]
            pi2_gain = pi_gain/2
            print(f'Pi gain from avgq data [dac units]: {int(pi_gain)}')
            print(f'\tPi/2 gain from avgq data [dac units]: {int(pi2_gain)}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi2_gain, color='0.2', linestyle='--')

        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname

# ====================================================== #
                      
class AmplitudeRabiOptimalCtrlChevronExperiment(Experiment):
    """
    Amplitude Rabi Experiment
    Experimental Config:
    expt = dict(
        starts: qubit gain [dac level] for each qubit in IQ_qubits (array of length IQ_qubits)
        steps: gain step [dac level] for each qubit
        expts: number steps for each qubit
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        Icontrols: 2D array with array of Icontrols for each of IQ_qubits
        Qcontrols: 2D array with array of Qcontrols for each of IQ_qubits
        times_us: same length as each array in Icontrols, Qcontrols
        IQ_qubits: qubits to play Icontrols, Qcontrols on
    )
    """

    def __init__(self, soccfg=None, path='', prefix='AmplitudeRabiOptCtrlChevron', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=True):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.readout.frequency)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update({key3: [value3]*num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})

        assert len(self.cfg.expt.IQ_qubits) == 2, 'not implemented loops for more than 2 qubits'
        data={"avgi":[[], []], "avgq":[[], []], "amps":[[], []], "phases":[[], []], 'counts_calib':[], 'counts_raw':[]}

        # ================= #
        # Get single shot calibration for 2 qubits
        # ================= #
        thresholds_q = ge_avgs_q = angles_q = fids_q = None
        if 'post_process' not in self.cfg.expt.keys(): # threshold or scale
            self.cfg.expt.post_process = None

        if self.cfg.expt.post_process is not None:
            if 'angles' in self.cfg.expt and 'thresholds' in self.cfg.expt and 'ge_avgs' in self.cfg.expt and 'counts_calib' in self.cfg.expt and self.cfg.expt.angles is not None and self.cfg.expt.thresholds is not None and self.cfg.expt.ge_avgs is not None and self.cfg.expt.counts_calib is not None:
                angles_q = self.cfg.expt.angles
                thresholds_q = self.cfg.expt.thresholds
                ge_avgs_q = np.asarray(self.cfg.expt.ge_avgs)
                data['counts_calib'] = self.cfg.expt.counts_calib
                if debug: print('Re-using provided angles, thresholds, ge_avgs')
            else:
                thresholds_q = [0]*4
                ge_avgs_q = [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]
                angles_q = [0]*4
                fids_q = [0]*4

                # We really just need the single shot plots here, but convenient to use the ErrorMitigation tomo to do it
                sscfg = AttrDict(deepcopy(self.cfg))
                sscfg.expt.reps = sscfg.expt.singleshot_reps
                sscfg.expt.rounds = 1
                sscfg.expt.tomo_qubits = self.cfg.expt.IQ_qubits

                calib_prog_dict = dict()
                calib_order = ['gg', 'ge', 'eg', 'ee']
                for prep_state in tqdm(calib_order):
                    # print(prep_state)
                    sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state, apply_q1_pi2=False)
                    err_tomo = ErrorMitigationStateTomo2QProgram(soccfg=self.soccfg, cfg=sscfg)
                    err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=False)
                    calib_prog_dict.update({prep_state:err_tomo})

                g_prog = calib_prog_dict['gg']
                Ig, Qg = g_prog.get_shots(verbose=False)

                # Get readout angle + threshold for qubits
                for qi, q in enumerate(sscfg.expt.tomo_qubits):
                    calib_e_state = 'gg'
                    calib_e_state = calib_e_state[:qi] + 'e' + calib_e_state[qi+1:]
                    e_prog = calib_prog_dict[calib_e_state]
                    Ie, Qe = e_prog.get_shots(verbose=False)
                    shot_data = dict(Ig=Ig[q], Qg=Qg[q], Ie=Ie[q], Qe=Qe[q])
                    print(f'Qubit  ({q})')
                    fid, threshold, angle = hist(data=shot_data, plot=debug, verbose=False)
                    thresholds_q[q] = threshold[0]
                    ge_avgs_q[q] = [np.average(Ig[q]), np.average(Qg[q]), np.average(Ie[q]), np.average(Qe[q])]
                    angles_q[q] = angle
                    fids_q[q] = fid[0]
                    print(f'ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[q]} \t threshold ge: {thresholds_q[q]}')

                # Process the shots taken for the confusion matrix with the calibration angles
                for prep_state in calib_order:
                    counts = calib_prog_dict[prep_state].collect_counts(angle=angles_q, threshold=thresholds_q)
                    data['counts_calib'].append(counts)
                # print(data['counts_calib'])

                if debug:
                    print(f'thresholds={thresholds_q},')
                    print(f'angles={angles_q},')
                    print(f'ge_avgs={ge_avgs_q},')
                    print(f"counts_calib={np.array(data['counts_calib']).tolist()}")

            data['thresholds'] = thresholds_q
            data['angles'] = angles_q
            data['ge_avgs'] = ge_avgs_q
            data['counts_calib'] = np.array(data['counts_calib'])

        # ================= #
        # Begin actual experiment
        # ================= #

        gainpts = []
        for q in self.cfg.expt.IQ_qubits:
            gainpts.append(self.cfg.expt.starts[q] + self.cfg.expt.steps[q]*np.arange(self.cfg.expt.expts[q]))
            data[f'gainpts{q}'] = gainpts[q]

        self.cfg.expt.start = self.cfg.expt.starts[0]
        self.cfg.expt.step = self.cfg.expt.steps[0]
        self.cfg.expt.expts = self.cfg.expt.expts[0]
        self.cfg.expt.gains = [0]*len(self.cfg.expt.IQ_qubits)
        for igain, gain1 in enumerate(tqdm(gainpts[1])): # 0th qubit gain sweep is covered inside the program
            self.cfg.expt.gains[1] = gain1
            if igain == 0: self.cfg.expt.plot_IQ = True
            else: self.cfg.expt.plot_IQ = False
            amprabi = AmplitudeRabiOptimalCtrlProgram(soccfg=self.soccfg, cfg=self.cfg)
        
            # xpts, avgi, avgq = amprabi.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False)
            avgi, avgq = amprabi.acquire_rotated(self.im[self.cfg.aliases.soc], angle=angles_q, threshold=thresholds_q, ge_avgs=ge_avgs_q, post_process=self.cfg.expt.post_process, progress=False, verbose=False)        
        
            for q in self.cfg.expt.IQ_qubits:
                data['avgi'][q].append(avgi[q])
                data['avgq'][q].append(avgq[q])
                data['amps'][q].append(np.abs(avgi[q] + 1j*avgq[q]))
                data['phases'][q].append(np.angle(avgi[q] + 1j*avgq[q])) # Calculating the phase        
        
        for k, a in data.items():
            data[k] = np.array(a)
        self.data=data
        return data

    def analyze(self, data=None, fit=True, gain0_range=None, gain1_range=None, opt_q0=0.5, opt_q1=0.5):
        if data is None:
            data=self.data
        if not fit: return

        if gain0_range is None: gain0_range = [np.min(data['gainpts0']), np.max(data['gainpts0'])]
        if gain1_range is None: gain1_range = [np.min(data['gainpts1']), np.max(data['gainpts1'])]

        gain0s = data['gainpts0']
        gain1s = data['gainpts1']
        gain0_range_indices = [np.argmin(np.abs(gain0s-gain0_range[0])), np.argmin(np.abs(gain0s-gain0_range[1]))]
        gain1_range_indices = [np.argmin(np.abs(gain1s-gain1_range[0])), np.argmin(np.abs(gain1s-gain1_range[1]))]
        # print(gain0_range_indices, gain1_range_indices)

        avgi_q0_search = data['avgi'][0][gain1_range_indices[0]:gain1_range_indices[1], gain0_range_indices[0]:gain0_range_indices[1]]
        avgi_q1_search = data['avgi'][1][gain1_range_indices[0]:gain1_range_indices[1], gain0_range_indices[0]:gain0_range_indices[1]]

        best_gain1, best_gain0 = np.unravel_index(np.argmin(np.abs(avgi_q0_search-opt_q0) + np.abs(avgi_q1_search-opt_q1)), avgi_q0_search.shape)

        data['best_gain1'] = best_gain1 + gain1_range_indices[0]
        data['best_gain0'] = best_gain0 + gain0_range_indices[0]
        pass

    def display(self, data=None, fit=True, plot_gain0=None, plot_gain1=None):
        if data is None:
            data=self.data 
        
        x_sweep = data['gainpts0']
        y_sweep = data['gainpts1']
        
        for q in self.cfg.expt.IQ_qubits:
            avgi = data['avgi'][q]
            # avgq = data['avgq'][q]

            plt.figure(figsize=(7,5))
            plt.subplot(111, title=f"Amplitude Rabi Qubit {q}", ylabel=f"Gain Q{self.cfg.expt.IQ_qubits[1]}")
            plt.imshow(
                np.flip(avgi, 0),
                cmap='viridis',
                extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
                aspect='auto')
            plt.colorbar(label='Population')
            plt.clim(vmin=None, vmax=None)
            plt.xlabel(f'Gain Q{self.cfg.expt.IQ_qubits[0]}')
            # plt.axvline(1684.92, color='k')
            # plt.axvline(1684.85, color='r')
            if plot_gain0 is not None and plot_gain1 is not None:
                plt.axvline(plot_gain0, color='r')
                plt.axhline(plot_gain1, color='r')
                popln = avgi[np.argmin(np.abs(y_sweep-plot_gain1)), np.argmin(np.abs(x_sweep-plot_gain0))]
                print(f'Q{q} popln at gain0 {plot_gain0}, gain1 {plot_gain1}:', popln)
            else:
                if fit:
                    plot_gain0 = data['gainpts0'][data['best_gain0']]
                    plot_gain1 = data['gainpts1'][data['best_gain1']]
                    plt.axvline(plot_gain0, color='r')
                    plt.axhline(plot_gain1, color='r')
                    popln = avgi[data['best_gain1'], data['best_gain0']]
                    print(f'Q{q} popln at gain0 {plot_gain0}, gain1 {plot_gain1}:', popln)
                    
            plt.tight_layout()
            plt.show()
        
        if fit: pass


        # plt.plot(y_sweep, data['amps'][:,-1])
        # plt.title(f'Gain {x_sweep[-1]}')
        # plt.show()

        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname



# ====================================================== #

class OptimalCtrlTomo2QProgram(AbstractStateTomo2QProgram):
   
    def state_prep_pulse(self, qubits, **kwargs):
        for q in self.cfg.expt.IQ_qubits:
            # play the I + Q component for each qubit in the IQ pulse
            self.handle_IQ_pulse(name=f'pulse_Q{q}', ch=self.qubit_chs[q], sync_after=False, play=True)
        self.sync_all()


    def initialize(self):
        super().initialize()

        # IQ pulse
        if 'plot_IQ' not in self.cfg.expt or self.cfg.expt.plot_IQ == None: self.cfg.expt.plot_IQ = False
        if 'ILC' not in self.cfg.expt: self.cfg.expt.ILC = False
        for iq, q in enumerate(self.cfg.expt.IQ_qubits):
            self.handle_IQ_pulse(name=f'pulse_Q{q}', ch=self.qubit_chs[q], I_mhz_vs_us=self.cfg.expt.Icontrols[iq], Q_mhz_vs_us=self.cfg.expt.Qcontrols[iq], times_us=self.cfg.expt.times_us, freq_MHz=self.f_ges[q, q], phase_deg=0, gain=self.cfg.expt.IQ_gain[iq], reload=True, play=False, plot_IQ=self.cfg.expt.plot_IQ, ILC=self.cfg.expt.ILC)
        
        self.sync_all(200)


class OptimalCtrlTomo2QExperiment(Experiment):
# outer outer loop over gain parameters
# outer loop over measurement bases
# set the state prep pulse to be preparing the gg, ge, eg, ee states for confusion matrix
    """
    Perform state tomography on the optimal ctrl prepared state with error mitigation.
    Experimental Config:
    expt = dict(
        starts: qubit gain [dac level] for each qubit in IQ_qubits (array of length IQ_qubits)
        steps: gain step [dac level] for each qubit
        expts: number steps for each qubit
        Icontrols: 2D array with array of Icontrols for each of IQ_qubits
        Qcontrols: 2D array with array of Qcontrols for each of IQ_qubits
        times_us: same length as each array in Icontrols, Qcontrols
        IQ_qubits: qubits to play Icontrols, Qcontrols on

        reps: number averages per measurement basis iteration
        singleshot_reps: number averages in single shot calibration
        tomo_qubits: the qubits to perform the two qubit tomography on (drive applied to the second qubit)
    )
    """

    def __init__(self, soccfg=None, path='', prefix='OptimalControl2QTomo', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=True):
        # expand entries in config that are length 1 to fill all qubits
        self.num_qubits_sample = len(self.cfg.device.readout.frequency)
        qA, qB = self.cfg.expt.tomo_qubits

        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update({key3: [value3]*self.num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*self.num_qubits_sample})

        assert len(self.cfg.expt.IQ_qubits) == 2, 'only implemented loops for IQ on 2 qubits'
        
        self.meas_order = ['ZZ', 'ZX', 'ZY', 'XZ', 'XX', 'XY', 'YZ', 'YX', 'YY']
        self.calib_order = ['gg', 'ge', 'eg', 'ee'] # should match with order of counts for each tomography measurement 
        data={'counts_tomo_gains':np.zeros(shape=(self.cfg.expt.expts[0], self.cfg.expt.expts[1], len(self.meas_order), len(self.calib_order))), 'counts_calib':[]}
        gainpts = []
        for q in self.cfg.expt.IQ_qubits:
            gainpts_q = self.cfg.expt.starts[q] + self.cfg.expt.steps[q]*np.arange(self.cfg.expt.expts[q])
            gainpts.append(gainpts_q)
            data[f'gainpts{q}'] = gainpts_q
            print(f'gainpts Q{q}', gainpts_q)
        
        self.readout_cool = False
        if 'readout_cool' in self.cfg.expt and self.cfg.expt.readout_cool:
            self.readout_cool = self.cfg.expt.readout_cool
        if not self.readout_cool: self.cfg.expt.n_init_readout = 0

        if 'n_init_readout' not in self.cfg.expt: self.cfg.expt.n_init_readout = 0
        data[f'ishots_raw'] = np.zeros(shape=(len(gainpts[0]), len(gainpts[1]), len(self.meas_order), self.num_qubits_sample, self.cfg.expt.n_init_readout+1, self.cfg.expt.reps)) # raw data tomo experiments
        data[f'qshots_raw'] = np.zeros(shape=(len(gainpts[0]), len(gainpts[1]), len(self.meas_order), self.num_qubits_sample, self.cfg.expt.n_init_readout+1, self.cfg.expt.reps)) # raw data tomo experiments

        self.ncalib = len(self.calib_order) + (2*(self.num_qubits_sample - len(self.cfg.expt.tomo_qubits)) if self.readout_cool else 0)
        data['calib_ishots_raw'] = np.zeros(shape=(self.ncalib, self.num_qubits_sample, self.cfg.expt.n_init_readout+1, self.cfg.expt.singleshot_reps)) # raw rotated g data for the calibration histograms for each of the measure experiments
        data['calib_qshots_raw'] = np.zeros(shape=(self.ncalib, self.num_qubits_sample, self.cfg.expt.n_init_readout+1, self.cfg.expt.singleshot_reps)) # raw rotated g data for the calibration histograms for each of the measure experiments
        
        # ================= #
        # Get single shot calibration for qubits
        # ================= #

        thresholds_q = ge_avgs_q = angles_q = fids_q = None

        if 'angles' in self.cfg.expt and 'thresholds' in self.cfg.expt and 'ge_avgs' in self.cfg.expt and 'counts_calib' in self.cfg.expt and None not in (self.cfg.expt.angles, self.cfg.expt.thresholds, self.cfg.expt.ge_avgs, self.cfg.expt.counts_calib):
            angles_q = self.cfg.expt.angles
            thresholds_q = self.cfg.expt.thresholds
            ge_avgs_q = self.cfg.expt.ge_avgs
            for q in range(self.num_qubits_sample):
                if ge_avgs_q[q] is None:
                    ge_avgs_q[q] = np.zeros_like(ge_avgs_q[self.cfg.expt.tomo_qubits[0]]) # just get the shape of the arrays correct by picking the old ge_avgs_q of a q that was definitely measured
            ge_avgs_q = np.array(ge_avgs_q)
            data['counts_calib'] = self.cfg.expt.counts_calib
            print('Re-using provided angles, thresholds, ge_avgs, counts_calib')

        else:
            thresholds_q = [0]*4
            ge_avgs_q = [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]
            angles_q = [0]*4
            fids_q = [0]*4

            sscfg = AttrDict(deepcopy(self.cfg))
            sscfg.expt.reps = sscfg.expt.singleshot_reps

            # Error mitigation measurements: prep in gg, ge, eg, ee to recalibrate measurement angle and measure confusion matrix
            calib_prog_dict = dict()
            for prep_state in tqdm(self.calib_order):
                # print(prep_state)
                sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state)
                err_tomo = ErrorMitigationStateTomo2QProgram(soccfg=self.soccfg, cfg=sscfg)
                err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=False)
                calib_prog_dict.update({prep_state:err_tomo})

            g_prog = calib_prog_dict['gg']
            Ig, Qg = g_prog.get_shots(verbose=False)
            threshold = [0]*self.num_qubits_sample
            angle = [0]*self.num_qubits_sample

            # Get readout angle + threshold for qubits
            for qi, q in enumerate(sscfg.expt.tomo_qubits):
                calib_e_state = 'gg'
                calib_e_state = calib_e_state[:qi] + 'e' + calib_e_state[qi+1:]
                e_prog = calib_prog_dict[calib_e_state]
                Ie, Qe = e_prog.get_shots(verbose=False)
                shot_data = dict(Ig=Ig[q], Qg=Qg[q], Ie=Ie[q], Qe=Qe[q])
                print(f'Qubit ({q})')
                fid, threshold, angle = hist(data=shot_data, plot=debug, verbose=False)
                thresholds_q[q] = threshold[0]
                ge_avgs_q[q] = [np.average(Ig[q]), np.average(Qg[q]), np.average(Ie[q]), np.average(Qe[q])]
                angles_q[q] = angle
                fids_q[q] = fid[0]
                print(f'ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[q]} \t threshold ge: {thresholds_q[q]}')
            
            # Process the shots taken for the confusion matrix with the calibration angles (for tomography)
            for iprep, prep_state in enumerate(self.calib_order):
                counts = calib_prog_dict[prep_state].collect_counts(angle=angles_q, threshold=thresholds_q)
                data['counts_calib'].append(counts)
                data[f'calib_ishots_raw'][iprep, :, :, :], data[f'calib_qshots_raw'][iprep, :, :, :] = calib_prog_dict[prep_state].get_multireadout_shots()

            # Do the calibration for the remaining qubits in case you want to do post selection
            if self.readout_cool or self.cfg.expt.expts > 1 or (self.cfg.expt.post_select and 1 not in self.cfg.expt.tomo_qubits):
                sscfg = AttrDict(deepcopy(self.cfg))
                sscfg.expt.reps = self.cfg.expt.singleshot_reps
                ps_calib_prog_dict = dict()
                iprep_temp = len(self.calib_order)
                for q in range(self.num_qubits_sample):
                    if q in self.cfg.expt.tomo_qubits: continue # already did these
                    sscfg.expt.qubit = q
                    for prep_state in tqdm(['g', 'e'], disable=not progress):
                        # print(prep_state)
                        sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state)
                        err_tomo = ErrorMitigationStateTomo1QProgram(soccfg=self.soccfg, cfg=sscfg)
                        err_tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                        ps_calib_prog_dict.update({prep_state:err_tomo})

                        # Save the full counts from the calibration experiments using the measured angles for all qubits in case you want to do post selection on the calibration
                        data[f'calib_ishots_raw'][iprep_temp, :, :, :], data[f'calib_qshots_raw'][iprep_temp, :, :, :] = err_tomo.get_multireadout_shots()
                        iprep_temp += 1

                    g_prog = ps_calib_prog_dict['g']
                    Ig, Qg = g_prog.get_shots(verbose=False)

                    # Get readout angle + threshold for qubit
                    e_prog = ps_calib_prog_dict['e']
                    Ie, Qe = e_prog.get_shots(verbose=False)
                    shot_data = dict(Ig=Ig[q], Qg=Qg[q], Ie=Ie[q], Qe=Qe[q])
                    print(f'Qubit ({q}) ge')
                    fid, threshold, angle = hist(data=shot_data, plot=debug, verbose=False)
                    thresholds_q[q] = threshold[0]
                    angles_q[q] = angle
                    ge_avgs_q[q] = [np.average(Ig[q]), np.average(Qg[q]), np.average(Ie[q]), np.average(Qe[q])]
                    print(f'ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[q]} \t threshold ge: {thresholds_q[q]}')

            data.update(dict(thresholds=thresholds_q, angles=angles_q, ge_avgs=ge_avgs_q)) 
        print(f'thresholds={thresholds_q},')
        print(f'angles={angles_q},')
        print(f'ge_avgs={ge_avgs_q},')
        print(f"counts_calib={np.array(data['counts_calib']).tolist()}")

        # ================= #
        # Begin actual experiment
        # ================= #
        self.pulse_dict = dict()

        tomo_cfg = AttrDict(deepcopy(self.cfg))
        tomo_cfg.expt.IQ_gain = [0]*len(self.cfg.expt.IQ_qubits)

        for igain1, gain1 in enumerate(tqdm(gainpts[1])):
            for igain0, gain0 in enumerate(tqdm(gainpts[0])):
                tomo_cfg.expt.IQ_gain = [gain0, gain1]

                # Tomography measurements
                for ibasis, basis in enumerate(self.meas_order):
                    # print(basis)
                    cfg = AttrDict(deepcopy(tomo_cfg))
                    cfg.expt.basis = basis
                    tomo = OptimalCtrlTomo2QProgram(soccfg=self.soccfg, cfg=cfg)
                    # print(tomo)
                    # from qick.helpers import progs2json
                    # print(progs2json([tomo.dump_prog()]))
                    # xpts, avgi, avgq = tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                    avgi, avgq = tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)

                    # print(basis)
                    adc_chs = self.cfg.hw.soc.adcs.readout.ch
                    # avgi, avgq = tomo.get_shots(angle=None, avg_shots=True)
                    # for q in self.cfg.expt.tomo_qubits:
                    #     print('q', q, 'avgi', avgi[adc_chs[q]])
                    #     print('q', q, 'avgq', avgq[adc_chs[q]])
                    #     print('q', q, 'amps', np.abs(avgi[adc_chs[q]]+1j*avgi[adc_chs[q]]))

                    counts = tomo.collect_counts(angle=angles_q, threshold=thresholds_q)
                    data['counts_tomo_gains'][igain0, igain1][ibasis] = counts
                    data[f'ishots_raw'][igain0, igain1, ibasis, :, :, :], data[f'qshots_raw'][igain0, igain1, ibasis, :, :, :] = tomo.get_multireadout_shots()
                    if igain0 == 0 and igain1 == 0: self.pulse_dict.update({basis:tomo.pulse_dict})

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


