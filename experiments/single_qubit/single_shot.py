import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from copy import deepcopy

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm

def hist(data, plot=True, span=None, verbose=True, title=None, fid_avg=False):
    """
    span: histogram limit is the mean +/- span
    fid_avg: if True, calculate fidelity F by the average mis-categorized e/g; otherwise count
        total number of miscategorized over total counts (gives F^2)
    """
    Ig = data['Ig']
    Qg = data['Qg']
    Ie = data['Ie']
    Qe = data['Qe']
    plot_f = False 
    if 'If' in data.keys():
        plot_f = True
        If = data['If']
        Qf = data['Qf']

    numbins = 200

    xg, yg = np.median(Ig), np.median(Qg)
    xe, ye = np.median(Ie), np.median(Qe)
    if plot_f: xf, yf = np.median(If), np.median(Qf)

    if verbose:
        print('Unrotated:')
        print(f'Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg+1j*yg)} +/- {np.std(np.abs(Ig + 1j*Qg))}')
        print(f'Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)} \t Amp e {np.abs(xe+1j*ye)} +/- {np.std(np.abs(Ig + 1j*Qe))}')
        if plot_f: print(f'If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)} \t Amp f {np.abs(xf+1j*yf)}')

    if plot:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 6))
        if title is not None: plt.suptitle(title)
        fig.tight_layout()

        axs[0,0].scatter(Ig, Qg, label='g', color='b', marker='.', edgecolor='None', alpha=0.2)
        axs[0,0].scatter(Ie, Qe, label='e', color='r', marker='.', edgecolor='None', alpha=0.2)
        if plot_f: axs[0,0].scatter(If, Qf, label='f', color='g', marker='.', edgecolor='None', alpha=0.2)
        axs[0,0].plot([xg], [yg], color='k', linestyle=':', marker='o', markerfacecolor='b', markersize=5)
        axs[0,0].plot([xe], [ye], color='k', linestyle=':', marker='o', markerfacecolor='r', markersize=5)    
        if plot_f:
            axs[0,0].plot([xf], [yf], color='k', linestyle=':', marker='o', markerfacecolor='g', markersize=5)    

        # axs[0,0].set_xlabel('I [ADC levels]')
        axs[0,0].set_ylabel('Q [ADC levels]')
        axs[0,0].tick_params(axis='both', which='major', labelsize=10)
        axs[0,0].legend(loc='upper right')
        axs[0,0].set_title('Unrotated', fontsize=14)
        axs[0,0].axis('equal')

    """Compute the rotation angle"""
    theta = -np.arctan2((ye-yg),(xe-xg))
    if plot_f: theta = -np.arctan2((yf-yg),(xf-xg))

    """
    Adjust rotation angle from g, e only
    """
    best_theta = theta
    I_tot = np.concatenate((Ie, Ig))
    span = (np.max(I_tot) - np.min(I_tot))/2
    midpoint = (np.max(I_tot) + np.min(I_tot))/2
    xlims = [midpoint-span, midpoint+span]
    ng, binsg = np.histogram(Ig, bins=numbins, range=xlims)
    ne, binse = np.histogram(Ie, bins=numbins, range=xlims)
    contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5*ng.sum() + 0.5*ne.sum())))
    best_fid = np.max(contrast)
    for theta_i in np.linspace(theta-np.pi/12, theta + np.pi/12, 10):
        Ig_new = Ig*np.cos(theta_i) - Qg*np.sin(theta_i)
        Qg_new = Ig*np.sin(theta) + Qg*np.cos(theta) 
        Ie_new = Ie*np.cos(theta_i) - Qe*np.sin(theta_i)
        Qe_new = Ie*np.sin(theta) + Qe*np.cos(theta)
        xg, yg = np.median(Ig_new), np.median(Qg_new)
        xe, ye = np.median(Ie_new), np.median(Qe_new)
        I_tot_new = np.concatenate((Ie_new, Ig_new))
        span = (np.max(I_tot_new) - np.min(I_tot_new))/2
        midpoint = (np.max(I_tot_new) + np.min(I_tot_new))/2
        xlims = [midpoint-span, midpoint+span]
        ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims)
        ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims)
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5*ng.sum() + 0.5*ne.sum())))
        fid = np.max(contrast)
        if fid > best_fid:
            best_theta = theta_i
            best_fid = fid
    theta = best_theta

    """Rotate the IQ data"""
    Ig_new = Ig*np.cos(theta) - Qg*np.sin(theta)
    Qg_new = Ig*np.sin(theta) + Qg*np.cos(theta) 

    Ie_new = Ie*np.cos(theta) - Qe*np.sin(theta)
    Qe_new = Ie*np.sin(theta) + Qe*np.cos(theta)

    if plot_f:
        If_new = If*np.cos(theta) - Qf*np.sin(theta)
        Qf_new = If*np.sin(theta) + Qf*np.cos(theta)

    """New means of each blob"""
    xg, yg = np.median(Ig_new), np.median(Qg_new)
    xe, ye = np.median(Ie_new), np.median(Qe_new)
    if plot_f: xf, yf = np.median(If_new), np.median(Qf_new)
    if verbose:
        print('Rotated:')
        print(f'Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg+1j*yg)} +/- {np.std(np.abs(Ig + 1j*Qg))}')
        print(f'Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)} \t Amp e {np.abs(xe+1j*ye)} +/- {np.std(np.abs(Ig + 1j*Qe))}')
        if plot_f: print(f'If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)} \t Amp f {np.abs(xf+1j*yf)}')


    span = (np.max(np.concatenate((Ie_new, Ig_new))) - np.min(np.concatenate((Ie_new, Ig_new))))/2
    lim_midpoint = (np.max(np.concatenate((Ie_new, Ig_new))) + np.min(np.concatenate((Ie_new, Ig_new))))/2
    if plot_f: 
        span = (np.max(np.concatenate((If_new, Ie_new, Ig_new))) - np.min(np.concatenate((If_new, Ie_new, Ig_new))))/2
        lim_midpoint = (np.max(np.concatenate((If_new, Ie_new, Ig_new))) + np.min(np.concatenate((If_new, Ie_new, Ig_new))))/2
    xlims = [lim_midpoint-span, lim_midpoint+span]
        

    if plot:
        axs[0,1].scatter(Ig_new, Qg_new, label='g', color='b', marker='.', edgecolor='None', alpha=0.3)
        axs[0,1].scatter(Ie_new, Qe_new, label='e', color='r', marker='.', edgecolor='None', alpha=0.3)
        if plot_f: axs[0, 1].scatter(If_new, Qf_new, label='f', color='g', marker='.', edgecolor='None', alpha=0.3)
        axs[0,1].plot([xg], [yg], color='k', linestyle=':', marker='o', markerfacecolor='b', markersize=5)
        axs[0,1].plot([xe], [ye], color='k', linestyle=':', marker='o', markerfacecolor='r', markersize=5)    
        if plot_f:
            axs[0,1].plot([xf], [yf], color='k', linestyle=':', marker='o', markerfacecolor='g', markersize=5)    

        # axs[0,1].set_xlabel('I [ADC levels]')
        axs[0,1].legend(loc='upper right')
        axs[0,1].set_title('Rotated', fontsize=14)
        axs[0,1].axis('equal')
        axs[0,1].tick_params(axis='both', which='major', labelsize=10)

        """X and Y ranges for histogram"""

        ng, binsg, pg = axs[1,0].hist(Ig_new, bins=numbins, range = xlims, color='b', label='g', alpha=0.5)
        ne, binse, pe = axs[1,0].hist(Ie_new, bins=numbins, range = xlims, color='r', label='e', alpha=0.5)
        if plot_f:
            nf, binsf, pf = axs[1,0].hist(If_new, bins=numbins, range = xlims, color='g', label='f', alpha=0.5)
        axs[1,0].set_ylabel('Counts', fontsize=14)
        axs[1,0].set_xlabel('I [ADC levels]', fontsize=14)
        axs[1,0].legend(loc='upper right')
        axs[1,0].tick_params(axis='both', which='major', labelsize=10)

    else:        
        ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims)
        ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims)
        if plot_f:
            nf, binsf = np.histogram(If_new, bins=numbins, range=xlims)

    """Compute the fidelity using overlap of the histograms"""
    fids = []
    thresholds = []
    contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5*ng.sum() + 0.5*ne.sum()))) # this method calculates fidelity as 1-2(Neg + Nge)/N
    tind=contrast.argmax()
    thresholds.append(binsg[tind])
    if not fid_avg: fids.append(contrast[tind])
    else: fids.append(0.5*(1-ng[tind:].sum()/ng.sum() + 1-ne[:tind].sum()/ne.sum())) # this method calculates fidelity as (Ngg+Nee)/N = Ngg/N + Nee/N=(0.5N-Nge)/N + (0.5N-Neg)/N = 1-(Nge+Neg)/N 
    if verbose:
        print(f'g correctly categorized: {100*(1-ng[tind:].sum()/ng.sum())}%')
        print(f'e correctly categorized: {100*(1-ne[:tind].sum()/ne.sum())}%')
    if plot_f:
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(nf)) / (0.5*ng.sum() + 0.5*nf.sum())))
        tind=contrast.argmax()
        thresholds.append(binsg[tind])
        if not fid_avg: fids.append(contrast[tind])
        else: fids.append(0.5*(1-ng[tind:].sum()/ng.sum() + 1-nf[:tind].sum()/nf.sum()))

        contrast = np.abs(((np.cumsum(ne) - np.cumsum(nf)) / (0.5*ne.sum() + 0.5*nf.sum())))
        tind=contrast.argmax()
        thresholds.append(binsg[tind])
        if not fid_avg: fids.append(contrast[tind])
        else: fids.append(0.5*(1-ne[tind:].sum()/ne.sum() + 1-nf[:tind].sum()/nf.sum()))
        
    if plot: 
        title = '$\overline{F}_{ge}$' if fid_avg else '$F_{ge}$'
        axs[1,0].set_title(f'Histogram ({title}: {100*fids[0]:.3}%)', fontsize=14)
        axs[1,0].axvline(thresholds[0], color='0.2', linestyle='--')
        if plot_f:
            axs[1,0].axvline(thresholds[1], color='0.2', linestyle='--')
            axs[1,0].axvline(thresholds[2], color='0.2', linestyle='--')

        axs[1,1].set_title('Cumulative Counts', fontsize=14)
        axs[1,1].plot(binsg[:-1], np.cumsum(ng), 'b', label='g')
        axs[1,1].plot(binse[:-1], np.cumsum(ne), 'r', label='e')
        axs[1,1].axvline(thresholds[0], color='0.2', linestyle='--')
        if plot_f:
            axs[1,1].plot(binsf[:-1], np.cumsum(nf), 'g', label='f')
            axs[1,1].axvline(thresholds[1], color='0.2', linestyle='--')
            axs[1,1].axvline(thresholds[2], color='0.2', linestyle='--')
        axs[1,1].legend()
        axs[1,1].set_xlabel('I [ADC levels]', fontsize=14)
        axs[1,1].tick_params(axis='both', which='major', labelsize=10)
        
        plt.subplots_adjust(hspace=0.25, wspace=0.15)        
        plt.show()

    return fids, thresholds, theta*180/np.pi # fids: ge, gf, ef

# ====================================================== #

class HistogramProgram(AveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        self.gen_delays = [0]*len(soccfg['gens']) # need to calibrate via oscilloscope

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        
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

        qTest = self.cfg.expt.qTest
        if 'qZZ' not in self.cfg.expt: self.cfg.expt.qZZ = None
        qZZ = self.cfg.expt.qZZ
        self.checkZZ = False
        if qZZ is not None: self.checkZZ = True
        else: qZZ = qTest

        self.num_qubits_sample = len(self.cfg.device.readout.frequency)
        
        self.adc_chs = self.cfg.hw.soc.adcs.readout.ch
        self.res_chs = self.cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = self.cfg.hw.soc.dacs.readout.type
        self.qubit_chs = self.cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = self.cfg.hw.soc.dacs.qubit.type
        if 'cool_qubits' in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            self.swap_f0g1_chs = self.cfg.hw.soc.dacs.swap_f0g1.ch
            self.swap_f0g1_ch_types = self.cfg.hw.soc.dacs.swap_f0g1.type
            mixer_freqs = self.cfg.hw.soc.dacs.swap_f0g1.mixer_freq
        
        self.f_ges = np.reshape(self.cfg.device.qubit.f_ge, (4,4))
        self.f_efs = np.reshape(self.cfg.device.qubit.f_ef, (4,4))
        self.pi_ge_gains = np.reshape(self.cfg.device.qubit.pulses.pi_ge.gain, (4,4))
        self.pi_ge_sigmas = np.reshape(self.cfg.device.qubit.pulses.pi_ge.sigma, (4,4))
        self.pi_ef_gains = np.reshape(self.cfg.device.qubit.pulses.pi_ef.gain, (4,4))
        self.pi_ef_sigmas = np.reshape(self.cfg.device.qubit.pulses.pi_ef.sigma, (4,4))

        self.f_res_regs = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(self.cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)]
        if 'cool_qubits' in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            self.f_f0g1_regs = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_f0g1, self.qubit_chs)]

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
                    self.declare_gen(ch=self.res_chs[q], nqz=cfg.hw.soc.dacs.readout.nyquist[q])
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

        # add readout pulses to respective channels
        if 'mux4' in self.res_ch_types:
            self.set_pulse_registers(ch=6, style="const", length=max(self.readout_lengths_dac), mask=mask)
        for q in range(self.num_qubits_sample):
            if self.res_ch_types[q] != 'mux4':
                if cfg.device.readout.gain[q] < 1:
                    gain = int(cfg.device.readout.gain[q] * 2**15)
                self.set_pulse_registers(ch=self.res_chs[q], style="const", freq=self.f_res_regs[q], phase=0, gain=gain, length=max(self.readout_lengths_dac))

        # declare qubit dacs
        for q in range(self.num_qubits_sample):
            mixer_freq = None
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in self.gen_chs:
                self.declare_gen(ch=self.qubit_chs[q], nqz=cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)

        # add qubit pulses to respective channels
        if self.checkZZ:
            self.pisigma_ge_qZZ = self.us2cycles(self.pi_ge_sigmas[qZZ, qZZ], gen_ch=self.qubit_chs[qZZ])
            self.add_gauss(ch=self.qubit_chs[qZZ], name='pi_qZZ', sigma=self.pisigma_ge_qZZ, length=self.pisigma_ge_qZZ*4)

        if self.cfg.expt.pulse_e:
            self.pi_ge_sigma_cycles = self.us2cycles(self.pi_ge_sigmas[qTest, qZZ], gen_ch=self.qubit_chs[qTest])
            self.add_gauss(ch=self.qubit_chs[qTest], name="pi_ge", sigma=self.pi_ge_sigma_cycles, length=self.pi_ge_sigma_cycles*4)

        if self.cfg.expt.pulse_f:
            self.pisigma_ef = self.us2cycles(self.pi_ef_sigmas[qTest, qZZ], gen_ch=self.qubit_chs[qTest])
            self.gain_ef = self.pi_ef_gains[qTest, qZZ]
            self.f_ef_reg = self.freq2reg(self.f_efs[qTest, qZZ], gen_ch=self.qubit_chs[qTest])
            self.add_gauss(ch=self.qubit_chs[qTest], name='pi_ef', sigma=self.pisigma_ef, length=self.pisigma_ef*4)

        if 'cool_qubits' in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            mixer_freq = None
            for q in self.cfg.expt.cool_qubits:
                if self.swap_f0g1_ch_types[q] == 'int4':
                    mixer_freq = mixer_freqs[q]
                if self.swap_f0g1_chs[q] not in self.gen_chs: 
                    self.declare_gen(ch=self.swap_f0g1_chs[q], nqz=self.cfg.hw.soc.dacs.swap_f0g1.nyquist[q], mixer_freq=mixer_freq)

                pisigma_ef_q = self.us2cycles(self.pi_ef_sigmas[q, q], gen_ch=self.qubit_chs[q]) # default pi_ef value
                self.add_gauss(ch=self.qubit_chs[q], name=f"pi_ef_qubit{q}", sigma=pisigma_ef_q, length=pisigma_ef_q*4)
                if self.cfg.device.qubit.pulses.pi_f0g1.type[q] == 'flat_top':
                    self.add_gauss(ch=self.swap_f0g1_chs[q], name=f"pi_f0g1_{q}", sigma=3, length=3*4)
                else: assert False, 'not implemented'

        self.set_gen_delays()
        self.sync_all(200)
    
    def body(self):
        cfg=AttrDict(self.cfg)

        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        if qZZ is None: qZZ = qTest

        self.reset_and_sync()

        if 'cool_qubits' in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            cool_idle = [self.cfg.device.qubit.pulses.pi_f0g1.idle[q] for q in self.cfg.expt.cool_qubits]
            cool_qubits = self.cfg.expt.cool_qubits
            if 'cool_idle' in self.cfg.expt and self.cfg.expt.cool_idle is not None:
                cool_idle = self.cfg.expt.cool_idle
            sorted_indices = np.argsort(cool_idle)[::-1] # sort cooling times longest first
            cool_qubits = np.array(cool_qubits)
            cool_idle = np.array(cool_idle)
            sorted_cool_qubits = cool_qubits[sorted_indices]
            sorted_cool_idle = cool_idle[sorted_indices]
            max_idle = sorted_cool_idle[0]
        
            last_pulse_len = 0
            remaining_idle = max_idle
            for q, idle in zip(sorted_cool_qubits, sorted_cool_idle):
                remaining_idle -= last_pulse_len

                last_pulse_len = 0
                self.setup_and_pulse(ch=self.qubit_chs[q], style="arb", phase=0, freq=self.freq2reg(self.f_efs[q, q], gen_ch=self.qubit_chs[q]), gain=self.pi_ef_gains[q, q], waveform=f"pi_ef_qubit{q}")
                self.sync_all()
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

            remaining_idle -= last_pulse_len
            last_idle = max((remaining_idle, sorted_cool_idle[-1]))
            self.sync_all(self.us2cycles(last_idle))

        # initializations as necessary
        if self.checkZZ:
            self.setup_and_pulse(ch=self.qubit_chs[qZZ], style="arb", phase=0, freq=self.freq2reg(self.f_ges[qZZ, qZZ], gen_ch=self.qubit_chs[qZZ]), gain=self.pi_ge_gains[qZZ, qZZ], waveform="pi_qZZ")
            self.sync_all()
            # print('check zz qubit', qZZ)

        if self.cfg.expt.pulse_e:
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.freq2reg(self.f_ges[qTest, qZZ], gen_ch=self.qubit_chs[qTest]), phase=0, gain=self.pi_ge_gains[qTest, qZZ], waveform=f"pi_ge")
            self.sync_all()

        if self.cfg.expt.pulse_f:
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.freq2reg(self.f_efs[qTest, qZZ], gen_ch=self.qubit_chs[qTest]), phase=0, gain=self.pi_ef_gains[qTest, qZZ], waveform=f"pi_ef")
        self.sync_all()

        self.measure(
            pulse_ch=self.measure_chs, 
            adcs=self.adc_chs,
            adc_trig_offset=cfg.device.readout.trig_offset[0],
            wait=True,
            syncdelay=self.us2cycles(max([cfg.device.readout.relax_delay[q] for q in range(4)])))

    def collect_shots(self, qubit=None):
        # collect shots for the relevant adc and I and Q channels
        cfg=AttrDict(self.cfg)
        # print(np.average(self.di_buf[0]))
        if qubit is None: qubit = self.cfg.expt.qTest
        else: assert qubit in range(self.num_qubits_sample), 'qubit out of range'
        shots_i0 = self.di_buf[qubit] / self.readout_lengths_adc[qubit]
        shots_q0 = self.dq_buf[qubit] / self.readout_lengths_adc[qubit]
        return shots_i0, shots_q0
        # return shots_i0[:5000], shots_q0[:5000]


class HistogramExperiment(Experiment):
    """
    Histogram Experiment
    expt = dict(
        reps: number of shots per expt
        check_e: whether to test the e state blob (true if unspecified)
        check_f: whether to also test the f state blob
    )
    """

    def __init__(self, soccfg=None, path='', prefix='Histogram', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.readout.frequency)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        if isinstance(value2, dict):
                            for key3, value3 in value2.items():
                                if not(isinstance(value3, list)):
                                    value2.update({key3: [value3]*num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})


        # rounds = 100
        # Ig = np.zeros(self.cfg.expt.reps)
        # Qg = np.zeros(self.cfg.expt.reps)
        # Ie = np.zeros(self.cfg.expt.reps)
        # Qe = np.zeros(self.cfg.expt.reps)
        # for r in tqdm(range(rounds)):
        #     x_pts, avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=False)
        #     i0, q0, i1, q1 = histpro.collect_shots()
        #     iq = ([i0, q0], [i1, q1])
        #     i, q = iq[0] # i/q[0]: ground state i/q, i/q[1]: excited state i/q
        #     Ig += i[0]
        #     Qg += q[0]
        #     Ie += i[1]
        #     Qe += q[1]

        data=dict()

        # Ground state shots
        cfg = AttrDict(deepcopy(self.cfg))
        cfg.expt.pulse_e = False
        cfg.expt.pulse_f = False
        cfg.expt.pulse_test = False
        histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
        avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress)
        self.prog = histpro
        data['Ig'], data['Qg'] = histpro.collect_shots()

        # Excited state shots
        if 'check_e' not in self.cfg.expt:
            self.check_e = True
        else: self.check_e = self.cfg.expt.check_e
        if self.check_e:
            cfg = AttrDict(deepcopy(self.cfg))
            cfg.expt.pulse_e = True 
            cfg.expt.pulse_f = False
            cfg.expt.pulse_test = False
            histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
            avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress)
            data['Ie'], data['Qe'] = histpro.collect_shots()

        # Excited f state shots
        self.check_f = self.cfg.expt.check_f
        if self.check_f:
            cfg = AttrDict(deepcopy(self.cfg))
            cfg.expt.pulse_e = True 
            cfg.expt.pulse_e = False
            print('WARNING TURNED OFF PULSE E FOR CHECK F')
            cfg.expt.pulse_f = True
            cfg.expt.pulse_test = False
            histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
            avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress)
            data['If'], data['Qf'] = histpro.collect_shots()

        # Test state shots
        if 'pulse_test' not in self.cfg.expt: self.cfg.expt.pulse_test = False
        self.check_test = self.cfg.expt.pulse_test
        if self.check_test:
            cfg = AttrDict(deepcopy(self.cfg))
            histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
            avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress)
            data['Itest'], data['Qtest'] = histpro.collect_shots()

        self.data = data
        return data

    def analyze(self, data=None, span=None, verbose=True, **kwargs):
        if data is None:
            data=self.data
        
        fids, thresholds, angle = hist(data=data, plot=False, span=span, verbose=verbose)
        data['fids'] = fids
        data['angle'] = angle
        data['thresholds'] = thresholds
        
        return data

    def display(self, data=None, span=None, verbose=True, **kwargs):
        if data is None:
            data=self.data 
        
        qTest = self.cfg.expt.qTest
        fids, thresholds, angle = hist(data=data, plot=True, verbose=verbose, span=span, title=f'Qubit {qTest}')
            
        print(f'ge fidelity (%): {100*fids[0]}')
        if self.cfg.expt.check_f:
            print(f'gf fidelity (%): {100*fids[1]}')
            print(f'ef fidelity (%): {100*fids[2]}')
        print(f'rotation angle (deg): {angle}')
        # print(f'set angle to (deg): {-angle}')
        print(f'threshold ge: {thresholds[0]}')
        if self.cfg.expt.check_f:
            print(f'threshold gf: {thresholds[1]}')
            print(f'threshold ef: {thresholds[2]}')

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname

# ====================================================== #

class SingleShotOptExperiment(Experiment):
    """
    Single Shot optimization experiment over readout parameters
    expt = dict(
        reps: number of shots per expt
        start_f: start frequency (MHz)
        step_f: frequency step (MHz)
        expts_f: number of experiments in frequency

        start_gain: start gain (dac units)
        step_gain: gain step (dac units)
        expts_gain: number of experiments in gain sweep

        start_len: start readout len (dac units)
        step_len: length step (dac units)
        expts_len: number of experiments in length sweep

        check_f: optimize fidelity for g/f (as opposed to g/e)
    )
    """

    def __init__(self, soccfg=None, path='', prefix='Histogram', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        fpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"]*np.arange(self.cfg.expt["expts_f"])
        gainpts = self.cfg.expt["start_gain"] + self.cfg.expt["step_gain"]*np.arange(self.cfg.expt["expts_gain"])
        lenpts = self.cfg.expt["start_len"] + self.cfg.expt["step_len"]*np.arange(self.cfg.expt["expts_len"])
        print(fpts)
        print(gainpts)
        print(lenpts)
        
        fid = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        threshold = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        angle = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))

        qTest = self.cfg.expt.qTest

        for f_ind, f in enumerate(tqdm(fpts, disable=not progress)):
            for g_ind, gain in enumerate(gainpts):
                for l_ind, l in enumerate(lenpts):
                    shot = HistogramExperiment(soccfg=self.soccfg, config_file=self.config_file)
                    shot.cfg = deepcopy(self.cfg)
                    shot.cfg.device.readout.frequency[qTest] = f
                    shot.cfg.device.readout.gain[qTest] = gain
                    shot.cfg.device.readout.readout_length = l 
                    check_e = True
                    if 'check_f' not in self.cfg.expt: check_f = False
                    else:
                        check_f = self.cfg.expt.check_f
                        check_e = not check_f
                    shot.cfg.expt = dict(reps=self.cfg.expt.reps, check_e=check_e, check_f=check_f, qTest=self.cfg.expt.qTest)
                    # print(shot.cfg)
                    shot.go(analyze=False, display=False, progress=False, save=False)
                    results = shot.analyze(verbose=False)
                    fid[f_ind, g_ind, l_ind] = results['fids'][0] if not check_f else results['fids'][1]
                    threshold[f_ind, g_ind, l_ind] = results['thresholds'][0] if not check_f else results['thresholds'][1]
                    angle[f_ind, g_ind, l_ind] = results['angle']
                    print(f'freq: {f}, gain: {gain}, len: {l}')
                    print(f'\tfid ge [%]: {100*results["fids"][0]}')
                    if check_f: print(f'\tfid gf [%]: {100*results["fids"][1]}')

        self.data = dict(fpts=fpts, gainpts=gainpts, lenpts=lenpts, fid=fid, threshold=threshold, angle=angle)
        return self.data

    def analyze(self, data=None, **kwargs):
        if data == None: data = self.data
        fid = data['fid']
        threshold = data['threshold']
        angle = data['angle']
        fpts = data['fpts']
        gainpts = data['gainpts']
        lenpts = data['lenpts']

        imax = np.unravel_index(np.argmax(fid), shape=fid.shape)
        print(imax)
        print(fpts)
        print(gainpts)
        print(lenpts)
        print(f'Max fidelity {100*fid[imax]} %')
        print(f'Set params: \n angle (deg) {-angle[imax]} \n threshold {threshold[imax]} \n freq [Mhz] {fpts[imax[0]]} \n gain [dac units] {gainpts[imax[1]]} \n readout length [us] {lenpts[imax[2]]}')

        return imax

    def display(self, data=None, **kwargs):
        if data is None:
            data=self.data 
        
        fid = data['fid']
        fpts = data['fpts'] # outer sweep, index 0
        gainpts = data['gainpts'] # middle sweep, index 1
        lenpts = data['lenpts'] # inner sweep, index 2

        # lenpts = [data['lenpts'][0]]
        for g_ind, gain in enumerate(gainpts):
            for l_ind, l in enumerate(lenpts):
                plt.plot(fpts, 100*fid[:,g_ind, l_ind], 'o-', label=f'gain: {gain:.2}, len [us]: {l}')
        plt.xlabel('Frequency [MHz]')
        plt.ylabel(f'Fidelity [%]')
        plt.legend()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname