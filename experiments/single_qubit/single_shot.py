import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from copy import deepcopy

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm

def hist(data, plot=True, span=None, verbose=True):
    """
    span: histogram limit is the mean +/- span
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
        print(f'Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg+1j*yg)}')
        print(f'Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)} \t Amp e {np.abs(xe+1j*ye)}')
        if plot_f: print(f'If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)} \t Amp f {np.abs(xf+1j*yf)}')

    if plot:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
        fig.tight_layout()

        axs[0,0].scatter(Ig, Qg, label='g', color='b', marker='.')
        axs[0,0].scatter(Ie, Qe, label='e', color='r', marker='.')
        if plot_f: axs[0,0].scatter(If, Qf, label='f', color='g', marker='.')
        axs[0,0].scatter(xg, yg, color='k', marker='o')
        axs[0,0].scatter(xe, ye, color='k', marker='o')
        if plot_f: axs[0,0].scatter(xf, yf, color='k', marker='o')

        # axs[0,0].set_xlabel('I [ADC levels]')
        axs[0,0].set_ylabel('Q [ADC levels]')
        axs[0,0].legend(loc='upper right')
        axs[0,0].set_title('Unrotated')
        axs[0,0].axis('equal')

    """Compute the rotation angle"""
    theta = -np.arctan2((ye-yg),(xe-xg))
    if plot_f: theta = -np.arctan2((yf-yg),(xf-xg))

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
        print(f'Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg+1j*yg)}')
        print(f'Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)} \t Amp e {np.abs(xe+1j*ye)}')
        if plot_f: print(f'If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)} \t Amp f {np.abs(xf+1j*yf)}')


    if span is None:
        span = (np.max(np.concatenate((Ie_new, Ig_new))) - np.min(np.concatenate((Ie_new, Ig_new))))/2
    xlims = [(xg+xe)/2-span, (xg+xe)/2+span]
    ylims = [yg-span, yg+span]

    if plot:
        axs[0,1].scatter(Ig_new, Qg_new, label='g', color='b', marker='.')
        axs[0,1].scatter(Ie_new, Qe_new, label='e', color='r', marker='.')
        if plot_f: axs[0, 1].scatter(If_new, Qf_new, label='f', color='g', marker='.')
        axs[0,1].scatter(xg, yg, color='k', marker='o')
        axs[0,1].scatter(xe, ye, color='k', marker='o')    
        if plot_f: axs[0, 1].scatter(xf, yf, color='k', marker='o')    

        # axs[0,1].set_xlabel('I [ADC levels]')
        axs[0,1].legend(loc='upper right')
        axs[0,1].set_title('Rotated')
        axs[0,1].axis('equal')

        """X and Y ranges for histogram"""

        ng, binsg, pg = axs[1,0].hist(Ig_new, bins=numbins, range = xlims, color='b', label='g', alpha=0.5)
        ne, binse, pe = axs[1,0].hist(Ie_new, bins=numbins, range = xlims, color='r', label='e', alpha=0.5)
        if plot_f:
            nf, binsf, pf = axs[1,0].hist(If_new, bins=numbins, range = xlims, color='g', label='f', alpha=0.5)
        axs[1,0].set_ylabel('Counts')
        axs[1,0].set_xlabel('I [ADC levels]')       
        axs[1,0].legend(loc='upper right')

    else:        
        ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims)
        ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims)
        if plot_f:
            nf, binsf = np.histogram(If_new, bins=numbins, range=xlims)

    """Compute the fidelity using overlap of the histograms"""
    fids = []
    thresholds = []
    contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5*ng.sum() + 0.5*ne.sum())))
    tind=contrast.argmax()
    thresholds.append(binsg[tind])
    fids.append(contrast[tind])
    if plot_f:
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(nf)) / (0.5*ng.sum() + 0.5*nf.sum())))
        tind=contrast.argmax()
        thresholds.append(binsg[tind])
        fids.append(contrast[tind])

        contrast = np.abs(((np.cumsum(ne) - np.cumsum(nf)) / (0.5*ne.sum() + 0.5*nf.sum())))
        tind=contrast.argmax()
        thresholds.append(binsg[tind])
        fids.append(contrast[tind])
        
    if plot: 
        axs[1,0].set_title(f'Histogram (Fidelity g-e: {100*fids[0]:.3}%)')
        axs[1,0].axvline(thresholds[0], color='0.2', linestyle='--')
        if plot_f:
            axs[1,0].axvline(thresholds[1], color='0.2', linestyle='--')
            axs[1,0].axvline(thresholds[2], color='0.2', linestyle='--')

        axs[1,1].set_title('Cumulative Counts')
        axs[1,1].plot(binsg[:-1], np.cumsum(ng), 'b', label='g')
        axs[1,1].plot(binse[:-1], np.cumsum(ne), 'r', label='e')
        axs[1,1].axvline(thresholds[0], color='0.2', linestyle='--')
        if plot_f:
            axs[1,1].plot(binsf[:-1], np.cumsum(nf), 'g', label='f')
            axs[1,1].axvline(thresholds[1], color='0.2', linestyle='--')
            axs[1,1].axvline(thresholds[2], color='0.2', linestyle='--')
        axs[1,1].legend()
        axs[1,1].set_xlabel('I [ADC levels]')
        
        plt.subplots_adjust(hspace=0.25, wspace=0.15)        
        plt.show()

    return fids, thresholds, theta*180/np.pi # fids: ge, gf, ef

# ====================================================== #

class HistogramProgram(AveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)

        qubit = self.cfg.expt.qubit
        self.num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        
        self.adc_chs = self.cfg.hw.soc.adcs.readout.ch
        self.res_chs = self.cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = self.cfg.hw.soc.dacs.readout.type
        self.qubit_chs = self.cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = self.cfg.hw.soc.dacs.qubit.type
        
        self.f_ge_regs = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(self.cfg.device.qubit.f_ge, self.qubit_chs)]
        self.f_res_regs = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(self.cfg.device.readout.frequency, self.res_chs)]
        self.f_ef_regs = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(self.cfg.device.qubit.f_ef, self.qubit_chs)]
        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [1+self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)]

        # declare res dacs, add readout pulses
        # declare res dacs
        mask = None
        mixer_freq = 0 # MHz
        mux_freqs = None # MHz
        mux_gains = None
        ro_ch = None
        if self.res_ch_types[qubit] == 'int4':
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[qubit]
        # elif self.res_ch_types[qubit] == 'mux4':
        #     assert self.res_chs[qubit] == 6
        #     mask = [0, 1, 2, 3] # indices of mux_freqs, mux_gains list to play
        #     mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[qubit]
        #     mux_freqs = cfg.device.readout.frequency
        #     mux_gains = cfg.device.readout.gain
        #     ro_ch=self.adc_chs[qubit]
       
    
        self.declare_gen(ch=self.res_chs[qubit], nqz=cfg.hw.soc.dacs.readout.nyquist[qubit], mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)
       
       

        self.declare_readout(ch=self.adc_chs[qubit], length=self.readout_lengths_adc[qubit], freq=cfg.device.readout.frequency[qubit], gen_ch=self.res_chs[qubit])

        # declare adcs - readout for all qubits everytime, defines number of buffers returned regardless of number of adcs triggered
        
       # for q in range(self.num_qubits_sample):
           # self.declare_readout(ch=self.adc_chs[q], length=self.readout_lengths_adc[q], freq=self.cfg.device.readout.frequency[q], gen_ch=self.res_chs[q])

        # add readout pulses to respective channels
        if self.res_ch_types[qubit] == 'mux4':
            self.set_pulse_registers(ch=self.res_chs[qubit], style="const", length=self.readout_lengths_dac[qubit], mask=mask)
        else: self.set_pulse_registers(ch=self.res_chs[qubit], style="const", freq=self.f_res_regs[qubit], phase=0, gain=cfg.device.readout.gain[qubit], length=self.readout_lengths_dac[qubit])

        # get aliases for the sigmas we need in clock cycles
        self.pi_sigmas_us = self.cfg.device.qubit.pulses.pi_ge.sigma
        self.pi_ef_sigmas_us = self.cfg.device.qubit.pulses.pi_ef.sigma
        #self.pi_Q1_ZZ_sigmas_us = self.cfg.device.qubit.pulses.pi_Q1_ZZ.sigma
        self.pi_ge_types = self.cfg.device.qubit.pulses.pi_ge.type
        self.pi_ef_types = self.cfg.device.qubit.pulses.pi_ef.type
        #self.pi_Q1_ZZ_types = self.cfg.device.qubit.pulses.pi_Q1_ZZ.type
        
        # declare qubit dacs, add qubit pi_ge pulses
        for q in range(len(self.pi_ge_types)):
            mixer_freq = 0
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = self.cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in self.gen_chs:
                self.declare_gen(ch=self.qubit_chs[q], nqz=self.cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)
            pi_ge_sigma_cycles = self.us2cycles(self.cfg.device.qubit.pulses.pi_ge.sigma[q], gen_ch=self.qubit_chs[q])
            self.add_gauss(ch=self.qubit_chs[q], name=f"qubit{q}", sigma=pi_ge_sigma_cycles, length=pi_ge_sigma_cycles*4)

            # assume ef pulses are gauss
            pi_ef_sigma_cycles = self.us2cycles(self.pi_ef_sigmas_us[q], gen_ch=self.qubit_chs[q])
            self.add_gauss(ch=self.qubit_chs[q], name=f"pi_ef_qubit{q}", sigma=pi_ef_sigma_cycles, length=pi_ef_sigma_cycles*4)
            # if q != 1:
            #     # pi_Q1_ZZ_sigma_cycles = self.us2cycles(self.pi_Q1_ZZ_sigmas_us[q], gen_ch=self.qubit_chs[1])
            #     self.add_gauss(ch=self.qubit_chs[1], name=f"qubit1_ZZ{q}", sigma=pi_Q1_ZZ_sigma_cycles, length=pi_Q1_ZZ_sigma_cycles*4)

        self.sync_all(200)
    
    def body(self):
        cfg=AttrDict(self.cfg)

        qubit = self.cfg.expt.qubit

        #Phase reset all channels
        # for ch in self.gen_chs.keys():
        #     if self.gen_chs[ch]['mux_freqs'] is None: # doesn't work for the mux channels
        #         # print('resetting', ch)
        #         self.setup_and_pulse(ch=ch, style='const', freq=100, phase=0, gain=100, length=10, phrst=1)
        #     self.sync_all()
        # self.sync_all(10)

        # if 'pulse_test' in self.cfg.expt and self.cfg.expt.pulse_test:
        #     # qDrive = 1
        #     # qNotDrive = 2
        #     # qSort = 2
        #     # setup_ZZ = None

        #     qDrive = 2
        #     qNotDrive = 1
        #     qSort = 2
        #     setup_ZZ = None

        #     # qDrive = 3
        #     # qNotDrive = 1
        #     # qSort = 3
        #     # setup_ZZ = 0

        #     self.swap_chs = self.cfg.hw.soc.dacs.swap.ch
        #     self.swap_ch_types = self.cfg.hw.soc.dacs.swap.type
        #     self.swap_Q_chs = self.cfg.hw.soc.dacs.swap_Q.ch
        #     self.swap_Q_ch_types = self.cfg.hw.soc.dacs.swap_Q.type

        #     if qDrive == 1:
        #         swap_ch = self.swap_chs[qSort]
        #         self.cfg.expt.gain = self.cfg.device.qubit.pulses.pi_EgGf.gain[qSort]
        #         self.cfg.expt.sigma = self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf.sigma[qSort], gen_ch=swap_ch)
        #         self.f_EgGf_reg = self.freq2reg(self.cfg.device.qubit.f_EgGf[qSort], gen_ch=swap_ch)
        #     else:
        #         swap_ch = self.swap_Q_chs[qSort] 
        #         self.cfg.expt.gain = self.cfg.device.qubit.pulses.pi_EgGf_Q.gain[qSort]
        #         self.cfg.expt.sigma = self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf_Q.sigma[qSort], gen_ch=swap_ch)
        #         self.f_EgGf_reg = self.freq2reg(self.cfg.device.qubit.f_EgGf_Q[qSort], gen_ch=swap_ch)
        #     sigma_ramp_cycles = 3
        #     self.add_gauss(ch=swap_ch, name=f"pi_EgGf_swap{qSort}_ramp", sigma=sigma_ramp_cycles, length=sigma_ramp_cycles*4)

        #     if setup_ZZ is None: setup_ZZ = 1
        #     if setup_ZZ != 1:
        #         pass
        #         assert qNotDrive == 1, 'qNotDrive != 1 and setup_ZZ != 1 not setup yet'
        #         pi_ge_sigma_cycles = self.us2cycles(self.cfg.device.qubit.pulses.pi_ge.sigma[setup_ZZ], gen_ch=self.qubit_chs[setup_ZZ])
        #         # self.f_Q1_ZZ_regs = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(self.cfg.device.qubit.f_Q1_ZZ, self.qubit_chs)]
        #         self.add_gauss(ch=self.qubit_chs[setup_ZZ], name=f"qubit{setup_ZZ}", sigma=pi_ge_sigma_cycles, length=pi_ge_sigma_cycles*4)
        #         self.setup_and_pulse(ch=self.qubit_chs[setup_ZZ], style='arb', freq=self.f_ge_regs[setup_ZZ], phase=0, gain=self.cfg.device.qubit.pulses.pi_ge.gain[setup_ZZ], waveform=f'qubit{setup_ZZ}')
        #         self.sync_all()

        #         # pi_Q1_ZZ_sigma_cycles = self.us2cycles(self.cfg.device.qubit.pulses.pi_Q1_ZZ.sigma[setup_ZZ], gen_ch=self.qubit_chs[qNotDrive])
        #         self.add_gauss(ch=self.qubit_chs[qNotDrive], name=f'qubit{qNotDrive}_ZZ{setup_ZZ}', sigma=pi_Q1_ZZ_sigma_cycles, length=pi_Q1_ZZ_sigma_cycles*4)
        #         self.setup_and_pulse(ch=self.qubit_chs[qNotDrive], style='arb', freq=self.f_Q1_ZZ_regs[setup_ZZ], phase=0, gain=self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[setup_ZZ], waveform=f'qubit{qNotDrive}_ZZ{setup_ZZ}')
        #         self.sync_all()
        #     else:
        #         # initialize qubit A to E: expect to end in Eg
        #         # self.setup_and_pulse(ch=self.qubit_chs[qSort], style="arb", phase=0, freq=self.f_ge_regs[qSort], gain=cfg.device.qubit.pulses.pi_ge.gain[qSort], waveform=f"qubit{qSort}") #, phrst=1)
        #         pi_ge_sigma_cycles = self.us2cycles(self.cfg.device.qubit.pulses.pi_ge.sigma[qNotDrive], gen_ch=self.qubit_chs[qNotDrive])
        #         self.add_gauss(ch=self.qubit_chs[qNotDrive], name=f"qubit{qNotDrive}", sigma=pi_ge_sigma_cycles, length=pi_ge_sigma_cycles*4)
        #         self.setup_and_pulse(ch=self.qubit_chs[qNotDrive], style='arb', freq=self.f_ge_regs[qNotDrive], phase=0, gain=self.cfg.device.qubit.pulses.pi_ge.gain[qNotDrive], waveform=f'qubit{qNotDrive}')
        #         self.sync_all()

        #     # apply Eg -> Gf pulse on qDrive: expect to end in Gf
        #     flat_length = self.cfg.expt.sigma - 3*4
        #     if flat_length >= 3:
        #         # print(self.cfg.expt.gain, flat_length, self.f_EgGf_reg)
        #         self.setup_and_pulse(
        #             ch=swap_ch,
        #             style="flat_top",
        #             freq=self.f_EgGf_reg,
        #             phase=0,
        #             gain=self.cfg.expt.gain,
        #             length=flat_length,
        #             waveform=f"pi_EgGf_swap{qSort}_ramp",
        #         )
        #     self.sync_all(5)

        #     # setup_measure = None
        #     # if 'setup_measure' in self.cfg.expt: setup_measure = self.cfg.expt.setup_measure

        #     # # take qDrive g->e: measure the population of just the e state when e/f are not distinguishable by checking the g population
        #     # if setup_measure == 'qDrive_ge':
        #     #     # print('playing ge pulse')
        #     #     self.X_pulse(q=qDrive, play=True)
        #     #     self.sync_all(5)
        
        #     # if setup_measure == None: pass # measure the real g population only

        #     # # take qDrive f->e: expect to end in Ge (or Eg if incomplete Eg-Gf)
        #     # # if setup_measure == 'qDrive_ef':
        #     # self.setup_and_pulse(ch=self.qubit_chs[qDrive], style="arb", freq=self.f_ef_regs[qDrive], phase=0, gain=cfg.device.qubit.pulses.pi_ef.gain[qDrive], waveform=f"pi_ef_qubit{qDrive}") #, phrst=1)

        
        # if self.cfg.expt.pulse_e or self.cfg.expt.pulse_f:
        #     if cfg.device.qubit.pulses.pi_ge.type[qubit] == 'gauss':
        #         self.setup_and_pulse(ch=self.qubit_chs[qubit], style="arb", freq=self.f_ge_regs[qubit], phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain[qubit], waveform=f"qubit{qubit}")
        #     else: # const pulse
        #         pi_ge_sigma_cycles = self.us2cycles(self.cfg.device.qubit.pulses.pi_ge.sigma[qubit], gen_ch=self.qubit_chs[qubit])
        #         self.setup_and_pulse(ch=self.qubit_chs[qubit], style="const", freq=self.f_ge_regs[qubit], phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain[qubit], length=pi_ge_sigma_cycles)

        # if self.cfg.expt.pulse_f:
        #     if cfg.device.qubit.pulses.pi_ef.type[qubit] == 'gauss':
        #         self.setup_and_pulse(ch=self.qubit_chs[qubit], style="arb", freq=self.f_ef_regs[qubit], phase=0, gain=cfg.device.qubit.pulses.pi_ef.gain[qubit], waveform=f"pi_ef_qubit{qubit}")
        #     else: # const pulse
        #         pi_ef_sigma_cycles = self.us2cycles(self.cfg.device.qubit.pi_ef.sigma[qubit], gen_ch=self.qubit_chs[qubit])
        #         self.setup_and_pulse(ch=self.qubit_chs[qubit], style="const", freq=self.f_ef_regs[qubit], phase=0, gain=cfg.device.qubit.pulses.pi_ef.gain[qubit], length=pi_ef_sigma_cycles)
        # self.sync_all()

        measure_chs = self.res_chs
        if self.res_ch_types[0] == 'mux4': measure_chs = self.res_chs[0]
        self.measure(
            pulse_ch=[measure_chs[qubit]], 
            adcs=[self.adc_chs[qubit]],
            adc_trig_offset=cfg.device.readout.trig_offset[0],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qubit]))

    def collect_shots(self):
        # collect shots for the relevant adc and I and Q channels
        cfg=AttrDict(self.cfg)
        # print(np.average(self.di_buf[0]))
        shots_i0 = self.di_buf[0] / self.readout_lengths_adc[self.cfg.expt.qubit]
        shots_q0 = self.dq_buf[0] / self.readout_lengths_adc[self.cfg.expt.qubit]
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

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
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
        #     x_pts, avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=False, debug=debug)
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
        avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug)
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
            avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug)
            data['Ie'], data['Qe'] = histpro.collect_shots()

        # Excited f state shots
        self.check_f = self.cfg.expt.check_f
        if self.check_f:
            cfg = AttrDict(deepcopy(self.cfg))
            cfg.expt.pulse_e = True 
            cfg.expt.pulse_f = True
            cfg.expt.pulse_test = False
            histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
            avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug)
            data['If'], data['Qf'] = histpro.collect_shots()

        # Test state shots
        if 'pulse_test' not in self.cfg.expt: self.cfg.expt.pulse_test = False
        self.check_test = self.cfg.expt.pulse_test
        if self.check_test:
            cfg = AttrDict(deepcopy(self.cfg))
            histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
            avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug)
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
        
        fids, thresholds, angle = hist(data=data, plot=True, verbose=verbose, span=span)
            
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

    def acquire(self, progress=False, debug=False):
        fpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"]*np.arange(self.cfg.expt["expts_f"])
        gainpts = self.cfg.expt["start_gain"] + self.cfg.expt["step_gain"]*np.arange(self.cfg.expt["expts_gain"])
        lenpts = self.cfg.expt["start_len"] + self.cfg.expt["step_len"]*np.arange(self.cfg.expt["expts_len"])
        print(fpts)
        print(gainpts)
        print(lenpts)
        
        fid = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        threshold = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        angle = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))

        qubit = self.cfg.expt.qubit

        for f_ind, f in enumerate(tqdm(fpts, disable=not progress)):
            for g_ind, gain in enumerate(gainpts):
                for l_ind, l in enumerate(lenpts):
                    shot = HistogramExperiment(soccfg=self.soccfg, config_file=self.config_file)
                    shot.cfg = self.cfg
                    shot.cfg.device.readout.frequency[qubit] = f
                    shot.cfg.device.readout.gain[qubit] = gain
                    shot.cfg.device.readout.readout_length = l 
                    check_e = True
                    if 'check_f' not in self.cfg.expt: check_f = False
                    else:
                        check_f = self.cfg.expt.check_f
                        check_e = not check_f
                    shot.cfg.expt = dict(reps=self.cfg.expt.reps, check_e=check_e, check_f=check_f, qubit=self.cfg.expt.qubit)
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