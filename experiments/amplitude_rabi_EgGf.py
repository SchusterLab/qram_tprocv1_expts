import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter

class AmplitudeRabiEgGfProgram(RAveragerProgram):
    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)

        self.num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.qubits = self.cfg.expt.qubits
        qA, qB = self.qubits

        # all of these saved self.whatever instance variables should be indexed by the actual qubit number as opposed to qubits_i. this means that more values are saved as instance variables than is strictly necessary, but this is overall less confusing
        self.adc_chs = cfg.hw.soc.adcs.readout.ch
        self.res_chs = self.cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = self.cfg.hw.soc.dacs.readout.type
        self.qubit_chs = self.cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = self.cfg.hw.soc.dacs.qubit.type
        self.swap_chs = self.cfg.hw.soc.dacs.swap.ch
        self.swap_ch_types = self.cfg.hw.soc.dacs.swap.type

        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs] # get register page for qubit_chs
        self.f_ge_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ge, self.qubit_chs)]
        self.f_ef_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ef, self.qubit_chs)]
        self.f_res_reg = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        self.f_EgGf_reg = self.freq2reg(cfg.device.qubit.f_EgGf[qA], gen_ch=self.swap_chs[qA])
        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [1+self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)]
        self.readout_length = [self.us2cycles(len) for len in self.cfg.device.readout.readout_length]

        # declare res dacs
        mask = None
        if self.res_ch_types[0] == 'mux4': # only supports having all resonators be on mux, or none
            assert np.all([ch == 6 for ch in self.res_chs])
            mask = range(4) # indices of mux_freqs, mux_gains list to play
            mux_freqs = [0 if i in self.qubits else cfg.device.readout.frequency[i] for i in range(4)]
            mux_gains = [0 if i in self.qubits else cfg.device.readout.gain[i] for i in range(4)]
            self.declare_gen(ch=6, nqz=cfg.hw.soc.dacs.readout.nyquist[0], mixer_freq=cfg.hw.soc.dacs.readout.mixer_freq[0], mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=0)
        else:
            for q in self.qubits:
                mixer_freq = 0
                if self.res_ch_types[q] == 'int4':
                    mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[q]
                self.declare_gen(ch=self.res_chs[q], nqz=cfg.hw.soc.dacs.readout.nyquist[q], mixer_freq=mixer_freq)

        # declare qubit dacs
        for q in self.qubits:
            mixer_freq = 0
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq[q]
            self.declare_gen(ch=self.qubit_chs[q], nqz=cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)

        # declare swap dac indexed by qA (since the the drive is always applied to qB)
        mixer_freq = 0
        if self.swap_ch_types[qA] == 'int4':
            mixer_freq = cfg.hw.soc.dacs.swap.mixer_freq[qA]
        self.declare_gen(ch=self.swap_chs[qA], nqz=cfg.hw.soc.dacs.swap.nyquist[qA], mixer_freq=mixer_freq)

        # declare adcs - readout for all qubits everytime
        for q in range(self.num_qubits_sample):
            self.declare_readout(ch=self.adc_chs[q], length=self.readout_lengths_adc[q], freq=cfg.device.readout.frequency[q], gen_ch=self.res_chs[q])

        # get gain register for swap ch
        if self.swap_ch_types[qA] == 'int4':
            self.r_gain_swap = self.sreg(self.swap_chs[qA], "addr")
        else: self.r_gain_swap = self.sreg(self.swap_chs[qA], "gain")
        # register to hold the current sweep gain
        self.r_gain_swap_update = 4
        # initialize gain
        self.safe_regwi(self.ch_page(self.swap_chs[qA]), self.r_gain_swap_update, self.cfg.expt.start)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds

        self.pi_sigmaA = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qA], gen_ch=self.qubit_chs[qA])
        self.pi_ef_sigmaB = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma[qB], self.qubit_chs[qB])
        self.pi_EgGf_sigma = self.us2cycles(cfg.expt.pi_EgGf_sigma, gen_ch=self.swap_chs[qA]) # defaults to eg-gf sigma in config

        # add qubit and swap pulses to respective channels
        self.add_gauss(ch=self.qubit_chs[qA], name="pi_qubitA", sigma=self.pi_sigmaA, length=self.pi_sigmaA*4)
        self.add_gauss(ch=self.qubit_chs[qB], name="pi_ef_qubitB", sigma=self.pi_ef_sigmaB, length=self.pi_ef_sigmaB*4)
        if cfg.expt.pulse_type.lower() == "gauss" and cfg.expt.pi_EgGf_sigma > 0:
            self.add_gauss(ch=self.swap_chs[qA], name="pi_EgGf_swap", sigma=self.pi_EgGf_sigma, length=self.pi_EgGf_sigma*4)

        # add readout pulses to respective channels
        if self.res_ch_types[0] == 'mux4':
            self.set_pulse_registers(ch=6, style="const", length=max(self.readout_lengths_dac), mask=mask)
        else:
            for q in self.qubits:
                self.set_pulse_registers(ch=self.res_chs[q], style="const", freq=self.f_res_reg[q], phase=0, gain=cfg.device.readout.gain[q], length=self.readout_lengths_dac[q])

        self.sync_all(200)

    def body(self):
        cfg=AttrDict(self.cfg)
        qA, qB = self.qubits

        # initialize qubit A to E: expect to end in Eg
        self.setup_and_pulse(ch=self.qubit_chs[qA], style="arb", phase=0, freq=self.f_ge_reg[qA], gain=cfg.device.qubit.pulses.pi_ge.gain[qA], waveform="pi_qubitA")
        self.sync_all(5)

        # apply Eg -> Gf pulse on B: expect to end in Gf
        if cfg.expt.pi_EgGf_sigma > 0:
            if cfg.expt.pulse_type.lower() == "gauss":
                self.set_pulse_registers(
                    ch=self.swap_chs[qA],
                    style="arb",
                    freq=self.f_EgGf_reg,
                    phase=0,
                    gain=0, # gain set by update
                    waveform="pi_EgGf_swap")
            else:
                self.set_pulse_registers(
                    ch=self.swap_chs[qA],
                    style="const",
                    freq=self.f_EgGf_reg,
                    phase=0,
                    gain=0, # gain set by update
                    length=self.pi_EgGf_sigma)
            self.mathi(self.ch_page(self.swap_chs[qA]), self.r_gain_swap, self.r_gain_swap_update, "+", 0)
            self.pulse(ch=self.swap_chs[qA])
        self.sync_all(5)

        # take qubit A G->E and qubit B f->e: expect to end in Ee (or Gg if incomplete Eg-Gf)
        self.setup_and_pulse(ch=self.qubit_chs[qA], style="arb", freq=self.f_ge_reg[qA], phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain[qA], waveform="pi_qubitA")
        self.sync_all(5)
        self.setup_and_pulse(ch=self.qubit_chs[qB], style="arb", freq=self.f_ef_reg[qB], phase=0, gain=cfg.device.qubit.pulses.pi_ef.gain[qB], waveform="pi_ef_qubitB")
        
        self.sync_all(5)
        if self.res_ch_types[0] == 'mux4': measure_chs = self.res_chs[0]
        self.measure(
            pulse_ch=measure_chs, 
            adcs=[0,1],
            adc_trig_offset=cfg.device.readout.trig_offset[0],
            wait=True,
            syncdelay=self.us2cycles(max([cfg.device.readout.relax_delay[q] for q in self.qubits])))
 
    def update(self):
        qA, qB = self.qubits
        step = self.cfg.expt.step
        if self.swap_ch_types[qA] == 'int4': step = step << 16
        self.mathi(self.ch_page(self.swap_chs[qA]), self.r_gain_swap_update, self.r_gain_swap_update, '+', step) # update test gain

class AmplitudeRabiEgGfExperiment(Experiment):
    """
    Amplitude Rabi Eg<->Gf Experiment
    Experimental Config:
    expt = dict(
        start: qubit gain [dac units]
        step: gain step [dac units]
        expts: number steps
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        pi_EgGf_sigma: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'const'
       singleshot: (optional) if true, uses threshold
    )
    """

    def __init__(self, soccfg=None, path='', prefix='AmplitudeRabiEgGf', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        qA, qB = self.cfg.expt.qubits

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

        adcA_ch = self.cfg.hw.soc.adcs.readout.ch[qA]
        adcB_ch = self.cfg.hw.soc.adcs.readout.ch[qB]
 
        if 'pi_EgGf_sigma' not in self.cfg.expt:
            self.cfg.expt.pi_EgGf_sigma = self.cfg.device.qubit.pulses.pi_EgGf.sigma

        threshold = None
        angle = None
        if 'singleshot' in self.cfg.expt.keys():
            if self.cfg.expt.singleshot:
                threshold = self.cfg.device.readout.threshold
                # angle = self.cfg.device.readout.phase
        
        amprabi = AmplitudeRabiEgGfProgram(soccfg=self.soccfg, cfg=self.cfg)
        x_pts, avgi, avgq = amprabi.acquire(self.im[self.cfg.aliases.soc], threshold=threshold, angle=angle, load_pulses=True, progress=progress, debug=debug)        
        self.prog = amprabi
        
        data=dict(
            xpts=x_pts,
            avgi=(avgi[adcA_ch][0], avgi[adcB_ch][0]),
            avgq=(avgq[adcA_ch][0], avgq[adcB_ch][0]),
            amps=(np.abs(avgi[adcA_ch][0]+1j*avgq[adcA_ch][0]),
                  np.abs(avgi[adcB_ch][0]+1j*avgq[adcB_ch][0])),
            phases=(np.angle(avgi[adcA_ch][0]+1j*avgq[adcA_ch][0]),
                    np.angle(avgi[adcB_ch][0]+1j*avgq[adcB_ch][0])),
        )
        self.data=data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data
        
        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), amp offset]
            # Remove the first and last point from fit in case weird edge measurements
            fitparams = [None, 1/max(data['xpts']), None, None]
            # fitparams = None
            pA_avgi, pCovA_avgi = fitter.fitsin(data['xpts'][:-1], data["avgi"][0][:-1], fitparams=fitparams)
            pA_avgq, pCovA_avgq = fitter.fitsin(data['xpts'][:-1], data["avgq"][0][:-1], fitparams=fitparams)
            pA_amps, pCovA_amps = fitter.fitsin(data['xpts'][:-1], data["amps"][0][:-1], fitparams=fitparams)
            data['fitA_avgi'] = pA_avgi   
            data['fitA_avgq'] = pA_avgq
            data['fitA_amps'] = pA_amps
            data['fitA_err_avgi'] = pCovA_avgi   
            data['fitA_err_avgq'] = pCovA_avgq
            data['fitA_err_amps'] = pCovA_amps

            pB_avgi, pCovB_avgi = fitter.fitsin(data['xpts'][:-1], data["avgi"][1][:-1], fitparams=fitparams)
            pB_avgq, pCovB_avgq = fitter.fitsin(data['xpts'][:-1], data["avgq"][1][:-1], fitparams=fitparams)
            pB_amps, pCovB_amps = fitter.fitsin(data['xpts'][:-1], data["amps"][1][:-1], fitparams=fitparams)
            data['fitB_avgi'] = pB_avgi   
            data['fitB_avgq'] = pB_avgq
            data['fitB_amps'] = pB_amps
            data['fitB_err_avgi'] = pCovB_avgi   
            data['fitB_err_avgq'] = pCovB_avgq
            data['fitB_err_amps'] = pCovB_amps
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 

        plt.figure(figsize=(20,6))
        plt.suptitle(f"Amplitude Rabi Eg-Gf (Drive Length {self.cfg.expt.pi_EgGf_sigma} us)")
        plt.subplot(121, title="Qubit A", ylabel="Amplitude [adc units]", xlabel='Gain [DAC units]')
        plt.plot(data["xpts"][0:-1], data["amps"][0][0:-1],'o-')
        if fit:
            p = data['fitA_amps']
            plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
            pi_gain = 1/p[1]/2
            print(f'Pi gain from amps data (qubit A) [DAC units]: {int(pi_gain)}')
            print(f'\tPi/2 gain from amps data (qubit A) [DAC units]: {int(pi_gain/2)}')
            # plt.axvline(pi_gain, color='0.2', linestyle='--')
            # plt.axvline(pi_gain/2, color='0.2', linestyle='--')
        plt.subplot(122, title="Qubit B", xlabel='Gain [DAC units]')
        plt.plot(data["xpts"][0:-1], data["amps"][1][0:-1],'o-')
        if fit:
            p = data['fitB_amps']
            plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
            pi_gain = 1/p[1]/2
            print()
            print(f'Pi gain from amps data (qubit A) [DAC units]: {int(pi_gain)}')
            print(f'\tPi/2 gain from amps data (qubit A) [DAC units]: {int(pi_gain/2)}')
            # plt.axvline(pi_gain, color='0.2', linestyle='--')
            # plt.axvline(pi_gain/2, color='0.2', linestyle='--')

        # plt.figure(figsize=(14,8))
        # plt.suptitle(f"Amplitude Rabi Eg-Gf (Drive Length {self.cfg.expt.pi_EgGf_sigma} us)")
        # if self.cfg.expt.singleshot: plt.subplot(221, title='Qubit A', ylabel=r"Probability of $|e\rangle$")
        # else: plt.subplot(221, title='Qubit A', ylabel="I [ADC units]")
        # plt.plot(data["xpts"][0:-1], data["avgi"][0][0:-1],'o-')
        # if fit:
        #     p = data['fitA_avgi']
        #     plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
        #     pi_gain = 1/p[1]/2
        #     print(f'Pi gain from avgi data (qubit A) [DAC units]: {int(pi_gain)}')
        #     print(f'\tPi/2 gain from avgi data (qubit A) [DAC units]: {int(pi_gain/2)}')
        #     plt.axvline(pi_gain, color='0.2', linestyle='--')
        #     plt.axvline(pi_gain/2, color='0.2', linestyle='--')
        # plt.subplot(223, xlabel="Gain [DAC units]", ylabel="Q [ADC units]")
        # plt.plot(data["xpts"][0:-1], data["avgq"][0][0:-1],'o-')
        # if fit:
        #     p = data['fitA_avgq']
        #     plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
        #     pi_gain = 1/p[1]/2
        #     print(f'Pi gain from avgq data (qubit A) [DAC units]: {int(pi_gain)}')
        #     print(f'\tPi/2 gain from avgq data (qubit A) [DAC units]: {int(pi_gain/2)}')
        #     plt.axvline(pi_gain, color='0.2', linestyle='--')
        #     plt.axvline(pi_gain/2, color='0.2', linestyle='--')

        # plt.subplot(222, title='Qubit B')
        # plt.plot(data["xpts"][0:-1], data["avgi"][1][0:-1],'o-')
        # if fit:
        #     p = data['fitB_avgi']
        #     plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
        #     pi_gain = 1/p[1]/2
        #     print()
        #     print(f'Pi gain from avgi data (qubit B) [DAC units]: {int(pi_gain)}')
        #     print(f'\tPi/2 gain from avgi data (qubit B) [DAC units]: {int(pi_gain/2)}')
        #     plt.axvline(pi_gain, color='0.2', linestyle='--')
        #     plt.axvline(pi_gain/2, color='0.2', linestyle='--')
        # plt.subplot(224, xlabel="Gain [DAC units]")
        # plt.plot(data["xpts"][0:-1], data["avgq"][1][0:-1],'o-')
        # if fit:
        #     p = data['fitB_avgq']
        #     plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
        #     pi_gain = 1/p[1]/2
        #     print(f'Pi gain from avgq data (qubit B) [DAC units]: {int(pi_gain)}')
        #     print(f'\tPi/2 gain from avgq data (qubit B) [DAC units]: {int(pi_gain/2)}')
        #     plt.axvline(pi_gain, color='0.2', linestyle='--')
        #     plt.axvline(pi_gain/2, color='0.2', linestyle='--')
        # plt.tight_layout()
        # plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)