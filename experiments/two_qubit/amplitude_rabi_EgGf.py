import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter

class AmplitudeRabiEgGfProgram(RAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
        
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)

        self.num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.qubits = self.cfg.expt.qubits
        qA, qB = self.qubits

        # all of these saved self.whatever instance variables should be indexed by the actual qubit number as opposed to qubits_i. this means that more values are saved as instance variables than is strictly necessary, but this is overall less confusing
        self.adc_chs = self.cfg.hw.soc.adcs.readout.ch
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

        gen_chs = []
        
        # declare res dacs
        mask = None
        if self.res_ch_types[0] == 'mux4': # only supports having all resonators be on mux, or none
            assert np.all([ch == 6 for ch in self.res_chs])
            mask = range(4) # indices of mux_freqs, mux_gains list to play
            mux_freqs = [0 if i not in self.qubits else cfg.device.readout.frequency[i] for i in range(4)]
            mux_gains = [0 if i not in self.qubits else cfg.device.readout.gain[i] for i in range(4)]
            self.declare_gen(ch=6, nqz=cfg.hw.soc.dacs.readout.nyquist[0], mixer_freq=cfg.hw.soc.dacs.readout.mixer_freq[0], mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=0)
            gen_chs.append(6)
        else:
            for q in self.qubits:
                mixer_freq = 0
                if self.res_ch_types[q] == 'int4':
                    mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[q]
                self.declare_gen(ch=self.res_chs[q], nqz=cfg.hw.soc.dacs.readout.nyquist[q], mixer_freq=mixer_freq)
                gen_chs.append(self.res_chs[q])

        # declare qubit dacs
        for q in self.qubits:
            mixer_freq = 0
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in gen_chs:
                self.declare_gen(ch=self.qubit_chs[q], nqz=cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)
                gen_chs.append(self.qubit_chs[q])

        # declare swap dac indexed by qA (since the the drive is always applied to qB) mixer_freq = 0
        if self.swap_ch_types[qA] == 'int4':
            mixer_freq = cfg.hw.soc.dacs.swap.mixer_freq[qA]
        if self.swap_chs[qA] not in gen_chs: 
            self.declare_gen(ch=self.swap_chs[qA], nqz=cfg.hw.soc.dacs.swap.nyquist[qA], mixer_freq=mixer_freq)
        gen_chs.append(self.swap_chs[qA])

        # declare adcs - readout for all qubits everytime, defines number of buffers returned regardless of number of adcs triggered
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

        self.pi_sigmaA = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qA], gen_ch=self.qubit_chs[qA])
        self.pi_ef_sigmaB = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma[qB], self.qubit_chs[qB])
        self.pi_EgGf_sigma = self.us2cycles(cfg.expt.pi_EgGf_sigma, gen_ch=self.swap_chs[qA]) # defaults to eg-gf sigma in config

        # add qubit and swap pulses to respective channels
        self.add_gauss(ch=self.qubit_chs[qA], name="pi_qubitA", sigma=self.pi_sigmaA, length=self.pi_sigmaA*4)
        self.add_gauss(ch=self.qubit_chs[qB], name="pi_ef_qubitB", sigma=self.pi_ef_sigmaB, length=self.pi_ef_sigmaB*4)
        if cfg.expt.pulse_type.lower() == "gauss" and cfg.expt.pi_EgGf_sigma > 0:
            self.add_gauss(ch=self.swap_chs[qA], name="pi_EgGf_swap", sigma=self.pi_EgGf_sigma, length=self.pi_EgGf_sigma*4)
        elif cfg.expt.pulse_type.lower() == "flat_top" and cfg.expt.pi_EgGf_sigma > 0:
            self.add_gauss(ch=self.swap_chs[qA], name="pi_EgGf_swap", sigma=5, length=20)

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
            pulse_type = cfg.expt.pulse_type.lower()
            if pulse_type == "gauss":
                self.set_pulse_registers(
                    ch=self.swap_chs[qA],
                    style="arb",
                    freq=self.f_EgGf_reg,
                    phase=0,
                    gain=0, # gain set by update
                    waveform="pi_EgGf_swap")
            elif pulse_type == 'flat_top':
                length = self.us2cycles(cfg.expt.pi_EgGf_sigma, gen_ch=self.swap_chs[qA])-5*4
                if length > 0:
                    self.set_pulse_registers(
                        ch=self.swap_chs[qA],
                        style="flat_top",
                        freq=self.f_EgGf_reg,
                        phase=0,
                        gain=0, # gain set by update
                        length=length,
                        waveform="pi_EgGf_swap")
            else: # const
                self.set_pulse_registers(
                    ch=self.swap_chs[qA],
                    style="const",
                    freq=self.f_EgGf_reg,
                    phase=0,
                    gain=0, # gain set by update
                    length=self.pi_EgGf_sigma)
            self.mathi(self.ch_page(self.swap_chs[qA]), self.r_gain_swap, self.r_gain_swap_update, "+", 0)
            if pulse_type != 'flat_top' or length > 0: self.pulse(ch=self.swap_chs[qA])
        self.sync_all(5)

        # take qubit B f->e: expect to end in Ge (or Eg if incomplete Eg-Gf)
        self.setup_and_pulse(ch=self.qubit_chs[qB], style="arb", freq=self.f_ef_reg[qB], phase=0, gain=cfg.device.qubit.pulses.pi_ef.gain[qB], waveform="pi_ef_qubitB")
        
        self.sync_all(5)
        measure_chs = self.res_chs
        if self.res_ch_types[0] == 'mux4': measure_chs = self.res_chs[0]
        self.measure(
            pulse_ch=measure_chs, 
            adcs=self.adc_chs,
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
        qubits: qubit 0 goes E->G, apply drive on qubit 1 (g->f)
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
            self.cfg.expt.pi_EgGf_sigma = self.cfg.device.qubit.pulses.pi_EgGf.sigma[qA]

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
            # fitparams = [None, 1/max(data['xpts']), None, None]
            fitparams = None
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

        # plt.figure(figsize=(20,6))
        # plt.suptitle(f"Amplitude Rabi Eg-Gf (Drive Length {self.cfg.expt.pi_EgGf_sigma} us)")
        # plt.subplot(121, title=f'Qubit A ({self.cfg.expt.qubits[0]})', ylabel="Amplitude [adc units]", xlabel='Gain [DAC units]')
        # plt.plot(data["xpts"][0:-1], data["amps"][0][0:-1],'o-')
        # if fit:
        #     p = data['fitA_amps']
        #     plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
        #     if p[2] > 180: p[2] = p[2] - 360
        #     elif p[2] < -180: p[2] = p[2] + 360
        #     if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
        #     else: pi_gain= (3/2 - p[2]/180)/2/p[1]
        #     pi2_gain = pi_gain/2
        #     print(f'Pi gain from amps data (qubit A) [dac units]: {int(pi_gain)}')
        #     print(f'\tPi/2 gain from amps data (qubit A) [dac units]: {int(pi2_gain)}')
        #     plt.axvline(pi_gain, color='0.2', linestyle='--')
        #     plt.axvline(pi2_gain, color='0.2', linestyle='--')
        # plt.subplot(122, title=f'Qubit B ({self.cfg.expt.qubits[1]})', xlabel='Gain [DAC units]')
        # plt.plot(data["xpts"][0:-1], data["amps"][1][0:-1],'o-')
        # if fit:
        #     p = data['fitB_amps']
        #     plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
        #     if p[2] > 180: p[2] = p[2] - 360
        #     elif p[2] < -180: p[2] = p[2] + 360
        #     if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
        #     else: pi_gain= (3/2 - p[2]/180)/2/p[1]
        #     pi2_gain = pi_gain/2
        #     print(f'Pi gain from amps data (qubit B) [dac units]: {int(pi_gain)}')
        #     print(f'\tPi/2 gain from amps data (qubit B) [dac units]: {int(pi2_gain)}')
        #     plt.axvline(pi_gain, color='0.2', linestyle='--')
        #     plt.axvline(pi2_gain, color='0.2', linestyle='--')

        plt.figure(figsize=(14,8))
        plt.suptitle(f"Amplitude Rabi Eg-Gf (Drive Length {self.cfg.expt.pi_EgGf_sigma} us)")
        if self.cfg.expt.singleshot: plt.subplot(221, title=f'Qubit A ({self.cfg.expt.qubits[0]})', ylabel=r"Probability of $|e\rangle$")
        else: plt.subplot(221, title=f'Qubit A ({self.cfg.expt.qubits[0]})', ylabel="I [ADC units]")
        plt.plot(data["xpts"][0:-1], data["avgi"][0][0:-1],'o-')
        if fit:
            p = data['fitA_avgi']
            plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
            else: pi_gain= (3/2 - p[2]/180)/2/p[1]
            pi2_gain = pi_gain/2
            print(f'Pi gain from avgi data (qubit A) [dac units]: {int(pi_gain)}')
            print(f'\tPi/2 gain from avgi data (qubit A) [dac units]: {int(pi2_gain)}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi2_gain, color='0.2', linestyle='--')
        plt.subplot(223, xlabel="Gain [DAC units]", ylabel="Q [ADC units]")
        plt.plot(data["xpts"][0:-1], data["avgq"][0][0:-1],'o-')
        if fit:
            p = data['fitA_avgq']
            plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
            else: pi_gain= (3/2 - p[2]/180)/2/p[1]
            pi2_gain = pi_gain/2
            print(f'Pi gain from avgq data (qubit A) [dac units]: {int(pi_gain)}')
            print(f'\tPi/2 gain from avgq data (qubit A) [dac units]: {int(pi2_gain)}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi2_gain, color='0.2', linestyle='--')

        plt.subplot(222, title=f'Qubit B ({self.cfg.expt.qubits[1]})')
        plt.plot(data["xpts"][0:-1], data["avgi"][1][0:-1],'o-')
        if fit:
            p = data['fitB_avgi']
            plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
            else: pi_gain= (3/2 - p[2]/180)/2/p[1]
            pi2_gain = pi_gain/2
            print(f'Pi gain from avgi data (qubit B) [dac units]: {int(pi_gain)}')
            print(f'\tPi/2 gain from avgi data (qubit B) [dac units]: {int(pi2_gain)}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi2_gain, color='0.2', linestyle='--')
        plt.subplot(224, xlabel="Gain [DAC units]")
        plt.plot(data["xpts"][0:-1], data["avgq"][1][0:-1],'o-')
        if fit:
            p = data['fitB_avgq']
            plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
            else: pi_gain= (3/2 - p[2]/180)/2/p[1]
            pi2_gain = pi_gain/2
            print(f'Pi gain from avgq data (qubit B) [dac units]: {int(pi_gain)}')
            print(f'\tPi/2 gain from avgq data (qubit B) [dac units]: {int(pi2_gain)}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi2_gain, color='0.2', linestyle='--')
        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)

# ===================================================================== #

class EgGfLenGainChevronExperiment(Experiment):
    """
    Rabi Eg<->Gf Experiment Chevron sweeping length vs. gain
    Experimental Config:
    expt = dict(
        start_gain: qubit gain [dac units]
        step_gain: gain step [dac units]
        expts_gain: number steps
        start_len: start length [us],
        step_len: length step, 
        expts_len: number of different length experiments, 
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        pulse_type: 'gauss' or 'const'
    )
    """
    def __init__(self, soccfg=None, path='', prefix='RabiEgGfLenGainChevron', config_file=None, progress=None):
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

        adc_chs = self.cfg.hw.soc.adcs.readout.ch
        
        lenpts = self.cfg.expt.start_len + self.cfg.expt.step_len * np.arange(self.cfg.expt.expts_len)
        gainpts = self.cfg.expt.start_gain + self.cfg.expt.step_gain * np.arange(self.cfg.expt.expts_gain)
        
        data={"gainpts":gainpts, "lenpts":lenpts, "avgi":[[],[]], "avgq":[[],[]], "amps":[[],[]], "phases":[[],[]]}

        self.cfg.expt.start = self.cfg.expt.start_gain
        self.cfg.expt.step = self.cfg.expt.step_gain
        self.cfg.expt.expts = self.cfg.expt.expts_gain

        threshold = None
        angle = None

        for length in tqdm(lenpts, disable=not progress):
            self.cfg.expt.pi_EgGf_sigma = float(length)

            amprabi = AmplitudeRabiEgGfProgram(soccfg=self.soccfg, cfg=self.cfg)
            gainpts, avgi, avgq = amprabi.acquire(self.im[self.cfg.aliases.soc], threshold=threshold, angle=angle, load_pulses=True, progress=False, debug=debug)        

            for q_ind, q in enumerate(self.cfg.expt.qubits):
                data['avgi'][q_ind].append(avgi[adc_chs[q], 0])
                data['avgq'][q_ind].append(avgq[adc_chs[q], 0])
                data['amps'][q_ind].append(np.abs(avgi[adc_chs[q], 0]+1j*avgi[adc_chs[q], 0]))
                data['phases'][q_ind].append(np.angle(avgi[adc_chs[q], 0]+1j*avgi[adc_chs[q], 0]))

        for k, a in data.items():
            data[k] = np.array(a)
        self.data=data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data
        pass
            

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 

        inner_sweep = data['gainpts']
        outer_sweep = data['lenpts']

        y_sweep = outer_sweep
        x_sweep = inner_sweep

        plt.figure(figsize=(14,8))
        plt.suptitle(f"Eg-Gf Chevron Length vs. Gain")


        plt.subplot(221, title=f'Qubit A ({self.cfg.expt.qubits[0]})', ylabel="Pulse Length [us]")
        plt.pcolormesh(x_sweep, y_sweep, data['avgi'][0], cmap='viridis', shading='auto')
        plt.colorbar(label='I [ADC level]')

        plt.subplot(223, xlabel="Gain [DAC units]", ylabel="Pulse Length [us]")
        plt.pcolormesh(x_sweep, y_sweep, data['avgq'][0], cmap='viridis', shading='auto')
        plt.colorbar(label='Q [ADC level]')


        plt.subplot(222, title=f'Qubit B ({self.cfg.expt.qubits[1]})')
        plt.pcolormesh(x_sweep, y_sweep, data['avgi'][1], cmap='viridis', shading='auto')
        plt.colorbar(label='I [ADC level]')

        plt.subplot(224, xlabel="Gain [DAC units]")
        plt.pcolormesh(x_sweep, y_sweep, data['avgq'][1], cmap='viridis', shading='auto')
        plt.colorbar(label='Q [ADC level]')


        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)

# ===================================================================== #

class EgGfFreqGainChevronExperiment(Experiment):
    """
    Rabi Eg<->Gf Experiment Chevron sweeping freq vs. gain
    Experimental Config:
    expt = dict(
        start_gain: qubit gain [dac units]
        step_gain: gain step [dac units]
        expts_gain: number steps
        start_f: start freq [MHz],
        step_f: freq step, 
        expts_f: number of different freq experiments, 
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        pulse_type: 'gauss' or 'const'
    )
    """
    def __init__(self, soccfg=None, path='', prefix='RabiEgGfFreqGainChevron', config_file=None, progress=None):
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

        adc_chs = self.cfg.hw.soc.adcs.readout.ch
        
        freqpts = self.cfg.expt.start_f + self.cfg.expt.step_f * np.arange(self.cfg.expt.expts_f)
        gainpts = self.cfg.expt.start_gain + self.cfg.expt.step_gain * np.arange(self.cfg.expt.expts_gain)
        
        data={"gainpts":gainpts, "freqpts":freqpts, "avgi":[[],[]], "avgq":[[],[]], "amps":[[],[]], "phases":[[],[]]}

        self.cfg.expt.start = self.cfg.expt.start_gain
        self.cfg.expt.step = self.cfg.expt.step_gain
        self.cfg.expt.expts = self.cfg.expt.expts_gain

        if 'pi_EgGf_sigma' not in self.cfg.expt:
            self.cfg.expt.pi_EgGf_sigma = self.cfg.device.qubit.pulses.pi_EgGf.sigma[qA]

        threshold = None
        angle = None

        for freq in tqdm(freqpts, disable=not progress):
            self.cfg.device.qubit.f_EgGf[qA] = float(freq)

            amprabi = AmplitudeRabiEgGfProgram(soccfg=self.soccfg, cfg=self.cfg)
            gainpts, avgi, avgq = amprabi.acquire(self.im[self.cfg.aliases.soc], threshold=threshold, angle=angle, load_pulses=True, progress=False, debug=debug)        

            for q_ind, q in enumerate(self.cfg.expt.qubits):
                data['avgi'][q_ind].append(avgi[adc_chs[q], 0])
                data['avgq'][q_ind].append(avgq[adc_chs[q], 0])
                data['amps'][q_ind].append(np.abs(avgi[adc_chs[q], 0]+1j*avgi[adc_chs[q], 0]))
                data['phases'][q_ind].append(np.angle(avgi[adc_chs[q], 0]+1j*avgi[adc_chs[q], 0]))

        for k, a in data.items():
            data[k] = np.array(a)
        self.data=data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data
        pass
            

    def display(self, data=None, fit=True, plot_freq=None, plot_gain=None, **kwargs):
        if data is None:
            data=self.data 

        inner_sweep = data['gainpts']
        outer_sweep = data['freqpts']

        y_sweep = outer_sweep
        x_sweep = inner_sweep

        plt.figure(figsize=(14,8))
        plt.suptitle(f"Eg-Gf Chevron Frequency vs. Gain")


        plt.subplot(221, title=f'Qubit A ({self.cfg.expt.qubits[0]})', ylabel="Pulse Frequency [MHz]")
        plt.pcolormesh(x_sweep, y_sweep, data['avgi'][0], cmap='viridis', shading='auto')
        if plot_freq is not None: plt.axhline(plot_freq, color='r')
        if plot_gain is not None: plt.axvline(plot_gain, color='r')
        plt.colorbar(label='I [ADC level]')

        plt.subplot(223, xlabel="Gain [DAC units]", ylabel="Pulse Frequency [MHz]")
        plt.pcolormesh(x_sweep, y_sweep, data['avgq'][0], cmap='viridis', shading='auto')
        if plot_freq is not None: plt.axhline(plot_freq, color='r')
        if plot_gain is not None: plt.axvline(plot_gain, color='r')
        plt.colorbar(label='Q [ADC level]')


        plt.subplot(222, title=f'Qubit B ({self.cfg.expt.qubits[1]})')
        plt.pcolormesh(x_sweep, y_sweep, data['avgi'][1], cmap='viridis', shading='auto')
        if plot_freq is not None: plt.axhline(plot_freq, color='r')
        if plot_gain is not None: plt.axvline(plot_gain, color='r')
        plt.colorbar(label='I [ADC level]')

        plt.subplot(224, xlabel="Gain [DAC units]")
        plt.pcolormesh(x_sweep, y_sweep, data['avgq'][1], cmap='viridis', shading='auto')
        if plot_freq is not None: plt.axhline(plot_freq, color='r')
        if plot_gain is not None: plt.axvline(plot_gain, color='r')
        plt.colorbar(label='Q [ADC level]')


        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname