import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

import experiments.fitting as fitter

"""
Measures Rabi oscillations by sweeping over the duration of the qubit drive pulse. This is a preliminary measurement to prove that we see Rabi oscillations. This measurement is followed up by the Amplitude Rabi experiment.
"""
class LengthRabiEgGfProgram(AveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        
        super().__init__(soccfg, self.cfg)

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

        # declare swap dac indexed by qA (since the the drive is always applied to qB)
        mixer_freq = 0
        if self.swap_ch_types[qA] == 'int4':
            mixer_freq = cfg.hw.soc.dacs.swap.mixer_freq[qA]
        if self.swap_chs[qA] not in gen_chs: 
            self.declare_gen(ch=self.swap_chs[qA], nqz=cfg.hw.soc.dacs.swap.nyquist[qA], mixer_freq=mixer_freq)
        gen_chs.append(self.swap_chs[qA])

        # declare adcs - readout for all qubits everytime, defines number of buffers returned regardless of number of adcs triggered
        for q in range(self.num_qubits_sample):
            self.declare_readout(ch=self.adc_chs[q], length=self.readout_lengths_adc[q], freq=cfg.device.readout.frequency[q], gen_ch=self.res_chs[q])

        self.pi_sigmaA = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qA], gen_ch=self.qubit_chs[qA])
        self.pi_ef_sigmaB = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma[qB], self.qubit_chs[qB])

        # update sigma in outer loop over averager program
        self.sigma_test = self.us2cycles(cfg.expt.sigma_test, gen_ch=self.swap_chs[qA])

        # add qubit pulses to respective channels
        self.add_gauss(ch=self.qubit_chs[qA], name="pi_qubitA", sigma=self.pi_sigmaA, length=self.pi_sigmaA*4)
        self.add_gauss(ch=self.qubit_chs[qB], name="pi_ef_qubitB", sigma=self.pi_ef_sigmaB, length=self.pi_ef_sigmaB*4)
        if cfg.expt.pulse_type.lower() == "gauss" and cfg.expt.sigma_test > 0:
            self.add_gauss(ch=self.swap_chs[qA], name="pi_EgGf_swap", sigma=self.sigma_test, length=self.sigma_test*4)

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
        if self.sigma_test > 0:
            if cfg.expt.pulse_type.lower() == "gauss":
                self.setup_and_pulse(ch=self.swap_chs[qA], style="arb", freq=self.f_EgGf_reg, phase=0, gain=cfg.expt.gain, waveform="pi_EgGf_swap")
            else:
                self.setup_and_pulse(ch=self.swap_chs[qA], style="const", freq=self.f_EgGf_reg, phase=0, gain=cfg.expt.gain, length=self.sigma_test)
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

# ===================================================================== #
        
class LengthRabiEgGfExperiment(Experiment):
    """
    Length Rabi EgGf Experiment
    Experimental Config
    expt = dict(
        start: start length [us],
        step: length step, 
        expts: number of different length experiments, 
        reps: number of reps,
        gain: gain to use for the qubit pulse
        pulse_type: 'gauss' or 'const'
        qubits: qubit 0 goes E->G, apply drive on qubit 1 (g->f)
        singleshot: (optional) if true, uses threshold
    )
    """

    def __init__(self, soccfg=None, path='', prefix='LengthRabiEgGf', config_file=None, progress=None):
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
        
        lengths = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        
        data={"xpts":[], "avgi":[[],[]], "avgq":[[],[]], "amps":[[],[]], "phases":[[],[]]}

        threshold = None
        if 'singleshot' in self.cfg.expt.keys():
            if self.cfg.expt.singleshot: threshold = self.cfg.device.readout.threshold

        if 'gain' not in self.cfg.expt:
            self.cfg.expt.gain = self.cfg.device.qubit.pulses.pi_EgGf.gain[qA]
        
        for length in tqdm(lengths, disable=not progress):
            self.cfg.expt.sigma_test = float(length)
            lengthrabi = LengthRabiEgGfProgram(soccfg=self.soccfg, cfg=self.cfg)
            avgi, avgq = lengthrabi.acquire(self.im[self.cfg.aliases.soc], threshold=threshold, load_pulses=True, progress=False, debug=debug)        
            # print(avgi[:,0], avgq[:,0])
            # print()
            data['avgi'][0].append(avgi[adcA_ch, 0])
            data['avgi'][1].append(avgi[adcB_ch, 0])
            data['avgq'][0].append(avgq[adcA_ch, 0])
            data['avgq'][1].append(avgq[adcB_ch, 0])
            data['amps'][0].append(np.abs(avgi[adcA_ch, 0]+1j*avgi[adcA_ch, 0]))
            data['amps'][1].append(np.abs(avgi[adcB_ch, 0]+1j*avgi[adcB_ch, 0]))
            data['phases'][0].append(np.angle(avgi[adcA_ch, 0]+1j*avgi[adcA_ch, 0]))
            data['phases'][1].append(np.angle(avgi[adcB_ch, 0]+1j*avgi[adcB_ch, 0]))
            data['xpts'].append(length)

        for k, a in data.items():
            data[k] = np.array(a)
        
        self.data = data

        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data
        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
            # Remove the first and last point from fit in case weird edge measurements
            fitparams = None

            pA_avgi, pCovA_avgi = fitter.fitdecaysin(data['xpts'][:-1], data["avgi"][0][:-1], fitparams=None)
            pA_avgq, pCovA_avgq = fitter.fitdecaysin(data['xpts'][:-1], data["avgq"][0][:-1], fitparams=None)
            pA_amps, pCovA_amps = fitter.fitdecaysin(data['xpts'][:-1], data["amps"][0][:-1], fitparams=None)
            data['fitA_avgi'] = pA_avgi   
            data['fitA_avgq'] = pA_avgq
            data['fitA_amps'] = pA_amps
            data['fitA_err_avgi'] = pCovA_avgi   
            data['fitA_err_avgq'] = pCovA_avgq
            data['fitA_err_amps'] = pCovA_amps

            pB_avgi, pCovB_avgi = fitter.fitdecaysin(data['xpts'][:-1], data["avgi"][1][:-1], fitparams=None)
            # fitparams = [20, 1/0.6, None, None, None, None]
            pB_avgq, pCovB_avgq = fitter.fitdecaysin(data['xpts'][:-1], data["avgq"][1][:-1], fitparams=fitparams)
            pB_amps, pCovB_amps = fitter.fitdecaysin(data['xpts'][:-1], data["amps"][1][:-1], fitparams=None)
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

        xpts_ns = data['xpts']*1e3

        # plt.figure(figsize=(18,6))
        # plt.suptitle(f"Length Rabi (Drive Gain {self.cfg.expt.gain})")
        # plt.subplot(121, title=f'Qubit A ({self.cfg.expt.qubits[0]})', ylabel="Amplitude [adc level]", xlabel='Length [ns]')
        # plt.plot(xpts_ns[0:-1], data["amps"][0][0:-1],'o-')
        # if fit:
        #     p = data['fitA_amps']
        #     plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
        #     if p[2] > 180: p[2] = p[2] - 360
        #     elif p[2] < -180: p[2] = p[2] + 360
        #     if p[2] < 0: pi_length = (1/2 - p[2]/180)/2/p[1]
        #     else: pi_length= (3/2 - p[2]/180)/2/p[1]
        #     pi2_length = pi_length/2
        #     print(f'Pi length from amps data (qubit A) [us]: {pi_length}')
        #     print(f'Pi/2 length from amps data (qubit A) [us]: {pi2_length}')
        #     plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
        #     plt.axvline(pi2_length*1e3, color='0.2', linestyle='--')
        # plt.subplot(122, title=f'Qubit B ({self.cfg.expt.qubits[1]})', xlabel='Length[ns]')
        # plt.plot(xpts_ns[0:-1], data["amps"][1][0:-1],'o-')
        # if fit:
        #     p = data['fitB_amps']
        #     plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
        #     if p[2] > 180: p[2] = p[2] - 360
        #     elif p[2] < -180: p[2] = p[2] + 360
        #     if p[2] < 0: pi_length = (1/2 - p[2]/180)/2/p[1]
        #     else: pi_length= (3/2 - p[2]/180)/2/p[1]
        #     pi2_length = pi_length/2
        #     print(f'Pi length from amps data (qubit B) [us]: {pi_length}')
        #     print(f'Pi/2 length from amps data (qubit B) [us]: {pi2_length}')
        #     plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
        #     plt.axvline(pi2_length*1e3, color='0.2', linestyle='--')

        plt.figure(figsize=(14,8))
        plt.suptitle(f"Length Rabi (Drive Gain {self.cfg.expt.gain})")
        plt.subplot(221, title=f'Qubit A ({self.cfg.expt.qubits[0]})', ylabel="I [adc level]")
        plt.plot(xpts_ns[0:-1], data["avgi"][0][0:-1],'o-')
        if fit:
            p = data['fitA_avgi']
            plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_length = (1/2 - p[2]/180)/2/p[1]
            else: pi_length= (3/2 - p[2]/180)/2/p[1]
            pi2_length = pi_length/2
            print(f'Pi length from avgi data (qubit A) [us]: {pi_length}')
            print(f'\tPi/2 length from avgi data (qubit A) [us]: {pi2_length}')
            plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
            plt.axvline(pi2_length*1e3, color='0.2', linestyle='--')
        plt.subplot(223, xlabel="Length [ns]", ylabel="Q [adc levels]")
        plt.plot(xpts_ns[0:-1], data["avgq"][0][0:-1],'o-')
        if fit:
            p = data['fitA_avgq']
            plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_length = (1/2 - p[2]/180)/2/p[1]
            else: pi_length= (3/2 - p[2]/180)/2/p[1]
            pi2_length = pi_length/2
            print(f'Pi length from avgq data (qubit A) [us]: {pi_length}')
            print(f'\tPi/2 length from avgq data (qubit A) [us]: {pi2_length}')
            plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
            plt.axvline(pi2_length*1e3, color='0.2', linestyle='--')

        plt.subplot(222, title=f'Qubit B ({self.cfg.expt.qubits[1]})')
        plt.plot(xpts_ns[0:-1], data["avgi"][1][0:-1],'o-')
        if fit:
            p = data['fitB_avgi']
            plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_length = (1/2 - p[2]/180)/2/p[1]
            else: pi_length= (3/2 - p[2]/180)/2/p[1]
            pi2_length = pi_length/2
            print(f'Pi length from avgi data (qubit B) [us]: {pi_length}')
            print(f'\tPi/2 length from avgi data (qubit B) [us]: {pi2_length}')
            plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
            plt.axvline(pi2_length*1e3, color='0.2', linestyle='--')
        plt.subplot(224, xlabel="Length [ns]")
        plt.plot(xpts_ns[0:-1], data["avgq"][1][0:-1],'o-')
        if fit:
            p = data['fitB_avgq']
            plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_length = (1/2 - p[2]/180)/2/p[1]
            else: pi_length= (3/2 - p[2]/180)/2/p[1]
            pi2_length = pi_length/2
            print(f'Pi length from avgq data (qubit B) [us]: {pi_length}')
            print(f'\tPi/2 length from avgq data (qubit B) [us]: {pi2_length}')
            plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
            plt.axvline(pi2_length*1e3, color='0.2', linestyle='--')

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)

# ===================================================================== #

class EgGfFreqLenChevronExperiment(Experiment):
    """
    Rabi Eg<->Gf Experiment Chevron sweeping freq vs. len
    Experimental Config:
    expt = dict(
        start_len: start length [us],
        step_len: length step, 
        expts_len: number of different length experiments, 
        start_f: start freq [MHz],
        step_f: freq step, 
        expts_f: number of different freq experiments, 
        gain: gain to use for the qubit pulse
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        pulse_type: 'gauss' or 'const'
    )
    """
    def __init__(self, soccfg=None, path='', prefix='RabiEgGfFreqLenChevron', config_file=None, progress=None):
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
        lenpts = self.cfg.expt.start_len + self.cfg.expt.step_len * np.arange(self.cfg.expt.expts_len)
        
        data={"lenpts":lenpts, "freqpts":freqpts, "avgi":[[],[]], "avgq":[[],[]], "amps":[[],[]], "phases":[[],[]]}

        self.cfg.expt.start = self.cfg.expt.start_len
        self.cfg.expt.step = self.cfg.expt.step_len
        self.cfg.expt.expts = self.cfg.expt.expts_len

        if 'gain' not in self.cfg.expt:
            self.cfg.expt.gain = self.cfg.device.qubit.pulses.pi_EgGf.gain[qA]

        threshold = None
        angle = None

        for freq in tqdm(freqpts, disable=not progress):
            self.cfg.device.qubit.f_EgGf[qA] = float(freq)

            for length in tqdm(lenpts, disable=True):
                self.cfg.expt.sigma_test = float(length)
                lenrabi = LengthRabiEgGfProgram(soccfg=self.soccfg, cfg=self.cfg)
                avgi, avgq = lenrabi.acquire(self.im[self.cfg.aliases.soc], threshold=threshold, angle=angle, load_pulses=True, progress=False, debug=debug)        

                for q_ind, q in enumerate(self.cfg.expt.qubits):
                    data['avgi'][q_ind].append(avgi[adc_chs[q], 0])
                    data['avgq'][q_ind].append(avgq[adc_chs[q], 0])
                    data['amps'][q_ind].append(np.abs(avgi[adc_chs[q], 0]+1j*avgi[adc_chs[q], 0]))
                    data['phases'][q_ind].append(np.angle(avgi[adc_chs[q], 0]+1j*avgi[adc_chs[q], 0]))

        for k, a in data.items():
            data[k] = np.array(a)
            if np.shape(data[k]) == (2, len(freqpts) * len(lenpts)):
                data[k] = np.reshape(data[k], (2, len(freqpts), len(lenpts)))
        self.data=data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data
        pass

    def display(self, data=None, fit=True, plot_freq=None, plot_len=None, **kwargs):
        if data is None:
            data=self.data 

        inner_sweep = data['lenpts']
        outer_sweep = data['freqpts']

        y_sweep = outer_sweep
        x_sweep = inner_sweep

        plt.figure(figsize=(14,8))
        plt.suptitle(f"Eg-Gf Chevron Frequency vs. Length")

        print(len(data['avgi'][0]))


        plt.subplot(221, title=f'Qubit A ({self.cfg.expt.qubits[0]})', ylabel="Pulse Frequency [MHz]")
        # plt.pcolormesh(x_sweep, y_sweep, np.reshape(data['avgi'][0], (len(outer_sweep), len(inner_sweep))), cmap='viridis', shading='auto')
        plt.pcolormesh(1e3*x_sweep, y_sweep, data['avgi'][0], cmap='viridis', shading='auto')
        if plot_freq is not None: plt.axhline(plot_freq, color='r')
        if plot_len is not None: plt.axvline(plot_len, color='r')
        plt.colorbar(label='I [ADC level]')

        plt.subplot(223, xlabel="Length [ns]", ylabel="Pulse Frequency [MHz]")
        plt.pcolormesh(1e3*x_sweep, y_sweep, data['avgq'][0], cmap='viridis', shading='auto')
        if plot_freq is not None: plt.axhline(plot_freq, color='r')
        if plot_len is not None: plt.axvline(plot_len, color='r')
        plt.colorbar(label='Q [ADC level]')


        plt.subplot(222, title=f'Qubit B ({self.cfg.expt.qubits[1]})')
        plt.pcolormesh(1e3*x_sweep, y_sweep, data['avgi'][1], cmap='viridis', shading='auto')
        if plot_freq is not None: plt.axhline(plot_freq, color='r')
        if plot_len is not None: plt.axvline(plot_len, color='r')
        plt.colorbar(label='I [ADC level]')

        plt.subplot(224, xlabel="Length [ns]")
        plt.pcolormesh(1e3*x_sweep, y_sweep, data['avgq'][1], cmap='viridis', shading='auto')
        if plot_freq is not None: plt.axhline(plot_freq, color='r')
        if plot_len is not None: plt.axvline(plot_len, color='r')
        plt.colorbar(label='Q [ADC level]')


        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname