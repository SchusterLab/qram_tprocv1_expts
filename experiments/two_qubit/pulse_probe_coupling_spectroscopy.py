import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter

class PulseProbeCouplingSpectroscopyProgram(RAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
        
        super().__init__(soccfg, self.cfg)

    """
    Qubit A: E<->G
    Qubit B: g<->f
    Drive applied on qubit B
    """
    def initialize(self):
        cfg=AttrDict(self.cfg)
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

        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs] # get register page for qubit_chs
        self.f_ge_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ge, self.qubit_chs)]
        self.f_res_reg = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(self.cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [1+self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)]

        gen_chs = []
        # declare res dacs
        mask = None
        if self.res_ch_types[0] == 'mux4': # only supports having all resonators be on mux, or none
            assert np.all([ch == 6 for ch in self.res_chs])
            mask = range(4) # indices of mux_freqs, mux_gains list to play
            # mux_freqs = [0 if i not in self.qubits else cfg.device.readout.frequency[i] for i in range(4)]
            # mux_gains = [0 if i not in self.qubits else cfg.device.readout.gain[i] for i in range(4)]
            mux_freqs = [cfg.device.readout.frequency[qA], 0, 0, 0]
            mux_gains = [cfg.device.readout.gain[qA], 0, 0, 0]
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

        # declare adc for qA
        self.declare_readout(ch=self.adc_chs[qA], length=self.readout_lengths_adc[qA], freq=cfg.device.readout.frequency[qA], gen_ch=self.res_chs[qA])

        # sweep drive is applied on qA
        self.r_freq_A = self.sreg(self.qubit_chs[qA], "freq")
        self.r_freq_A_update = 4 # register to hold the current sweep frequency

        self.pi_sigmaB = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qB], gen_ch=self.qubit_chs[qB])

        self.f_start = self.freq2reg(cfg.expt.start, gen_ch=self.qubit_chs[qA])
        self.f_step = self.freq2reg(cfg.expt.step, gen_ch=self.qubit_chs[qA])

        # send start frequency to r_freq_A_update
        self.safe_regwi(self.q_rps[qA], self.r_freq_A_update, self.f_start)

        # add qubit pulses to respective channels
        if self.cfg.expt.pulse_type == 'flat_top':
            self.add_gauss(ch=self.qubit_chs[qA], name="qubit", sigma=3, length=3*4)
        elif self.cfg.expt.pulse_type == 'gauss':
            length = self.us2cycles(cfg.expt.length, gen_ch=self.qubit_chs[qA])
            self.add_gauss(ch=self.qubit_chs[qA], name="qubit", sigma=length, length=length*4)

        self.add_gauss(ch=self.qubit_chs[qB], name="pi_qubitB", sigma=self.pi_sigmaB, length=self.pi_sigmaB*4)

        # add readout pulses to respective channels
        if self.res_ch_types[0] == 'mux4':
            self.set_pulse_registers(ch=6, style="const", length=max(self.readout_lengths_dac), mask=mask)
        else:
            self.set_pulse_registers(ch=self.res_chs[qA], style="const", freq=self.f_res_reg[qA], phase=0, gain=cfg.device.readout.gain[qA], length=self.readout_lengths_dac[qA])

        self.sync_all(200)

    def body(self):
        cfg=AttrDict(self.cfg)
        qA, qB = self.qubits
 
        # initialize qubit B to E
        if self.cfg.expt.pulseB:
            self.setup_and_pulse(ch=self.qubit_chs[qB], style="arb", phase=0, freq=self.f_ge_reg[qB], gain=cfg.device.qubit.pulses.pi_ge.gain[qB], waveform="pi_qubitB")
            self.sync_all(5)

        # sweep qubit A frequency
        length = self.us2cycles(cfg.expt.length, gen_ch=self.qubit_chs[qA])
        if self.cfg.expt.pulse_type == 'flat_top':
            self.set_pulse_registers(ch=self.qubit_chs[qA], style="flat_top", phase=0, freq=self.f_start, gain=cfg.expt.gain, length=length, waveform="qubit") # play probe pulse
        elif self.cfg.expt.pulse_type == 'gauss':
            self.set_pulse_registers(ch=self.qubit_chs[qA], style="arb", phase=0, freq=self.f_start, gain=cfg.expt.gain, waveform="qubit") # play probe pulse
        elif self.cfg.expt.pulse_type == 'const':
            self.set_pulse_registers(ch=self.qubit_chs[qA], style="const", freq=self.f_start, phase=0, gain=cfg.expt.gain, length=self.us2cycles(cfg.expt.length, gen_ch=self.qubit_chs[qA]))
        self.mathi(self.q_rps[qA], self.r_freq_A, self.r_freq_A_update, "+", 0)
        self.pulse(ch=self.qubit_chs[qA])
        self.sync_all(5)

        self.sync_all(5)
        measure_chs = self.res_chs
        if self.res_ch_types[0] == 'mux4': measure_chs = self.res_chs[0]
        self.measure(
            pulse_ch=measure_chs, 
            adcs=[self.adc_chs[qA]],
            adc_trig_offset=cfg.device.readout.trig_offset[0],
            wait=True,
            syncdelay=self.us2cycles(max([cfg.device.readout.relax_delay[q] for q in self.qubits])))

    def update(self):
        qA, qB = self.qubits
        self.mathi(self.q_rps[qA], self.r_freq_A_update, self.r_freq_A_update, '+', self.f_step) # update frequency list index
        

class PulseProbeCouplingSpectroscopyExperiment(Experiment):
    """
    Pulse Probe Eg-Gf Spectroscopy Experiment
    Experimental Config:
    expt = dict(
        start: start ef probe frequency [MHz]
        step: step ef probe frequency
        expts: number experiments stepping from start
        reps: number averages per experiment
        rounds: number repetitions of experiment sweep
        pulseB: True/False if apply the pulse on qubit B
        length: ef const pulse length [us]
        gain: ef const pulse gain [dac units]
        qubits: [qA, qB], sweep qA and optionally apply pulse on qB
    )
    """

    def __init__(self, soccfg=None, path='', prefix='PulseProbeCouplingSpectroscopy', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

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

        qspec = PulseProbeCouplingSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
        xpts, avgi, avgq = qspec.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug)        

        avgi = avgi[0][0]
        avgq = avgq[0][0]
        amps = np.abs(avgi+1j*avgq)
        phases = np.angle(avgi+1j*avgq) # Calculating the phase        
        
        data={'xpts':xpts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases}
        self.data=data
        return data

    def analyze(self, data=None, fit=True, signs=[1,1], **kwargs):
        if data is None:
            data=self.data
        if fit:
            xdata = data['xpts'][1:-1]
            data['fit_amps'], data['fit_err_amps'] = fitter.fitlor(xdata, signs[0]*data['amps'][1:-1])
            data['fit_avgi'], data['fit_err_avgi'] = fitter.fitlor(xdata, signs[1]*data['avgi'][1:-1])
            data['fit_avgq'], data['fit_err_avgq'] = fitter.fitlor(xdata, signs[2]*data['avgq'][1:-1])
        return data

    def display(self, data=None, fit=True, signs=[1,1], **kwargs):
        if data is None:
            data=self.data 

        qA, qB = self.cfg.expt.qubits
        xpts = self.cfg.hw.soc.dacs.qubit.mixer_freq[qA] + data['xpts'][1:-1]

        plt.figure(figsize=(9, 11))
        plt.subplot(311, title=f"Qubit Spectroscopy on Q{qA} with Pi Pulse on Q{qB}", xlabel="Pulse Frequency [MHz]", ylabel="Amplitude [ADC units]")
        plt.plot(xpts, data["amps"][1:-1],'o-')
        if fit:
            plt.plot(xpts, signs[0]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_amps"]))
            print(f'Found peak in amps at [MHz] {data["fit_amps"][2]}, HWHM {data["fit_amps"][3]}')

        plt.subplot(312, title=f"Qubit {qA} Spectroscopy (Q{qB} in {'e' if self.cfg.expt.pulseB else 'g'})", ylabel="I [ADC units]")
        plt.plot(xpts, data["avgi"][1:-1],'o-')
        if fit:
            plt.plot(xpts, signs[1]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_avgi"]))
            print(f'Found peak in I at [MHz] {data["fit_avgi"][2]}, HWHM {data["fit_avgi"][3]}')
        plt.subplot(313, xlabel="Pulse Frequency (MHz)", ylabel="Q [ADC units]")
        plt.plot(xpts, data["avgq"][1:-1],'o-')
        if fit:
            plt.plot(xpts, signs[2]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_avgq"]))
            # plt.axvline(4828.13, c='k', ls='--')
            print(f'Found peak in Q at [MHz] {data["fit_avgq"][2]}, HWHM {data["fit_avgq"][3]}')

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname