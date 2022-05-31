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
    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        
        self.qubits = self.cfg.expt.qubits
        qA, qB = self.qubits
        
        # all of these saved self.whatever instance variables should be indexed by the actual qubit number. this means that more values are saved as instance variables than is strictly necessary, but this is overall less confusing
        self.res_chs = self.cfg.hw.soc.dacs.readout.ch
        self.qubit_chs = self.cfg.hw.soc.dacs.qubit.ch
        self.adc_chs = self.cfg.hw.soc.adcs.readout.ch

        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs] # get register page for qubit_chs
        self.f_ge = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ge, self.qubit_chs)]
        self.f_ef = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ef, self.qubit_chs)]
        self.f_res = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.readout.frequency, self.res_chs)]
        self.f_EgGf = self.freq2reg(cfg.device.qubit.f_EgGf, gen_ch=self.qubit_chs[qB])
        self.readout_length = [self.us2cycles(len) for len in self.cfg.device.readout.readout_length]

        for q in self.qubits:
            self.declare_gen(ch=self.res_chs[q], nqz=self.cfg.hw.soc.dacs.readout.nyquist[q])
            self.declare_gen(ch=self.qubit_chs[q], nqz=self.cfg.hw.soc.dacs.qubit.nyquist[q])
            self.declare_readout(ch=self.adc_chs[q], length=self.readout_length[q], freq=self.cfg.device.readout.frequency[q], gen_ch=self.res_chs[q])
        
        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps

        self.pi_sigmaA = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qA])
        self.pi_ef_sigmaB = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma[qB])

        # update sigma in outer loop over averager program
        self.sigma_test = self.us2cycles(cfg.expt.sigma_test)
        # print(self.sigma_test)

        # add qubit and readout pulses to respective channels
        self.add_gauss(ch=self.qubit_chs[qA], name="pi_qubitA", sigma=self.pi_sigmaA, length=self.pi_sigmaA*4)
        self.add_gauss(ch=self.qubit_chs[qB], name="pi_ef_qubitB", sigma=self.pi_ef_sigmaB, length=self.pi_ef_sigmaB*4)
        if cfg.expt.pulse_type.lower() == "gauss" and cfg.expt.sigma_test > 0:
            self.add_gauss(ch=self.qubit_chs[qB], name="pi_EgGf_qubitB", sigma=self.sigma_test, length=self.sigma_test*4)

        for q in self.qubits:
            self.set_pulse_registers(ch=self.res_chs[q], style="const", freq=self.f_res[q], phase=self.deg2reg(cfg.device.readout.phase[q], gen_ch=self.res_chs[q]), gain=cfg.device.readout.gain[q], length=self.readout_length[q])

        self.sync_all(self.us2cycles(0.2))
    
    def body(self):
        cfg=AttrDict(self.cfg)
        qA, qB = self.qubits

        # initialize qubit A to E: expect to end in Eg
        self.setup_and_pulse(ch=self.qubit_chs[qA], style="arb", phase=0, freq=self.f_ge[qA], gain=cfg.device.qubit.pulses.pi_ge.gain[qA], waveform="pi_qubitA")
        self.sync_all()

        # apply Eg -> Gf pulse on B: expect to end in Gf
        if self.sigma_test > 0:
            if cfg.expt.pulse_type.lower() == "gauss":
                self.setup_and_pulse(ch=self.qubit_chs[qB], style="arb", freq=self.f_EgGf, phase=0, gain=cfg.expt.gain, waveform="pi_EgGf_qubitB")
            else:
                self.setup_and_pulse(ch=self.qubit_chs[qB], style="const", freq=self.f_EgGf, phase=0, gain=cfg.expt.gain, length=self.sigma_test)
        self.sync_all()

        # take qubit A G->E and qubit B f->e: expect to end in Ee (or Gg if incomplete Eg-Gf)
        self.setup_and_pulse(ch=self.qubit_chs[qA], style="arb", freq=self.f_ge[qA], phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain[qA], waveform="pi_qubitA")
        self.setup_and_pulse(ch=self.qubit_chs[qB], style="arb", freq=self.f_ef[qB], phase=0, gain=cfg.device.qubit.pulses.pi_ef.gain[qB], waveform="pi_ef_qubitB")
        
        self.sync_all(self.us2cycles(0.01)) # align channels and wait 10ns
        self.measure(pulse_ch=self.res_chs, 
             adcs=[0,1],
             adc_trig_offset=cfg.device.readout.trig_offset,
             wait=True,
             syncdelay=self.us2cycles(max([cfg.device.readout.relax_delay[q] for q in self.qubits])))

        
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
       singleshot: (optional) if true, uses threshold
    )
    """

    def __init__(self, soccfg=None, path='', prefix='LengthRabiEgGf', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        qA, qB = self.cfg.expt.qubits
        adcA_ch = self.cfg.hw.soc.adcs.readout.ch[qA]
        adcB_ch = self.cfg.hw.soc.adcs.readout.ch[qB]
        
        lengths = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        
        data={"xpts":[], "avgi":[[],[]], "avgq":[[],[]], "amps":[[],[]], "phases":[[],[]]}

        threshold = None
        if 'singleshot' in self.cfg.expt.keys():
            if self.cfg.expt.singleshot: threshold = self.cfg.device.readout.threshold
        
        for length in tqdm(lengths, disable=not progress):
            self.cfg.expt.sigma_test = float(length)
            lengthrabi = LengthRabiEgGfProgram(soccfg=self.soccfg, cfg=self.cfg)
            avgi, avgq = lengthrabi.acquire(self.im[self.cfg.aliases.soc], threshold=threshold, load_pulses=True, progress=False, debug=debug)        
            # print(avgi, avgq)
            data['avgi'][0].append(avgi[adcA_ch][0])
            data['avgi'][1].append(avgi[adcB_ch][0])
            data['avgq'][0].append(avgq[adcA_ch][0])
            data['avgq'][1].append(avgq[adcB_ch][0])
            data['amps'][0].append(np.abs(avgi[adcA_ch][0]+1j*avgi[adcA_ch][0]))
            data['amps'][1].append(np.abs(avgi[adcB_ch][0]+1j*avgi[adcB_ch][0]))
            data['phases'][0].append(np.angle(avgi[adcA_ch][0]+1j*avgi[adcA_ch][0]))
            data['phases'][1].append(np.angle(avgi[adcB_ch][0]+1j*avgi[adcB_ch][0]))
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
            pB_avgq, pCovB_avgq = fitter.fitdecaysin(data['xpts'][:-1], data["avgq"][1][:-1], fitparams=None)
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
        # plt.suptitle(f"Length Rabi Eg-Gf") # (Drive Gain {self.cfg.expt.gain})")
        # plt.subplot(121, title="Qubit A", ylabel="Amplitude [adc level]", xlabel='Length [ns]')
        # plt.plot(xpts_ns[0:-1], data["amps"][0][0:-1],'o-')
        # if fit:
        #     p = data['fitA_amps']
        #     plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
        #     pi_length = 1/p[1]/2
        #     print(f'Pi length from amps data (qubit A) [us]: {int(pi_length)}')
        #     print(f'\tPi/2 length from amps data (qubit A) [us]: {int(pi_length/2)}')
        #     plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
        #     plt.axvline(pi_length*1e3/2, color='0.2', linestyle='--')
        # plt.subplot(122, title="Qubit B", xlabel='Length[ns]')
        # plt.plot(xpts_ns[0:-1], data["amps"][1][0:-1],'o-')
        # if fit:
        #     p = data['fitB_amps']
        #     plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
        #     pi_length = 1/p[1]/2
        #     print()
        #     print(f'Pi length from amps data (qubit B) [us]: {int(pi_length)}')
        #     print(f'\tPi/2 length from amps data (qubit B) [us]: {int(pi_length/2)}')
        #     plt.axvline(pi_length, color='0.2', linestyle='--')
        #     plt.axvline(pi_length/2, color='0.2', linestyle='--')

        plt.figure(figsize=(14,8))
        plt.suptitle(f"Length Rabi Eg-Gf") # (Drive Gain {self.cfg.expt.gain})")
        plt.subplot(221, title="Qubit A", ylabel="I [adc level]")
        plt.plot(xpts_ns[0:-1], data["avgi"][0][0:-1],'o-')
        if fit:
            p = data['fitA_avgi']
            plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
            pi_length = 1/p[1]/2
            print(f'Pi length from avgi data (qubit A) [us]: {int(pi_length)}')
            print(f'\tPi/2 length from avgi data (qubit A) [us]: {int(pi_length/2)}')
            plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
            plt.axvline(pi_length*1e3/2, color='0.2', linestyle='--')
        plt.subplot(223, xlabel="Length [ns]", ylabel="Q [adc levels]")
        plt.plot(xpts_ns[0:-1], data["avgq"][0][0:-1],'o-')
        if fit:
            p = data['fitA_avgq']
            plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
            pi_length = 1/p[1]/2
            pi_length = 1/data['fitA_avgq'][1]/2
            print(f'Pi length from avgq data (qubit A) [us]: {int(pi_length)}')
            print(f'\tPi/2 length from avgq data (qubit A) [us]: {int(pi_length/2)}')
            plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
            plt.axvline(pi_length/2, color='0.2', linestyle='--')

        plt.subplot(222, title="Qubit B")
        plt.plot(xpts_ns[0:-1], data["avgi"][1][0:-1],'o-')
        if fit:
            p = data['fitB_avgi']
            plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
            pi_length = 1/p[1]/2
            print()
            print(f'Pi length from avgi data (qubit B) [us]: {int(pi_length)}')
            print(f'\tPi/2 length from avgi data (qubit B) [us]: {int(pi_length/2)}')
            plt.axvline(pi_length, color='0.2', linestyle='--')
            plt.axvline(pi_length/2, color='0.2', linestyle='--')
        plt.subplot(224, xlabel="Length [ns]")
        plt.plot(xpts_ns[0:-1], data["avgq"][1][0:-1],'o-')
        if fit:
            p = data['fitB_avgq']
            plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
            pi_length = 1/p[1]/2
            print(f'Pi length from avgq data (qubit B) [us]: {int(pi_length)}')
            print(f'\tPi/2 length from avgq data (qubit B) [us]: {int(pi_length/2)}')
            plt.axvline(pi_length, color='0.2', linestyle='--')
            plt.axvline(pi_length/2, color='0.2', linestyle='--')

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)