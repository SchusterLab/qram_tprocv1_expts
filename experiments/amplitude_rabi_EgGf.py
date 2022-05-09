import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm


class AmplitudeRabiEgGfProgram(RAveragerProgram):
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

        # drive is applied on qubit B
        self.r_gainB = self.sreg(self.qubit_chs[qB], "gain") # get gain register for qubit_chs    
        self.r_gainB_update = 4 # register to hold the current sweep gain
        # initialize gain
        self.safe_regwi(self.q_rps[qB], self.r_gainB_update, self.cfg.expt.start)
        
        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
        
        self.pi_sigmaA = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qA])
        self.pi_ef_sigmaB = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma[qB])
        self.sigma_test = self.us2cycles(cfg.expt.sigma_test) # defaults to eg-gf sigma in config
        
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
        if cfg.expt.sigma_test > 0:
            if cfg.expt.pulse_type.lower() == "gauss":
                self.set_pulse_registers(
                    ch=self.qubit_chs[qB],
                    style="arb",
                    freq=self.f_EgGf,
                    phase=0,
                    gain=0, # gain set by update
                    waveform="pi_EgGf_qubitB")
            else:
                self.set_pulse_registers(
                    ch=self.qubit_chs[qB],
                    style="const",
                    freq=self.f_EgGf,
                    phase=0,
                    gain=0, # gain set by update
                    length=self.sigma_test)
            self.mathi(self.q_rps[qB], self.r_gainB, self.r_gainB_update, "+", 0)
            self.pulse(ch=self.qubit_chs[qB])
        self.sync_all()

        # take qubit A G->E and qubit B f->e: expect to end in Ee (or Gg if incomplete Eg-Gf)
        self.setup_and_pulse(ch=self.qubit_chs[qA], style="arb", freq=self.f_ge[qA], phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain[qA], waveform="pi_qubitA")
        self.sync_all()
        self.setup_and_pulse(ch=self.qubit_chs[qB], style="arb", freq=self.f_ef[qB], phase=0, gain=cfg.device.qubit.pulses.pi_ef.gain[qB], waveform="pi_ef_qubitB")
        
        self.sync_all(self.us2cycles(0.01)) # align channels and wait 10ns
        self.measure(pulse_ch=self.res_chs, 
             adcs=[0,1],
             adc_trig_offset=cfg.device.readout.trig_offset,
             wait=True,
             syncdelay=self.us2cycles(max([cfg.device.readout.relax_delay[q] for q in self.qubits])))
        
    def update(self):
        qA, qB = self.qubits
        self.mathi(self.q_rps[qB], self.r_gainB_update, self.r_gainB_update, '+', self.cfg.expt.step) # update test gain
                      
                      
class AmplitudeRabiEgGfExperiment(Experiment):
    """
    Amplitude Rabi Eg<->Gf Experiment
    Experimental Config:
    expt = dict(
        start: qubit gain [dac level]
        step: gain step [dac level]
        expts: number steps
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        sigma_test: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'const'
       singleshot: (optional) if true, uses threshold
    )
    """

    def __init__(self, soccfg=None, path='', prefix='AmplitudeRabiEgGf', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        qA, qB = self.cfg.expt.qubits
        adcA_ch = self.cfg.hw.soc.adcs.readout.ch[qA]
        adcB_ch = self.cfg.hw.soc.adcs.readout.ch[qB]
        
        if 'sigma_test' not in self.cfg.expt:
            self.cfg.expt.sigma_test = self.cfg.device.qubit.pulses.pi_EgGf.sigma

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
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
            # Remove the first and last point from fit in case weird edge measurements
            pA_I = dsfit.fitdecaysin(data['xpts'][0:-1], data["avgi"][0][0:-1], fitparams=None, showfit=False)
            pA_Q = dsfit.fitdecaysin(data['xpts'][0:-1], data["avgq"][0][0:-1], fitparams=None, showfit=False)
            pB_I = dsfit.fitdecaysin(data['xpts'][0:-1], data["avgi"][1][0:-1], fitparams=None, showfit=False)
            pB_Q = dsfit.fitdecaysin(data['xpts'][0:-1], data["avgq"][1][0:-1], fitparams=None, showfit=False)
            # adding this due to extra parameter in decaysin that is not in fitdecaysin
            data['fitA_avgi'] = np.append(pA_I, data['xpts'][0])
            data['fitA_avgq'] = np.append(pA_Q, data['xpts'][0])
            data['fitB_avgi'] = np.append(pB_I, data['xpts'][0])
            data['fitB_avgq'] = np.append(pB_Q, data['xpts'][0])
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 

        plt.figure(figsize=(14,8))
        plt.suptitle(f"Amplitude Rabi Eg-Gf (Drive Length {self.cfg.expt.sigma_test} us)")

        if self.cfg.expt.singleshot: plt.subplot(221, title='Qubit A', ylabel=r"Probability of $|e\rangle$")
        else: plt.subplot(221, title='Qubit A', ylabel="I [adc level]")
        plt.plot(data["xpts"][0:-1], data["avgi"][0][0:-1],'o-')
        if fit:
            plt.plot(data["xpts"][0:-1], dsfit.decaysin(data["fitA_avgi"], data["xpts"][0:-1]))
            pi_gain = 1/data['fitA_avgi'][1]/2
            print(f'Pi gain from avgi data (qubit A) [dac units]: {int(pi_gain)}')
            print(f'\tPi/2 gain from avgi data (qubit A) [dac units]: {int(pi_gain/2)}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi_gain/2, color='0.2', linestyle='--')
        plt.subplot(223, xlabel="Gain [dac units]", ylabel="Q [adc levels]")
        plt.plot(data["xpts"][0:-1], data["avgq"][0][0:-1],'o-')
        if fit:
            plt.plot(data["xpts"][0:-1], dsfit.decaysin(data["fitA_avgq"], data["xpts"][0:-1]))
            pi_gain = 1/data['fitA_avgq'][1]/2
            print(f'Pi gain from avgq data (qubit A) [dac units]: {int(pi_gain)}')
            print(f'\tPi/2 gain from avgq data (qubit A) [dac units]: {int(pi_gain/2)}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi_gain/2, color='0.2', linestyle='--')

        plt.subplot(222, title='Qubit B')
        plt.plot(data["xpts"][0:-1], data["avgi"][1][0:-1],'o-')
        if fit:
            plt.plot(data["xpts"][0:-1], dsfit.decaysin(data["fitB_avgi"], data["xpts"][0:-1]))
            pi_gain = 1/data['fitB_avgi'][1]/2
            print()
            print(f'Pi gain from avgi data (qubit B) [dac units]: {int(pi_gain)}')
            print(f'\tPi/2 gain from avgi data (qubit B) [dac units]: {int(pi_gain/2)}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi_gain/2, color='0.2', linestyle='--')
        plt.subplot(224, xlabel="Gain [dac units]")
        plt.plot(data["xpts"][0:-1], data["avgq"][1][0:-1],'o-')
        if fit:
            plt.plot(data["xpts"][0:-1], dsfit.decaysin(data["fitB_avgq"], data["xpts"][0:-1]))
            pi_gain = 1/data['fitB_avgq'][1]/2
            print(f'Pi gain from avgq data (qubit B) [dac units]: {int(pi_gain)}')
            print(f'\tPi/2 gain from avgq data (qubit B) [dac units]: {int(pi_gain/2)}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi_gain/2, color='0.2', linestyle='--')
        plt.tight_layout()
        plt.show()
        
        # plt.figure(figsize=(10,8))
        # plt.subplot(211,title="Amplitude Rabi Eg-Gf", ylabel="Amp Qubit A [adc level]")
        # plt.plot(data["xpts"][1:-1], data["amps"][0][1:-1], 'o-')
        
        # if fit:
        #     plt.plot(data["xpts"][1:-1], dsfit.decaysin(data["fitA"], data["xpts"][1:-1]))
        #     pi_gain = 1/data['fitA'][1]/2
        #     print(f'Pi gain [dac units]: {int(pi_gain)}')
        #     print(f'Pi/2 gain [dac units]: {int(pi_gain/2)}')
        #     plt.axvline(pi_gain, color='0.2', linestyle='--')
        #     plt.axvline(pi_gain/2, color='0.2', linestyle='--')
            
        # plt.subplot(212, xlabel='Gain [dac units]', ylabel='Amp Qubit B [adc level]' )
        # plt.plot(data["xpts"][1:-1], data["amps"][1][1:-1], 'o-')
        
        # if fit:
        #     plt.plot(data["xpts"][1:-1], dsfit.decaysin(data["fitB"], data["xpts"][1:-1]))
        # plt.tight_layout()
        # plt.show()
    
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)