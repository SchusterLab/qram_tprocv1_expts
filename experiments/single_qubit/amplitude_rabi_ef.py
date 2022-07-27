import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter

class AmplitudeRabiEFProgram(RAveragerProgram):
    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        
        self.adc_ch = cfg.hw.soc.adcs.readout.ch
        self.res_ch = cfg.hw.soc.dacs.readout.ch
        self.res_ch_type = cfg.hw.soc.dacs.readout.type
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type

        self.q_rp = self.ch_page(self.qubit_ch) # get register page for qubit_ch
        if self.qubit_ch_type == 'int4':
            self.r_gain = self.sreg(self.qubit_ch, "addr") # get gain register for qubit_ch    
        else: self.r_gain = self.sreg(self.qubit_ch, "gain") # get gain register for qubit_ch    
        self.r_gain2 = 4
        
        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_ch)
        self.f_ef = self.freq2reg(cfg.device.qubit.f_ef, gen_ch=self.qubit_ch)
        self.f_res_reg = self.freq2reg(cfg.device.readout.frequency, gen_ch=self.res_ch, ro_ch=self.adc_ch)
        self.readout_length_dac = self.us2cycles(cfg.device.readout.readout_length, gen_ch=self.res_ch)
        self.readout_length_adc = self.us2cycles(cfg.device.readout.readout_length, ro_ch=self.adc_ch)
        self.readout_length_adc += 1 # ensure the rounding of the clock ticks calculation doesn't mess up the buffer

        # declare res dacs
        mask = None
        mixer_freq = 0 # MHz
        mux_freqs = None # MHz
        mux_gains = None
        ro_ch = None
        if self.res_ch_type == 'int4':
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
        elif self.res_ch_type == 'mux4':
            assert self.res_ch == 6
            mask = [0, 1, 2, 3] # indices of mux_freqs, mux_gains list to play
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
            mux_freqs = [0]*4
            mux_freqs[cfg.expt.qubit] = cfg.device.readout.frequency
            mux_gains = [0]*4
            mux_gains[cfg.expt.qubit] = cfg.device.readout.gain
            ro_ch=self.adc_ch
        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist, mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)

        # declare qubit dacs
        mixer_freq = 0
        if self.qubit_ch_type == 'int4':
            mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist, mixer_freq=mixer_freq)

        # declare adcs
        self.declare_readout(ch=self.adc_ch, length=self.readout_length_adc, freq=cfg.device.readout.frequency, gen_ch=self.res_ch)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds

        self.pisigma_ge = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        self.sigma_test = self.us2cycles(cfg.expt.sigma_test, gen_ch=self.qubit_ch)

        # add qubit and readout pulses to respective channels
        self.add_gauss(ch=self.qubit_ch, name="pi_qubit", sigma=self.pisigma_ge, length=self.pisigma_ge*4)
        if cfg.expt.pulse_type.lower() == "gauss" and cfg.expt.sigma_test > 0:
            self.add_gauss(ch=self.qubit_ch, name="pi_ef", sigma=self.sigma_test, length=self.sigma_test*4)

        if self.res_ch_type == 'mux4':
            self.set_pulse_registers(ch=self.res_ch, style="const", length=self.readout_length_dac, mask=mask)
        else: self.set_pulse_registers(ch=self.res_ch, style="const", freq=self.f_res_reg, phase=0, gain=cfg.device.readout.gain, length=self.readout_length_dac)

        # initialize gain
        self.safe_regwi(self.q_rp, self.r_gain2, self.cfg.expt.start)

        self.sync_all(200)
    
    def body(self):
        cfg=AttrDict(self.cfg)
        
        # init to qubit excited state
        if self.cfg.expt.pulse_ge:
            self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain, waveform="pi_qubit")

        # play test ef pulse
        if cfg.expt.sigma_test > 0:
            if cfg.expt.pulse_type.lower() == "gauss":
                self.set_pulse_registers(
                    ch=self.qubit_ch,
                    style="arb",
                    freq=self.f_ef,
                    phase=0,
                    gain=0, # gain set by update
                    waveform="pi_ef")
            else:
                self.set_pulse_registers(
                    ch=self.qubit_ch,
                    style="const",
                    freq=self.f_ef,
                    phase=0,
                    gain=0, # gain set by update
                    length=self.sigma_test)
        self.mathi(self.q_rp, self.r_gain, self.r_gain2, "+", 0)
        self.pulse(ch=self.qubit_ch)

        # go back to ground state if not in f
        self.setup_and_pulse(
            ch=self.qubit_ch,
            style="arb",
            freq=self.f_ge,
            phase=0,
            gain=cfg.device.qubit.pulses.pi_ge.gain,
            waveform="pi_qubit")

        self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns
        self.measure(pulse_ch=self.res_ch, 
             adcs=[self.adc_ch],
             adc_trig_offset=cfg.device.readout.trig_offset,
             wait=True,
             syncdelay=self.us2cycles(cfg.device.readout.relax_delay))

    def update(self):
        step = self.cfg.expt.step
        if self.qubit_ch_type == 'int4': step = step << 16
        self.mathi(self.q_rp, self.r_gain2, self.r_gain2, '+', step) # update test gain
                      
                      
class AmplitudeRabiEFExperiment(Experiment):
    """
    Amplitude Rabi EF Experiment
    Experimental Config:
    expt = dict(
        start: qubit gain [dac level]
        step: gain step [dac level]
        expts: number steps
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        sigma_test: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_ge: set to False if doing qubit temperature experiment
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(self, soccfg=None, path='', prefix='AmplitudeRabiEF', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        q_ind = self.cfg.expt.qubit
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, list):
                    subcfg.update({key: value[q_ind]})
                elif isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if isinstance(value3, list):
                                value2.update({key3: value3[q_ind]})                                

        if 'sigma_test' not in self.cfg.expt:
            self.cfg.expt.sigma_test = self.cfg.device.qubit.pulses.pi_ef.sigma
        if 'pulse_ge' not in self.cfg.expt:
            self.cfg.expt.pulse_ge = True

        amprabiEF = AmplitudeRabiEFProgram(soccfg=self.soccfg, cfg=self.cfg)

        x_pts, avgi, avgq = amprabiEF.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress, debug=debug)        

        avgi = avgi[0][0]
        avgq = avgq[0][0]
        amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
        phases = np.angle(avgi+1j*avgq) # Calculating the phase        

        data={'xpts': x_pts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases}
        self.data=data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data

        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
            # Remove the first and last point from fit in case weird edge measurements
            # fitparams = [None, 1/max(data['xpts']), None, None]
            fitparams = None
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

        # plt.figure(figsize=(12, 8))
        # plt.subplot(111, title=f"Amplitude Rabi EF (Pulse Length {self.cfg.expt.sigma_test})", xlabel="Gain [DAC units]", ylabel="Amplitude [ADC units]")
        # plt.plot(data["xpts"][1:-1], data["amps"][1:-1],'o-')
        # if fit:
        #     p = data['fit_amps']
        #     plt.plot(data["xpts"][1:-1], fitter.sinfunc(data["xpts"][1:-1], *p))

        plt.figure(figsize=(10,10))
        plt.subplot(211, title=f"Amplitude Rabi EF (Pulse Length {self.cfg.expt.sigma_test})", ylabel="I [ADC levels]")
        plt.plot(data["xpts"][1:-1], data["avgi"][1:-1],'o-')
        if fit:
            p = data['fit_avgi']
            plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
            else: pi_gain = (3/2 - p[2]/180)/2/p[1]
            pi2_gain = pi_gain/2
            print(f'Pi gain from avgi data [dac units]: {int(pi_gain)}')
            print(f'\tPi/2 gain from avgi data [dac units]: {int(pi2_gain)}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi2_gain, color='0.2', linestyle='--')
        plt.subplot(212, xlabel="Gain [dac units]", ylabel="Q [ADC levels]")
        plt.plot(data["xpts"][1:-1], data["avgq"][1:-1],'o-')
        if fit:
            p = data['fit_avgq']
            plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
            else: pi_gain = (3/2 - p[2]/180)/2/p[1]
            pi2_gain = pi_gain/2
            print(f'Pi gain from avgq data [dac units]: {int(pi_gain)}')
            print(f'\tPi/2 gain from avgq data [dac units]: {int(pi2_gain)}')
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi2_gain, color='0.2', linestyle='--')
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)