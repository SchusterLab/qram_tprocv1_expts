import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm


class AmplitudeRabiEFProgram(RAveragerProgram):
    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        
        q_ind = self.cfg.expt.qubit
        self.res_ch = cfg.hw.soc.dacs.readout.ch[q_ind]
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch[q_ind]

        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist[q_ind])
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist[q_ind])
        
        self.q_rp = self.ch_page(self.qubit_ch) # get register page for qubit_ch
        self.r_gain = self.sreg(self.qubit_ch, "gain") # get gain register for qubit_ch   
        self.r_gain2 = 4
        
        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_ch)
        self.f_ef = self.freq2reg(cfg.device.qubit.f_ef, gen_ch=self.qubit_ch)
        self.f_res=self.freq2reg(cfg.device.readout.frequency, gen_ch=self.res_ch) # convert f_res to dac register value
        self.readout_length=self.us2cycles(cfg.device.readout.readout_length)
        for ch in [0,1]: # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=cfg.device.readout.frequency, gen_ch=self.res_ch)
        
        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
        
        self.pisigma_ge = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma)
        self.sigma_test = self.us2cycles(cfg.expt.sigma_test)
        
        # add qubit and readout pulses to respective channels
        self.add_gauss(ch=self.qubit_ch, name="pi_qubit", sigma=self.pisigma_ge, length=self.pisigma_ge*4)
        if cfg.expt.pulse_type.lower() == "gauss" and cfg.expt.sigma_test > 0:
            self.add_gauss(ch=self.qubit_ch, name="pi_ef", sigma=self.sigma_test, length=self.sigma_test*4)
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.f_res,
            phase=self.deg2reg(cfg.device.readout.phase, gen_ch=self.res_ch),
            gain=cfg.device.readout.gain,
            length=self.readout_length)

        # initialize gain
        self.safe_regwi(self.q_rp, self.r_gain2, self.cfg.expt.start)
           
        self.sync_all(self.us2cycles(0.2))
    
    def body(self):
        cfg=AttrDict(self.cfg)
        
        # init to qubit excited state
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
             adcs=[0,1],
             adc_trig_offset=cfg.device.readout.trig_offset,
             wait=True,
             syncdelay=self.us2cycles(cfg.device.readout.relax_delay))
    
    def update(self):
        self.mathi(self.q_rp, self.r_gain2, self.r_gain2, '+', self.cfg.expt.step) # update test gain
                      
                      
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
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(self, soccfg=None, path='', prefix='AmplitudeRabiEF', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        fpts = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        
        q_ind = self.cfg.expt.qubit
        for key, value in self.cfg.device.readout.items():
            if isinstance(value, list):
                self.cfg.device.readout.update({key: value[q_ind]})
        for key, value in self.cfg.device.qubit.items():
            if isinstance(value, list):
                self.cfg.device.qubit.update({key: value[q_ind]})
            elif isinstance(value, dict):
                for key2, value2 in value.items():
                    for key3, value3 in value2.items():
                        if isinstance(value3, list):
                            value2.update({key3: value3[q_ind]})                        
        if 'sigma_test' not in self.cfg.expt:
            self.cfg.expt.sigma_test = self.cfg.device.qubit.pulses.pi_ef.sigma
        
        amprabiEF = AmplitudeRabiEFProgram(soccfg=self.soccfg, cfg=self.cfg)
        adc_ch = self.cfg.hw.soc.adcs.readout.ch[q_ind]
        
        x_pts, avgi, avgq = amprabiEF.acquire(self.im[self.cfg.aliases.soc], threshold=None,load_pulses=True,progress=progress, debug=debug)        
        
        avgi = avgi[adc_ch][0]
        avgq = avgq[adc_ch][0]
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
            p = dsfit.fitdecaysin(data['xpts'][1:-1], data["amps"][1:-1], fitparams=None, showfit=False)
            # add due to extra parameter in decaysin that is not in fitdecaysin
            p = np.append(p, data['xpts'][0])
            data['fit_amps'] = p       

            p = dsfit.fitdecaysin(data['xpts'][1:-1], data["phases"][1:-1], fitparams=None, showfit=False)
            # add due to extra parameter in decaysin that is not in fitdecaysin
            p = np.append(p, data['xpts'][0])
            data['fit_phases'] = p    
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data

        plt.figure(figsize=(10,10))
        plt.subplot(211, title="Amplitude Rabi EF", ylabel="Amps [adc level]")
        plt.plot(data["xpts"][1:-1], data["amps"][1:-1],'o-')
        if fit:
            plt.plot(data["xpts"][1:-1], dsfit.decaysin(data["fit_amps"], data["xpts"][1:-1]))
            pi_gain = 1/data['fit_amps'][1]/2
            print(f'Pi gain from amp [dac units]: {int(pi_gain)}')
            print(f'Pi/2 gain from amp [dac units]: {int(pi_gain/2)}')
            print()
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi_gain/2, color='0.2', linestyle='--')
        plt.subplot(212, xlabel="Gain [dac units]", ylabel="Phases [radians]")
        plt.plot(data["xpts"][1:-1], data["phases"][1:-1],'o-')
        if fit:
            plt.plot(data["xpts"][1:-1], dsfit.decaysin(data["fit_phases"], data["xpts"][1:-1]))
            pi_gain = 1/data['fit_phases'][1]/2
            print(f'Pi gain from phase [dac units]: {int(pi_gain)}')
            print(f'Pi/2 gain from phase [dac units]: {int(pi_gain/2)}')
            print()
            plt.axvline(pi_gain, color='0.2', linestyle='--')
            plt.axvline(pi_gain/2, color='0.2', linestyle='--')
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)