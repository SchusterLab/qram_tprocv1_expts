import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

"""
Measures Rabi oscillations by sweeping over the duration of the qubit drive pulse. This is a preliminary measurement to prove that we see Rabi oscillations. This measurement is followed up by the Amplitude Rabi experiment.
"""
class LengthRabiProgram(AveragerProgram):
    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        
        q_ind = self.cfg.expt.qubit
        self.res_ch = cfg.hw.soc.dacs.readout.ch[q_ind]
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch[q_ind]

        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist[q_ind])
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist[q_ind])
        
        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_ch)
        self.f_res=self.freq2reg(cfg.device.readout.frequency, gen_ch=self.res_ch) # convert f_res to dac register value
        self.readout_length=self.us2cycles(cfg.device.readout.readout_length)

        for ch in [0,1]: # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=cfg.device.readout.frequency, gen_ch=self.res_ch)
        
        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds

        # update sigma in outer loop over averager program
        self.pi_sigma = self.us2cycles(cfg.expt.length_placeholder)
        # print(self.pi_sigma)

        # add qubit and readout pulses to respective channels
        if cfg.expt.pulse_type.lower() == "gauss" and self.pi_sigma > 0:
            self.add_gauss(ch=self.qubit_ch, name="qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.f_ge,
                phase=0,
                gain=cfg.expt.gain,
                waveform="qubit")
        elif self.pi_sigma > 0:
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="const",
                freq=self.f_ge,
                phase=0,
                gain=cfg.expt.gain,
                length=self.pi_sigma)
            
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.f_res,
            phase=self.deg2reg(cfg.device.readout.phase, gen_ch=self.res_ch),
            gain=cfg.device.readout.gain,
            length=self.readout_length)

        self.sync_all(self.us2cycles(0.2))
    
    def body(self):
        cfg=AttrDict(self.cfg)
        if self.pi_sigma > 0:
            self.pulse(ch=self.qubit_ch)
        self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns
        self.measure(pulse_ch=self.res_ch, 
             adcs=[0,1],
             adc_trig_offset=cfg.device.readout.trig_offset,
             wait=True,
             syncdelay=self.us2cycles(cfg.device.readout.relax_delay))

        
class LengthRabiExperiment(Experiment):
    """
    Length Rabi Experiment
    Experimental Config
    expt = dict(
       start: start length [us],
       step: length step, 
       expts: number of different length experiments, 
       reps: number of reps,
       gain: gain to use for the qubit pulse
       pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(self, soccfg=None, path='', prefix='LengthRabi', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
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
        
        lengths = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        
        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
        
        adc_ch = self.cfg.hw.soc.adcs.readout.ch[q_ind]
        for length in tqdm(lengths, disable=not progress):
            self.cfg.expt.length_placeholder = float(length)
            lengthrabi = LengthRabiProgram(soccfg=self.soccfg, cfg=self.cfg)
            self.prog = lengthrabi
            avgi, avgq = lengthrabi.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False, debug=debug)        
            avgi = avgi[adc_ch][0]
            avgq = avgq[adc_ch][0]
            amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phase = np.angle(avgi+1j*avgq) # Calculating the phase
            data["xpts"].append(length)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)

        for k, a in data.items():
            data[k]=np.array(a)
        
        self.data = data

        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data
        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
            # Remove the first and last point from fit in case weird edge measurements
            p_avgi = dsfit.fitdecaysin(data['xpts'][1:-1], data["avgi"][1:-1], fitparams=None, showfit=False)
            p_avgq = dsfit.fitdecaysin(data['xpts'][1:-1], data["avgq"][1:-1], fitparams=None, showfit=False)
            # adding this due to extra parameter in decaysin that is not in fitdecaysin
            p_avgi = np.append(p_avgi, data['xpts'][0])
            p_avgq = np.append(p_avgq, data['xpts'][0])
            data['fit_avgi'] = p_avgi   
            data['fit_avgq'] = p_avgq
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 
        xpts_ns = data['xpts']*1e3
        plt.figure(figsize=(10,8))
        plt.subplot(211, title=f"Length Rabi (Qubit Gain {self.cfg.expt.gain})", ylabel="I [adc level]")
        plt.plot(xpts_ns[1:-1], data["avgi"][1:-1],'o-')
        if fit:
            plt.plot(xpts_ns[1:-1], dsfit.decaysin(data["fit_avgi"], data["xpts"][1:-1]))
            pi_len = 1/data['fit_avgi'][1]/2
            print(f'Pi length from avgi data [us]: {pi_len}')
            print(f'Pi/2 length from avgi data [dac units]: {pi_len/2}')
            plt.axvline(pi_len*1e3, color='0.2', linestyle='--')
            plt.axvline(pi_len*1e3/2, color='0.2', linestyle='--')
        plt.subplot(212, xlabel="Pulse length [ns]", ylabel="Q [adc levels]")
        plt.plot(xpts_ns[1:-1], data["avgq"][1:-1],'o-')
        if fit:
            plt.plot(xpts_ns[1:-1], dsfit.decaysin(data["fit_avgq"], data["xpts"][1:-1]))
            pi_len = 1/data['fit_avgq'][1]/2
            print(f'Pi length from avgq data [us]: {pi_len}')
            print(f'Pi/2 length from avgq data [dac units]: {pi_len/2}')
            plt.axvline(pi_len*1e3, color='0.2', linestyle='--')
            plt.axvline(pi_len*1e3/2, color='0.2', linestyle='--')
        plt.tight_layout()
        plt.show()
    
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)