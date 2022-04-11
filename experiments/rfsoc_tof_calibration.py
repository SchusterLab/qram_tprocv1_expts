import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

from qick import *
from slab import Experiment, dsfit, AttrDict

"""
Run this calibration when the wiring of the setup is changed.

This calibration measures the time of flight of measurement pulse so we only start capturing data from this point in time onwards. Time of flight (tof) is stored in parameter cfg.device.readout.trig_offset.
"""
class ToFCalibrationProgram(AveragerProgram):
    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(self.cfg.expt)

        self.dac_ch = cfg.hw.soc.dacs.readout.ch[cfg.device.readout.dac]
        self.declare_gen(ch=self.dac_ch, nqz=cfg.hw.soc.dacs.readout.nyquist[cfg.device.readout.dac])

        self.frequency = cfg.expt.frequency
        self.freqreg = self.freq2reg(self.frequency) # convert frequency to dac frequency (ensuring it is an available adc frequency)
        self.gain = cfg.expt.gain
        
        self.pulse_length = self.us2cycles(cfg.expt.pulse_length)
        self.readout_length = self.us2cycles(cfg.expt.readout_length)

        for ch in [0,1]: # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=self.frequency, gen_ch=self.dac_ch)
        
        # copy over parameters for the acquire method
        self.cfg.soft_avgs = cfg.expt.reps # same as reps
        self.cfg.reps = 1 # not used for acquire_decimated

        self.set_pulse_registers(
            ch=self.dac_ch,
            style="const",
            freq=self.freqreg,
            phase=0,
            gain=self.gain,
            length=self.pulse_length)
        self.synci(self.us2cycles(500)) # give processor some time to configure pulses
    
    def body(self):
        cfg=AttrDict(self.cfg)
        self.measure(pulse_ch=self.dac_ch, 
             adcs=[0,1],
             adc_trig_offset=cfg.device.readout.trig_offset,
             wait=True,
             syncdelay=self.us2cycles(cfg.device.readout.relax_delay))

# ====================================================== #

class ToFCalibrationExperiment(Experiment):
    """
    Time of flight experiment
    Experimental Config
    expt_cfg = dict(
        pulse_length [us]
        readout_length [us]
        gain [DAC units]
        frequency [MHz]
        adc_trig_offset [Clock ticks]
    } 
    """

    def __init__(self, soccfg=None, path='', prefix='ToFCalibration', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        q_ind = self.cfg.expt.qubit_i
        for key, value in self.cfg.device.readout.items():
            if isinstance(value, list):
                self.cfg.device.readout.update({key: value[q_ind]})
        
        data={"i":[], "q":[], "amps":[], "phases":[]}
        adc_ch = self.cfg.hw.soc.adcs.readout.ch[self.cfg.device.readout.adc]
        tof = ToFCalibrationProgram(soccfg=self.soccfg, cfg=self.cfg)
        self.prog = tof 
        iq = tof.acquire_decimated(self.im[self.cfg.aliases.soc], load_pulses=True, progress=True)
        i, q = iq[adc_ch]
        amp = np.abs(i+1j*q) # Calculating the magnitude
        phase = np.angle(i+1j*q) # Calculating the phase

        data = dict(
            i=i,
            q=q,
            amps=amp,
            phases=phase
        )
        
        for k, a in data.items():
            data[k]=np.array(a)
        
        self.data=data
        return data

    def analyze(self, data=None, fit=False, findpeaks=False, **kwargs):
        if data is None:
            data=self.data
        return data

    def display(self, data=None, adc_trig_offset=0, **kwargs):
        if data is None:
            data=self.data 
        
        adc_ch = self.cfg.hw.soc.adcs.readout.ch[self.cfg.device.readout.adc]
        dac_ch = self.cfg.hw.soc.dacs.readout.ch[self.cfg.device.readout.dac]
        plt.subplot(111, title=f"Time of flight calibration: dac ch {dac_ch} to adc ch {adc_ch}", xlabel="Clock ticks", ylabel="Transmission (adc level)")
        
        plt.plot(data["i"], label='I')
        plt.plot(data["q"], label='Q')
        plt.axvline(adc_trig_offset, c='k', ls='--')
        plt.legend()
        plt.show()
        
                
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)