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

        self.adc_ch = cfg.hw.soc.adcs.readout.ch
        self.dac_ch = cfg.hw.soc.dacs.readout.ch
        self.dac_ch_type = cfg.hw.soc.dacs.readout.type

        self.frequency = cfg.expt.frequency
        self.freqreg = self.freq2reg(self.frequency, gen_ch=self.dac_ch, ro_ch=self.adc_ch) # convert frequency to dac frequency (ensuring it is an available adc frequency)
        self.gain = cfg.expt.gain
        self.pulse_length = self.us2cycles(cfg.expt.pulse_length, gen_ch=self.dac_ch)
        self.readout_length = self.us2cycles(cfg.expt.readout_length, ro_ch=self.adc_ch)
        print(self.pulse_length, self.readout_length)

        mask = None
        mixer_freq = 0 # MHz
        mux_freqs = None # MHz
        mux_gains = None
        ro_ch = None
        if self.dac_ch_type == 'int4':
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
        elif self.dac_ch_type == 'mux4':
            assert self.dac_ch == 6
            mask = [0, 1, 2, 3] # indices of mux_freqs, mux_gains list to play
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
            mux_freqs= [cfg.expt.frequency, 0, 0, 0]
            mux_gains=[cfg.expt.gain, 0, 0, 0]
            ro_ch=self.adc_ch
        self.declare_gen(ch=self.dac_ch, nqz=cfg.hw.soc.dacs.readout.nyquist, mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)
        print(f'readout freq {mixer_freq} +/- {cfg.expt.frequency}')

        self.declare_readout(ch=self.adc_ch, length=self.readout_length, freq=self.frequency, gen_ch=self.dac_ch) # gen_ch links to the mixer_freq being used on the mux

        # copy over parameters for the acquire method
        self.cfg.soft_avgs = cfg.expt.reps # same as reps
        self.cfg.reps = 1 # not used for acquire_decimated

        if self.dac_ch_type == 'mux4':
            self.set_pulse_registers(ch=self.dac_ch, style="const", length=self.pulse_length, mask=mask)
        else: self.set_pulse_registers(ch=self.dac_ch, style="const", freq=self.freqreg, phase=0, gain=self.gain, length=self.pulse_length)
        self.synci(200) # give processor some time to configure pulses
    
    def body(self):
        cfg=AttrDict(self.cfg)
        self.measure(pulse_ch=self.dac_ch, adcs=[self.adc_ch], adc_trig_offset=cfg.expt.trig_offset, wait=True, syncdelay=self.us2cycles(cfg.device.readout.relax_delay))
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

        data={"i":[], "q":[], "amps":[], "phases":[]}
        tof = ToFCalibrationProgram(soccfg=self.soccfg, cfg=self.cfg)
        # print(tof)
        iq = tof.acquire_decimated(self.im[self.cfg.aliases.soc], load_pulses=True, progress=True)
        i, q = iq[0]
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
        
        q_ind = self.cfg.expt.qubit
        adc_ch = self.cfg.hw.soc.adcs.readout.ch
        dac_ch = self.cfg.hw.soc.dacs.readout.ch
        plt.subplot(111, title=f"Time of flight calibration: dac ch {dac_ch} to adc ch {adc_ch}", xlabel="Clock ticks", ylabel="Transmission [ADC units]")
        
        plt.plot(data["i"], label='I')
        plt.plot(data["q"], label='Q')
        plt.axvline(adc_trig_offset, c='k', ls='--')
        # plt.ylim(-100, 100)
        plt.legend()
        plt.show()
        
                
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)