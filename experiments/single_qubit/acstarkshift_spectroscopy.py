import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from qick import *

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm
import time

import experiments.fitting as fitter

class ACStarkShiftProgram(RAveragerProgram):
    def initialize(self):
        cfg=AttrDict(self.cfg)
        self.cfg.update(cfg.expt)

        self.adc_ch = cfg.hw.soc.adcs.readout.ch
        self.res_ch = cfg.hw.soc.dacs.readout.ch
        self.res_ch_type = cfg.hw.soc.dacs.readout.type
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type
        self.pump = cfg.expt.pump_params
        self.pump_ch = self.pump.ch
        self.pump_ch_type = self.pump.type

        self.q_rp=self.ch_page(self.qubit_ch) # get register page for qubit_ch
        self.f_res_reg = self.freq2reg(cfg.device.readout.frequency, gen_ch=self.res_ch, ro_ch=self.adc_ch)
        
        self.readout_length_dac = self.us2cycles(cfg.device.readout.readout_length, gen_ch=self.res_ch)
        self.readout_length_adc = self.us2cycles(cfg.device.readout.readout_length, ro_ch=self.adc_ch)
        self.readout_length_adc += 1 # ensure the rounding of the clock ticks calculation doesn't mess up the buffer
        self.pump_length_dac = self.us2cycles(cfg.expt.pump_length, gen_ch=self.pump_ch)
        self.qubit_length_dac = self.us2cycles(cfg.expt.qubit_length, gen_ch=self.qubit_ch)

        # declare res dacs
        mask = None
        mixer_freq = 0 # MHz
        mux_freqs = None # MHz
        mux_gains = None
        ro_ch = self.adc_ch
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
        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist, mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)

        # declare qubit dacs
        mixer_freq = 0
        if self.qubit_ch_type == 'int4':
            mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist, mixer_freq=mixer_freq)

        # declare pump dacs
        # assert self.pump_ch != self.qubit_ch
        mixer_freq = 0
        if self.pump.type == 'int4':
            mixer_freq = self.pump.mixer_freq
        if self.pump_ch not in self.gen_chs: self.declare_gen(ch=self.pump_ch, nqz=self.pump.nyquist, mixer_freq=mixer_freq)

        # declare adcs
        self.declare_readout(ch=self.adc_ch, length=self.readout_length_adc, freq=cfg.device.readout.frequency, gen_ch=self.res_ch)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        
        self.f_start = self.freq2reg(cfg.expt.start, gen_ch=self.qubit_ch) # get start/step frequencies for qubit spec
        self.f_step = self.freq2reg(cfg.expt.step, gen_ch=self.qubit_ch)

        self.f_pump = self.freq2reg(cfg.expt.pump_freq, gen_ch=self.pump_ch)

        # add qubit and readout pulses to respective channels
        if self.cfg.expt.pulse_type == 'flat_top':
            self.add_gauss(ch=self.qubit_ch, name="qubit", sigma=3, length=3*4)
        elif self.cfg.expt.pulse_type == 'gauss':
            length = self.qubit_length_dac
            self.add_gauss(ch=self.qubit_ch, name="qubit", sigma=length, length=length*4)

        if self.res_ch_type == 'mux4':
            self.set_pulse_registers(ch=self.res_ch, style="const", length=self.readout_length_dac, mask=mask)
        else: self.set_pulse_registers(ch=self.res_ch, style="const", freq=self.f_res_reg, phase=0, gain=cfg.device.readout.gain, length=self.readout_length_dac)


        # initialize registers
        if self.qubit_ch_type == 'int4':
            self.r_freq = self.sreg(self.qubit_ch, "freq") # get freq register for qubit_ch    
        else: self.r_freq = self.sreg(self.qubit_ch, "freq") # get freq register for qubit_ch    
        self.r_freq2 = 4
        self.safe_regwi(self.q_rp, self.r_freq2, self.f_start)

        self.synci(200) # give processor some time to configure pulses
    
    def body(self):
        cfg=AttrDict(self.cfg)

        # Phase reset all channels
        for ch in self.gen_chs.keys():
            if self.gen_chs[ch]['mux_freqs'] is None: # doesn't work for the mux channels
                # print('resetting', ch)
                self.setup_and_pulse(ch=ch, style='const', freq=100, phase=0, gain=100, length=10, phrst=1)
        self.sync_all(10)

        # pump tone (always const pulse)
        if cfg.expt.pump_gain > 0:
            self.setup_and_pulse(ch=self.pump_ch, style="const", freq=self.f_pump, phase=0, gain=cfg.expt.pump_gain, length=self.pump_length_dac)
        # self.sync_all()

        # self.setup_and_pulse(ch=self.qubit_ch, t=self.pump_length_dac - self.qubit_length_dac) # play qubit pulse at end of pump tone
        length = self.qubit_length_dac
        if self.cfg.expt.pulse_type == 'flat_top':
            self.set_pulse_registers(ch=self.qubit_ch, style="flat_top", phase=0, freq=self.f_start, gain=cfg.expt.qubit_gain, length=length, waveform="qubit") # play probe pulse
        elif self.cfg.expt.pulse_type == 'gauss':
            self.set_pulse_registers(ch=self.qubit_ch, style="arb", phase=0, freq=self.f_start, gain=cfg.expt.qubit_gain, waveform="qubit") # play probe pulse
        elif self.cfg.expt.pulse_type == 'const':
            self.set_pulse_registers(ch=self.qubit_ch, style="const", freq=self.f_start, phase=0, gain=cfg.expt.qubit_gain, length=length)
        self.mathi(self.q_rp, self.r_freq, self.r_freq2, "+", 0)
        self.pulse(ch=self.qubit_ch) # play probe pulse

        self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns
        self.measure(pulse_ch=self.res_ch, 
             adcs=[self.adc_ch],
             adc_trig_offset=cfg.device.readout.trig_offset,
             wait=True,
             syncdelay=self.us2cycles(cfg.device.readout.relax_delay))
    
    def update(self):
        self.mathi(self.q_rp, self.r_freq2, self.r_freq2, '+', self.f_step) # update frequency list index
 
# ====================================================== #

class ACStarkShiftPulseProbeExperiment(Experiment):
    """
    Experimental Config
        start_f: start probe frequency sweep [MHz]
        step_f
        expts_f
        start_gain: start pump gain sweep [dac units]
        step_gain
        expts_gain
        pump_params = dict(
            ch
            type
            mixer_freq
            nyquist
            )
        pump_length
        qubit_gain
        qubit_length
        reps: number averages per experiment
        rounds: number repetitions of experiment sweep
        qubit
    )
    """

    def __init__(self, soccfg=None, path='', prefix='ACStarkShiftPulseProbe', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        q_ind = self.cfg.expt.qubit
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc, self.cfg.expt.pump_params):
            for key, value in subcfg.items() :
                if isinstance(value, list):
                    subcfg.update({key: value[q_ind]})
                elif isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if isinstance(value3, list):
                                value2.update({key3: value3[q_ind]})                                

        freqpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"]*np.arange(self.cfg.expt["expts_f"])
        gainpts = self.cfg.expt["start_gain"] + self.cfg.expt["step_gain"]*np.arange(self.cfg.expt["expts_gain"])
        data={"gainpts":gainpts, "freqpts":freqpts, "avgi":[], "avgq":[], "amps":[], "phases":[]}

        self.cfg.expt.start = self.cfg.expt.start_f
        self.cfg.expt.step = self.cfg.expt.step_f
        self.cfg.expt.expts = self.cfg.expt.expts_f
        for gain in tqdm(gainpts):
            self.cfg.expt.pump_gain = gain
            acspec = ACStarkShiftProgram(soccfg=self.soccfg, cfg=self.cfg)
        
            freqpts, avgi, avgq = acspec.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False)
        
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amps = np.abs(avgi+1j*avgq)
            phases = np.angle(avgi+1j*avgq)

            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amps)
            data["phases"].append(phases)
        
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
        
        inner_sweep = data['freqpts']
        outer_sweep = data['gainpts']

        y_sweep = outer_sweep
        x_sweep = inner_sweep
        avgi = np.copy(data['avgi'])
        avgq = np.copy(data['avgq'])

        for i in range(len(avgi)):
            # amps_gain = (amps_gain - np.average(amps_gain)) / np.average(amps_gain)
            avgi[i] -= np.average(avgi[i])
            avgq[i] -= np.average(avgq[i])

        # THIS IS CORRECT EXTENT LIMITS FOR 2D PLOTS
        plt.figure(figsize=(10,8))
        plt.subplot(211, title=f"Qubit {self.cfg.expt.qubit} AC Stark Shift (Pump Freq {self.cfg.expt.pump_freq} MHz)", ylabel="Pump Gain [dac units]")
        plt.pcolormesh(x_sweep, y_sweep, avgi, cmap='viridis', shading='auto')
        plt.colorbar(label='I [ADC level]')
        # plt.clim(vmin=-6, vmax=6)
        # plt.axvline(1684.92, color='k')
        # plt.axvline(1684.85, color='r')

        plt.subplot(212, xlabel="Frequency [MHz]", ylabel="Pump Gain [dac units]")
        plt.pcolormesh(x_sweep, y_sweep, avgq, cmap='viridis', shading='auto')
        plt.colorbar(label='Q [ADC level]')
        plt.clim(vmin=None, vmax=None)
        
        if fit: pass

        plt.tight_layout()
        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)