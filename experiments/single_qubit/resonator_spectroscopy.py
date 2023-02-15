import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import time

from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

import experiments.fitting as fitter

"""
Measures the resonant frequency of the readout resonator when the qubit is in its ground state: sweep readout pulse frequency and look for the frequency with the maximum measured amplitude.

The resonator frequency is stored in the parameter cfg.device.readouti.frequency.

Note that harmonics of the clock frequency (6144 MHz) will show up as "infinitely"  narrow peaks!
"""
class ResonatorSpectroscopyProgram(AveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(self.cfg.expt)
        
        self.adc_ch = cfg.hw.soc.adcs.readout.ch
        self.res_ch = cfg.hw.soc.dacs.readout.ch
        self.res_ch_type = cfg.hw.soc.dacs.readout.type
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type

        self.frequency = cfg.expt.frequency
        self.freqreg = self.freq2reg(self.frequency, gen_ch=self.res_ch, ro_ch=self.adc_ch)
        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_ch)
        if self.cfg.expt.pulse_f: 
            self.f_ef = self.freq2reg(cfg.device.qubit.f_ef, gen_ch=self.qubit_ch)
        self.res_gain = cfg.device.readout.gain
        self.readout_length_dac = self.us2cycles(cfg.device.readout.readout_length, gen_ch=self.res_ch)
        self.readout_length_adc = self.us2cycles(cfg.device.readout.readout_length, ro_ch=self.adc_ch)
        self.readout_length_adc += 1 # ensure the rounding of the clock ticks calculation doesn't mess up the buffer

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
            mux_freqs[cfg.expt.qubit] = self.frequency
            mux_gains = [0]*4
            mux_gains[cfg.expt.qubit] = self.res_gain
            ro_ch=self.adc_ch
        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist, mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)
        # print(f'readout freq {mixer_freq} +/- {self.frequency}')

        mixer_freq = 0
        if self.qubit_ch_type == 'int4':
            mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist, mixer_freq=mixer_freq)

        self.declare_readout(ch=self.adc_ch, length=self.readout_length_adc, freq=self.frequency, gen_ch=self.res_ch)

        self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        self.pi_gain = cfg.device.qubit.pulses.pi_ge.gain
        if self.cfg.expt.pulse_f:
            self.pi_ef_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)
            self.pi_ef_gain = cfg.device.qubit.pulses.pi_ef.gain
        
        if self.cfg.expt.pulse_e or self.cfg.expt.pulse_f:
            self.add_gauss(ch=self.qubit_ch, name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)
        if self.cfg.expt.pulse_f:
            self.add_gauss(ch=self.qubit_ch, name="pi_ef_qubit", sigma=self.pi_ef_sigma, length=self.pi_ef_sigma*4)

        if self.res_ch_type == 'mux4':
            self.set_pulse_registers(ch=self.res_ch, style="const", length=self.readout_length_dac, mask=mask)
        else: self.set_pulse_registers(ch=self.res_ch, style="const", freq=self.freqreg, phase=0, gain=self.res_gain, length=self.readout_length_dac)
        self.synci(200) # give processor some time to configure pulses

    def body(self):
        # pass
        cfg=AttrDict(self.cfg)
        if self.cfg.expt.pulse_e or self.cfg.expt.pulse_f:
            self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=self.pi_gain, waveform="pi_qubit")
            self.sync_all() # align channels
        if self.cfg.expt.pulse_f:
            self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ef, phase=0, gain=self.pi_ef_gain, waveform="pi_ef_qubit")
            self.sync_all() # align channels
        self.measure(
            pulse_ch=self.res_ch,
            adcs=[self.adc_ch],
            adc_trig_offset=cfg.device.readout.trig_offset,
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay))

# ====================================================== #

class ResonatorSpectroscopyExperiment(Experiment):
    """
    Resonator Spectroscopy Experiment
    Experimental Config
    expt = dict(
        start: start frequency (MHz), 
        step: frequency step (MHz), 
        expts: number of experiments, 
        pulse_e: boolean to add e pulse prior to measurement
        pulse_f: boolean to add f pulse prior to measurement
        reps: number of reps
        )
    """

    def __init__(self, soccfg=None, path='', prefix='ResonatorSpectroscopy', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        xpts=self.cfg.expt["start"] + self.cfg.expt["step"]*np.arange(self.cfg.expt["expts"])
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

        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
        for f in tqdm(xpts, disable=not progress):
            self.cfg.expt.frequency = f
            rspec = ResonatorSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
            # print(rspec)
            avgi, avgq = rspec.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phase = np.angle(avgi+1j*avgq) # Calculating the phase

            data["xpts"].append(f)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)
        
        for k, a in data.items():
            data[k]=np.array(a)
        
        self.data=data

        return data

    def analyze(self, data=None, fit=False, findpeaks=False, verbose=True, **kwargs):
        if data is None:
            data=self.data
            
        if fit:
            # fitparams = [f0, Qi, Qe, phi, scale]
            xdata = data["xpts"][1:-1]
            # ydata = data["avgi"][1:-1] + 1j*data["avgq"][1:-1]
            ydata = data['amps'][1:-1]
            fitparams=None
            data['fit'], data['fit_err'] = fitter.fithanger(xdata, ydata, fitparams=fitparams)
            if isinstance(data['fit'], (list, np.ndarray)):
                f0, Qi, Qe, phi, scale, a0, slope = data['fit']
                if verbose:
                    print(f'\nFreq with minimum transmission: {xdata[np.argmin(ydata)]}')
                    print(f'Freq with maximum transmission: {xdata[np.argmax(ydata)]}')
                    print('From fit:')
                    print(f'\tf0: {f0}')
                    print(f'\tQi: {Qi}')
                    print(f'\tQe: {Qe}')
                    print(f'\tQ0: {1/(1/Qi+1/Qe)}')
                    print(f'\tkappa [MHz]: {f0*(1/Qi+1/Qe)}')
                    print(f'\tphi [radians]: {phi}')
            
        if findpeaks:
            maxpeaks, minpeaks = dsfit.peakdetect(data['amps'][1:-1], x_axis=data['xpts'][1:-1], lookahead=30, delta=5*np.std(data['amps'][:5]))
            data['maxpeaks'] = maxpeaks
            data['minpeaks'] = minpeaks
            
        return data

    def display(self, data=None, fit=True, findpeaks=False, **kwargs):
        if data is None:
            data=self.data 
        xpts = float(self.cfg.hw.lo.readout.frequency)*1e-6 + self.cfg.device.readout.lo_sideband*(self.cfg.hw.soc.dacs.readout.mixer_freq + data['xpts'][1:-1])

        plt.figure(figsize=(16,16))
        plt.subplot(311, title=f"Resonator Spectroscopy at gain {self.cfg.device.readout.gain}",  ylabel="Amps [ADC units]")
        plt.plot(xpts, data['amps'][1:-1],'o-')
        if fit:
            plt.plot(xpts, fitter.hangerS21func_sloped(data["xpts"][1:-1], *data["fit"]))
        if findpeaks:
            # for peak in np.concatenate((data['maxpeaks'], data['minpeaks'])):
            for peak in data['minpeaks']:
                plt.axvline(peak[0], linestyle='--', color='0.2')
                print(f'Found peak [MHz]: {peak[0]}')
        # plt.axvline(float(self.cfg.hw.lo.readout.frequency)*1e-6 + self.cfg.device.readout.lo_sideband*(self.cfg.hw.soc.dacs.readout.mixer_freq + 812.37), c='k', ls='--')
        # plt.axvline(7687.5, c='k', ls='--')

        plt.subplot(312, xlabel="Readout Frequency [MHz]", ylabel="I [ADC units]")
        plt.plot(xpts, data["avgi"][1:-1],'o-')

        plt.subplot(313, xlabel="Readout Frequency [MHz]", ylabel="Q [ADC units]")
        plt.plot(xpts, data["avgq"][1:-1],'o-')
        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)


# ====================================================== #

class ResonatorPowerSweepSpectroscopyExperiment(Experiment):
    """Resonator Power Sweep Spectroscopy Experiment
       Experimental Config
       expt_cfg={
       "start_f": start frequency (MHz), 
       "step_f": frequency step (MHz), 
       "expts_f": number of experiments in frequency,
       "start_gain": start frequency (dac units), 
       "step_gain": frequency step (dac units), 
       "expts_gain": number of experiments in gain sweep,
       "reps": number of reps, 
        } 
    """

    def __init__(self, soccfg=None, path='', prefix='ResonatorPowerSweepSpectroscopy', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        xpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"]*np.arange(self.cfg.expt["expts_f"])
        gainpts = self.cfg.expt["start_gain"] + self.cfg.expt["step_gain"]*np.arange(self.cfg.expt["expts_gain"])

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

        data={"xpts":[], "gainpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
        for gain in tqdm(gainpts, disable=not progress):
            self.cfg.device.readout.gain = gain
            data["avgi"].append([])
            data["avgq"].append([])
            data["amps"].append([])
            data["phases"].append([])

            for f in tqdm(xpts, disable=True):
                self.cfg.expt.frequency = f
                rspec = ResonatorSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
                self.prog = rspec
                avgi, avgq = rspec.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                avgi = avgi[0][0]
                avgq = avgq[0][0]
                amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
                phase = np.angle(avgi+1j*avgq) # Calculating the phase
                data["avgi"][-1].append(avgi)
                data["avgq"][-1].append(avgq)
                data["amps"][-1].append(amp)
                data["phases"][-1].append(phase)
        
        data["xpts"] = xpts
        data["gainpts"] = gainpts
        
        for k, a in data.items():
            data[k] = np.array(a)
        
        self.data = data
        return data

    def analyze(self, data=None, fit=True, highgain=None, lowgain=None, **kwargs):
        if data is None:
            data=self.data
        
        # Lorentzian fit at highgain [DAC units] and lowgain [DAC units]
        if fit:
            if highgain == None: highgain = data['gainpts'][-1]
            if lowgain == None: lowgain = data['gainpts'][0]
            i_highgain = np.argmin(np.abs(data['gainpts']-highgain))
            i_lowgain = np.argmin(np.abs(data['gainpts']-lowgain))
            fit_highpow=dsfit.fitlor(data["xpts"], data["amps"][i_highgain])
            fit_lowpow=dsfit.fitlor(data["xpts"], data["amps"][i_lowgain])
            data['fit'] = [fit_highpow, fit_lowpow]
            data['fit_gains'] = [highgain, lowgain]
            data['lamb_shift'] = fit_highpow[2] - fit_lowpow[2]
        
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 

        inner_sweep = float(self.cfg.hw.lo.readout.frequency)*1e-6 + self.cfg.device.readout.lo_sideband*(self.cfg.hw.soc.dacs.readout.mixer_freq + data['xpts'])
        outer_sweep = data['gainpts']

        amps = data['amps']
        for amps_gain in amps:
            amps_gain -= np.average(amps_gain)
        
        y_sweep = outer_sweep
        x_sweep = inner_sweep

        # THIS IS CORRECT EXTENT LIMITS FOR 2D PLOTS
        plt.figure(figsize=(10,8))
        plt.pcolormesh(x_sweep, y_sweep, amps, cmap='viridis', shading='auto')
        
        if fit:
            fit_highpow, fit_lowpow = data['fit']
            highgain, lowgain = data['fit_gains']
            plt.axvline(fit_highpow[2], linewidth=0.5, color='0.2')
            plt.axvline(fit_lowpow[2], linewidth=0.5, color='0.2')
            plt.plot(x_sweep, [highgain]*len(x_sweep), linewidth=0.5, color='0.2')
            plt.plot(x_sweep, [lowgain]*len(x_sweep), linewidth=0.5, color='0.2')
            print(f'High power peak [MHz]: {fit_highpow[2]}')
            print(f'Low power peak [MHz]: {fit_lowpow[2]}')
            print(f'Lamb shift [MHz]: {data["lamb_shift"]}')
            
        plt.title(f"Resonator Spectroscopy Power Sweep")
        plt.xlabel("Resonator Frequency [MHz]")
        plt.ylabel("Resonator Gain [DAC level]")
        # plt.clim(vmin=-0.2, vmax=0.2)
        plt.clim(vmin=-10, vmax=5)
        plt.colorbar(label='Amps-Avg [ADC level]')
        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)

# ====================================================== #

class ResonatorVoltSweepSpectroscopyExperiment(Experiment):
    """Resonator Volt Sweep Spectroscopy Experiment
       Experimental Config
       expt_cfg={
       "start_f": start frequency (MHz), 
       "step_f": frequency step (MHz), 
       "expts_f": number of experiments in frequency,
       "start_volt": start volt, 
       "step_volt": voltage step, 
       "expts_volt": number of experiments in voltage sweep,
       "reps": number of reps, 
       "dc_ch": channel on dc_instr to sweep voltage
        } 
    """

    def __init__(self, soccfg=None, path='', dc_instr=None, dc_ch=None, prefix='ResonatorVoltSweepSpectroscopy', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)
        self.dc_instr = dc_instr

    def acquire(self, progress=False):
        xpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"]*np.arange(self.cfg.expt["expts_f"])
        voltpts = self.cfg.expt["start_volt"] + self.cfg.expt["step_volt"]*np.arange(self.cfg.expt["expts_volt"])
        
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

        data={"xpts":[], "voltpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}

        self.dc_instr.set_mode('CURR')
        self.dc_instr.set_current_limit(max(abs(voltpts)*5))
        print(f'Setting current limit {self.dc_instr.get_current_limit()*1e6} uA')
        self.dc_instr.set_output(True)

        for volt in tqdm(voltpts, disable=not progress):
            # self.dc_instr.set_voltage(channel=self.cfg.expt.dc_ch, voltage=volt)
            self.dc_instr.set_current(volt)
            print(f'current set to {self.dc_instr.get_current() * 1e6} uA')
            time.sleep(0.5)
            data["avgi"].append([])
            data["avgq"].append([])
            data["amps"].append([])
            data["phases"].append([])
            
            for f in tqdm(xpts, disable=True):
                self.cfg.device.readout.frequency = f
                rspec = ResonatorSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
                self.prog = rspec
                avgi, avgq = rspec.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                avgi = avgi[0][0]
                avgq = avgq[0][0]
                amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
                phase = np.angle(avgi+1j*avgq) # Calculating the phase
                
                data["avgi"][-1].append(avgi)
                data["avgq"][-1].append(avgq)
                data["amps"][-1].append(amp)
                data["phases"][-1].append(phase)
            time.sleep(0.5)
        # self.dc_instr.initialize()
        # self.dc_instr.set_voltage(channel=self.cfg.expt.dc_ch, voltage=0)

        self.dc_instr.set_current(0)
        print(f'current set to {self.dc_instr.get_current() * 1e6} uA')
        
        data["xpts"] = xpts
        data["voltpts"] = voltpts
        
        for k, a in data.items():
            data[k] = np.array(a)
        
        self.data = data
        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data
        pass

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 
        
        plt.figure(figsize=(12, 8))
        x_sweep = 1e3*data['voltpts']
        y_sweep = data['xpts']
        amps = data['amps']
        # for amps_volt in amps:
        #     amps_volt -= np.average(amps_volt)
        
        plt.pcolormesh(x_sweep, y_sweep, np.flip(np.rot90(data['amps']), 0), cmap='viridis')
        if 'add_data' in kwargs:
            for add_data in kwargs['add_data']:
                plt.pcolormesh(
                    1e3*add_data['voltpts'], add_data['xpts'], np.flip(np.rot90(add_data['amps']), 0), cmap='viridis')
        if fit: pass
            
        plt.title(f"Resonator {self.cfg.expt.qubit} sweeping DAC box ch {self.cfg.expt.dc_ch}")
        plt.ylabel("Resonator frequency")
        plt.xlabel("DC current [mA]")
        # plt.ylabel("DC voltage [V]")
        plt.clim(vmin=None, vmax=None)
        plt.colorbar(label='Amps [ADC level]')

        # plt.plot(x_sweep, amps[1])
        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname