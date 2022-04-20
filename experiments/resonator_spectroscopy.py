import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import time

from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

"""
Measures the resonant frequency of the readout resonator when the qubit is in its ground state: sweep readout pulse frequency and look for the frequency with the maximum measured amplitude.

The resonator frequency is stored in the parameter cfg.device.readouti.frequency.

Note that harmonics of the clock frequency (6144 MHz) will show up as "infinitely"  narrow peaks!
"""
class ResonatorSpectroscopyProgram(AveragerProgram):
    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(self.cfg.expt)
        
        q_ind = self.cfg.expt.qubit
        self.res_ch = cfg.hw.soc.dacs.readout.ch[q_ind]
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch[q_ind]

        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist[q_ind])
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist[q_ind])

        self.frequency = cfg.device.readout.frequency
        self.freqreg = self.freq2reg(self.frequency, gen_ch=self.res_ch, ro_ch=None)
        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_ch)
        self.res_gain = cfg.device.readout.gain
        self.readout_length = self.us2cycles(cfg.device.readout.readout_length)

        for ch in [0,1]: # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=self.frequency, gen_ch=self.res_ch)

        self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma)
        
        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        
        if self.cfg.expt.pi_pulse:
            self.add_gauss(ch=self.qubit_ch, name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.f_ge,
                phase=0,
                gain=cfg.device.qubit.pulses.pi_ge.gain, waveform="pi_qubit")
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.freqreg,
            phase=self.deg2reg(cfg.device.readout.phase, gen_ch=self.res_ch),
            gain=self.res_gain,
            length=self.readout_length)
        self.synci(self.us2cycles(500)) # give processor some time to configure pulses
    
    def body(self):
        cfg=AttrDict(self.cfg)
        if self.cfg.expt.pi_pulse:
            self.pulse(ch=self.qubit_ch) # play probe pulse
            self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns
        self.measure(pulse_ch=self.res_ch, 
             adcs=[0,1],
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
        pi_pulse: boolean to add pi pulse prior to measurement
        reps: number of reps
        )
    """

    def __init__(self, soccfg=None, path='', prefix='ResonatorSpectroscopy', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        xpts=self.cfg.expt["start"] + self.cfg.expt["step"]*np.arange(self.cfg.expt["expts"])
        
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
        
        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
        adc_ch = self.cfg.hw.soc.adcs.readout.ch[q_ind]
        for f in tqdm(xpts, disable=not progress):
            self.cfg.device.readout.frequency = f
            rspec = ResonatorSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
            self.prog = rspec
            avgi, avgq = rspec.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
            avgi = avgi[adc_ch][0]
            avgq = avgq[adc_ch][0]
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
            # fitparams = [f0,Qi,Qc,df,scale]
            fit=dsfit.fithanger(data["xpts"][1:-1], data["amps"][1:-1], printresult=verbose)
            f0, Qi, Qe = (fit[0], fit[1], fit[2])
            if verbose:
                print(f'Q: {1/(1/Qi+1/Qe)}')
                print(f'kappa [MHz]: {f0*(1/Qi+1/Qe)}')
            data["fit"]=fit

            # p = dsfit.fitdecaysin(data['xpts'][1:-1], data["amps"][1:-1], fitparams=None, showfit=False)
            # p = np.append(p, data['xpts'][0])
            # data['fit'] = p        
            
        if findpeaks:
            maxpeaks, minpeaks = dsfit.peakdetect(data['amps'][1:-1], x_axis=data['xpts'][1:-1], lookahead=30, delta=5*np.std(data['amps'][:5]))
            data['maxpeaks'] = maxpeaks
            data['minpeaks'] = minpeaks
            
        return data

    def display(self, data=None, fit=True, findpeaks=False, **kwargs):
        if data is None:
            data=self.data 
        plt.figure(figsize=(18,6))
        plt.subplot(111, title=f"Resonator Spectroscopy at gain {self.cfg.device.readout.gain}", xlabel="Resonator Frequency (MHz)", ylabel="Amps (adc level)")
        
        plt.plot(data["xpts"][1:-1], data["amps"][1:-1],'o-')
        
        if fit:
            plt.plot(data["xpts"][1:-1], dsfit.hangerfunc(data["fit"], data["xpts"][1:-1]))
            # plt.plot(data["xpts"][1:-1], dsfit.decaysin(data["fit"], data["xpts"][1:-1]))
            # print(data['fit'][1])
            
        if findpeaks:
            # for peak in np.concatenate((data['maxpeaks'], data['minpeaks'])):
            for peak in data['minpeaks']:
                plt.axvline(peak[0], linestyle='--', color='0.2')
                print(f'Found peak [MHz]: {peak[0]}')
            
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
        
        data={"xpts":[], "gainpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
        adc_ch = self.cfg.hw.soc.adcs.readout.ch[q_ind]
        for gain in tqdm(gainpts, disable=not progress):
            self.cfg.device.readout.gain = gain
            data["avgi"].append([])
            data["avgq"].append([])
            data["amps"].append([])
            data["phases"].append([])
            
            for f in tqdm(xpts, disable=True):
                self.cfg.device.readout.frequency = f
                rspec = ResonatorSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
                self.prog = rspec
                avgi, avgq = rspec.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                avgi = avgi[adc_ch][0]
                avgq = avgq[adc_ch][0]
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
        
        x_sweep = data['xpts']
        y_sweep = data['gainpts'] 
        amps = data['amps']
        for amps_gain in amps:
            amps_gain -= np.average(amps_gain)
        
        plt.imshow(
            np.flip(amps, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        
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
            
        plt.title(f"Resonator spectroscopy power sweep")
        plt.xlabel("Resonator frequency")
        plt.ylabel("Resonator gain [DAC level]")
        # plt.clim(vmin=-0.2, vmax=0.2)
        plt.clim(vmin=None, vmax=None)
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
        
        data={"xpts":[], "voltpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
        adc_ch = self.cfg.hw.soc.adcs.readout.ch[q_ind]

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
                avgi = avgi[adc_ch][0]
                avgq = avgq[adc_ch][0]
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
        # plt.clim(vmin=-0.10, vmax=0.07)
        plt.clim(vmin=None, vmax=None)
        plt.colorbar(label='Amps [ADC level]')

        # plt.plot(x_sweep, amps[1])
        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)