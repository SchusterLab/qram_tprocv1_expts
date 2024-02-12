import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import time
from copy import deepcopy

from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

import experiments.fitting as fitter
from experiments.single_qubit.single_shot import HistogramProgram
from scipy.signal import find_peaks

class ResonatorSpectroscopyExperiment(Experiment):
    """
    Resonator Spectroscopy Experiment - just reuses histogram experiment because somehow it's better
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
        self.qubit = self.cfg.expt.qubit
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        if isinstance(value2, dict):
                            for key3, value3 in value2.items():
                                if not(isinstance(value3, list)):
                                    value2.update({key3: [value3]*num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})

        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
        for f in tqdm(xpts, disable=not progress):
            self.cfg.expt.frequency = f
            cfg = AttrDict(deepcopy(self.cfg))
            cfg.device.readout.frequency[self.qubit] = f
            rspec = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
            avgi, avgq = rspec.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
            datai, dataq = rspec.collect_shots()
            avgi = np.average(datai)
            avgq = np.average(dataq)
            amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phase = np.angle(avgi+1j*avgq) # Calculating the phase
            self.prog = rspec
           
            data["xpts"].append(f)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)
        
        for k, a in data.items():
            data[k]=np.array(a)
        
        self.data=data

        return data

    def analyze(self, data=None, fit=False, findpeaks=False, verbose=True, coarse_scan = False, **kwargs):
        if data is None:
            data=self.data
            
        if fit:
            xdata = data["xpts"][1:-1]
            ydata = data['amps'][1:-1]
            #data['fit'], data['fit_err'] = fitter.fithanger(xdata, ydata, fitparams=fitparams)
            fitparams = [max(ydata), -(max(ydata)-min(ydata)), xdata[np.argmin(ydata)], 0.1 ]
            # print(fitparams)
            data["lorentz_fit"]=dsfit.fitlor(xdata, ydata, fitparams=fitparams)
            print('From Fit:')
            print(f'\tf0: {data["lorentz_fit"][2]}')
            print(f'\tkappa[MHz]: {data["lorentz_fit"][3]*2}')
            # if isinstance(data['fit'], (list, np.ndarray)):
            #     f0, Qi, Qe, phi, scale, a0, slope = data['fit']
            #     if 'lo' in self.cfg.hw:
            #         print(float(self.cfg.hw.lo.readout.frequency)*1e-6)
            #         print(f0)
            #     if verbose:
            #         print(f'\nFreq with minimum transmission: {xdata[np.argmin(ydata)]}')
            #         print(f'Freq with maximum transmission: {xdata[np.argmax(ydata)]}')
            #         print('From fit:')
            #         print(f'\tf0: {f0}')
            #         print(f'\tQi: {Qi}')
            #         print(f'\tQe: {Qe}')
            #         print(f'\tQ0: {1/(1/Qi+1/Qe)}')
            #         print(f'\tkappa [MHz]: {f0*(1/Qi+1/Qe)}')
            #         print(f'\tphi [radians]: {phi}')
            
        if findpeaks:
            maxpeaks, minpeaks = dsfit.peakdetect(data['amps'][1:-1], x_axis=data['xpts'][1:-1], lookahead=10, delta=0.01)
            data['maxpeaks'] = maxpeaks
            data['minpeaks'] = minpeaks
        
        if coarse_scan: 
            xdata = data["xpts"][1:-1]
            ydata = data['amps'][1:-1]
            coarse_peaks = find_peaks(-ydata, distance=100, prominence= 0.9, width=3, threshold = 0.9, rel_height=1)  
            data['coarse_peaks_index'] = coarse_peaks 
            data['coarse_peaks'] = xdata[coarse_peaks[0]]
        return data

    def display(self, data=None, fit=True, findpeaks=False, coarse_scan = False, **kwargs):
        if data is None:
            data=self.data 

        if 'lo' in self.cfg.hw:
            xpts = float(self.cfg.hw.lo.readout.frequency)*1e-6 + self.cfg.device.readout.lo_sideband[self.qubit]*(self.cfg.hw.soc.dacs.readout.mixer_freq[self.qubit] + data['xpts'][1:-1])
            
        else:
            xpts = data['xpts'][1:-1]

        plt.figure(figsize=(16,16))
        plt.subplot(311, title=f"Resonator {self.qubit}  Spectroscopy at gain {self.cfg.device.readout.gain[self.qubit]}",  ylabel="Amps [ADC units]")
        plt.plot(xpts, data['amps'][1:-1],'o-')
        # if fit:
        #     plt.plot(xpts, fitter.hangerS21func_sloped(data["xpts"][1:-1], *data["fit"]))
        if fit:
            plt.plot(xpts, dsfit.lorfunc(data["lorentz_fit"], xpts), label='Lorentzian fit')
        if findpeaks:
            for peak in data['minpeaks']:
                plt.axvline(peak[0], linestyle='--', color='0.2')
                print(f'Found peak [MHz]: {peak[0]}')

        if coarse_scan:
            num_peaks = len(data['coarse_peaks_index'][0])
            print('number of peaks:', num_peaks)
            peak_indicies = data['coarse_peaks_index'][0]
            for i in range(num_peaks):
                peak = peak_indicies[i]
                plt.axvline(xpts[peak], linestyle='--', color='0.2')
           
 
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

        self.qubit = self.cfg.expt.qubit
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        if isinstance(value2, dict):
                            for key3, value3 in value2.items():
                                if not(isinstance(value3, list)):
                                    value2.update({key3: [value3]*num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})

        data={"xpts":[], "gainpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
        for gain in tqdm(gainpts, disable=not progress):
            self.cfg.device.readout.gain[self.qubit] = gain
            data["avgi"].append([])
            data["avgq"].append([])
            data["amps"].append([])
            data["phases"].append([])

            for f in tqdm(xpts, disable=True):
                cfg = AttrDict(deepcopy(self.cfg))
                cfg.device.readout.frequency[self.qubit] = f
                rspec = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
                avgi, avgq = rspec.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                datai, dataq = rspec.collect_shots()
                avgi = np.average(datai)
                avgq = np.average(dataq)
                self.prog = rspec
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

        if 'lo' in self.cfg.hw:
            inner_sweep = float(self.cfg.hw.lo.readout.frequency)*1e-6 + self.cfg.device.readout.lo_sideband[self.qubit]*(self.cfg.hw.soc.dacs.readout.mixer_freq[self.qubit] + data['xpts'])
        else: inner_sweep = data['xpts']
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

        plt.xlabel("Resonator Frequency [MHz]")
        plt.ylabel("Resonator Gain [DAC level]")
        plt.colorbar(label='Amps-Avg [ADC level]')
        plt.show()

        print(y_sweep[-1])
        plt.plot(x_sweep, amps[-1,:])
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