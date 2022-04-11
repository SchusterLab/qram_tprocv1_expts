import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm

class PulseProbeEFSpectroscopyProgram(RAveragerProgram):
    def initialize(self):
        cfg=AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        
        self.res_ch = cfg.hw.soc.dacs.readout.ch[cfg.device.readout.dac]
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch[cfg.device.qubit.dac]

        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist[cfg.device.readout.dac])
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist[cfg.device.qubit.dac])
    
        self.q_rp=self.ch_page(self.qubit_ch) # get register page for qubit_ch
        self.r_freq=self.sreg(self.qubit_ch, "freq") # get frequency register for qubit_ch 
        self.r_freq2 = 4
        
        self.f_res=self.freq2reg(cfg.device.readout.frequency) # conver f_res to dac register value
        self.readout_length=self.us2cycles(cfg.device.readout.readout_length)

        for ch in [0,1]: # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=self.readout_length,
                                 freq=cfg.device.readout.frequency, gen_ch=self.res_ch)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
        
        self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma)
        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge)
        self.f_start = self.freq2reg(cfg.expt.start)
        self.f_step = self.freq2reg(cfg.expt.step)
        
        self.safe_regwi(self.q_rp, self.r_freq2, self.f_start) # send start frequency to r_freq2
        
        # add pre-defined qubit and readout pulses to respective channels
        self.add_gauss(ch=self.qubit_ch, name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)
        self.set_pulse_registers(
            ch=self.res_ch,
            style="const",
            freq=self.f_res,
            phase=self.deg2reg(cfg.device.readout.phase, gen_ch=self.res_ch),
            gain=cfg.device.readout.gain,
            length=self.readout_length)
           
        self.synci(self.us2cycles(1)) # give processor some time to configure pulses
    
    def body(self):
        cfg=AttrDict(self.cfg)

        self.set_pulse_registers(
            ch=self.qubit_ch,
            style="arb",
            freq=self.f_ge,
            phase=0,
            gain=cfg.device.qubit.pulses.pi_ge.gain,
            waveform="pi_qubit")
        self.pulse(ch=self.qubit_ch)

        # setup and play ef probe pulse
        self.set_pulse_registers(
            ch=self.qubit_ch,
            style="const",
            freq=0, # freq set by update
            phase=0,
            gain=cfg.expt.gain,
            length=self.us2cycles(cfg.expt.length))
        self.mathi(self.q_rp, self.r_freq, self.r_freq2, "+", 0)
        self.pulse(ch=self.qubit_ch)

        # go back to ground state if in e
        self.set_pulse_registers(
            ch=self.qubit_ch,
            style="arb",
            freq=self.f_ge,
            phase=0,
            gain=cfg.device.qubit.pulses.pi_ge.gain,
            waveform="pi_qubit")
        self.pulse(ch=self.qubit_ch)

        self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns
        self.measure(pulse_ch=self.res_ch, 
             adcs=[0,1],
             adc_trig_offset=cfg.device.readout.trig_offset,
             wait=True,
             syncdelay=self.us2cycles(cfg.device.readout.relax_delay))
    
    def update(self):
        self.mathi(self.q_rp, self.r_freq2, self.r_freq2, '+', self.f_step) # update frequency list index
        

class PulseProbeEFSpectroscopyExperiment(Experiment):
    """
    PulseProbe EF Spectroscopy Experiment
    Experimental Config:
    expt = dict(
        start: start ef probe frequency [MHz]
        step: step ef probe frequency
        expts: number experiments stepping from start
        reps: number averages per experiment
        rounds: number repetitions of experiment sweep
        length: ef const pulse length [us]
        gain: ef const pulse gain [dac units]
    )
    """

    def __init__(self, soccfg=None, path='', prefix='PulseProbeEFSpectroscopy', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

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
        adc_ch = self.cfg.hw.soc.adcs.readout.ch[self.cfg.device.readout.adc]

        qspec_ef=PulseProbeEFSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
        x_pts, avgi, avgq = qspec_ef.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug)        
        
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
            data['fit_avgi']=dsfit.fitlor(data["xpts"][1:-1], data['avgi'][1:-1])
            data['fit_avgq']=dsfit.fitlor(data["xpts"][1:-1], -data['avgq'][1:-1])
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 
        plt.figure(figsize=(10,8))
        plt.subplot(211, title="Pulse Probe EF Spectroscopy", ylabel="I [adc level]")
        plt.plot(data["xpts"][1:-1], data["avgi"][1:-1],'o-')
        if fit:
            plt.plot(data["xpts"][1:-1], dsfit.lorfunc(data["fit_avgi"], data["xpts"][1:-1]))
            print(f'Found peak in avgi at [MHz] {data["fit_avgi"][2]}, HWHM {data["fit_avgi"][3]}')
        plt.subplot(212, xlabel="Pulse Frequency (MHz)", ylabel="Q [adc level]")
        plt.plot(data["xpts"][1:-1], data["avgq"][1:-1],'o-')
        if fit:
            plt.plot(data["xpts"][1:-1], -dsfit.lorfunc(data["fit_avgq"], data["xpts"][1:-1]))
            # plt.axvline(3593.2, c='k', ls='--')
            print(f'Found peak in avgq at [MHz] {data["fit_avgq"][2]}, HWHM {data["fit_avgq"][3]}')
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)


class PulseProbeEFPowerSweepSpectroscopyExperiment(Experiment):
    """
    Pulse probe EF power wweep spectroscopy experiment
    Experimental Config
        expt = dict(
        start_f: start ef probe frequency [MHz]
        step_f: step ef probe frequency
        expts_f: number experiments freq stepping from start
        start_gain: start ef const pulse gain (dac units)
        step_gain
        expts_gain
        reps: number averages per experiment
        rounds: number repetitions of experiment sweep
        length: ef const pulse length [us]
    )
    """

    def __init__(self, soccfg=None, path='', prefix='PulseProbeEFPowerSweepSpectroscopy', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        fpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"]*np.arange(self.cfg.expt["expts_f"])
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
        
        data={"fpts":[], "gainpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
        adc_ch = self.cfg.hw.soc.adcs.readout.ch[self.cfg.device.readout.adc]
        for gain in tqdm(gainpts):
            self.cfg.expt.gain = gain
            self.cfg.expt.start = self.cfg.expt.start_f
            self.cfg.expt.step = self.cfg.expt.step_f
            self.cfg.expt.expts = self.cfg.expt.expts_f
            spec = PulseProbeEFSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
            self.prog = spec
            x_pts, avgi, avgq = spec.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
            avgi = avgi[adc_ch][0]
            avgq = avgq[adc_ch][0]
            amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phase = np.angle(avgi+1j*avgq) # Calculating the phase
                
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)
        
        data["fpts"] = fpts
        data["gainpts"] = gainpts
        
        for k, a in data.items():
            data[k] = np.array(a)
        
        self.data = data
        return data

    def analyze(self, data=None, fit=True, highgain=None, lowgain=None, **kwargs):
        if data is None:
            data=self.data
        
        # Lorentzian fit at highgain [DAC units] and lowgain [DAC units]
        # if fit:
        #     if highgain == None: highgain = data['gainpts'][-1]
        #     if lowgain == None: lowgain = data['gainpts'][0]
        #     i_highgain = np.argmin(np.abs(data['gainpts']-highgain))
        #     i_lowgain = np.argmin(np.abs(data['gainpts']-lowgain))
        #     fit_highpow=dsfit.fitlor(data["fpts"], data["avgi"][i_highgain])
        #     fit_lowpow=dsfit.fitlor(data["fpts"], data["avgi"][i_lowgain])
        #     data['fit'] = [fit_highpow, fit_lowpow]
        #     data['fit_gains'] = [highgain, lowgain]
        #     data['lamb_shift'] = fit_highpow[2] - fit_lowpow[2]
        
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 
        
        x_sweep = data['fpts']
        y_sweep = data['gainpts'] 
        avgi = data['avgi']
        avgq = data['avgq']
        for avgi_gain in avgi:
            avgi_gain -= np.average(avgi_gain)
        for avgq_gain in avgq:
            avgq_gain -= np.average(avgq_gain)


        plt.figure(figsize=(10,12))
        plt.subplot(211, title="Pulse Probe EF Spectroscopy Power Sweep", ylabel="Pulse Gain [adc level]")
        plt.imshow(
            np.flip(avgi, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        plt.clim(vmin=None, vmax=None)
        plt.colorbar(label='Amps-Avg [adc level]')
        
        plt.subplot(212, xlabel="Pulse Frequency (MHz)", ylabel="Pulse Gain [adc level]")
        plt.imshow(
            np.flip(avgi, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        plt.clim(vmin=None, vmax=None)
        plt.colorbar(label='Phases-Avg [radians]')
        
        plt.show()    
        
        # if fit:
        #     fit_highpow, fit_lowpow = data['fit']
        #     highgain, lowgain = data['fit_gains']
        #     plt.axvline(fit_highpow[2], linewidth=0.5, color='0.2')
        #     plt.axvline(fit_lowpow[2], linewidth=0.5, color='0.2')
        #     plt.plot(x_sweep, [highgain]*len(x_sweep), linewidth=0.5, color='0.2')
        #     plt.plot(x_sweep, [lowgain]*len(x_sweep), linewidth=0.5, color='0.2')
        #     print(f'High power peak [MHz]: {fit_highpow[2]}')
        #     print(f'Low power peak [MHz]: {fit_lowpow[2]}')
        #     print(f'Lamb shift [MHz]: {data["lamb_shift"]}')
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)