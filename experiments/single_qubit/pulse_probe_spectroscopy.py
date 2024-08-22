import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from qick import *

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm
import time

import experiments.fitting as fitter

class PulseProbeSpectroscopyProgram(RAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        self.gen_delays = [0]*len(soccfg['gens']) # need to calibrate via oscilloscope

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        
        super().__init__(soccfg, self.cfg)

    def reset_and_sync(self):
        # Phase reset all channels except readout DACs (since mux ADCs can't be phase reset)
        for ch in self.gen_chs.keys():
            if ch not in self.measure_chs: # doesn't work for the mux ADCs
                # print('resetting', ch)
                self.setup_and_pulse(ch=ch, style='const', freq=100, phase=0, gain=100, length=10, phrst=1)
        self.sync_all(10)

    def set_gen_delays(self):
        for ch in self.gen_chs:
            delay_ns = self.cfg.hw.soc.dacs.delay_chs.delay_ns[np.argwhere(np.array(self.cfg.hw.soc.dacs.delay_chs.ch) == ch)[0][0]]
            delay_cycles = self.us2cycles(delay_ns*1e-3, gen_ch=ch)
            self.gen_delays[ch] = delay_cycles

    def sync_all(self, t=0):
        super().sync_all(t=t, gen_t0=self.gen_delays)


    def initialize(self):
        cfg=AttrDict(self.cfg)
        self.cfg.update(cfg.expt)

        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        self.checkZZ = False
        if qZZ is not None: self.checkZZ = True
        else: qZZ = qTest
        self.checkEF = self.cfg.expt.checkEF

        self.num_qubits_sample = len(self.cfg.device.readout.frequency)

        # all of these saved self.whatever instance variables should be indexed by the actual qubit number as opposed to qubits_i. this means that more values are saved as instance variables than is strictly necessary, but this is overall less confusing
        self.adc_chs = cfg.hw.soc.adcs.readout.ch
        self.res_chs = self.cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = self.cfg.hw.soc.dacs.readout.type
        self.qubit_chs = self.cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = self.cfg.hw.soc.dacs.qubit.type

        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs] # get register page for qubit_chs

        self.f_ges = np.reshape(self.cfg.device.qubit.f_ge, (4,4))
        self.f_efs = np.reshape(self.cfg.device.qubit.f_ef, (4,4))
        self.pi_ge_gains = np.reshape(self.cfg.device.qubit.pulses.pi_ge.gain, (4,4))
        self.pi_ge_sigmas = np.reshape(self.cfg.device.qubit.pulses.pi_ge.sigma, (4,4))
        self.pi_ge_half_gains = np.reshape(self.cfg.device.qubit.pulses.pi_ge.half_gain, (4,4))
        self.pi_ge_half_gain_pi_sigmas = np.reshape(self.cfg.device.qubit.pulses.pi_ge.half_gain_pi_sigma, (4,4))
        self.pi_ef_gains = np.reshape(self.cfg.device.qubit.pulses.pi_ef.gain, (4,4))
        self.pi_ef_sigmas = np.reshape(self.cfg.device.qubit.pulses.pi_ef.sigma, (4,4))
        self.pi_ef_half_gains = np.reshape(self.cfg.device.qubit.pulses.pi_ef.half_gain, (4,4))
        self.pi_ef_half_gain_pi_sigmas = np.reshape(self.cfg.device.qubit.pulses.pi_ef.half_gain_pi_sigma, (4,4))

        self.f_res_regs = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(self.cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [1+self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)]

        # declare all res dacs
        self.measure_chs = []
        mask = [] # indices of mux_freqs, mux_gains list to play
        mux_mixer_freq = None
        mux_freqs = [0]*4 # MHz
        mux_gains = [0]*4
        mux_ro_ch = None
        mux_nqz = None
        for q in range(self.num_qubits_sample):
            assert self.res_ch_types[q] in ['full', 'mux4']
            if self.res_ch_types[q] == 'full':
                if self.res_chs[q] not in self.measure_chs:
                    self.declare_gen(ch=self.res_chs[q], nqz=cfg.hw.soc.dacs.readout.nyquist[q])
                    self.measure_chs.append(self.res_chs[q])
                
            elif self.res_ch_types[q] == 'mux4':
                assert self.res_chs[q] == 6
                mask.append(q)
                if mux_mixer_freq is None: mux_mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[q]
                else: assert mux_mixer_freq == cfg.hw.soc.dacs.readout.mixer_freq[q] # ensure all mux channels have specified the same mixer freq
                mux_freqs[q] = cfg.device.readout.frequency[q]
                mux_gains[q] = cfg.device.readout.gain[q]
                mux_ro_ch = self.adc_chs[q]
                mux_nqz = cfg.hw.soc.dacs.readout.nyquist[q]
                if self.res_chs[q] not in self.measure_chs:
                    self.measure_chs.append(self.res_chs[q])
        if 'mux4' in self.res_ch_types: # declare mux4 channel
            self.declare_gen(ch=6, nqz=mux_nqz, mixer_freq=mux_mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=mux_ro_ch)

        # declare adcs - readout for all qubits everytime, defines number of buffers returned regardless of number of adcs triggered
        for q in range(self.num_qubits_sample):
            if self.adc_chs[q] not in self.ro_chs:
                self.declare_readout(ch=self.adc_chs[q], length=self.readout_lengths_adc[q], freq=self.cfg.device.readout.frequency[q], gen_ch=self.res_chs[q])

        # declare qubit dacs
        for q in range(self.num_qubits_sample):
            mixer_freq = None
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in self.gen_chs:
                self.declare_gen(ch=self.qubit_chs[q], nqz=cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)

        self.f_start = self.freq2reg(cfg.expt.start, gen_ch=self.qubit_chs[qTest]) # get start/step frequencies
        self.f_step = self.freq2reg(cfg.expt.step, gen_ch=self.qubit_chs[qTest])
        
        # add qubit pulses to respective channels
        self.length_cycles = self.us2cycles(cfg.expt.length, gen_ch=self.qubit_chs[qTest])
        if self.cfg.expt.pulse_type == 'flat_top':
            self.add_gauss(ch=self.qubit_chs[qTest], name="pulse_test", sigma=3, length=3*4)
        elif self.cfg.expt.pulse_type == 'gauss':
            self.add_gauss(ch=self.qubit_chs[qTest], name="pulse_test", sigma=self.length_cycles, length=self.length_cycles*4)
        
        if self.checkZZ:
            self.pisigma_ge_qZZ = self.us2cycles(self.pi_ge_sigmas[qZZ, qZZ], gen_ch=self.qubit_chs[qZZ])
            self.add_gauss(ch=self.qubit_chs[qZZ], name='pi_qZZ', sigma=self.pisigma_ge_qZZ, length=self.pisigma_ge_qZZ*4)
        if self.checkEF:
            self.pisigma_ge_init = self.us2cycles(self.pi_ge_sigmas[qTest, qZZ], gen_ch=self.qubit_chs[qTest])
            self.gain_ge_init = self.pi_ge_gains[qTest, qZZ] if self.pi_ge_gains[qTest, qZZ] > 0 else self.pi_ge_gains[qTest, qTest] # this contingency is possible if the ge pulse is not calibrated but we want to calibrate the EF pulse for a specific ZZ configuration
            self.f_ge_init_reg = self.freq2reg(self.f_ges[qTest, qZZ], gen_ch=self.qubit_chs[qTest])
            self.add_gauss(ch=self.qubit_chs[qTest], name='pi_ge', sigma=self.pisigma_ge_init, length=self.pisigma_ge_init*4)

        # add readout pulses to respective channels
        if 'mux4' in self.res_ch_types:
            self.set_pulse_registers(ch=6, style="const", length=max(self.readout_lengths_dac), mask=mask)
        for q in range(self.num_qubits_sample):
            if self.res_ch_types[q] != 'mux4':
                if cfg.device.readout.gain[q] < 1:
                    gain = int(cfg.device.readout.gain[q] * 2**15)
                self.set_pulse_registers(ch=self.res_chs[q], style="const", freq=self.f_res_regs[q], phase=0, gain=gain, length=max(self.readout_lengths_dac))

        # initialize registers
        if self.qubit_ch_types[qTest] == 'int4':
            self.r_freq = self.sreg(self.qubit_chs[qTest], "freq") # get freq register for qubit_ch    
        else: self.r_freq = self.sreg(self.qubit_chs[qTest], "freq") # get freq register for qubit_ch    
        self.r_freq2 = 4
        self.safe_regwi(self.q_rps[qTest], self.r_freq2, self.f_start)

        self.set_gen_delays()
        self.sync_all(200)
    
    def body(self):
        cfg=AttrDict(self.cfg)
        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        if qZZ is None: qZZ = qTest

        self.reset_and_sync()

        # initializations as necessary
        if self.checkZZ:
            assert self.pi_ge_gains[qZZ, qZZ] > 0
            self.setup_and_pulse(ch=self.qubit_chs[qZZ], style="arb", phase=0, freq=self.freq2reg(self.f_ges[qZZ, qZZ], gen_ch=self.qubit_chs[qZZ]), gain=self.pi_ge_gains[qZZ, qZZ], waveform="pi_qZZ")
            self.sync_all()
        if self.checkEF:
            assert self.gain_ge_init > 0
            assert self.f_ge_init_reg > 0
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_init_reg, phase=0, gain=self.gain_ge_init, waveform="pi_ge")
            # print(self.reg2freq(self.f_ge_init_reg, gen_ch=self.qubit_chs[qTest]), self.gain_ge_init, self.pisigma_ge_init)
            self.sync_all()

        # play probe pulse
        if self.cfg.expt.pulse_type == 'flat_top':
            self.set_pulse_registers(ch=self.qubit_chs[qTest], style="flat_top", phase=0, freq=self.f_start, gain=cfg.expt.gain, length=self.length_cycles, waveform="pulse_test")
        elif self.cfg.expt.pulse_type == 'gauss':
            self.set_pulse_registers(ch=self.qubit_chs[qTest], style="arb", phase=0, freq=self.f_start, gain=cfg.expt.gain, waveform="pulse_test")
        elif self.cfg.expt.pulse_type == 'const':
            self.set_pulse_registers(ch=self.qubit_chs[qTest], style="const", freq=self.f_start, phase=0, gain=cfg.expt.gain, length=self.length_cycles)
        self.mathi(self.q_rps[qTest], self.r_freq, self.r_freq2, "+", 0)
        self.pulse(ch=self.qubit_chs[qTest])
        self.sync_all()

        if self.checkEF: # map excited back to qubit ground state for measurement
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_init_reg, phase=0, gain=self.gain_ge_init, waveform="pi_ge")

        # align channels and measure
        self.sync_all()
        self.measure(
            pulse_ch=self.measure_chs, 
            adcs=self.adc_chs,
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(max([cfg.device.readout.relax_delay[q] for q in range(4)]))
        )

    def update(self):
        qTest = self.cfg.expt.qTest
        self.mathi(self.q_rps[qTest], self.r_freq2, self.r_freq2, '+', self.f_step) # update freq

        # self.mathi(self.q_rp, self.r_freq, self.r_freq, '+', self.f_step) # update frequency list index
 
# ====================================================== #

class PulseProbeSpectroscopyExperiment(Experiment):
    """
    PulseProbe Spectroscopy Experiment
    Experimental Config:
        start: Qubit frequency [MHz]
        step
        expts: Number of experiments stepping from start
        reps: Number of averages per point
        rounds: Number of start to finish sweeps to average over
        length: Qubit probe constant pulse length [us]
        gain: Qubit pulse gain [DAC units]

        qTest
        qZZ
        checkEF
    """

    def __init__(self, soccfg=None, path='', prefix='PulseProbeSpectroscopy', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.readout.frequency)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update({key3: [value3]*num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})

        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        self.checkZZ = False
        if qZZ is not None: self.checkZZ = True
        else: qZZ = qTest
        self.checkEF = self.cfg.expt.checkEF

        print(f'Running pulse probe {"EF " if self.cfg.expt.checkEF else ""}on Q{qTest} {"with ZZ Q" + str(qZZ) if self.checkZZ else ""}')

        qspec = PulseProbeSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
        xpts, avgi, avgq = qspec.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress)
        avgi = avgi[qTest][0]
        avgq = avgq[qTest][0]
        amps = np.abs(avgi+1j*avgq)
        phases = np.angle(avgi+1j*avgq) # Calculating the phase        
        
        data={'xpts':xpts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases}
        self.data=data
        return data

    def analyze(self, data=None, fit=True, signs=[1,1,1], **kwargs):
        if data is None:
            data=self.data
        if fit:
            xdata = data['xpts'][1:-1]
            data['fit_amps'], data['fit_err_amps'] = fitter.fitlor(xdata, signs[0]*data['amps'][1:-1])
            data['fit_avgi'], data['fit_err_avgi'] = fitter.fitlor(xdata, signs[1]*data['avgi'][1:-1])
            data['fit_avgq'], data['fit_err_avgq'] = fitter.fitlor(xdata, signs[2]*data['avgq'][1:-1])
        return data

    def display(self, data=None, fit=True, signs=[1,1,1], **kwargs):
        if data is None:
            data=self.data 

        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        self.checkZZ = False
        if qZZ is not None: self.checkZZ = True
        else: qZZ = qTest

        if 'mixer_freq' in self.cfg.hw.soc.dacs.qubit:
            xpts = self.cfg.hw.soc.dacs.qubit.mixer_freq[qTest] + data['xpts'][1:-1]
        else: 
            xpts = data['xpts'][1:-1]

        plt.figure(figsize=(9, 11))
        title = f"Qubit Spectroscopy {'EF ' if self.cfg.expt.checkEF else ''}on Q{qTest} (Gain {self.cfg.expt.gain}, {self.cfg.expt.length} us {self.cfg.expt.pulse_type}){(', ZZ Q'+str(qZZ)) if self.checkZZ else ''}"
        plt.subplot(311, title=title, ylabel="Amplitude [ADC units]")
        plt.plot(xpts, data["amps"][1:-1],'.-')
        # plt.plot(xpts, (data["amps"][1:-1]-np.min(data['amps'][1:-1]))/(np.max(data["amps"][1:-1])-np.min(data['amps'][1:-1])),'.-')
        # plt.yscale('log')
        if fit:
            plt.plot(xpts, signs[0]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_amps"]))
            print(f'Found peak in amps at [MHz] {data["fit_amps"][2]}, HWHM {data["fit_amps"][3]}')

        plt.subplot(312, ylabel="I [ADC units]")
        plt.plot(xpts, data["avgi"][1:-1],'.-')
        if fit:
            plt.plot(xpts, signs[1]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_avgi"]))
            print(f'Found peak in I at [MHz] {data["fit_avgi"][2]}, HWHM {data["fit_avgi"][3]}')
        # freq = 4386.167531612781
        # shift = 0.494
        # plt.axvline(freq, c='k', ls='--')
        # plt.axvline(freq-shift, c='k', ls='--')
        # plt.axvline(freq-2*shift, c='k', ls='--')
        # plt.axvline(freq-3*shift, c='k', ls='--')
        # plt.axvline(freq+shift, c='k', ls='--')
        # plt.axvline(freq+2*shift, c='k', ls='--')
        # plt.axvline(freq+3*shift, c='k', ls='--')
        plt.subplot(313, xlabel="Pulse Frequency (MHz)", ylabel="Q [ADC units]")
        plt.plot(xpts, data["avgq"][1:-1],'.-')
        if fit:
            plt.plot(xpts, signs[2]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_avgq"]))
            # plt.axvline(3593.2, c='k', ls='--')
            print(f'Found peak in Q at [MHz] {data["fit_avgq"][2]}, HWHM {data["fit_avgq"][3]}')

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname

# ====================================================== #

from experiments.single_qubit.resonator_spectroscopy import ResonatorSpectroscopyExperiment
class PulseProbeVoltSweepSpectroscopyExperiment(Experiment):
    """
    PulseProbe Spectroscopy Experiment Sweep Voltage
    Experimental Config:
        start_qf: start qubit frequency (MHz), 
        step_qf: frequency step (MHz), 
        expts_qf: number of experiments in frequency,
        length: Qubit probe constant pulse length [us]
        gain: Qubit pulse gain [DAC units]
        dc_ch: channel on dc_instr to sweep voltage

        start_rf: start resonator frequency (MHz), 
        step_rf: frequency step (MHz), 
        expts_rf: number of experiments in frequency,

        start_volt: start volt, 
        step_volt: voltage step, 
        expts_volt: number of experiments in voltage sweep,

        reps_q: Number of averages per point for pulse probe
        rounds_q: Number of start to finish freq sweeps to average over

        reps_r: Number of averages per point for resonator spectroscopy
    """

    def __init__(self, soccfg=None, path='', dc_instr=None, prefix='PulseProbeVoltSweepSpectroscopy', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)
        self.dc_instr = dc_instr
        self.path = path
        self.config_file = config_file

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
        
        voltpts = self.cfg.expt["start_volt"] + self.cfg.expt["step_volt"]*np.arange(self.cfg.expt["expts_volt"])
        data=dict(
            xpts=[],
            voltpts=[],
            avgi=[],
            avgq=[],
            amps=[],
            phases=[],
            rspec_avgi=[],
            rspec_avgq=[],
            rspec_amps=[],
            rspec_phases=[],
            rspec_fits=[]
        )

        self.cfg.expt.start = self.cfg.expt.start_qf
        self.cfg.expt.step = self.cfg.expt.step_qf
        self.cfg.expt.expts = self.cfg.expt.expts_qf
        self.cfg.expt.reps = self.cfg.expt.reps_q
        self.cfg.expt.rounds = self.cfg.expt.rounds_q

        for volt in tqdm(voltpts):
            self.dc_instr.set_voltage(channel=self.cfg.expt.dc_ch, voltage=volt)
            time.sleep(0.5)

            # Get readout frequency
            rspec = ResonatorSpectroscopyExperiment(
                soccfg=self.soccfg,
                path=self.path,
                config_file=self.config_file,
            )
            rspec.cfg.expt = dict(
                start=self.cfg.expt.start_rf,
                step=self.cfg.expt.step_rf,
                expts=self.cfg.expt.expts_rf,
                reps=self.cfg.expt.reps_r,
                pi_pulse=False,
                qubit=self.cfg.expt.qubit,
            )
            rspec.go(analyze=False, display=False, progress=False, save=False)
            rspec.analyze(fit=True, verbose=False)
            readout_freq = rspec.data['fit'][0]

            self.cfg.device.readout.frequency = readout_freq
            print(f'readout at {readout_freq} at voltage {volt}')

            qspec = PulseProbeSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
            xpts, avgi, avgq = qspec.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False)
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amps = np.abs(avgi+1j*avgq)
            phases = np.angle(avgi+1j*avgq) # Calculating the phase        

            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amps)
            data["phases"].append(phases)

            data["rspec_avgi"].append(rspec.data['avgi'])
            data["rspec_avgq"].append(rspec.data['avgq'])
            data["rspec_amps"].append(rspec.data['amps'])
            data["rspec_phases"].append(rspec.data['phases'])
            data["rspec_fits"].append(rspec.data['fit'])

            time.sleep(0.5)
        # self.dc_instr.initialize()
        self.dc_instr.set_voltage(channel=self.cfg.expt.dc_ch, voltage=0)

        data["rspec_xpts"] = rspec.data['xpts']
        data['xpts'] = xpts
        data['voltpts'] = voltpts
        for k, a in data.items():
            data[k] = np.array(a)
        self.data=data
        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data

        # data.update(
        #     dict(
        #     rspec_avgi=[],
        #     rspec_avgq=[],
        #     rspec_amps=[],
        #     rspec_phases=[],
        #     rspec_fits=[]
        #     )
        # )
        # data["rspec_xpts"] = data['rspec_data'][0]['xpts']
        # for rspec_data in data['rspec_data']:
        #     data["rspec_avgi"].append(rspec_data['avgi'])
        #     data["rspec_avgq"].append(rspec_data['avgq'])
        #     data["rspec_amps"].append(rspec_data['amps'])
        #     data["rspec_phases"].append(rspec_data['phases'])
        #     data["rspec_fits"].append(rspec_data['fit'])

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 
        
        freqs_q = data['xpts']
        freqs_r = data['rspec_xpts']
        x_sweep = 1e3*data['voltpts']
        amps = data['amps']
        # for amps_volt in amps:
        #     amps_volt -= np.average(amps_volt)
        
        # THIS IS THE FIXED EXTENT LIMITS FOR 2D PLOTS
        plt.figure(figsize=(12,12))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1,2])
        plt.subplot(gs[0], title="Pulse Probe Voltage Sweep", ylabel="Resonator Frequency [MHz]")
        y_sweep = freqs_r
        plt.pcolormesh(x_sweep, y_sweep, np.flip(np.rot90(data['rspec_amps']), 0), cmap='viridis')
        rfreqs = [data['rspec_fits'][i][0] for i in range(len(data['voltpts']))]
        plt.scatter(x_sweep, rfreqs, marker='o', color='r')
        if 'add_data' in kwargs:
            for add_data in kwargs['add_data']:
                plt.pcolormesh(
                    1e3*add_data['voltpts'], add_data['rspec_xpts'], np.flip(np.rot90(add_data['rspec_amps']), 0), cmap='viridis')
                rfreqs = [add_data['rspec_fits'][i][0] for i in range(len(add_data['voltpts']))]
                plt.scatter(1e3*add_data['voltpts'], rfreqs, marker='o', color='r')
        plt.xlim(min(x_sweep), max(x_sweep))
        # plt.clim(vmin=None, vmax=None)
        plt.colorbar(label='Amps [ADC level]')

        plt.subplot(gs[1], xlabel=f"DC Voltage (DAC ch {self.cfg.expt.dc_ch}) [mV]", ylabel="Qubit Frequency [MHz]")
        y_sweep = freqs_q
        plt.pcolormesh(x_sweep, y_sweep, np.flip(np.rot90(amps), 0), cmap='viridis')
        plt.xlim(min(x_sweep), max(x_sweep))
        if 'add_data' in kwargs:
            for add_data in kwargs['add_data']:
                y_sweep = add_data['xpts']
                x_sweep = 1e3*add_data['voltpts']
                amps = add_data['amps']
                # for amps_volt in amps:
                #     amps_volt -= np.average(amps_volt)
                plt.pcolormesh(x_sweep, y_sweep, np.flip(np.rot90(amps), 0), cmap='viridis')
        plt.axvline(2.55)
        # plt.clim(vmin=None, vmax=None)
        plt.colorbar(label='Amps [ADC level]')
        
        # if fit: pass
        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname

# ====================================================== #

class PhotonCountingSpectroscopyProgram(RAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        self.gen_delays = [0]*len(soccfg['gens']) # need to calibrate via oscilloscope

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        
        super().__init__(soccfg, self.cfg)

    def reset_and_sync(self):
        # Phase reset all channels except readout DACs (since mux ADCs can't be phase reset)
        for ch in self.gen_chs.keys():
            if ch not in self.measure_chs: # doesn't work for the mux ADCs
                # print('resetting', ch)
                self.setup_and_pulse(ch=ch, style='const', freq=100, phase=0, gain=100, length=10, phrst=1)
        self.sync_all(10)

    def set_gen_delays(self):
        for ch in self.gen_chs:
            delay_ns = self.cfg.hw.soc.dacs.delay_chs.delay_ns[np.argwhere(np.array(self.cfg.hw.soc.dacs.delay_chs.ch) == ch)[0][0]]
            delay_cycles = self.us2cycles(delay_ns*1e-3, gen_ch=ch)
            self.gen_delays[ch] = delay_cycles

    def sync_all(self, t=0):
        super().sync_all(t=t, gen_t0=self.gen_delays)


    def initialize(self):
        cfg=AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        self.qubit = self.cfg.expt.qubit

        self.adc_ch = cfg.hw.soc.adcs.readout.ch
        self.res_meas_ch = cfg.hw.soc.dacs.readout.ch
        self.res_meas_ch_type = cfg.hw.soc.dacs.readout.type
        self.res_pump_ch = cfg.hw.soc.dacs.res_pump.ch
        self.res_pump_type = cfg.hw.soc.dacs.res_pump.type
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type

        self.q_rp=self.ch_page(self.qubit_ch) # get register page for qubit_ch
        self.f_res_pump = float(self.cfg.hw.lo.readout.frequency)*1e-6 + self.cfg.device.readout.lo_sideband*(self.cfg.hw.soc.dacs.readout.mixer_freq + self.cfg.device.readout.frequency)
        print('pump freq', self.f_res_pump)
        self.f_res_pump_reg = self.freq2reg(self.f_res_pump, gen_ch=self.res_pump_ch)
        self.f_res_meas_reg = self.freq2reg(cfg.device.readout.frequency, gen_ch=self.res_meas_ch, ro_ch=self.adc_ch)
        
        self.readout_length_dac = self.us2cycles(cfg.device.readout.readout_length, gen_ch=self.res_meas_ch)
        self.readout_length_adc = self.us2cycles(cfg.device.readout.readout_length, ro_ch=self.adc_ch)

        # declare res meas dac
        mask = None
        mixer_freq = None # MHz
        mux_freqs = None # MHz
        mux_gains = None
        ro_ch = None
        if self.res_meas_ch_type == 'int4':
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
        elif self.res_meas_ch_type == 'mux4':
            assert self.res_meas_ch == 6
            mask = [0, 1, 2, 3] # indices of mux_freqs, mux_gains list to play
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
            mux_freqs = [0]*4
            mux_freqs[cfg.expt.qubit] = cfg.device.readout.frequency
            mux_gains = [0]*4
            mux_gains[cfg.expt.qubit] = cfg.device.readout.gain
            ro_ch=self.adc_ch
        else: assert False
        self.declare_gen(ch=self.res_meas_ch, nqz=cfg.hw.soc.dacs.readout.nyquist, mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)

        # declare res pump dac
        self.declare_gen(ch=self.res_pump_ch, nqz=cfg.hw.soc.dacs.res_pump.nyquist, mixer_freq=0)

        # declare qubit dacs
        mixer_freq = None
        if self.qubit_ch_type == 'int4':
            mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist, mixer_freq=mixer_freq)

        # declare adcs
        self.declare_readout(ch=self.adc_ch, length=self.readout_length_adc, freq=cfg.device.readout.frequency, gen_ch=self.res_meas_ch)

        self.f_start = self.freq2reg(cfg.expt.start, gen_ch=self.qubit_ch) # get start/step frequencies
        self.f_step = self.freq2reg(cfg.expt.step, gen_ch=self.qubit_ch)

        # add qubit and readout pulses to respective channels
        if self.cfg.expt.pulse_type == 'flat_top':
            self.add_gauss(ch=self.qubit_ch, name="qubit", sigma=3, length=3*4)
        elif self.cfg.expt.pulse_type == 'gauss':
            length = self.us2cycles(cfg.expt.length, gen_ch=self.qubit_ch)
            self.add_gauss(ch=self.qubit_ch, name="qubit", sigma=length, length=length*4)

        self.res_pump_length = self.us2cycles(cfg.expt.pump_length, gen_ch=self.res_pump_ch)
        print('pump ch', self.res_pump_ch)
        print('pump length', cfg.expt.pump_length)
        print('pump gain', cfg.expt.pump_gain)
        self.set_pulse_registers(ch=self.res_pump_ch, style="const", freq=self.f_res_pump_reg, phase=0, gain=cfg.expt.pump_gain, length=self.res_pump_length)

        if self.res_meas_ch_type == 'mux4':
            self.set_pulse_registers(ch=self.res_meas_ch, style="const", length=self.readout_length_dac, mask=mask)
        else: self.set_pulse_registers(ch=self.res_meas_ch, style="const", freq=self.f_res_meas_reg, phase=0, gain=cfg.device.readout.gain, length=self.readout_length_dac)

        # initialize registers
        if self.qubit_ch_type == 'int4':
            self.r_freq = self.sreg(self.qubit_ch, "freq") # get freq register for qubit_ch    
        else: self.r_freq = self.sreg(self.qubit_ch, "freq") # get freq register for qubit_ch    
        self.r_freq2 = 4
        self.safe_regwi(self.q_rp, self.r_freq2, self.f_start)

        self.set_gen_delays()
        self.sync_all(200)
    
    def body(self):
        cfg=AttrDict(self.cfg)

        self.reset_and_sync()

        # play res pump pulse
        if self.cfg.expt.pump_gain > 0:
            self.setup_and_pulse(ch=self.res_pump_ch, style="const", freq=self.f_res_pump_reg, phase=0, gain=cfg.expt.pump_gain, length=self.res_pump_length)

        # offset qubit pulse from beginning of pump pulse
        if self.cfg.expt.offset == -1: self.sync_all()
        else: self.synci(self.us2cycles(self.cfg.expt.offset))

        # play qubit probe pulse
        length = self.us2cycles(cfg.expt.length, gen_ch=self.qubit_ch)
        if self.cfg.expt.pulse_type == 'flat_top':
            self.set_pulse_registers(ch=self.qubit_ch, style="flat_top", phase=0, freq=self.f_start, gain=cfg.expt.gain, length=length, waveform="qubit") # play probe pulse
        elif self.cfg.expt.pulse_type == 'gauss':
            self.set_pulse_registers(ch=self.qubit_ch, style="arb", phase=0, freq=self.f_start, gain=cfg.expt.gain, waveform="qubit") # play probe pulse
        elif self.cfg.expt.pulse_type == 'const':
            self.set_pulse_registers(ch=self.qubit_ch, style="const", freq=self.f_start, phase=0, gain=cfg.expt.gain, length=self.us2cycles(cfg.expt.length, gen_ch=self.qubit_ch))
        self.mathi(self.q_rp, self.r_freq, self.r_freq2, "+", 0)
        self.pulse(ch=self.qubit_ch) # play probe pulse

        self.sync_all(self.us2cycles(cfg.expt.meas_delay))

        self.measure(pulse_ch=self.res_meas_ch, 
             adcs=[self.adc_ch],
             adc_trig_offset=cfg.device.readout.trig_offset,
             wait=True,
             syncdelay=self.us2cycles(cfg.device.readout.relax_delay))
    
    def update(self):
        self.mathi(self.q_rp, self.r_freq2, self.r_freq2, '+', self.f_step) # update freq

        # self.mathi(self.q_rp, self.r_freq, self.r_freq, '+', self.f_step) # update frequency list index
 
# ====================================================== #

class PhotonCountingExperiment(Experiment):
    """
    PulseProbe Spectroscopy Experiment
    Experimental Config:
        start: Qubit frequency [MHz]
        step
        expts: Number of experiments stepping from start
        reps: Number of averages per point
        rounds: Number of start to finish sweeps to average over
        length: Qubit probe constant pulse length [us]
        gain: Qubit pulse gain [DAC units]
        pump_gain
        pump_length (us)
        offset: delay start of qubit pulse from start of pump; -1 indicates to start qubit pulse at end of pump
        meas_delay: delay between qubit tone and measurement
    """

    def __init__(self, soccfg=None, path='', prefix='PhotonCounting', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        q_ind = self.cfg.expt.qubit

        self.num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, list) and len(value) == self.num_qubits_sample:
                    subcfg.update({key: value[q_ind]})
                elif isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if isinstance(value3, list) and len(value3) == self.num_qubits_sample:
                                value2.update({key3: value3[q_ind]})                                

        qspec = PhotonCountingSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
        xpts, avgi, avgq = qspec.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress)
        avgi = avgi[0][0]
        avgq = avgq[0][0]
        amps = np.abs(avgi+1j*avgq)
        phases = np.angle(avgi+1j*avgq) # Calculating the phase        
        
        data={'xpts':xpts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases}
        self.data=data
        return data

    def analyze(self, data=None, fit=True, npeaks=None, chi=None, search_span=None, signs=[1, 1, 1]):
        if data is None:
            data=self.data
        if fit:
            assert None not in [npeaks, chi]

            for i_axis, axis in enumerate(['amps', 'avgi', 'avgq']):
                xfit = data['xpts']
                ypts = np.copy(data[axis])
                fitparams=[None]*6
                fit_peak_loc, fit_err_peak_loc = fitter.fitnlor(xfit, ypts**2, npeaks=npeaks, chi_guess=chi, f0_guess=self.cfg.device.qubit.f_ge)
                data[f'fit_peaks_{axis}'] = fit_peak_loc
                # print(axis, data[f'fit_peaks_{axis}'])
                data[f'fit_peaks_err_{axis}'] = fit_err_peak_loc

            # xpts = np.copy(data['xpts'])
            # for i_axis, axis in enumerate(['amps', 'avgi', 'avgq']):
            #     data[f'fit_peaks_{axis}'] = []
            #     data[f'fit_peaks_err_{axis}'] = []
            # data['fit_peak_range'] = []
            # fq = self.cfg.device.qubit.f_ge

            # peak_heights = [[], [], []]
            # peak_locs = [[], [], []]
            # for i in range(npeaks):
            #     est_peak_loc = np.argmin(np.abs(xpts - (fq - i*chi)))
            #     search_span_units = np.argmin(np.abs(xpts - (search_span + xpts[0])))
            #     xfit = data['xpts'][est_peak_loc-search_span_units//2:est_peak_loc+search_span_units//2]
            #     for i_axis, axis in enumerate(['amps', 'avgi', 'avgq']):
            #         ypts = np.copy(data[axis])
            #         max_i = est_peak_loc-search_span_units//2 + np.argmax(ypts[est_peak_loc-search_span_units//2:est_peak_loc+search_span_units//2]) 
            #         yfit = signs[i_axis]*data[axis][est_peak_loc-search_span_units//2:est_peak_loc+search_span_units//2]
            #         fit_peak_loc, fit_err_peak_loc = fitter.fitlor(xfit, yfit**2)
            #         data[f'fit_peaks_{axis}'].append(fit_peak_loc)
            #         data[f'fit_peaks_err_{axis}'].append(fit_err_peak_loc)
            #         peak_heights[i_axis].append(ypts[max_i])
            #         peak_locs[i_axis].append(fit_peak_loc[2])
            #     data['fit_peak_range'].append(xfit)

            # for i_axis, axis in enumerate(['amps', 'avgi', 'avgq']):
            #     ypts = np.copy(data[axis])
            #     peak_heights[i_axis] -= np.min(ypts)
            #     peak_heights[i_axis] *= peak_heights[i_axis]
            #     peak_heights[i_axis] /= sum(peak_heights[i_axis])
            #     print(f'peak heights {axis} as probabilities', peak_heights[i_axis])
            #     print(f'peak locs {axis}', peak_locs)

            #     data[f'fit_photon_{axis}'], data[f'fit_photon_err_{axis}'] = fitter.fit_poisson(np.arange(npeaks, dtype=np.int16), peak_heights[i_axis])

        return data

    def display(self, data=None, fit=True, npeaks=None, signs=[1,1,1]):
        if data is None:
            data=self.data 

        if 'mixer_freq' in self.cfg.hw.soc.dacs.qubit:
            xpts = self.cfg.hw.soc.dacs.qubit.mixer_freq + data['xpts']
        else: 
            xpts = data['xpts']


        # plt.figure(figsize=(10, 11))
        title = f"Q{self.cfg.expt.qubit} (Res Pump Gain {self.cfg.expt.pump_gain}, Qubit Gain {self.cfg.expt.gain}, {self.cfg.expt.length} us {self.cfg.expt.pulse_type})"
        # plt.subplot(311, title=title, ylabel="Power [ADC units sqrd]")
        # plt.plot(xpts, data["amps"],'.')
        # if fit:
        #     for i in range(len(data['fit_peaks_amps'])):
        #         plt.plot(data['fit_peak_range'][i], signs[0]*np.sqrt(fitter.lorfunc(data['fit_peak_range'][i], *data['fit_peaks_amps'][i])))
        #         plt.axvline(data['fit_peaks_amps'][i][2], color='r', linestyle='--', alpha=0.5)
        #     print(f"fit photon number {data['fit_photon_amps'][0]} +/- {data['fit_photon_err_amps'][0][0]}")
        # # plt.plot(xpts, (data["amps"][1:-1]-np.min(data['amps'][1:-1]))/(np.max(data["amps"][1:-1])-np.min(data['amps'][1:-1])),'.-')
        # # plt.yscale('log')

        plt.figure()
        plt.title(title)
        plt.ylabel("Power [ADC units sqrd]")
        plt.xlabel('Pulse Frequency (MHz)')
        plt.plot(xpts, data["amps"]**2,'.')
        if fit:
            plt.plot(xpts, fitter.nlorfunc(xpts, npeaks, *data['fit_peaks_amps']), label='$\\bar{n}$: '+f'{data["fit_peaks_amps"][4]:.2f} $\pm$ {data["fit_peaks_err_amps"][4][4]:.5f}')
            print(f"fit photon number {data['fit_peaks_amps'][4]} +/- {data['fit_peaks_err_amps'][4][4]}")
        plt.legend()
        

        # plt.subplot(312, ylabel="I [ADC units]")
        # plt.plot(xpts, data["avgi"],'.')
        # # if fit:
        # #     for i in range(len(data['fit_peaks_avgi'])):
        # #         plt.plot(data['fit_peak_range'][i], signs[1]*np.sqrt(fitter.lorfunc(data['fit_peak_range'][i], *data['fit_peaks_avgi'][i])))
        # #         plt.axvline(data['fit_peaks_avgi'][i][2], color='r', linestyle='--', alpha=0.5)
        # #     print(f"fit photon number {data['fit_photon_avgi'][0]} +/- {data['fit_photon_err_avgi'][0][0]}")

        # plt.subplot(313, xlabel="Pulse Frequency (MHz)", ylabel="Q [ADC units]")
        # plt.plot(xpts, data["avgq"],'.')
        # # if fit:
        # #     for i in range(len(data['fit_peaks_avgq'])):
        # #         plt.plot(data['fit_peak_range'][i], signs[2]*np.sqrt(fitter.lorfunc(data['fit_peak_range'][i], *data['fit_peaks_avgq'][i])))
        # #         plt.axvline(data['fit_peaks_avgq'][i][2], color='r', linestyle='--', alpha=0.5)
        # #     print(f"fit photon number {data['fit_photon_avgq'][0]} +/- {data['fit_photon_err_avgq'][0][0]}")

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
