import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter

class T1Program(RAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        self.gen_delays = [0]*len(soccfg['gens']) # need to calibrate via oscilloscope

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
        
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
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        self.checkEF = self.cfg.expt.checkEF

        qTest = self.cfg.expt.qTest
        self.num_qubits_sample = len(self.cfg.device.readout.frequency)
        
        self.adc_chs = cfg.hw.soc.adcs.readout.ch
        self.res_chs = cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = cfg.hw.soc.dacs.readout.type
        self.qubit_chs = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = cfg.hw.soc.dacs.qubit.type

        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs] # get register page for qubit_chs
        self.r_wait = 3
        self.safe_regwi(self.q_rps[qTest], self.r_wait, self.us2cycles(cfg.expt.start))
        
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

        self.f_res_regs = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)]

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

        self.pi_sigma = self.us2cycles(self.pi_ge_sigmas[qTest, qTest], gen_ch=self.qubit_chs[qTest])

        # add qubit pulses to respective channels
        if self.cfg.device.qubit.pulses.pi_ge.type[qTest].lower() == 'gauss' and self.pi_sigma > 0:
            self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)

        if self.checkEF:
            self.pi_ef_sigma = self.us2cycles(self.pi_ef_sigmas[qTest, qTest], gen_ch=self.qubit_chs[qTest])
            self.add_gauss(ch=self.qubit_chs[qTest], name="pi_ef_qubit", sigma=self.pi_ef_sigma, length=self.pi_ef_sigma*4)

        # add readout pulses to respective channels
        if 'mux4' in self.res_ch_types:
            self.set_pulse_registers(ch=6, style="const", length=max(self.readout_lengths_dac), mask=mask)
        for q in range(self.num_qubits_sample):
            if self.res_ch_types[q] != 'mux4':
                if cfg.device.readout.gain[q] < 1:
                    gain = int(cfg.device.readout.gain[q] * 2**15)
                self.set_pulse_registers(ch=self.res_chs[q], style="const", freq=self.f_res_regs[q], phase=0, gain=gain, length=max(self.readout_lengths_dac))


        self.set_gen_delays()
        self.sync_all(200)

    def body(self):
        cfg=AttrDict(self.cfg)

        qTest = self.cfg.expt.qTest

        self.reset_and_sync()

        f_ge_reg = self.freq2reg(self.f_ges[qTest, qTest], gen_ch=self.qubit_chs[qTest])
        if self.cfg.device.qubit.pulses.pi_ge.type[qTest].lower() == 'gauss':
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=f_ge_reg, phase=0, gain=self.pi_ge_gains[qTest, qTest], waveform="pi_qubit")
        else:
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="const", freq=f_ge_reg, phase=0, gain=cfg.expt.start, length=self.pi_sigma)
        self.sync_all()

        if self.checkEF:
            f_ef_reg = self.freq2reg(self.f_efs[qTest, qTest], gen_ch=self.qubit_chs[qTest])
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=f_ef_reg, phase=0, gain=self.pi_ef_gains[qTest, qTest], waveform="pi_ef_qubit")
        self.sync_all()

        # wait for the time stored in the wait variable register
        self.sync(self.q_rps[qTest], self.r_wait)

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
        self.mathi(self.q_rps[qTest], self.r_wait, self.r_wait, '+', self.us2cycles(self.cfg.expt.step)) # update wait time


class T1Experiment(Experiment):
    """
    T1 Experiment
    Experimental Config:
    expt = dict(
        start: wait time sweep start [us]
        step: wait time sweep step
        expts: number steps in sweep
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
        qTest
    )
    """

    def __init__(self, soccfg=None, path='', prefix='T1', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
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
        if 'checkEF' not in self.cfg.expt:
            self.cfg.expt.checkEF = False
        if self.cfg.expt.checkEF:
            self.cfg.device.readout.frequency = self.cfg.device.readout.frequency_ef
            self.cfg.device.readout.readout_length = self.cfg.device.readout.readout_length

        t1 = T1Program(soccfg=self.soccfg, cfg=self.cfg)
        x_pts, avgi, avgq = t1.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress)

        avgi = avgi[qTest][0]
        avgq = avgq[qTest][0]
        amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
        phases = np.angle(avgi+1j*avgq) # Calculating the phase        

        data={'xpts': x_pts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases}
        self.data=data
        return data

    def analyze(self, data=None, fit_log=False, fit_slice=None):
        if data is None:
            data=self.data
            
        # fitparams=[y-offset, amp, x-offset, decay rate]
        # Remove the last point from fit in case weird edge measurements
        xpts = data['xpts']

        for fit_axis in ['avgi', 'avgq', 'amps']:
            ypts_fit = data[fit_axis]

            data[f'fit_{fit_axis}'], data[f'fit_err_{fit_axis}'] = fitter.fitexp(xpts, ypts_fit, fitparams=None)

            if not fit_log: continue

            ypts_fit = np.copy(ypts_fit)
            if ypts_fit[0] > ypts_fit[-1]: ypts_fit = (ypts_fit - min(ypts_fit))/(max(ypts_fit) - min(ypts_fit))
            else: ypts_fit = (ypts_fit - max(ypts_fit))/(min(ypts_fit) - max(ypts_fit))

            # need to get rid of the 0 at the minimum
            xpts_fit = xpts
            min_ind = np.argmin(ypts_fit)
            ypts_fit[min_ind] = ypts_fit[min_ind-1] + ypts_fit[min_ind+(1 if min_ind+1<len(xpts) else 0)]

            if fit_slice is None:
                fit_slice = (0, len(xpts_fit))
            xpts_fit = xpts_fit[fit_slice[0]:fit_slice[1]]
            ypts_fit = ypts_fit[fit_slice[0]:fit_slice[1]]

            ypts_logscale = np.log(ypts_fit)

            data[f'fit_log_{fit_axis}'], data[f'fit_log_err_{fit_axis}'] = fitter.fitlogexp(xpts_fit, ypts_logscale, fitparams=None)
        return data

    def display(self, data=None, fit=True, fit_log=False):
        if data is None:
            data=self.data 

        qTest = self.cfg.expt.qTest
        
        plt.figure(figsize=(10, 5))
        plt.subplot(111,title="$T_1$", xlabel="Wait Time [us]", ylabel="Amplitude [ADC level]")
        plt.plot(data["xpts"], data["amps"],'.-')
        if fit:
            p = data['fit_amps']
            pCov = data['fit_err_amps']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(data["xpts"], fitter.expfunc(data["xpts"], *data["fit_amps"]), label=captionStr)
            plt.legend()
            print(f'Fit T1 amps [us]: {data["fit_amps"][3]}')

        xpts = data["xpts"]
        avgi = data["avgi"]
        avgq = data["avgq"]

        plt.figure(figsize=(10,10))
        title = "$T_1$" + (' EF' if self.cfg.expt.checkEF else '') + f' on Q{qTest}'
        plt.subplot(211, title=title, ylabel="I [ADC units]")
        plt.plot(xpts, avgi,'.-')
        if fit:
            p = data['fit_avgi']
            pCov = data['fit_err_avgi']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            fit_data = fitter.expfunc(xpts, *data["fit_avgi"])
            plt.plot(xpts, fit_data, label=captionStr)
            plt.legend()
            print(f'Fit T1 avgi [us]: {data["fit_avgi"][3]}')
        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC units]")
        plt.plot(xpts, avgq,'.-')
        if fit:
            p = data['fit_avgq']
            pCov = data['fit_err_avgq']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            fit_data = fitter.expfunc(xpts, *data["fit_avgq"])
            plt.plot(xpts, fit_data, label=captionStr)
            plt.legend()
            print(f'Fit T1 avgq [us]: {data["fit_avgq"][3]}')
        plt.show()

        if not fit_log: return

        plt.figure(figsize=(10,10))
        plt.subplot(211, title="$T_1$", ylabel="I [ADC units]")
        plt.yscale('log')
        ypts_scaled = np.copy(data['avgi'])
        if ypts_scaled[0] > ypts_scaled[-1]: ypts_scaled = (ypts_scaled - min(ypts_scaled))/(max(ypts_scaled) - min(ypts_scaled))
        else: ypts_scaled = (ypts_scaled - max(ypts_scaled))/(min(ypts_scaled) - max(ypts_scaled))
        plt.plot(xpts, ypts_scaled,'.-')
        if fit:
            p = data['fit_log_avgi']
            pCov = data['fit_log_err_avgi']
            captionStr = '$T_{1}$'+ f' fit [us]: {p[0]:.3} $\pm$ {np.sqrt(pCov[0][0]):.3}'
            fit_data = fitter.expfunc(xpts, 0, 1, 0, *data["fit_log_avgi"])
            plt.plot(xpts, fit_data, label=captionStr)
            plt.legend()
            print(f'Fit T1 avgq [us]: {data["fit_log_avgi"][0]} $\pm$ {np.sqrt(pCov[0][0])}')
        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC units]")
        plt.yscale('log')
        ypts_scaled = np.copy(data['avgq'])
        if ypts_scaled[0] > ypts_scaled[-1]: ypts_scaled = (ypts_scaled - min(ypts_scaled))/(max(ypts_scaled) - min(ypts_scaled))
        else: ypts_scaled = (ypts_scaled - max(ypts_scaled))/(min(ypts_scaled) - max(ypts_scaled))
        plt.plot(xpts, ypts_scaled,'.-')
        if fit:
            p = data['fit_log_avgq']
            pCov = data['fit_log_err_avgq']
            captionStr = '$T_{1}$'+ f' fit [us]: {p[0]:.3} $\pm$ {np.sqrt(pCov[0][0]):.3}'
            fit_data = fitter.expfunc(xpts, 0, 1, 0, *data["fit_log_avgq"])
            plt.plot(xpts, fit_data, label=captionStr)
            plt.legend()
            print(f'Fit T1 avgq [us]: {data["fit_log_avgq"][0]} $\pm$ {np.sqrt(pCov[0][0])}')
        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname
