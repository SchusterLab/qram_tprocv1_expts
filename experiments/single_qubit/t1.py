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

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
        
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        self.checkEF = self.cfg.expt.checkEF
        
        self.adc_ch = cfg.hw.soc.adcs.readout.ch
        self.res_ch = cfg.hw.soc.dacs.readout.ch
        self.res_ch_type = cfg.hw.soc.dacs.readout.type
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type

        self.q_rp = self.ch_page(self.qubit_ch) # get register page for qubit_ch
        self.r_wait = 3
        self.safe_regwi(self.q_rp, self.r_wait, self.us2cycles(cfg.expt.start))
        
        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_ch)
        self.f_ef_reg = self.freq2reg(cfg.device.qubit.f_ef, gen_ch=self.qubit_ch)
        self.f_res_reg = self.freq2reg(cfg.device.readout.frequency, gen_ch=self.res_ch, ro_ch=self.adc_ch)
        self.readout_length_dac = self.us2cycles(cfg.device.readout.readout_length, gen_ch=self.res_ch)
        self.readout_length_adc = self.us2cycles(cfg.device.readout.readout_length, ro_ch=self.adc_ch)
        self.readout_length_adc += 1 # ensure the rounding of the clock ticks calculation doesn't mess up the buffer

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

        # declare adcs
        self.declare_readout(ch=self.adc_ch, length=self.readout_length_adc, freq=cfg.device.readout.frequency, gen_ch=self.res_ch)

        self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)

        # add qubit and readout pulses to respective channels
        if self.cfg.device.qubit.pulses.pi_ge.type.lower() == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)

        if self.checkEF:
            self.pi_ef_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)
            self.add_gauss(ch=self.qubit_ch, name="pi_ef_qubit", sigma=self.pi_ef_sigma, length=self.pi_ef_sigma*4)

        if self.res_ch_type == 'mux4':
            self.set_pulse_registers(ch=self.res_ch, style="const", length=self.readout_length_dac, mask=mask)
        else: self.set_pulse_registers(ch=self.res_ch, style="const", freq=self.f_res_reg, phase=0, gain=cfg.device.readout.gain, length=self.readout_length_dac)

        self.sync_all(200)

    def body(self):
        cfg=AttrDict(self.cfg)

        if self.cfg.device.qubit.pulses.pi_ge.type.lower() == 'gauss':
            self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain, waveform="pi_qubit")
        else:
            self.setup_and_pulse(ch=self.qubit_ch, style="const", freq=self.f_ge, phase=0, gain=cfg.expt.start, length=self.pi_sigma)
        self.sync_all()

        if self.checkEF:
            self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ef_reg, phase=0, gain=cfg.device.qubit.pulses.pi_ef.gain, waveform="pi_ef_qubit")
        self.sync_all()

        # wait for the time stored in the wait variable register
        self.sync(self.q_rp, self.r_wait)

        self.sync_all()
        
        self.measure(pulse_ch=self.res_ch, 
             adcs=[self.adc_ch],
             adc_trig_offset=cfg.device.readout.trig_offset,
             wait=True,
             syncdelay=self.us2cycles(cfg.device.readout.relax_delay))

    def update(self):
        self.mathi(self.q_rp, self.r_wait, self.r_wait, '+', self.us2cycles(self.cfg.expt.step)) # update wait time


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
    )
    """

    def __init__(self, soccfg=None, path='', prefix='T1', config_file=None, progress=None):
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

        if 'checkEF' not in self.cfg.expt:
            self.cfg.expt.checkEF = False
        if self.cfg.expt.checkEF:
            self.cfg.device.readout.frequency = self.cfg.device.readout.frequency_ef
        t1 = T1Program(soccfg=self.soccfg, cfg=self.cfg)
        x_pts, avgi, avgq = t1.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress)

        avgi = avgi[0][0]
        avgq = avgq[0][0]
        amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
        phases = np.angle(avgi+1j*avgq) # Calculating the phase        

        data={'xpts': x_pts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases}
        self.data=data
        return data

    def analyze(self, data=None, fit_log=True, fit_slice=None):
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

    def display(self, data=None, fit=True, fit_log=True):
        if data is None:
            data=self.data 
        
        # plt.figure(figsize=(12, 8))
        # plt.subplot(111,title="$T_1$", xlabel="Wait Time [us]", ylabel="Amplitude [ADC level]")
        # plt.plot(data["xpts"][:-1], data["amps"][:-1],'o-')
        # if fit:
        #     p = data['fit_amps']
        #     pCov = data['fit_err_amps']
        #     captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
        #     plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_amps"]), label=captionStr)
        #     plt.legend()
        #     print(f'Fit T1 amps [us]: {data["fit_amps"][3]}')

        xpts = data["xpts"]
        avgi = data["avgi"]
        avgq = data["avgq"]

        plt.figure(figsize=(10,10))
        qubit = self.cfg.expt.qubit
        title = "$T_1$" + (' EF' if self.cfg.expt.checkEF else '') + f' on Q{qubit}'
        plt.subplot(211, title=title, ylabel="I [ADC units]")
        plt.plot(xpts, avgi,'o-')
        if fit:
            p = data['fit_avgi']
            pCov = data['fit_err_avgi']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            fit_data = fitter.expfunc(xpts, *data["fit_avgi"])
            plt.plot(xpts, fit_data, label=captionStr)
            plt.legend()
            print(f'Fit T1 avgi [us]: {data["fit_avgi"][3]}')
        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC units]")
        plt.plot(xpts, avgq,'o-')
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
        plt.plot(xpts, ypts_scaled,'o-')
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
        plt.plot(xpts, ypts_scaled,'o-')
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
