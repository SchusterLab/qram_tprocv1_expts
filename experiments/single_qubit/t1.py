import matplotlib.patches as mpl_patches
import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from slab import AttrDict, Experiment
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter
from experiments.clifford_averager_program import QutritRAveragerProgram


class T1Program(QutritRAveragerProgram):
    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)

        self.cfg.update(cfg.expt)
        self.num_qubits_sample = len(self.cfg.device.readout.frequency)

        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        self.checkZZ = False
        if qZZ is not None and qZZ != qTest:
            self.checkZZ = True
        else:
            qZZ = qTest
        self.checkEF = self.cfg.expt.checkEF

        super().initialize()

        self.r_wait = 3
        self.safe_regwi(self.q_rps[qTest], self.r_wait, self.us2cycles(cfg.expt.start))

    def body(self):
        cfg = AttrDict(self.cfg)
        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        if qZZ is None:
            qZZ = qTest

        self.reset_and_sync()

        if "cool_qubits" in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            cool_idle = [self.cfg.device.qubit.pulses.pi_f0g1.idle[q] for q in self.cfg.expt.cool_qubits]
            if "cool_idle" in self.cfg.expt and self.cfg.expt.cool_idle is not None:
                cool_idle = self.cfg.expt.cool_idle
            self.active_cool(cool_qubits=self.cfg.expt.cool_qubits, cool_idle=cool_idle)

        if self.readout_cool:
            self.measure_readout_cool()

        # initializations as necessary
        if self.checkZZ:
            self.X_pulse(q=qZZ, play=True)

        self.X_pulse(q=qTest, ZZ_qubit=qZZ, play=True)
        if self.checkEF:
            self.Xef_pulse(q=qTest, ZZ_qubit=qZZ, play=True)

        # wait for the time stored in the wait variable register
        self.sync(self.q_rps[qTest], self.r_wait)

        self.sync_all()
        self.measure(
            pulse_ch=self.measure_chs,
            adcs=self.adc_chs,
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(max([cfg.device.readout.relax_delay[q] for q in range(self.num_qubits_sample)])),
        )

    def update(self):
        qTest = self.cfg.expt.qTest
        self.mathi(
            self.q_rps[qTest], self.r_wait, self.r_wait, "+", self.us2cycles(self.cfg.expt.step)
        )  # update wait time


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

    def __init__(self, soccfg=None, path="", prefix="T1", config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        num_qubits_sample = len(self.cfg.device.readout.frequency)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not (isinstance(value3, list)):
                                value2.update({key3: [value3] * num_qubits_sample})
                elif not (isinstance(value, list)):
                    subcfg.update({key: [value] * num_qubits_sample})

        full_mux_expt = False
        if "full_mux_expt" in self.cfg.expt:
            full_mux_expt = self.cfg.expt.full_mux_expt

        qTest = self.cfg.expt.qTest

        if "checkEF" not in self.cfg.expt:
            self.cfg.expt.checkEF = False
        if self.cfg.expt.checkEF:
            self.cfg.device.readout.frequency = self.cfg.device.readout.frequency_ef
            self.cfg.device.readout.gain = self.cfg.device.readout.gain_ef
            self.cfg.device.readout.readout_length = self.cfg.device.readout.readout_length_ef

        t1 = T1Program(soccfg=self.soccfg, cfg=self.cfg)
        x_pts, idata, qdata = t1.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=progress)
        _, ishots, qshots = t1.get_shots()

        avgi = idata[qTest][0]
        avgq = qdata[qTest][0]
        amps = np.average(np.abs(ishots + 1j * qshots), axis=2)[qTest]  # Calculating the magnitude
        phases = np.average(np.angle(ishots + 1j * qshots), axis=2)[qTest]  # Calculating the phase

        data = {"xpts": x_pts, "avgi": avgi, "avgq": avgq, "amps": amps, "phases": phases}
        self.data = data
        return data

    def analyze(self, data=None, fit_log=False, fit_slice=None):
        if data is None:
            data = self.data

        # fitparams=[y-offset, amp, x-offset, decay rate]
        # Remove the last point from fit in case weird edge measurements
        xpts = data["xpts"]

        for fit_axis in ["avgi", "avgq", "amps"]:
            ypts_fit = data[fit_axis]

            data[f"fit_{fit_axis}"], data[f"fit_err_{fit_axis}"] = fitter.fitexp(xpts, ypts_fit, fitparams=None)

            if not fit_log:
                continue

            ypts_fit = np.copy(ypts_fit)
            if ypts_fit[0] > ypts_fit[-1]:
                ypts_fit = (ypts_fit - min(ypts_fit)) / (max(ypts_fit) - min(ypts_fit))
            else:
                ypts_fit = (ypts_fit - max(ypts_fit)) / (min(ypts_fit) - max(ypts_fit))

            # need to get rid of the 0 at the minimum
            xpts_fit = xpts
            min_ind = np.argmin(ypts_fit)
            ypts_fit[min_ind] = ypts_fit[min_ind - 1] + ypts_fit[min_ind + (1 if min_ind + 1 < len(xpts) else 0)]

            if fit_slice is None:
                fit_slice = (0, len(xpts_fit))
            xpts_fit = xpts_fit[fit_slice[0] : fit_slice[1]]
            ypts_fit = ypts_fit[fit_slice[0] : fit_slice[1]]

            ypts_logscale = np.log(ypts_fit)

            data[f"fit_log_{fit_axis}"], data[f"fit_log_err_{fit_axis}"] = fitter.fitlogexp(
                xpts_fit, ypts_logscale, fitparams=None
            )
        return data

    def display(self, data=None, fit=True, fit_log=False):
        if data is None:
            data = self.data

        qTest = self.cfg.expt.qTest

        checkEF = False
        if "checkEF" in self.cfg.expt:
            checkEF = self.cfg.expt.checkEF

        plt.figure(figsize=(10, 5))
        plt.subplot(
            111,
            title="$T_1$" + (" EF" if checkEF else "") + f" on Q{qTest}",
            xlabel="Wait Time [us]",
            ylabel="Amplitude [ADC level]",
        )
        plt.plot(data["xpts"], data["amps"], ".-")
        if fit:
            p = data["fit_amps"]
            pCov = data["fit_err_amps"]
            captionStr = f"$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}"
            plt.plot(data["xpts"], fitter.expfunc(data["xpts"], *data["fit_amps"]), label=captionStr)
            plt.legend()
            print(f'Fit T1 amps [us]: {data["fit_amps"][3]}')

        xpts = data["xpts"]
        avgi = data["avgi"]
        avgq = data["avgq"]

        plt.figure(figsize=(10, 10))
        title = "$T_1$" + (" EF" if self.cfg.expt.checkEF else "") + f" on Q{qTest}"
        plt.subplot(211, title=title, ylabel="I [ADC units]")
        plt.plot(xpts, avgi, ".-")
        if fit:
            p = data["fit_avgi"]
            pCov = data["fit_err_avgi"]
            captionStr = f"$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}"
            fit_data = fitter.expfunc(xpts, *data["fit_avgi"])
            plt.plot(xpts, fit_data, label=captionStr)
            plt.legend()
            print(f'Fit T1 avgi [us]: {data["fit_avgi"][3]}')
        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC units]")
        plt.plot(xpts, avgq, ".-")
        if fit:
            p = data["fit_avgq"]
            pCov = data["fit_err_avgq"]
            captionStr = f"$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}"
            fit_data = fitter.expfunc(xpts, *data["fit_avgq"])
            plt.plot(xpts, fit_data, label=captionStr)
            plt.legend()
            print(f'Fit T1 avgq [us]: {data["fit_avgq"][3]}')
        plt.show()

        if not fit_log:
            return

        plt.figure(figsize=(10, 10))
        plt.subplot(211, title="$T_1$", ylabel="I [ADC units]")
        plt.yscale("log")
        ypts_scaled = np.copy(data["avgi"])
        if ypts_scaled[0] > ypts_scaled[-1]:
            ypts_scaled = (ypts_scaled - min(ypts_scaled)) / (max(ypts_scaled) - min(ypts_scaled))
        else:
            ypts_scaled = (ypts_scaled - max(ypts_scaled)) / (min(ypts_scaled) - max(ypts_scaled))
        plt.plot(xpts, ypts_scaled, ".-")
        if fit:
            p = data["fit_log_avgi"]
            pCov = data["fit_log_err_avgi"]
            captionStr = "$T_{1}$" + f" fit [us]: {p[0]:.3} $\pm$ {np.sqrt(pCov[0][0]):.3}"
            fit_data = fitter.expfunc(xpts, 0, 1, 0, *data["fit_log_avgi"])
            plt.plot(xpts, fit_data, label=captionStr)
            plt.legend()
            print(f'Fit T1 avgq [us]: {data["fit_log_avgi"][0]} $\pm$ {np.sqrt(pCov[0][0])}')
        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC units]")
        plt.yscale("log")
        ypts_scaled = np.copy(data["avgq"])
        if ypts_scaled[0] > ypts_scaled[-1]:
            ypts_scaled = (ypts_scaled - min(ypts_scaled)) / (max(ypts_scaled) - min(ypts_scaled))
        else:
            ypts_scaled = (ypts_scaled - max(ypts_scaled)) / (min(ypts_scaled) - max(ypts_scaled))
        plt.plot(xpts, ypts_scaled, ".-")
        if fit:
            p = data["fit_log_avgq"]
            pCov = data["fit_log_err_avgq"]
            captionStr = "$T_{1}$" + f" fit [us]: {p[0]:.3} $\pm$ {np.sqrt(pCov[0][0]):.3}"
            fit_data = fitter.expfunc(xpts, 0, 1, 0, *data["fit_log_avgq"])
            plt.plot(xpts, fit_data, label=captionStr)
            plt.legend()
            print(f'Fit T1 avgq [us]: {data["fit_log_avgq"][0]} $\pm$ {np.sqrt(pCov[0][0])}')
        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname
