import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from slab import AttrDict, Experiment, dsfit
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter
from experiments.clifford_averager_program import ReadoutResetPulser
from experiments.single_qubit.single_shot import HistogramProgram


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

    def __init__(self, soccfg=None, path="", prefix="ResonatorSpectroscopy", config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        xpts = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        qTest = self.cfg.expt.qTest

        num_qubits_sample = len(self.cfg.device.readout.frequency)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        if isinstance(value2, dict):
                            for key3, value3 in value2.items():
                                if not (isinstance(value3, list)):
                                    value2.update({key3: [value3] * num_qubits_sample})
                elif not (isinstance(value, list)):
                    subcfg.update({key: [value] * num_qubits_sample})

        data = {"xpts": [], "avgi": [], "avgq": [], "amps": [], "phases": []}
        for f in tqdm(xpts, disable=not progress):
            # self.cfg.expt.frequency = f
            cfg = AttrDict(deepcopy(self.cfg))
            cfg.device.readout.frequency[qTest] = f
            rspec = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
            avgi, avgq = rspec.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
            datai, dataq = rspec.collect_shots()
            avgi = np.average(datai)
            avgq = np.average(dataq)
            amp = np.average(np.abs((datai + 1j * dataq)))  # Calculating the magnitude
            phase = np.average(np.angle(datai + 1j * dataq))  # Calculating the phase
            self.prog = rspec
            # if f > 822.426:
            #     print('amps', amp)
            #     print('hello', rspec.cfg)
            #     break

            data["xpts"].append(f)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)

        for k, a in data.items():
            data[k] = np.array(a)

        self.data = data

        return data

    def analyze(self, data=None, fit=False, findpeaks=False, verbose=True, **kwargs):
        if data is None:
            data = deepcopy(self.data)
        qTest = self.cfg.expt.qTest

        # Fix the "electrical delay": (doesn't work for mux currently as the mux gen is not phase coherent) - see https://github.com/meeg/qick_demos_sho/blob/main/2023-01-12_qick-workshop/qick_workshop_new.ipynb

        freqs = self.soccfg.adcfreq(data["xpts"])
        freqs = data["xpts"]
        means = data["avgi"] + 1j * data["avgq"]
        a = np.vstack([freqs, np.ones_like(freqs)]).T
        phase_correction = np.linalg.lstsq(a, np.unwrap(np.angle(means)), rcond=None)[0][0] / (2 * np.pi)
        # print('phase correction (deg)', phase_correction)
        means_rotated = means * np.exp(-1j * freqs * 2 * np.pi * phase_correction)
        phase_trim = np.linalg.lstsq(a, np.unwrap(np.angle(means_rotated)), rcond=None)[0][0] / (2 * np.pi)
        # print('phase trim (deg)', phase_trim)
        phase_correction += phase_trim
        print("electrical delay phase correction (no mux support) (deg)", phase_correction)

        means_corrected = means * np.exp(-1j * freqs * 2 * np.pi * phase_correction)
        data["avgi"] = np.real(means_corrected)
        data["avgq"] = np.imag(means_corrected)
        data["phases"] = np.angle(means_corrected)

        if fit:
            # fitparams = [f0, Qi, Qe, phi, scale]
            xdata = data["xpts"][1:-1]
            # ydata = data["avgi"][1:-1] + 1j*data["avgq"][1:-1]
            ydata = data["amps"][1:-1]
            fitparams = None
            # fitparams = [xdata[np.argmin(ydata)], None, 5000, # [f0, Qi, Qe, phi, slope]
            if "lo" in self.cfg.hw:
                # print('lo freq', float(self.cfg.hw.lo.readout.frequency)*1e-6)
                # print('mux mixer', self.cfg.device.readout.lo_sideband[qTest]*(self.cfg.hw.soc.dacs.readout.mixer_freq[qTest]))
                xdata = float(self.cfg.hw.lo.readout.frequency) * 1e-6 + self.cfg.device.readout.lo_sideband[qTest] * (
                    self.cfg.hw.soc.dacs.readout.mixer_freq[qTest] + xdata
                )
            baseline = np.mean(np.sort(ydata)[-20:])
            data["fit"], data["fit_err"] = fitter.fithanger(xdata, ydata / baseline, fitparams=fitparams)
            if isinstance(data["fit"], (list, np.ndarray)):
                f0, Qi, Qe, phi, slope, a0 = data["fit"]
                # f0, Qi, Qe, phi = data['fit']
                if verbose:
                    print(f'\nFreq with minimum transmission: {data["xpts"][np.argmin(ydata)]}')
                    print(f'Freq with maximum transmission: {data["xpts"][np.argmax(ydata)]}')
                    print("From fit:")
                    print(f"\tf0: {f0}")
                    print(f"\tQi: {Qi} \t kappa_i/2pi: {f0/Qi}")
                    print(f"\tQe: {Qe} \t kappa_e/2pi: {f0/Qe}")
                    print(f"\tQ0: {1/(1/Qi+1/Qe)}")
                    print(f"\tkappa [MHz]: {f0*(1/Qi+1/Qe)}")
                    print(f"\tphi [radians]: {phi}")

        if findpeaks:
            maxpeaks, minpeaks = dsfit.peakdetect(
                data["amps"][1:-1], x_axis=data["xpts"][1:-1], lookahead=30, delta=5 * np.std(data["amps"][:5])
            )
            data["maxpeaks"] = maxpeaks
            data["minpeaks"] = minpeaks

        return data

    def display(self, data=None, fit=True, findpeaks=False, **kwargs):
        if data is None:
            data = self.data
        qTest = self.cfg.expt.qTest

        if "lo" in self.cfg.hw:
            xpts = float(self.cfg.hw.lo.readout.frequency) * 1e-6 + self.cfg.device.readout.lo_sideband[qTest] * (
                self.cfg.hw.soc.dacs.readout.mixer_freq[qTest] + data["xpts"][1:-1]
            )
        else:
            xpts = data["xpts"][1:-1]
        ydata = data["amps"][1:-1]

        plt.figure(figsize=(10, 10))
        plt.subplot(
            311,
            title=f"Resonator Spectroscopy Q{qTest}{' pulse E' if self.cfg.expt.pulse_e else ''}{' pulse F' if self.cfg.expt.pulse_f else ''} at gain {self.cfg.device.readout.gain[qTest]}",
            ylabel="Amps [ADC units]",
        )
        baseline = np.mean(np.sort(ydata)[-20:])
        plt.plot(xpts, ydata, ".-")
        if fit:
            print("baseline", baseline)
            plt.plot(xpts, baseline * fitter.hangerS21func_sloped(xpts, *data["fit"]))
        if findpeaks:
            # for peak in np.concatenate((data['maxpeaks'], data['minpeaks'])):
            for peak in data["minpeaks"]:
                plt.axvline(peak[0], linestyle="--", color="0.2")
                print(f"Found peak [MHz]: {peak[0]}")
        minfreq = xpts[np.argmin(ydata)]
        plt.axvline(minfreq, c="k", ls="--")  # |0>|1>
        plt.axvline(minfreq - 0.1, c="k", ls="--")  # |0>|1>

        if fit:
            f0, Qi, Qe, phi, slope, a0 = data["fit"]
            plt.axvline(f0, c="r", ls="--")  # |0>|1>

        plt.subplot(312, xlabel="Readout Frequency [MHz]", ylabel="I [ADC units]")
        plt.plot(xpts, data["avgi"][1:-1], ".-")
        # plt.ylim(0, None)

        # plt.subplot(313, xlabel="Readout Frequency [MHz]", ylabel="Q [ADC units]")
        # plt.plot(xpts, data["avgq"][1:-1],'o-')
        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
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

    def __init__(
        self, soccfg=None, path="", prefix="ResonatorPowerSweepSpectroscopy", config_file=None, progress=None
    ):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        xpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"] * np.arange(self.cfg.expt["expts_f"])
        gainpts = self.cfg.expt["start_gain"] + self.cfg.expt["step_gain"] * np.arange(self.cfg.expt["expts_gain"])

        qTest = self.cfg.expt.qTest
        num_qubits_sample = len(self.cfg.device.readout.frequency)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        if isinstance(value2, dict):
                            for key3, value3 in value2.items():
                                if not (isinstance(value3, list)):
                                    value2.update({key3: [value3] * num_qubits_sample})
                elif not (isinstance(value, list)):
                    subcfg.update({key: [value] * num_qubits_sample})

        data = {"xpts": [], "gainpts": [], "avgi": [], "avgq": [], "amps": [], "phases": []}
        for gain in tqdm(gainpts, disable=not progress):
            self.cfg.device.readout.gain[qTest] = gain
            data["avgi"].append([])
            data["avgq"].append([])
            data["amps"].append([])
            data["phases"].append([])

            for f in tqdm(xpts, disable=True):
                cfg = AttrDict(deepcopy(self.cfg))
                cfg.device.readout.frequency[qTest] = f
                rspec = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
                avgi, avgq = rspec.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                datai, dataq = rspec.collect_shots()
                avgi = np.average(datai)
                avgq = np.average(dataq)
                # rspec = ResonatorSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
                self.prog = rspec
                # avgi, avgq = rspec.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                # avgi = avgi[0][0]
                # avgq = avgq[0][0]

                amp = np.average(np.abs((datai + 1j * dataq)))  # Calculating the magnitude
                phase = np.average(np.angle(datai + 1j * dataq))  # Calculating the phase
                # amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
                # phase = np.angle(avgi + 1j * avgq)  # Calculating the phase
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
            data = self.data

        # Lorentzian fit at highgain [DAC units] and lowgain [DAC units]
        if fit:
            if highgain == None:
                highgain = data["gainpts"][-1]
            if lowgain == None:
                lowgain = data["gainpts"][0]
            i_highgain = np.argmin(np.abs(data["gainpts"] - highgain))
            i_lowgain = np.argmin(np.abs(data["gainpts"] - lowgain))
            fit_highpow = dsfit.fitlor(data["xpts"], data["amps"][i_highgain])
            fit_lowpow = dsfit.fitlor(data["xpts"], data["amps"][i_lowgain])
            data["fit"] = [fit_highpow, fit_lowpow]
            data["fit_gains"] = [highgain, lowgain]
            data["lamb_shift"] = fit_highpow[2] - fit_lowpow[2]

        return data

    def display(self, data=None, fit=True, select=None, **kwargs):
        if data is None:
            data = self.data
        qTest = self.cfg.expt.qTest

        if "lo" in self.cfg.hw:
            inner_sweep = float(self.cfg.hw.lo.readout.frequency) * 1e-6 + self.cfg.device.readout.lo_sideband[
                qTest
            ] * (self.cfg.hw.soc.dacs.readout.mixer_freq[qTest] + data["xpts"])
        else:
            inner_sweep = data["xpts"]
        outer_sweep = data["gainpts"]

        amps = np.copy(data["amps"])
        for amps_gain in amps:
            # amps_gain = (amps_gain - np.average(amps_gain)) / np.average(amps_gain)
            amps_gain -= np.average(amps_gain)

        y_sweep = outer_sweep
        x_sweep = inner_sweep

        # for iy, y in enumerate(y_sweep):
        #     plt.plot(x_sweep, amps[iy])

        # THIS IS CORRECT EXTENT LIMITS FOR 2D PLOTS
        plt.figure(figsize=(10, 8))
        plt.title(f"Resonator Power Spectroscopy (Qubit {self.cfg.expt.qTest})")
        plt.pcolormesh(x_sweep, y_sweep, amps, cmap="viridis", shading="auto")

        if fit:
            fit_highpow, fit_lowpow = data["fit"]
            highgain, lowgain = data["fit_gains"]
            plt.axvline(fit_highpow[2], linewidth=0.5, color="0.2")
            plt.axvline(fit_lowpow[2], linewidth=0.5, color="0.2")
            plt.plot(x_sweep, [highgain] * len(x_sweep), linewidth=0.5, color="0.2")
            plt.plot(x_sweep, [lowgain] * len(x_sweep), linewidth=0.5, color="0.2")
            print(f"High power peak [MHz]: {fit_highpow[2]}")
            print(f"Low power peak [MHz]: {fit_lowpow[2]}")
            print(f'Lamb shift [MHz]: {data["lamb_shift"]}')

        # plt.title(f"Resonator Spectroscopy Power Sweep")
        plt.xlabel("Resonator Frequency [MHz]")
        plt.ylabel("Resonator Gain [DAC level]")
        # # plt.clim(vmin=-0.2, vmax=0.2)
        # # plt.clim(vmin=-10, vmax=5)
        plt.colorbar(label="Amps-Avg [ADC level]")
        plt.show()

        if select is not None:
            fig, ax = plt.subplots()
            y_closest_i = np.argmin(abs(y_sweep - select))
            y_closest = y_sweep[y_closest_i]
            y_plot = data["amps"][y_closest_i, :] / np.max(data["amps"][y_closest_i, :])
            print("plotting at gain", y_closest, "index", y_closest_i)
            ax.plot(x_sweep, y_plot, "o-", label=f"Gain {y_closest}")
            ax.legend()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
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

    def __init__(
        self,
        soccfg=None,
        path="",
        dc_instr=None,
        dc_ch=None,
        prefix="ResonatorVoltSweepSpectroscopy",
        config_file=None,
        progress=None,
    ):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)
        self.dc_instr = dc_instr

    def acquire(self, progress=False):
        xpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"] * np.arange(self.cfg.expt["expts_f"])
        voltpts = self.cfg.expt["start_volt"] + self.cfg.expt["step_volt"] * np.arange(self.cfg.expt["expts_volt"])

        q_ind = self.cfg.expt.qubit

        num_qubits_sample = len(self.cfg.device.readout.frequency)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, list) and len(value) == self.num_qubits_sample:
                    subcfg.update({key: value[q_ind]})
                elif isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if isinstance(value3, list) and len(value3) == self.num_qubits_sample:
                                value2.update({key3: value3[q_ind]})

        data = {"xpts": [], "voltpts": [], "avgi": [], "avgq": [], "amps": [], "phases": []}

        self.dc_instr.set_mode("CURR")
        self.dc_instr.set_current_limit(max(abs(voltpts) * 5))
        print(f"Setting current limit {self.dc_instr.get_current_limit()*1e6} uA")
        self.dc_instr.set_output(True)

        for volt in tqdm(voltpts, disable=not progress):
            # self.dc_instr.set_voltage(channel=self.cfg.expt.dc_ch, voltage=volt)
            self.dc_instr.set_current(volt)
            print(f"current set to {self.dc_instr.get_current() * 1e6} uA")
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
                amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
                phase = np.angle(avgi + 1j * avgq)  # Calculating the phase

                data["avgi"][-1].append(avgi)
                data["avgq"][-1].append(avgq)
                data["amps"][-1].append(amp)
                data["phases"][-1].append(phase)
            time.sleep(0.5)
        # self.dc_instr.initialize()
        # self.dc_instr.set_voltage(channel=self.cfg.expt.dc_ch, voltage=0)

        self.dc_instr.set_current(0)
        print(f"current set to {self.dc_instr.get_current() * 1e6} uA")

        data["xpts"] = xpts
        data["voltpts"] = voltpts

        for k, a in data.items():
            data[k] = np.array(a)

        self.data = data
        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data
        pass

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        plt.figure(figsize=(12, 8))
        x_sweep = 1e3 * data["voltpts"]
        y_sweep = data["xpts"]
        amps = data["amps"]
        # for amps_volt in amps:
        #     amps_volt -= np.average(amps_volt)

        plt.pcolormesh(x_sweep, y_sweep, np.flip(np.rot90(data["amps"]), 0), cmap="viridis")
        if "add_data" in kwargs:
            for add_data in kwargs["add_data"]:
                plt.pcolormesh(
                    1e3 * add_data["voltpts"], add_data["xpts"], np.flip(np.rot90(add_data["amps"]), 0), cmap="viridis"
                )
        if fit:
            pass

        plt.title(f"Resonator {self.cfg.expt.qubit} sweeping DAC box ch {self.cfg.expt.dc_ch}")
        plt.ylabel("Resonator frequency")
        plt.xlabel("DC current [mA]")
        # plt.ylabel("DC voltage [V]")
        plt.clim(vmin=None, vmax=None)
        plt.colorbar(label="Amps [ADC level]")

        # plt.plot(x_sweep, amps[1])
        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname


class ResonatorRingDownExperiment(Experiment):
    """Resonator Ring Down Experiment
    Experimental Config
    expt_cfg={
        "start_time": start time (us),
        "step_time": time step (us),
        "expts": number of experiments in time,
        "freq": frequency of the drive,
        "gain": gain of the readout,
        "reps": number of reps,
    """

    def __init__(self, soccfg=None, path="", prefix="ResonatorRingDown", config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        xpts = self.cfg.expt.start_time + self.cfg.expt.step_time * np.arange(self.cfg.expt.expts)
        qTest = self.cfg.expt.qTest

        if "freq" in self.cfg.expt and self.cfg.expt.freq is not None:
            self.cfg.device.readout.frequency[qTest] = self.cfg.expt.freq

        if "gain" in self.cfg.expt and self.cfg.expt.gain is not None:
            self.cfg.device.readout.gain[qTest] = self.cfg.expt.gain

        assert "len_readout_adc" in self.cfg.expt and self.cfg.expt.len_readout_adc is not None

        num_qubits_sample = len(self.cfg.device.readout.frequency)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        if isinstance(value2, dict):
                            for key3, value3 in value2.items():
                                if not (isinstance(value3, list)):
                                    value2.update({key3: [value3] * num_qubits_sample})
                elif not (isinstance(value, list)):
                    subcfg.update({key: [value] * num_qubits_sample})

        data = {"xpts": [], "avgi": [], "avgq": [], "amps": [], "phases": []}

        if "full_mux_expt" in self.cfg.expt and self.cfg.expt.full_mux_expt:
            if "resonator_reset" in self.cfg.expt and (
                "pulse_I_shapes" not in self.cfg.expt or self.cfg.expt.pulse_I_shapes is None
            ):
                pulse_I_shapes = []
                pulse_Q_shapes = []
                Tp = self.cfg.device.readout.readout_length[0]
                dt_us = 0.1e-3
                times_us = np.linspace(0, Tp, int(Tp / dt_us))
                kappa_ext = self.cfg.device.readout.kappa_ext
                kerr = self.cfg.device.readout.kerr
                t_rise = self.cfg.device.readout.t_rise_reset
                readout_length = self.cfg.device.readout.readout_length
                self.cfg.expt.full_mux_chs = self.cfg.hw.soc.dacs.readout.full_mux_chs
                self.cfg.expt.mask = [0, 1, 2, 3]
                for q in range(num_qubits_sample):
                    if q in self.cfg.expt.resonator_reset:
                        readout_pulser = ReadoutResetPulser(
                            kappa_ext_MHz=2 * np.pi * kappa_ext[q],
                            kappa_int_MHz=0,
                            chi_MHz=0,
                            kerr_MHz=2 * np.pi * kerr[q],
                        )
                        _, (I_pulse, Q_pulse) = readout_pulser.flat_top_kerr_drive(
                            t_rise=t_rise[q], Tp=Tp, num_filter=1, nstep=int(Tp / dt_us)
                        )
                    else:
                        I_pulse, Q_pulse = np.ones((2, int(Tp / dt_us)))
                    pulse_I_shapes.append(I_pulse)
                    pulse_Q_shapes.append(Q_pulse)
                self.cfg.expt.pulse_I_shapes = np.array(pulse_I_shapes)
                self.cfg.expt.pulse_Q_shapes = np.array(pulse_Q_shapes)
                self.cfg.expt.times_us = times_us

        for t in tqdm(xpts, disable=not progress):
            cfg = AttrDict(deepcopy(self.cfg))

            # Time to play the readout pulse
            if "full_mux_expt" in self.cfg.expt and self.cfg.expt.full_mux_expt:
                if "pulse_I_shapes" in self.cfg.expt and self.cfg.expt.pulse_I_shapes is not None:
                    self.cfg.device.readout.readout_length = [self.cfg.expt.times_us[-1]] * num_qubits_sample

            t_readout = min(t, self.cfg.device.readout.readout_length[qTest])

            # If requested is past the end of the (original) readout length, figure out the offset time for the trigger
            t_offset = max(t, t_readout)
            cycles_off_baseline = cfg.device.readout.trig_offset[0]
            cfg.device.readout.trig_offset = [self.soc.us2cycles(t_offset) + cycles_off_baseline] * num_qubits_sample
            # print(cfg.device.readout.trig_offset)
            cfg.device.readout.readout_length = [t_readout] * num_qubits_sample  # set for all qubits

            # Handle slicing IQ pulses
            if "full_mux_expt" in self.cfg.expt and self.cfg.expt.full_mux_expt:
                if "pulse_I_shapes" in self.cfg.expt and self.cfg.expt.pulse_I_shapes is not None:
                    t_index = np.argmin(np.abs(t - self.cfg.expt.times_us))
                    assert (
                        self.cfg.expt.times_us[t_index] > 0
                    ), "Trying to run a pulse of total length 0, check the time step size of your pulse shape"
                    cfg.expt.pulse_I_shapes = self.cfg.expt.pulse_I_shapes[:, :t_index]
                    cfg.expt.pulse_Q_shapes = self.cfg.expt.pulse_Q_shapes[:, :t_index]
                    cfg.expt.times_us = self.cfg.expt.times_us[:t_index]
                else:
                    assert (
                        "lengths" in self.cfg.expt and self.cfg.expt.lengths is not None
                    ), "Need to specify either pulse_I_shapes or lengths for this experiment"
                    cfg.expt.lengths = cfg.device.readout.readout_length

            # print(f"Readout time: {t_readout} us, Offset time: {t_offset} us")

            rspec = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
            avgi, avgq = rspec.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
            datai, dataq = rspec.collect_shots()
            avgi = np.average(datai)
            avgq = np.average(dataq)
            # amp = np.abs((avgi + 1j * avgq))  # Calculating the magnitude
            amp = np.average(np.abs((datai + 1j * dataq)))  # Calculating the magnitude
            phase = np.average(np.angle(datai + 1j * dataq))  # Calculating the phase
            # plt.figure()
            # plt.plot(datai, label="I")
            # plt.axhline(np.average(datai))
            # plt.plot(dataq, label="Q")
            # plt.plot(np.abs(datai + 1j * dataq), label="Amp")
            # plt.axhline(np.average(datai), color="b", label="I")
            # plt.axhline(np.average(dataq), color="g", label="Q")
            # plt.axhline(np.average(np.abs(datai + 1j * dataq)), label="Amp")
            # plt.axhline(np.abs(avgi + 1j * avgq), color="r", label="Amp")
            # plt.legend()
            # plt.show()

            self.prog = rspec
            # print("i", datai[50], dataq[50], np.abs((datai[50] + 1j * dataq[50])))  # Calculating the magnitude

            data["xpts"].append(t)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)

        for k, a in data.items():
            data[k] = np.array(a)

        self.data = data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data
        qTest = self.cfg.expt.qTest

        fit_start = np.argmin(np.abs(data["xpts"] - self.cfg.device.readout.readout_length[qTest]))
        xpts = data["xpts"][fit_start:]
        ypts_fit = data["amps"][fit_start:]
        data["fit_amps"], data["fit_err_amps"] = fitter.fitexp(xpts, ypts_fit, fitparams=None)

        return data

    def display(self, data=None, fit=False, **kwargs):

        if data is None:
            data = self.data
        qTest = self.cfg.expt.qTest

        fig, ax = plt.subplots(4, 1, figsize=(8, 11))
        ax[0].plot(data["xpts"], data["amps"], ".-")
        if fit:
            fit_start = np.argmin(np.abs(data["xpts"] - self.cfg.device.readout.readout_length[qTest]))
            p = data["fit_amps"]
            pCov = data["fit_err_amps"]
            decay_time = p[3]
            kappa_kHz = 1e3 / (decay_time * 2 * np.pi)
            kappa_err = (1 / p[3] ** 2) * np.sqrt(pCov[3][3]) * 1e3 / (2 * np.pi)
            captionStr = f"$\kappa$ fit [linear kHz]: {kappa_kHz:.3} $\pm$ {kappa_err:.3}"
            fit_xpts = data["xpts"][fit_start:]
            ax[0].plot(fit_xpts, fitter.expfunc(fit_xpts, *data["fit_amps"]), label=captionStr)
            plt.sca(ax[0])
            plt.legend()
        ax[0].set_title(f"Resonator Ring Down Q{qTest} at gain {self.cfg.device.readout.gain[qTest]}")
        ax[0].set_ylabel("Amps [ADC units]")
        ax[0].set_xlabel("Time [us]")
        ax[1].plot(data["xpts"], data["avgi"], ".-")
        ax[1].set_ylabel("I [ADC units]")
        ax[2].plot(data["xpts"], data["avgq"], ".-")
        ax[2].set_ylabel("Q [ADC units]")
        ax[3].plot(data["xpts"], data["phases"], ".-")
        ax[3].set_ylabel("Phase [radians]")
        ax[3].set_xlabel("Time [us]")
        fig.tight_layout()
        plt.show()

        # plot the IQ trajectory in the complex plane
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        plt.title(f"IQ trajectory Q{qTest} at gain {self.cfg.device.readout.gain[qTest]}")
        ax.plot(data["avgi"], data["avgq"], ".-")
        plt.plot(data["avgi"][0], data["avgq"][0], marker="o", markerfacecolor="g", markersize=5)
        plt.plot(data["avgi"][-1], data["avgq"][-1], marker="o", markerfacecolor="r", markersize=5)
        plt.xlabel("I [ADC units]")
        plt.ylabel("Q [ADC units]")
        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname
