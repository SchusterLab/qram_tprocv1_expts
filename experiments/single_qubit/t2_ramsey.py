import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from slab import AttrDict, Experiment, dsfit
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter
from experiments.clifford_averager_program import QutritRAveragerProgram


class RamseyProgram(QutritRAveragerProgram):
    def initialize(self):
        cfg = AttrDict(self.cfg)
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

        # Override the default length parameters before initializing the clifford averager program
        self.qubit_chs = self.cfg.hw.soc.dacs.qubit.ch
        calib_index = qTest * self.num_qubits_sample + qZZ
        if "sigma_test" not in self.cfg.expt:
            self.cfg.expt.sigma_test = self.cfg.device.qubit.pulses.pi_ge.sigma[calib_index]
            if self.checkEF:
                self.cfg.expt.sigma_test = self.cfg.device.qubit.pulses.pi_ef.sigma[calib_index]
        sigma_test_cycles = self.us2cycles(self.cfg.expt.sigma_test, gen_ch=self.qubit_chs[qTest])
        if sigma_test_cycles > 3:
            self.num_qubits_sample = len(self.cfg.device.readout.frequency)
            if self.checkEF:
                self.cfg.device.qubit.pulses.pi_ef.sigma[calib_index] = self.cfg.expt.sigma_test
            else:
                self.cfg.device.qubit.pulses.pi_ge.sigma[calib_index] = self.cfg.expt.sigma_test

        super().initialize()

        # declare registers for phase incrementing
        self.r_wait = 5
        self.r_phase2 = 6
        if self.qubit_ch_types[qTest] == "int4":
            self.r_phase = self.sreg(self.qubit_chs[qTest], "freq")
            self.r_phase3 = 5  # for storing the left shifted value
        else:
            self.r_phase = self.sreg(self.qubit_chs[qTest], "phase")

        # calibrate the pi/2 pulse instead of the pi pulse by taking half the sigma and calibrating the gain
        self.test_pi_half = False
        self.divide_len = True
        if "divide_len" in self.cfg.expt:
            self.divide_len = self.cfg.expt.divide_len
        if "test_pi_half" in self.cfg.expt and self.cfg.expt.test_pi_half:
            self.test_pi_half = self.cfg.expt.test_pi_half
        self.use_pi2_for_pi = "use_pi2_for_pi" in self.cfg.expt and self.cfg.expt.use_pi2_for_pi

        if not self.test_pi_half:
            self.use_pi2_for_pi = True  # always using 2 pi/2 pulses for pi now

        # initialize wait registers
        self.safe_regwi(self.q_rps[qTest], self.r_wait, self.us2cycles(cfg.expt.start))
        self.safe_regwi(self.q_rps[qTest], self.r_phase2, 0)

        self.set_gen_delays()
        self.sync_all(200)

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
        if self.checkEF:
            self.X_pulse(q=qTest, ZZ_qubit=qZZ, play=True)

        if self.checkEF:
            test_X_pulse = self.Xef_pulse
        else:
            test_X_pulse = self.X_pulse

        # play pi/2 pulse with the freq that we want to calibrate (phase = 0)
        test_X_pulse(q=qTest, ZZ_qubit=qZZ, pihalf=True, divide_len=self.divide_len, play=True)

        # handle echo
        num_pi = 0
        if "num_pi" in self.cfg.expt:
            num_pi = self.cfg.expt.num_pi
        if num_pi >= 1:
            assert "echo_type" in self.cfg.expt
            assert self.cfg.expt.echo_type in ["cp", "cpmg"]
            echo_type = self.cfg.expt.echo_type
        if "echo_type" in self.cfg.expt and self.cfg.expt.echo_type in ["cp", "cpmg"]:
            assert "num_pi" in self.cfg.expt
            assert self.cfg.expt.num_pi >= 1
        for i in range(num_pi):
            self.sync(self.q_rps[qTest], self.r_wait)
            if echo_type == "cp":
                phase = 0
            elif echo_type == "cpmg":
                phase = 90

            test_X_pulse(q=qTest, ZZ_qubit=qZZ, extra_phase=phase, divide_len=self.divide_len, play=True)
            self.sync(self.q_rps[qTest], self.r_wait)

        # wait advanced wait time (ramsey wait time)
        if num_pi == 0:
            self.sync(self.q_rps[qTest], self.r_wait)

        # play pi/2 pulse with advanced phase
        test_X_pulse(q=qTest, ZZ_qubit=qZZ, pihalf=True, divide_len=self.divide_len, play=False, set_reg=True)
        if self.qubit_ch_types[qTest] == "int4":
            self.bitwi(self.q_rps[qTest], self.r_phase3, self.r_phase2, "<<", 16)
            self.bitwi(self.q_rps[qTest], self.r_phase3, self.r_phase3, "|", self.f_pi_test_reg)
            self.mathi(self.q_rps[qTest], self.r_phase, self.r_phase3, "+", 0)
        else:
            self.mathi(self.q_rps[qTest], self.r_phase, self.r_phase2, "+", 0)
        self.pulse(ch=self.qubit_chs[qTest])

        if self.checkEF:  # map excited back to qubit ground state for measurement
            if (
                "frequency_ef" not in self.cfg.device.readout
                or self.cfg.device.readout.frequency[qTest] != self.cfg.device.readout.frequency_ef[qTest]
            ):
                self.X_pulse(q=qTest, ZZ_qubit=qZZ, play=True)

        # align channels and measure
        self.sync_all()
        self.measure(
            pulse_ch=self.measure_chs,
            adcs=self.adc_chs,
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest]),
        )

    def update(self):
        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        if qZZ is None:
            qZZ = qTest

        num_pi = 0
        if "num_pi" in self.cfg.expt:
            num_pi = self.cfg.expt.num_pi

        if num_pi > 0:
            self.mathi(
                self.q_rps[qTest], self.r_wait, self.r_wait, "+", self.us2cycles(self.cfg.expt.step / 2 / num_pi)
            )  # update the time between two π/2 pulses
        else:
            self.mathi(
                self.q_rps[qTest], self.r_wait, self.r_wait, "+", self.us2cycles(self.cfg.expt.step)
            )  # update the time between two π/2 pulses

        phase_step = self.deg2reg(
            360 * self.cfg.expt.ramsey_freq * self.cfg.expt.step, gen_ch=self.qubit_chs[qTest]
        )  # phase step [deg] = 360 * f_Ramsey [MHz] * tau_step [us]
        self.mathi(
            self.q_rps[qTest], self.r_phase2, self.r_phase2, "+", phase_step
        )  # advance the phase of the LO for the second π/2 pulse


class RamseyExperiment(Experiment):
    """
    Ramsey experiment
    Experimental Config:
    expt = dict(
        start: wait time start sweep [us]
        step: wait time step - make sure nyquist freq = 0.5 * (1/step) > ramsey (signal) freq!
        expts: number experiments stepping from start
        ramsey_freq: frequency by which to advance phase [MHz]
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
        checkZZ: True/False for putting another qubit in e (specify as qA)
        checkEF: does ramsey on the EF transition instead of ge
        qubits: if not checkZZ, just specify [1 qubit]. if checkZZ: [qA in e , qB sweeps length rabi]

        to run echo experiment:
        num_pi
        echo_type: cp or cpmg
    )
    """

    def __init__(self, soccfg=None, path="", prefix="Ramsey", config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        # expand entries in config that are length 1 to fill all qubits
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

        ramsey = RamseyProgram(soccfg=self.soccfg, cfg=self.cfg)

        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        if qZZ is None:
            qZZ = qTest
        num_pi = 0
        if "num_pi" in self.cfg.expt:
            num_pi = self.cfg.expt.num_pi
        print(
            f'Running Ramsey {"EF " if self.cfg.expt.checkEF else ""}{"Echo " if num_pi > 0 else ""}on Q{qTest} {"with ZZ Q" + str(qZZ) if qZZ != qTest else ""}'
        )

        x_pts, idata, qdata = ramsey.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=progress)
        _, ishots, qshots = ramsey.get_shots()

        avgi = idata[qTest][0]
        avgq = qdata[qTest][0]

        if "full_mux_expt" in self.cfg.expt and self.cfg.expt.full_mux_expt:
            amps = np.average(np.abs(ishots + 1j * qshots), axis=2)[qTest]  # Calculating the magnitude
            phases = np.average(np.angle(ishots + 1j * qshots), axis=2)[qTest]  # Calculating the phase
        else:
            amps = np.abs(avgi + 1j * avgq)
            phases = np.angle(avgi + 1j * avgq)

        data = {"xpts": x_pts, "avgi": avgi, "avgq": avgq, "amps": amps, "phases": phases}
        self.data = data
        return data

    def analyze(self, data=None, fit=True, fit_num_sin=1):
        if data is None:
            data = self.data

        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset]
            # fitparams=[yscale0, freq0, phase_deg0, decay0, yscale1, freq1, phase_deg1, y0] # two fit freqs
            # fitparams=[yscale0, freq0, phase_deg0, decay0, y00, x00, yscale1, freq1, phase_deg1, y01, yscale2, freq2, phase_deg2, y02] # three fit freqs

            # Remove the first and last point from fit in case weird edge measurements
            fitparams = None
            if fit_num_sin == 2:
                fitfunc = fitter.fittwofreq_decaysin
                fitparams = [None] * 8
                fitparams[1] = self.cfg.expt.ramsey_freq
                fitparams[3] = 15  # decay
                fitparams[4] = 0.05  # yscale1 (ratio relative to base oscillations)
                fitparams[5] = 1 / 12.5  # freq1
                # print('FITPARAMS', fitparams[7])
            elif fit_num_sin == 3:
                fitfunc = fitter.fitthreefreq_decaysin
                fitparams = [None] * 14
                fitparams[1] = self.cfg.expt.ramsey_freq
                fitparams[3] = 15  # decay
                fitparams[6] = 1.1  # yscale1
                fitparams[7] = 0.415  # freq1
                fitparams[-4] = 1.1  # yscale2
                fitparams[-3] = 0.494  # freq2
                # print('FITPARAMS', fitparams[7])
            else:
                fitfunc = fitter.fitdecaysin
                fitparams = [None, self.cfg.expt.ramsey_freq, 0, None, None]
            p_avgi, pCov_avgi = fitfunc(data["xpts"][:-1], data["avgi"][:-1], fitparams=fitparams)
            p_avgq, pCov_avgq = fitfunc(data["xpts"][:-1], data["avgq"][:-1], fitparams=fitparams)
            p_amps, pCov_amps = fitfunc(data["xpts"][:-1], data["amps"][:-1], fitparams=fitparams)
            data["fit_avgi"] = p_avgi
            data["fit_avgq"] = p_avgq
            data["fit_amps"] = p_amps
            data["fit_err_avgi"] = pCov_avgi
            data["fit_err_avgq"] = pCov_avgq
            data["fit_err_amps"] = pCov_amps

            # print('p avgi', p_avgi)
            # print('p avgq', p_avgq)
            # print('p amps', p_amps)

            if isinstance(p_avgi, (list, np.ndarray)):
                data["f_adjust_ramsey_avgi"] = sorted(
                    (self.cfg.expt.ramsey_freq - p_avgi[1], -self.cfg.expt.ramsey_freq - p_avgi[1]), key=abs
                )
            if isinstance(p_avgq, (list, np.ndarray)):
                data["f_adjust_ramsey_avgq"] = sorted(
                    (self.cfg.expt.ramsey_freq - p_avgq[1], -self.cfg.expt.ramsey_freq - p_avgq[1]), key=abs
                )
            if isinstance(p_amps, (list, np.ndarray)):
                data["f_adjust_ramsey_amps"] = sorted(
                    (self.cfg.expt.ramsey_freq - p_amps[1], -self.cfg.expt.ramsey_freq - p_amps[1]), key=abs
                )

            if fit_num_sin == 2:
                data["f_adjust_ramsey_avgi2"] = sorted(
                    (self.cfg.expt.ramsey_freq - p_avgi[5], -self.cfg.expt.ramsey_freq - p_avgi[5]), key=abs
                )
                data["f_adjust_ramsey_avgq2"] = sorted(
                    (self.cfg.expt.ramsey_freq - p_avgq[5], -self.cfg.expt.ramsey_freq - p_avgq[5]), key=abs
                )
                data["f_adjust_ramsey_amps2"] = sorted(
                    (self.cfg.expt.ramsey_freq - p_amps[5], -self.cfg.expt.ramsey_freq - p_amps[5]), key=abs
                )
        return data

    def display(self, data=None, fit=True, fit_num_sin=1):
        if data is None:
            data = self.data

        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        self.checkZZ = False
        if qZZ is not None and qZZ != qTest:
            self.checkZZ = True
        else:
            qZZ = qTest
        self.checkEF = self.cfg.expt.checkEF

        num_qubits_sample = len(self.cfg.device.readout.frequency)
        f_pi_test = np.reshape(self.cfg.device.qubit.f_ge, (num_qubits_sample, num_qubits_sample))[qTest, qZZ]
        if self.checkEF:
            f_pi_test = np.reshape(self.cfg.device.qubit.f_ef, (num_qubits_sample, num_qubits_sample))[qTest, qZZ]

        num_pi = 0
        if "num_pi" in self.cfg.expt:
            num_pi = self.cfg.expt.num_pi

        title = (
            ("EF" if self.checkEF else "")
            + f'Ramsey {"Echo " if num_pi > 0 else ""}on Q{qTest}'
            + (f" with Q{qZZ} in e" if self.checkZZ else "")
        )

        if fit_num_sin == 2:
            fitfunc = fitter.twofreq_decaysin
        elif fit_num_sin == 3:
            fitfunc = fitter.threefreq_decaysin
        else:
            fitfunc = fitter.decaysin

        plt.figure(figsize=(8, 5))
        plt.subplot(
            111,
            title=f"{title} (Ramsey {'Echo ' if num_pi > 0 else ''}Freq: {self.cfg.expt.ramsey_freq} MHz)",
            xlabel="Wait Time [us]",
            ylabel="Amplitude [ADC level]",
        )
        plt.plot(data["xpts"][:-1], data["amps"][:-1], ".-")
        if fit:
            p = data["fit_amps"]
            if isinstance(p, (list, np.ndarray)):
                pCov = data["fit_err_amps"]
                captionStr = f"$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}"
                plt.plot(data["xpts"][:-1], fitfunc(data["xpts"][:-1], *p), label=captionStr)
                plt.plot(
                    data["xpts"][:-1],
                    fitter.expfunc(data["xpts"][:-1], p[-1], p[0], 0, p[3]),
                    color="0.2",
                    linestyle="--",
                )
                plt.plot(
                    data["xpts"][:-1],
                    fitter.expfunc(data["xpts"][:-1], p[-1], -p[0], 0, p[3]),
                    color="0.2",
                    linestyle="--",
                )
                plt.legend()
                print(f"Current pi pulse frequency: {f_pi_test}")
                print(f"Fit frequency from amps [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}")
                if p[1] > 2 * self.cfg.expt.ramsey_freq:
                    print("WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!")
                print(
                    f"Suggested new pi pulse frequencies from fit amps [MHz]:\n",
                    f'\t{f_pi_test + data["f_adjust_ramsey_amps"][0]}\n',
                    f'\t{f_pi_test + data["f_adjust_ramsey_amps"][1]}',
                )
                if fit_num_sin == 2:
                    print(
                        "Beating frequencies from fit amps [MHz]:\n",
                        f"\tyscale base: {1-p[4]}",
                        f"\tfit freq {p[1]}\n",
                        f"\tyscale1: {p[4]}" f"\tfit freq {p[5]}\n",
                    )
                print(f"T2 Ramsey from fit amps [us]: {p[3]} +/- {np.sqrt(pCov[3][3])}")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 9))
        plt.subplot(211, title=f"{title} (Ramsey Freq: {self.cfg.expt.ramsey_freq} MHz)", ylabel="I [ADC level]")
        plt.plot(data["xpts"][:-1], data["avgi"][:-1], ".-")
        if fit:
            p = data["fit_avgi"]
            if isinstance(p, (list, np.ndarray)):
                pCov = data["fit_err_avgi"]
                captionStr = f"$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}"
                plt.plot(data["xpts"][:-1], fitfunc(data["xpts"][:-1], *p), label=captionStr)
                plt.plot(
                    data["xpts"][:-1],
                    fitter.expfunc(data["xpts"][:-1], p[-1], p[0], 0, p[3]),
                    color="0.2",
                    linestyle="--",
                )
                plt.plot(
                    data["xpts"][:-1],
                    fitter.expfunc(data["xpts"][:-1], p[-1], -p[0], 0, p[3]),
                    color="0.2",
                    linestyle="--",
                )
                plt.legend()
                print(f"Current pi pulse frequency: {f_pi_test}")
                print(f"Fit frequency from I [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}")
                if p[1] > 2 * self.cfg.expt.ramsey_freq:
                    print("WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!")
                print(
                    "Suggested new pi pulse frequency from fit I [MHz]:\n",
                    f'\t{f_pi_test + data["f_adjust_ramsey_avgi"][0]}\n',
                    f'\t{f_pi_test + data["f_adjust_ramsey_avgi"][1]}',
                )
                if fit_num_sin == 2:
                    print(
                        "Beating frequencies from fit avgi [MHz]:\n",
                        f"\tyscale base: {1-p[4]}",
                        f"\tfit freq {p[1]}\n",
                        f"\tyscale1: {p[4]}" f"\tfit freq {p[5]}\n",
                    )
                print(f"T2 Ramsey from fit I [us]: {p[3]} +/- {np.sqrt(pCov[3][3])}")
        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC level]")
        plt.plot(data["xpts"][:-1], data["avgq"][:-1], ".-")
        if fit:
            p = data["fit_avgq"]
            if isinstance(p, (list, np.ndarray)):
                pCov = data["fit_err_avgq"]
                captionStr = f"$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}"
                plt.plot(data["xpts"][:-1], fitfunc(data["xpts"][:-1], *p), label=captionStr)
                plt.plot(
                    data["xpts"][:-1],
                    fitter.expfunc(data["xpts"][:-1], p[-1], p[0], 0, p[3]),
                    color="0.2",
                    linestyle="--",
                )
                plt.plot(
                    data["xpts"][:-1],
                    fitter.expfunc(data["xpts"][:-1], p[-1], -p[0], 0, p[3]),
                    color="0.2",
                    linestyle="--",
                )
                plt.legend()
                print(f"Fit frequency from Q [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}")
                if p[1] > 2 * self.cfg.expt.ramsey_freq:
                    print("WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!")
                print(
                    "Suggested new pi pulse frequencies from fit Q [MHz]:\n",
                    f'\t{f_pi_test + data["f_adjust_ramsey_avgq"][0]}\n',
                    f'\t{f_pi_test + data["f_adjust_ramsey_avgq"][1]}',
                )
                if fit_num_sin == 2:
                    print(
                        "Beating frequencies from fit avgq [MHz]:\n",
                        f"\tyscale base: {1-p[4]}",
                        f"\tfit freq {p[1]}\n",
                        f"\tyscale1: {p[4]}" f"\tfit freq {p[5]}\n",
                    )
                print(f"T2 Ramsey from fit Q [us]: {p[3]} +/- {np.sqrt(pCov[3][3])}")

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname
