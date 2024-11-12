from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from slab import AttrDict, Experiment, dsfit
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter
from experiments.clifford_averager_program import (
    QutritAveragerProgram,
    post_select_shots,
    ps_threshold_adjust,
    rotate_and_threshold,
)
from experiments.single_qubit.single_shot import hist
from experiments.two_qubit.twoQ_state_tomography import (
    ErrorMitigationStateTomo1QProgram,
)
from TomoAnalysis import TomoAnalysis


class LengthRabiProgram(QutritAveragerProgram):
    """
    Measures Rabi oscillations by sweeping over the duration of the qubit drive pulse. This is a preliminary measurement to prove that we see Rabi oscillations. This measurement is followed up by the Amplitude Rabi experiment.
    """

    def initialize(self):
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
        sigma_test_cycles = self.us2cycles(self.cfg.expt.sigma_test, gen_ch=self.qubit_chs[qTest])
        if "sigma_test" in self.cfg.expt and sigma_test_cycles > 3:
            self.num_qubits_sample = len(self.cfg.device.readout.frequency)
            if self.checkZZ:
                self.cfg.device.qubit.pulses.pi_ef.sigma[
                    qTest * self.num_qubits_sample + qZZ
                ] = self.cfg.expt.sigma_test
            else:
                self.cfg.device.qubit.pulses.pi_ge.sigma[
                    qTest * self.num_qubits_sample + qZZ
                ] = self.cfg.expt.sigma_test

        super().initialize()
        # calibrate the pi/2 pulse instead of the pi pulse by taking half the sigma and calibrating the gain
        self.test_pi_half = False
        self.divide_len = True
        if "divide_len" in self.cfg.expt:
            self.divide_len = self.cfg.expt.divide_len
        if "test_pi_half" in self.cfg.expt and self.cfg.expt.test_pi_half:
            self.test_pi_half = self.cfg.expt.test_pi_half

        if self.checkEF:
            if "pulse_ge" not in self.cfg.expt:
                self.pulse_ge = True
            else:
                self.pulse_ge = self.cfg.expt.pulse_ge
            if "readout_ge" not in self.cfg.expt:
                self.readout_ge = True
            else:
                self.readout_ge = self.cfg.expt.readout_ge

        self.error_amp = "error_amp" in self.cfg.expt and self.cfg.expt.error_amp
        self.qTest_rphase = self.sreg(self.qubit_chs[qTest], "phase")

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
        if self.checkEF and self.pulse_ge:
            self.X_pulse(q=qTest, ZZ_qubit=qZZ, play=True)

        self.pi_minuspi = "pi_minuspi" in self.cfg.expt and self.cfg.expt.pi_minuspi
        self.check_I_distort = "check_I_distort" in self.cfg.expt and self.cfg.expt.check_I_distort
        self.check_C_distort = "check_C_distort" in self.cfg.expt and self.cfg.expt.check_C_distort
        if self.check_C_distort:
            assert self.pi_minuspi
        if self.check_I_distort:
            assert "check_I_phase" in self.cfg.expt
            self.check_I_phase = self.cfg.expt.check_I_phase
        self.use_pi2_for_pi = "use_pi2_for_pi" in self.cfg.expt and self.cfg.expt.use_pi2_for_pi
        if self.check_I_distort or self.check_C_distort:
            assert "delay_error_amp" in self.cfg.expt

        # play pi pulse that we want to calibrate
        play_pulse = True
        if "sigma_test" in self.cfg.expt and self.cfg.expt.sigma_test == 0:
            play_pulse = False
        self.cfg.expt.gain = 0

        if "pulse_type" in self.cfg.expt:
            if "pulse_type" != "gauss" and "pulse_type" != "const":
                special = self.cfg.expt.pulse_type

        if "skip_first_pi2" in self.cfg.expt:
            skip_first_pi2 = self.cfg.expt.skip_first_pi2
        else:
            skip_first_pi2 = False

        if not self.test_pi_half:
            self.use_pi2_for_pi = True  # always using 2 pi/2 pulses for pi now

        if play_pulse:
            if self.error_amp:
                assert "n_pulses" in self.cfg.expt and self.cfg.expt.n_pulses is not None
                n_pulses = self.cfg.expt.n_pulses
                # print('init pi/2 freq', self.reg2freq(self.f_pi_test_reg, gen_ch=self.qubit_chs[qTest]), 'gain', self.pi_test_half_gain)

                if not self.pi_minuspi or self.check_I_distort:
                    if not skip_first_pi2:
                        # play initial pi/2 pulse if you're just doing error amplification and not the pi/-pi sweep
                        if not self.checkEF:
                            self.X_pulse(q=qTest, ZZ_qubit=qZZ, pihalf=True, play=True, special=special)
                        else:
                            self.Xef_pulse(q=qTest, ZZ_qubit=qZZ, pihalf=True, play=True, special=special)
                    self.sync_all()

                # setup pulse regs to save memory for iteration
                if self.checkEF:
                    self.Xef_pulse(
                        q=qTest,
                        ZZ_qubit=qZZ,
                        pihalf=self.test_pi_half or self.use_pi2_for_pi,
                        divide_len=self.divide_len,
                        play=False,
                        set_reg=True,
                        special=special,
                    )
                    name = "X_ef"
                else:
                    self.X_pulse(
                        q=qTest,
                        ZZ_qubit=qZZ,
                        pihalf=self.test_pi_half or self.use_pi2_for_pi,
                        divide_len=self.divide_len,
                        play=False,
                        set_reg=True,
                        special=special,
                    )
                    name = "X"
                if self.cfg.expt.pulse_type == "robust":
                    name += "_robust"
                if self.checkZZ:
                    name += f"_ZZ{qZZ}"
                if self.test_pi_half or self.cfg.expt.pulse_type == "robust" or self.use_pi2_for_pi:
                    name += "_half"
                self.cfg.expt.gain = self.pulse_dict[f"{name}_q{qTest}"]["gain"]

                # print("n_pulses", n_pulses)
                for i in range(int(n_pulses)):  # n_pulses is the number of cycle sets
                    phase = 0
                    if "use_Y" in self.cfg.expt:
                        use_Y = self.cfg.expt.use_Y
                    else:
                        use_Y = False

                    if i % 2 == 1:
                        if self.pi_minuspi:
                            phase = -180
                        if self.check_I_distort:
                            phase = -90 + self.check_I_phase

                    if use_Y:
                        phase = -90 + phase

                    num_test_pulses = 1

                    if self.use_pi2_for_pi:
                        num_test_pulses = 2

                    self.safe_regwi(
                        self.q_rps[qTest], self.qTest_rphase, self.deg2reg(phase, gen_ch=self.qubit_chs[qTest])
                    )
                    for j in range(num_test_pulses):
                        # print("phase", phase)
                        self.pulse(ch=self.qubit_chs[qTest])
                        # self.sync_all(20) # MAY NEED TO ADD DELAY IF PULSE IS SHORT!!

                        # print(
                        #     "pulse pi test freq",
                        #     self.reg2freq(self.f_pi_test_reg, gen_ch=self.qubit_chs[qTest]),
                        #     "qtest",
                        #     qTest,
                        #     "gain",
                        #     self.gain_pi_test,
                        #     "phase",
                        #     phase,
                        # )
                    delay_cycles = 0
                    if self.check_C_distort or self.check_I_distort:
                        delay_cycles = self.cfg.expt.delay_error_amp
                    if delay_cycles > 0:
                        self.sync_all(delay_cycles)

                if self.check_C_distort or self.check_I_distort:
                    if self.checkEF:
                        self.Yef_pulse(
                            q=qTest, ZZ_qubit=qZZ, pihalf=self.test_pi_half, divide_len=self.divide_len, play=True
                        )
                    else:
                        self.Y_pulse(
                            q=qTest, ZZ_qubit=qZZ, pihalf=self.test_pi_half, divide_len=self.divide_len, play=True
                        )

            else:
                # n_pulses = 1
                # if self.test_pi_half:
                n_pulses = 2
                for i in range(int(n_pulses)):
                    if self.checkEF:
                        self.Xef_pulse(
                            q=qTest, ZZ_qubit=qZZ, pihalf=self.test_pi_half, divide_len=self.divide_len, play=True
                        )
                        name = "X_ef"
                    else:
                        self.X_pulse(
                            q=qTest, ZZ_qubit=qZZ, pihalf=self.test_pi_half, divide_len=self.divide_len, play=True
                        )
                        name = "X"
                if self.cfg.expt.pulse_type == "robust":
                    name += "_robust"
                if self.checkZZ:
                    name += f"_ZZ{qZZ}"
                if self.test_pi_half or self.cfg.expt.pulse_type == "robust" or self.use_pi2_for_pi:
                    name += "_half"
                self.cfg.expt.gain = self.pulse_dict[f"{name}_q{qTest}"]["gain"]

        if self.checkEF and self.readout_ge:  # map excited back to qubit ground state for measurement
            self.X_pulse(q=qTest, ZZ_qubit=qZZ, play=True)

        # align channels and measure
        self.sync_all()
        self.measure(
            pulse_ch=self.measure_chs,
            adcs=self.adc_chs,
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(max([cfg.device.readout.relax_delay[q] for q in range(4)])),
        )


# ====================================================== #


class LengthRabiExperiment(Experiment):
    """
    Length Rabi Experiment
    Experimental Config
    expt = dict(
        start: start length [us],
        step: length step,
        expts: number of different length experiments,
        reps: number of reps,
        pulse_type: 'gauss' or 'const'
        checkEF: does ramsey on the EF transition instead of ge
        qTest: qubit on which to do the test pulse
        qZZ: None if not checkZZ, else specify other qubit to pi pulse
        pulse_ge: whether to pulse ge before the test pulse
        readout_ge: whether to readout at the g/e set point or e/f set point
    )
    """

    def __init__(self, soccfg=None, path="", prefix="LengthRabi", config_file=None, progress=None):
        super().__init__(
            path=path,
            soccfg=soccfg,
            prefix=prefix,
            config_file=config_file,
            progress=progress,
        )

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

        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        self.checkZZ = False
        if qZZ is not None and qZZ != qTest:
            self.checkZZ = True
        else:
            qZZ = qTest
        self.checkEF = self.cfg.expt.checkEF

        print(
            f'Running length rabi {"EF " if self.cfg.expt.checkEF else ""}on Q{qTest} {"with ZZ Q" + str(qZZ) if self.checkZZ else ""}'
        )

        lengths = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        if "gain" in self.cfg.expt:
            assert (
                False
            ), "WARNING: gain is no longer supported as a parameter in cfg.expt, set it directly to the cfg parameter"

        data = {"xpts": [], "avgi": [], "avgq": [], "amps": [], "phases": []}

        if "readout_ge" not in self.cfg.expt:
            self.cfg.expt.readout_ge = True
        for length in tqdm(lengths, disable=not progress):
            self.cfg.expt.sigma_test = float(length)
            if not self.cfg.expt.readout_ge:
                self.cfg.device.readout.frequency = self.cfg.device.readout.frequency_ef
                self.cfg.device.readout.readout_length = self.cfg.device.readout.readout_length_ef
            lengthrabi = LengthRabiProgram(soccfg=self.soccfg, cfg=self.cfg)
            self.prog = lengthrabi

            # print('\n\n', length)
            # from qick.helpers import progs2json
            # print(progs2json([self.prog.dump_prog()]))

            avgi, avgq = lengthrabi.acquire(self.im[self.cfg.aliases.soc])
            self.cfg.expt.gain = lengthrabi.cfg.expt.gain
            avgi = avgi[qTest][0]
            avgq = avgq[qTest][0]
            amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
            phase = np.angle(avgi + 1j * avgq)  # Calculating the phase
            data["xpts"].append(length)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)

        for k, a in data.items():
            data[k] = np.array(a)

        self.data = data

        return data

    def analyze(self, data=None, fit=True, fit_func="decaysin"):
        if data is None:
            data = self.data
        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
            # Remove the first and last point from fit in case weird edge measurements
            # fitparams = [None, 1/max(data['xpts']), None, None]
            xdata = data["xpts"]
            fitparams = None
            if fit_func == "sin":
                fitparams = [None] * 4
            elif fit_func == "decaysin":
                fitparams = [None] * 5
            fitparams[1] = 2.0 / xdata[-1]
            if fit_func == "decaysin":
                fit_fitfunc = fitter.fitdecaysin
            elif fit_func == "sin":
                fit_fitfunc = fitter.fitsin
            p_avgi, pCov_avgi = fit_fitfunc(data["xpts"][:-1], data["avgi"][:-1], fitparams=fitparams)
            p_avgq, pCov_avgq = fit_fitfunc(data["xpts"][:-1], data["avgq"][:-1], fitparams=fitparams)
            p_amps, pCov_amps = fit_fitfunc(data["xpts"][:-1], data["amps"][:-1], fitparams=fitparams)
            data["fit_avgi"] = p_avgi
            data["fit_avgq"] = p_avgq
            data["fit_amps"] = p_amps
            data["fit_err_avgi"] = pCov_avgi
            data["fit_err_avgq"] = pCov_avgq
            data["fit_err_amps"] = pCov_amps
        return data

    def display(self, data=None, fit=True, fit_func="decaysin"):
        if data is None:
            data = self.data

        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        self.checkZZ = False
        if qZZ is not None and qZZ != qTest:
            self.checkZZ = True
        else:
            qZZ = qTest

        xpts_ns = data["xpts"] * 1e3
        if fit_func == "decaysin":
            fit_func = fitter.decaysin
        elif fit_func == "sin":
            fit_func = fitter.sinfunc

        gain = self.cfg.expt.gain
        title = f"Length Rabi {'EF ' if self.cfg.expt.checkEF else ''}on Q{qTest} (Gain {gain}) {(', ZZ Q'+str(qZZ)) if self.checkZZ else ''}"

        plt.figure(figsize=(8, 5))
        plt.subplot(111, title=title, xlabel="Length [ns]", ylabel="Amplitude [ADC units]")
        plt.plot(xpts_ns[:-1], data["amps"][:-1], ".-")
        if fit:
            p = data["fit_amps"]
            plt.plot(xpts_ns[:-1], fit_func(data["xpts"][:-1], *p))
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 9))
        plt.subplot(211, title=title, ylabel="I [adc level]")
        plt.plot(xpts_ns[1:-1], data["avgi"][1:-1], ".-")
        if fit:
            p = data["fit_avgi"]
            plt.plot(xpts_ns[0:-1], fit_func(data["xpts"][0:-1], *p))
            if p[2] > 180:
                p[2] = p[2] - 360
            elif p[2] < -180:
                p[2] = p[2] + 360
            if p[2] < 0:
                pi_length = (1 / 2 - p[2] / 180) / 2 / p[1]
            else:
                pi_length = (3 / 2 - p[2] / 180) / 2 / p[1]
            pi2_length = pi_length / 2
            if fit_func == "decaysin":
                print("Decay from avgi [us]", p[3])
            print(f"Pi length from avgi data [us]: {pi_length}")
            print(f"\tPi/2 length from avgi data [us]: {pi2_length}")
            plt.axvline(pi_length * 1e3, color="0.2", linestyle="--")
            plt.axvline(pi2_length * 1e3, color="0.2", linestyle="--")

        print()
        plt.subplot(212, xlabel="Pulse length [ns]", ylabel="Q [adc levels]")
        plt.plot(xpts_ns[1:-1], data["avgq"][1:-1], ".-")
        if fit:
            p = data["fit_avgq"]
            plt.plot(xpts_ns[0:-1], fit_func(data["xpts"][0:-1], *p))
            if p[2] > 180:
                p[2] = p[2] - 360
            elif p[2] < -180:
                p[2] = p[2] + 360
            if p[2] < 0:
                pi_length = (1 / 2 - p[2] / 180) / 2 / p[1]
            else:
                pi_length = (3 / 2 - p[2] / 180) / 2 / p[1]
            pi2_length = pi_length / 2
            if fit_func == "decaysin":
                print("Decay from avgq [us]", p[3])
            print(f"Pi length from avgq data [us]: {pi_length}")
            print(f"Pi/2 length from avgq data [us]: {pi2_length}")
            plt.axvline(pi_length * 1e3, color="0.2", linestyle="--")
            plt.axvline(pi2_length * 1e3, color="0.2", linestyle="--")
        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname


# ====================================================== #


class NPulseExperiment(Experiment):
    """
    Play a pi/2 or pi pulse variable N times
    Experimental Config
    expt = dict(
        start: start N [us],
        step
        expts
        reps: number of reps,
        gain: gain to use for the calibration pulse (uses config value by default, calculated based on flags)
        pulse_type: 'gauss' or 'const' (uses config value by default)
        checkZZ: True/False for putting another qubit in e (specify as qZZ)
        checkEF: does ramsey on the EF transition instead of ge
        qubits: if not checkZZ, just specify [1 qubit]. if checkZZ: [qZZ in e , qB sweeps length rabi]
        test_pi_half: calibrate the pi/2 instead of pi pulse by dividing length cycles // 2
        readout_ge: whether to readout at the g/e set point or e/f set point
    )
    """

    def __init__(self, soccfg=None, path="", prefix="NPulseExpt", config_file=None, progress=None):
        super().__init__(
            path=path,
            soccfg=soccfg,
            prefix=prefix,
            config_file=config_file,
            progress=progress,
        )

    def acquire(self, progress=False, debug=True):
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

        cycles = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])

        data = {
            "xpts": [],
            "avgi": [],
            "avgq": [],
            "amps": [],
            "phases": [],
            "counts_calib": [],
        }

        self.checkEF = self.cfg.expt.checkEF
        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        if qZZ is None:
            qZZ = qTest

        if "readout_ge" not in self.cfg.expt:
            self.cfg.expt.readout_ge = True
        if not self.cfg.expt.readout_ge:
            self.cfg.device.readout.frequency = self.cfg.device.readout.frequency_ef
            self.cfg.device.readout.readout_length = self.cfg.device.readout.readout_length_ef

        # ================= #
        # Get single shot calibration for 1 qubit
        # ================= #
        thresholds_q = ge_avgs_q = angles_q = fids_q = None
        if "post_process" not in self.cfg.expt.keys():  # threshold or scale
            self.cfg.expt.post_process = None
            assert False, "you probably want to be doing this experiment with post processing or the fit will be weird"

        if self.cfg.expt.post_process is not None:
            if (
                "angles" in self.cfg.expt
                and "thresholds" in self.cfg.expt
                and "ge_avgs" in self.cfg.expt
                and "counts_calib" in self.cfg.expt
                and self.cfg.expt.angles is not None
                and self.cfg.expt.thresholds is not None
                and self.cfg.expt.ge_avgs is not None
                and self.cfg.expt.counts_calib is not None
            ):
                angles_q = self.cfg.expt.angles
                thresholds_q = self.cfg.expt.thresholds
                ge_avgs_q = np.asarray(self.cfg.expt.ge_avgs)
                data["counts_calib"] = self.cfg.expt.counts_calib
                print("Re-using provided angles, thresholds, ge_avgs")
            else:
                thresholds_q = [0] * 4
                ge_avgs_q = [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]
                angles_q = [0] * 4
                fids_q = [0] * 4

                # We really just need the single shot plots here, but convenient to use the ErrorMitigation tomo to do it
                sscfg = AttrDict(deepcopy(self.cfg))
                sscfg.expt.reps = sscfg.expt.singleshot_reps
                sscfg.expt.qubit = qTest

                calib_prog_dict = dict()
                calib_order = ["g", "e" if self.cfg.expt.readout_ge else "f"]
                for prep_state in tqdm(calib_order):
                    # print(prep_state)
                    sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state, apply_q1_pi2=False)
                    err_tomo = ErrorMitigationStateTomo1QProgram(soccfg=self.soccfg, cfg=sscfg)
                    err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=False)
                    calib_prog_dict.update({prep_state: err_tomo})

                g_prog = calib_prog_dict["g"]
                Ig, Qg = g_prog.get_shots(verbose=False)

                # Get readout angle + threshold for qubits
                e_prog = calib_prog_dict["e" if self.cfg.expt.readout_ge else "f"]
                Ie, Qe = e_prog.get_shots(verbose=False)
                shot_data = dict(Ig=Ig[qTest], Qg=Qg[qTest], Ie=Ie[qTest], Qe=Qe[qTest])
                print(f"Qubit  ({qTest})")
                fid, threshold, angle = hist(data=shot_data, plot=debug, verbose=False)
                thresholds_q[qTest] = threshold[0]
                ge_avgs_q[qTest] = [
                    np.average(Ig[qTest]),
                    np.average(Qg[qTest]),
                    np.average(Ie[qTest]),
                    np.average(Qe[qTest]),
                ]
                angles_q[qTest] = angle
                fids_q[qTest] = fid[0]
                print(
                    f"ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[qTest]} \t threshold ge: {thresholds_q[qTest]}"
                )

                # Process the shots taken for the confusion matrix with the calibration angles
                for prep_state in calib_order:
                    counts = calib_prog_dict[prep_state].collect_counts(angle=angles_q, threshold=thresholds_q)
                    data["counts_calib"].append(counts)

                if debug:
                    print(f"thresholds={thresholds_q},")
                    print(f"angles={angles_q},")
                    print(f"ge_avgs={ge_avgs_q},")
                    print(f"counts_calib={np.array(data['counts_calib']).tolist()}")

            data["thresholds"] = thresholds_q
            data["angles"] = angles_q
            data["ge_avgs"] = ge_avgs_q
            data["counts_calib"] = np.array(data["counts_calib"])

        # ================= #
        # Begin actual experiment
        # ================= #
        # define as the length for the pi pulse ** this is still the specification even when calibrating the pi/2 pulse
        length = np.reshape(self.cfg.device.qubit.pulses.pi_ge.sigma, (4, 4))[
            qTest, qZZ
        ]  # length of pulse whose error we are trying to find
        if self.checkEF:
            length = np.reshape(self.cfg.device.qubit.pulses.pi_ef.sigma, (4, 4))[qTest, qZZ]

        self.cfg.expt.sigma_test = float(length)

        if "loops" not in self.cfg.expt:
            self.cfg.expt.loops = 1
        for loop in tqdm(range(self.cfg.expt.loops), disable=not progress or self.cfg.expt.loops == 1):
            for n_cycle in tqdm(cycles, disable=not progress or self.cfg.expt.loops > 1):
                self.cfg.expt.n_pulses = n_cycle
                # print('n cycle', n_cycle)
                lengthrabi = LengthRabiProgram(soccfg=self.soccfg, cfg=self.cfg)
                self.prog = lengthrabi
                avgi, avgq = lengthrabi.acquire_rotated(
                    self.im[self.cfg.aliases.soc],
                    angle=angles_q,
                    threshold=thresholds_q,
                    ge_avgs=ge_avgs_q,
                    post_process=self.cfg.expt.post_process,
                    progress=False,
                    verbose=False,
                )
                self.cfg.expt.gain = self.prog.cfg.expt.gain
                avgi = avgi[qTest]
                avgq = avgq[qTest]
                amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
                phase = np.angle(avgi + 1j * avgq)  # Calculating the phase
                data["avgi"].append(avgi)
                data["avgq"].append(avgq)
                data["amps"].append(amp)
                data["phases"].append(phase)
        data["xpts"] = cycles

        for k, a in data.items():
            data[k] = np.array(a)

        data["avgi"] = np.average(np.reshape(data["avgi"], (self.cfg.expt.loops, len(cycles))), axis=0)
        data["avgq"] = np.average(np.reshape(data["avgq"], (self.cfg.expt.loops, len(cycles))), axis=0)
        data["amps"] = np.average(np.reshape(data["amps"], (self.cfg.expt.loops, len(cycles))), axis=0)
        data["phases"] = np.average(np.reshape(data["phases"], (self.cfg.expt.loops, len(cycles))), axis=0)

        self.data = data

        return data

    def analyze(self, data=None, fit=True, scale=None):
        # scale should be [Ig, Qg, Ie, Qe] single shot experiment

        self.checkEF = self.cfg.expt.checkEF
        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        if qZZ is None:
            qZZ = qTest

        if data is None:
            data = self.data
        if fit:
            xdata = data["xpts"]
            fitparams = None
            fitparams = [None, 1 * np.pi / 180]
            print("fitparams", fitparams)
            fit_fitfunc = fitter.fit_probg_Xhalf
            # if self.cfg.expt.test_pi_half:
            #     fit_fitfunc = fitter.fit_probg_Xhalf
            # else:
            #     fit_fitfunc = fitter.fit_probg_X
            if scale is not None:
                Ig, Qg, Ie, Qe = scale[self.qTest]
                reformatted_scale = [
                    (Ig, Ie),
                    (Qg, Qe),
                    (np.abs(Ig + 1j * Qg), np.abs(Ie + 1j * Qe)),
                ]
            for fit_i, fit_axis in enumerate(["avgi", "avgq", "amps"]):
                fit_data = data[fit_axis]
                if scale is not None:
                    g_avg, e_avg = reformatted_scale[fit_i]
                    fit_data = (data[fit_axis] - g_avg) / (e_avg - g_avg)
                p, pCov = fit_fitfunc(xdata, fit_data, fitparams=fitparams)
                data[f"fit_{fit_axis}"] = p
                data[f"fit_err_{fit_axis}"] = pCov
        return data

    def display(self, data=None, fit=True, scale=None):
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

        xdata = data["xpts"]
        fit_func = fitter.probg_Xhalf
        # if self.cfg.expt.test_pi_half:
        #     fit_func = fitter.probg_Xhalf
        # else:
        #     fit_func = fitter.probg_X

        title = (
            f"Angle Error Q{qTest}"
            + (f" ZZ Q{qZZ}" if self.checkZZ else "")
            + (" EF" if self.checkEF else "")
            + (" $\pi/2$" if self.cfg.expt.test_pi_half else " $\pi$")
        )

        if scale is not None:
            Ig, Qg, Ie, Qe = scale[self.qTest]
            reformatted_scale = [
                (Ig, Ie),
                (Qg, Qe),
                (np.abs(Ig + 1j * Qg), np.abs(Ie + 1j * Qe)),
            ]

        current_gain = self.cfg.expt.gain

        plt.figure(figsize=(8, 5))
        label = "($X_{\pi/2}, X_{" + ("\pi" if not self.cfg.expt.test_pi_half else "\pi/2") + "}^{"+ (str(2) if self.cfg.expt.test_pi_half else "") +"n}$)"
        plt.subplot(
            111,
            title=title,
            xlabel=f"Number repeated gates {label} [n]",
            ylabel="Amplitude (scaled)",
        )
        plot_data = data["amps"]
        if scale is not None:
            g_avg, e_avg = reformatted_scale[2]
            plot_data = (plot_data - g_avg) / (e_avg - g_avg)
        plt.plot(xdata, plot_data, ".-")
        if fit:
            p = data["fit_amps"]
            pCov = data["fit_err_amps"]
            captionStr = f"$\epsilon$ fit [deg]: {p[1]:.3} $\pm$ {np.sqrt(pCov[1][1]):.3}"
            plt.plot(xdata, fit_func(xdata, *p), label=captionStr)
            plt.legend()
            if self.cfg.expt.test_pi_half:
                amp_ratio = (90 - p[1]) / 90
            else:
                amp_ratio = (180 - p[1]) / 180
            print(f"From amps: adjust amplitude to {current_gain} / {amp_ratio} = {current_gain/amp_ratio}")
        plt.ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 9))
        plt.subplot(211, title=title, ylabel="I (scaled)")
        plot_data = data["avgi"]
        if scale is not None:
            g_avg, e_avg = reformatted_scale[0]
            plot_data = (plot_data - g_avg) / (e_avg - g_avg)
        plt.plot(xdata, plot_data, ".-")
        if fit:
            p = data["fit_avgi"]
            pCov = data["fit_err_avgi"]
            captionStr = f"$\epsilon$ fit [deg]: {p[1]:.3} $\pm$ {np.sqrt(pCov[1][1]):.3}"
            plt.plot(xdata, fit_func(xdata, *p), label=captionStr)
            plt.legend()
            if self.cfg.expt.test_pi_half:
                amp_ratio = (90 - p[1]) / 90
            else:
                amp_ratio = (180 - p[1]) / 180
            print(f"From avgi: adjust amplitude to {current_gain} / {amp_ratio} = {current_gain/amp_ratio}")
        plt.ylim(-0.1, 1.1)

        label = "($X_{\pi/2}, X_{" + ("\pi" if not self.cfg.expt.test_pi_half else "\pi/2") + "}^{2n}$)"
        plt.subplot(212, xlabel=f"Number repeated gates {label} [n]", ylabel="Q (scaled)")
        plot_data = data["avgq"]
        if scale is not None:
            g_avg, e_avg = reformatted_scale[1]
            plot_data = (plot_data - g_avg) / (e_avg - g_avg)
        plt.plot(xdata, plot_data, ".-")
        if fit:
            p = data["fit_avgq"]
            pCov = data["fit_err_avgq"]
            captionStr = f"$\epsilon$ fit [deg]: {p[1]:.3} $\pm$ {np.sqrt(pCov[1][1]):.3}"
            plt.plot(xdata, fit_func(xdata, *p), label=captionStr)
            plt.legend()
            if self.cfg.expt.test_pi_half:
                amp_ratio = (90 + p[1]) / 90
            else:
                amp_ratio = (180 - p[1]) / 180
            print(f"From avgq: adjust amplitude to {current_gain} / {amp_ratio} = {current_gain/amp_ratio}")
        plt.ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname


# ====================================================== #


class PiMinusPiExperiment(Experiment):
    """
    Play Nx(pi, minus pi) sweeping different drive frequencies to calibrate the stark shifted frequency

    Experimental Config
    expt = dict(
        start_N: start N sequences of pi, minus pi
        step_N
        expts_N
        start_f: start f
        step_f
        expts_f
        reps: number of reps,
        gain: gain to use for the calibration pulse (uses config value by default, calculated based on flags)
        pulse_type: 'gauss' or 'const' (uses config value by default)
        checkZZ: True/False for putting another qubit in e (specify as qZZ)
        checkEF: does ramsey on the EF transition instead of ge
        qubits: if not checkZZ, just specify [1 qubit]. if checkZZ: [qZZ in e , qB sweeps length rabi]
        readout_ge: whether to readout at the g/e set point or e/f set point
    )

    See https://arxiv.org/pdf/2406.08295 Appendix E
    """

    def __init__(
        self,
        soccfg=None,
        path="",
        prefix="PiMinusPiExpt",
        config_file=None,
        progress=None,
    ):
        super().__init__(
            path=path,
            soccfg=soccfg,
            prefix=prefix,
            config_file=config_file,
            progress=progress,
        )

    def acquire(self, progress=False, debug=True):
        # expand entries in config that are length 1 to fill all qubits
        self.num_qubits_sample = len(self.cfg.device.readout.frequency)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not (isinstance(value3, list)):
                                value2.update({key3: [value3] * self.num_qubits_sample})
                elif not (isinstance(value, list)):
                    subcfg.update({key: [value] * self.num_qubits_sample})

        self.checkEF = self.cfg.expt.checkEF
        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        if qZZ is None:
            qZZ = qTest

        self.cfg.expt.pi_minuspi = True

        data = dict()

        if "readout_ge" not in self.cfg.expt:
            self.cfg.expt.readout_ge = True
        if not self.cfg.expt.readout_ge:
            self.cfg.device.readout.frequency = self.cfg.device.readout.frequency_ef
            self.cfg.device.readout.readout_length = self.cfg.device.readout.readout_length_ef

        # ================= #
        # Get single shot calibration for 1 qubit
        # ================= #
        data["thresholds"] = None
        data["angles"] = None
        data["ge_avgs"] = None
        data["counts_calib"] = []

        thresholds_q = ge_avgs_q = angles_q = fids_q = None
        if "post_process" not in self.cfg.expt.keys():  # threshold or scale
            self.cfg.expt.post_process = None
            assert False, "you probably want to be doing this experiment with post processing or the fit will be weird"

        if self.cfg.expt.post_process is not None:
            if (
                "angles" in self.cfg.expt
                and "thresholds" in self.cfg.expt
                and "ge_avgs" in self.cfg.expt
                and "counts_calib" in self.cfg.expt
                and self.cfg.expt.angles is not None
                and self.cfg.expt.thresholds is not None
                and self.cfg.expt.ge_avgs is not None
                and self.cfg.expt.counts_calib is not None
            ):
                angles_q = self.cfg.expt.angles
                thresholds_q = self.cfg.expt.thresholds
                ge_avgs_q = np.asarray(self.cfg.expt.ge_avgs)
                data["counts_calib"] = self.cfg.expt.counts_calib
                print("Re-using provided angles, thresholds, ge_avgs")
            else:
                thresholds_q = [0] * 4
                ge_avgs_q = [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]
                angles_q = [0] * 4
                fids_q = [0] * 4

                # We really just need the single shot plots here, but convenient to use the ErrorMitigation tomo to do it
                sscfg = AttrDict(deepcopy(self.cfg))
                sscfg.expt.reps = sscfg.expt.singleshot_reps
                sscfg.expt.qubit = qTest

                calib_prog_dict = dict()
                calib_order = ["g", "e" if self.cfg.expt.readout_ge else "f"]
                for prep_state in tqdm(calib_order):
                    # print(prep_state)
                    sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state, apply_q1_pi2=False)
                    err_tomo = ErrorMitigationStateTomo1QProgram(soccfg=self.soccfg, cfg=sscfg)
                    err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=False)
                    calib_prog_dict.update({prep_state: err_tomo})

                g_prog = calib_prog_dict["g"]
                Ig, Qg = g_prog.get_shots(verbose=False)

                # Get readout angle + threshold for qubits
                e_prog = calib_prog_dict["e" if self.cfg.expt.readout_ge else "f"]
                Ie, Qe = e_prog.get_shots(verbose=False)
                shot_data = dict(Ig=Ig[qTest], Qg=Qg[qTest], Ie=Ie[qTest], Qe=Qe[qTest])
                print(f"Qubit  ({qTest})")
                fid, threshold, angle = hist(data=shot_data, plot=debug, verbose=False)
                thresholds_q[qTest] = threshold[0]
                ge_avgs_q[qTest] = [
                    np.average(Ig[qTest]),
                    np.average(Qg[qTest]),
                    np.average(Ie[qTest]),
                    np.average(Qe[qTest]),
                ]
                angles_q[qTest] = angle
                fids_q[qTest] = fid[0]
                print(
                    f"ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[qTest]} \t threshold ge: {thresholds_q[qTest]}"
                )

                # Process the shots taken for the confusion matrix with the calibration angles
                for prep_state in calib_order:
                    counts = calib_prog_dict[prep_state].collect_counts(angle=angles_q, threshold=thresholds_q)
                    data["counts_calib"].append(counts)

                if debug:
                    print(f"thresholds={thresholds_q},")
                    print(f"angles={angles_q},")
                    print(f"ge_avgs={ge_avgs_q},")
                    print(f"counts_calib={np.array(data['counts_calib']).tolist()}")

        data["thresholds"] = thresholds_q
        data["angles"] = angles_q
        data["ge_avgs"] = ge_avgs_q
        data["counts_calib"] = np.array(data["counts_calib"])

        # ================= #
        # Begin actual experiment
        # ================= #

        cycle_sweep = self.cfg.expt["start_N"] + self.cfg.expt["step_N"] * np.arange(self.cfg.expt["expts_N"])
        freq_sweep = self.cfg.expt["start_f"] + self.cfg.expt["step_f"] * np.arange(self.cfg.expt["expts_f"])
        if "loops" not in self.cfg.expt:
            self.cfg.expt.loops = 1

        self.cfg.expt.error_amp = True

        data.update(
            {
                "cycle_sweep": cycle_sweep,
                "freq_sweep": freq_sweep,
                "avgi": np.zeros((self.cfg.expt.loops, len(cycle_sweep), len(freq_sweep))),
                "avgq": np.zeros((self.cfg.expt.loops, len(cycle_sweep), len(freq_sweep))),
                "amps": np.zeros((self.cfg.expt.loops, len(cycle_sweep), len(freq_sweep))),
                "phases": np.zeros((self.cfg.expt.loops, len(cycle_sweep), len(freq_sweep))),
            }
        )

        # define as the length for the pi pulse ** this is still the specification even when calibrating the pi/2 pulse
        # length of pulse whose error we are trying to find
        if self.checkEF:
            length = np.reshape(self.cfg.device.qubit.pulses.pi_ef.sigma, (4, 4))[qTest, qZZ]
        else:
            length = np.reshape(self.cfg.device.qubit.pulses.pi_ge.sigma, (4, 4))[qTest, qZZ]

        self.cfg.expt.sigma_test = float(length)

        cfg = deepcopy(self.cfg)

        for loop in tqdm(range(self.cfg.expt.loops), disable=not progress or self.cfg.expt.loops == 1):
            for icycle, n_cycle in enumerate(tqdm(cycle_sweep, disable=not progress or self.cfg.expt.loops > 1)):
                for ifreq, freq in enumerate(freq_sweep):
                    cfg.expt.n_pulses = n_cycle

                    if self.checkEF:
                        cfg.device.qubit.f_ef[qTest * self.num_qubits_sample + qZZ] = freq
                    else:
                        if self.cfg.expt.pulse_type != "robust":
                            cfg.device.qubit.f_ge[qTest * self.num_qubits_sample + qZZ] = freq
                        else:
                            cfg.device.qubit.f_ge_robust[qTest * self.num_qubits_sample + qZZ] = freq

                    # print('n cycle', n_cycle)
                    lengthrabi = LengthRabiProgram(soccfg=self.soccfg, cfg=cfg)
                    self.prog = lengthrabi
                    avgi, avgq = lengthrabi.acquire_rotated(
                        self.im[self.cfg.aliases.soc],
                        angle=angles_q,
                        threshold=thresholds_q,
                        ge_avgs=ge_avgs_q,
                        post_process=self.cfg.expt.post_process,
                        progress=False,
                        verbose=False,
                    )
                    self.cfg.expt.gain = self.prog.cfg.expt.gain
                    avgi = avgi[qTest]
                    avgq = avgq[qTest]
                    amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
                    phase = np.angle(avgi + 1j * avgq)  # Calculating the phase
                    data["avgi"][loop, icycle, ifreq] = avgi
                    data["avgq"][loop, icycle, ifreq] = avgq
                    data["amps"][loop, icycle, ifreq] = amp
                    data["phases"][loop, icycle, ifreq] = phase

        for k, a in data.items():
            data[k] = np.array(a)

        data["avgi"] = np.average(data["avgi"], axis=0)
        data["avgq"] = np.average(data["avgq"], axis=0)
        data["amps"] = np.average(data["amps"], axis=0)
        data["phases"] = np.average(data["phases"], axis=0)

        self.data = data

        return data

    def analyze(self, data=None, fit=True, scale=None):
        # scale should be [Ig, Qg, Ie, Qe] single shot experiment

        self.checkEF = self.cfg.expt.checkEF
        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        if qZZ is None:
            qZZ = qTest

        if data is None:
            data = self.data

        if self.cfg.expt.post_process == "threshold":
            tomo_analysis = TomoAnalysis(nb_qubits=1)
            old_shape = data["avgi"].shape
            n_raw = np.zeros((np.prod(old_shape), 2))
            n_raw[:, 1] = data["avgi"].flatten()
            n_raw[:, 0] = 1 - data["avgi"].flatten()
            data["popln"] = np.reshape(tomo_analysis.correct_readout_err(n_raw, data["counts_calib"])[:, 1], old_shape)
        else:
            data["popln"] = np.copy(data["avgi"])

        prods = []
        for col in range(len(data["freq_sweep"])):
            col_data = data["popln"][:, col]
            prod = np.round(np.prod(1 - col_data), decimals=5)
            prods.append(np.sqrt(prod))

        plt.figure(figsize=(8, 4))
        plt.plot(data["freq_sweep"], prods, ".-")
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("$\sqrt{\Pi_n (1-P(e))}$")

        if not fit:
            plt.show()
            return data

        popt, pcov = fitter.fit_gauss(data["freq_sweep"], np.array(prods))
        fit_freq = popt[1]
        data["best_freq"] = fit_freq
        if self.checkEF:
            old_freq = self.cfg.device.qubit.f_ef[qTest * self.num_qubits_sample + qZZ]
        else:
            if self.cfg.expt.pulse_type != "robust":
                old_freq = self.cfg.device.qubit.f_ge[qTest * self.num_qubits_sample + qZZ]
            else:
                old_freq = self.cfg.device.qubit.f_ge_robust[qTest * self.num_qubits_sample + qZZ]
        print("Fit best freq", fit_freq, "which is", fit_freq - old_freq, "away from old freq", old_freq)

        plt.plot(data["freq_sweep"], fitter.gaussian(data["freq_sweep"], *popt))
        plt.show()

        return data

    def display(self, data=None, fit=True, scale=None):
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

        title = (
            f"Frequency Error Q{qTest}"
            + (f" ZZ Q{qZZ}" if (self.checkZZ and qZZ != qTest) else "")
            + (" EF" if self.checkEF else "")
            + (" $\pi/2$" if self.cfg.expt.test_pi_half else " $\pi$")
        )

        data = deepcopy(data)
        inner_sweep = data["freq_sweep"]
        outer_sweep = data["cycle_sweep"]

        y_sweep = outer_sweep
        x_sweep = inner_sweep

        label = "($X_{\pi}, X_{-\pi})^N$"
        if self.cfg.expt.test_pi_half:
            label = "($X_{\pi/2}, X_{-\pi/2})^N$"

        rows = 1
        cols = 1
        index = rows * 100 + cols * 10
        plt.figure(figsize=(7 * cols, 6))
        plt.suptitle(title)

        data_name = "popln"
        if self.checkEF:
            old_freq = self.cfg.device.qubit.f_ef[qTest * self.num_qubits_sample + qZZ]
        else:
            if self.cfg.expt.pulse_type != "robust":
                old_freq = self.cfg.device.qubit.f_ge[qTest * self.num_qubits_sample + qZZ]
            else:
                old_freq = self.cfg.device.qubit.f_ge_robust[qTest * self.num_qubits_sample + qZZ]

        ax = plt.gca()
        ax.set_ylabel(f"N {label}", fontsize=18)
        ax.set_xlabel("$f-f_0$ [MHz]", fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=16)
        plt.pcolormesh(x_sweep - old_freq, y_sweep, data[data_name], cmap="viridis", shading="auto")
        if fit:
            plt.axvline(data["best_freq"] - old_freq, color="r", linestyle="--")

        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname


# ====================================================== #


class PreSelectionPiMinusPiExperiment(Experiment):
    """
    Varying a readout cooling pulse wait time, play Nx(pi, minus pi) sweeping different drive frequencies to calibrate the stark shifted frequency. Frequency is fixed at the pi/-pi calibrated frequency from a standard experiment.
    Goal is we are looking qualitatively to make sure the resonator is empty before starting a standard experiment.

    Experimental Config
    expt = dict(
        start_N: start N sequences of pi, minus pi
        step_N
        expts_N
        start_t: start wait time sweep
        step_t
        expts_t
        reps: number of reps,
        gain: gain to use for the calibration pulse (uses config value by default, calculated based on flags)
        pulse_type: 'gauss' or 'const' (uses config value by default)
        checkZZ: True/False for putting another qubit in e (specify as qZZ)
        checkEF: does ramsey on the EF transition instead of ge
        qubits: if not checkZZ, just specify [1 qubit]. if checkZZ: [qZZ in e , qB sweeps length rabi]
        readout_ge: whether to readout at the g/e set point or e/f set point

        These params will be set automatically if they are not specified:
        readout_cool=True
        n_init_readout=1
        n_trig=1
        avg_trigs=True
        use_gf_readout=False
    )

    """

    def __init__(
        self,
        soccfg=None,
        path="",
        prefix="PreSelectionPiMinusPiExpt",
        config_file=None,
        progress=None,
    ):
        super().__init__(
            path=path,
            soccfg=soccfg,
            prefix=prefix,
            config_file=config_file,
            progress=progress,
        )

    def acquire(self, progress=False, debug=True):
        # expand entries in config that are length 1 to fill all qubits
        self.num_qubits_sample = len(self.cfg.device.readout.frequency)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not (isinstance(value3, list)):
                                value2.update({key3: [value3] * self.num_qubits_sample})
                elif not (isinstance(value, list)):
                    subcfg.update({key: [value] * self.num_qubits_sample})

        self.checkEF = self.cfg.expt.checkEF
        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        if qZZ is None:
            qZZ = qTest

        self.cfg.expt.pi_minuspi = True

        data = dict()

        if "readout_ge" not in self.cfg.expt:
            self.cfg.expt.readout_ge = True
        if not self.cfg.expt.readout_ge:
            self.cfg.device.readout.frequency = self.cfg.device.readout.frequency_ef
            self.cfg.device.readout.readout_length = self.cfg.device.readout.readout_length_ef

        if "readout_cool" not in self.cfg.expt:
            self.cfg.expt.readout_cool = True
        if "n_init_readout" not in self.cfg.expt:
            self.cfg.expt.n_init_readout = 1
        if "n_trig" not in self.cfg.expt:
            self.cfg.expt.n_trig = 1
        if "avg_trigs" not in self.cfg.expt:
            self.cfg.expt.avg_trigs = True
        if "use_gf_readout" not in self.cfg.expt:
            self.cfg.expt.use_gf_readout = False
        assert "init_read_wait_us" not in self.cfg.expt, "init_read_wait_us is a sweep variable in this experiment!"

        full_mux_expt = False
        if "full_mux_expt" in self.cfg.expt:
            full_mux_expt = self.cfg.expt.full_mux_expt

        # ================= #
        # Get single shot calibration for 1 qubit
        # ================= #
        data["thresholds"] = None
        data["angles"] = None
        data["ge_avgs"] = None
        data["counts_calib"] = []

        thresholds_q = ge_avgs_q = angles_q = fids_q = None
        if "post_process" not in self.cfg.expt.keys():  # threshold or scale
            self.cfg.expt.post_process = None
            assert False, "you probably want to be doing this experiment with post processing or the fit will be weird"

        if self.cfg.expt.post_process is not None:
            if (
                "angles" in self.cfg.expt
                and "thresholds" in self.cfg.expt
                and "ge_avgs" in self.cfg.expt
                and "counts_calib" in self.cfg.expt
                and self.cfg.expt.angles is not None
                and self.cfg.expt.thresholds is not None
                and self.cfg.expt.ge_avgs is not None
                and self.cfg.expt.counts_calib is not None
            ):
                angles_q = self.cfg.expt.angles
                thresholds_q = self.cfg.expt.thresholds
                ge_avgs_q = np.asarray(self.cfg.expt.ge_avgs)
                data["counts_calib"] = self.cfg.expt.counts_calib
                print("Re-using provided angles, thresholds, ge_avgs")
            else:
                thresholds_q = [0] * 4
                ge_avgs_q = [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]
                angles_q = [0] * 4
                fids_q = [0] * 4

                # We really just need the single shot plots here, but convenient to use the ErrorMitigation tomo to do it
                sscfg = AttrDict(deepcopy(self.cfg))
                sscfg.expt.reps = sscfg.expt.singleshot_reps
                sscfg.expt.qubit = qTest

                # OVERRIDING THE READOUT COOLING SINCE THAT'S THE POINT OF THE PRE SELECTION EXPERIMENT!
                sscfg.expt.readout_cool = False

                calib_prog_dict = dict()
                calib_order = ["g", "e" if self.cfg.expt.readout_ge else "f"]
                for prep_state in tqdm(calib_order):
                    # print(prep_state)
                    sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state, apply_q1_pi2=False)
                    err_tomo = ErrorMitigationStateTomo1QProgram(soccfg=self.soccfg, cfg=sscfg)
                    err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=False)
                    calib_prog_dict.update({prep_state: err_tomo})

                g_prog = calib_prog_dict["g"]
                Ig, Qg = g_prog.get_shots(verbose=False)

                # Get readout angle + threshold for qubits
                e_prog = calib_prog_dict["e" if self.cfg.expt.readout_ge else "f"]
                Ie, Qe = e_prog.get_shots(verbose=False)
                shot_data = dict(Ig=Ig[qTest], Qg=Qg[qTest], Ie=Ie[qTest], Qe=Qe[qTest])
                print(f"Qubit  ({qTest})")
                fid, threshold, angle = hist(data=shot_data, plot=debug, verbose=False, amplitude_mode=full_mux_expt)
                thresholds_q[qTest] = threshold[0]
                ge_avgs_q[qTest] = [
                    np.average(Ig[qTest]),
                    np.average(Qg[qTest]),
                    np.average(Ie[qTest]),
                    np.average(Qe[qTest]),
                ]
                angles_q[qTest] = angle
                fids_q[qTest] = fid[0]
                print(
                    f"ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[qTest]} \t threshold ge: {thresholds_q[qTest]}"
                )

                # Process the shots taken for the confusion matrix with the calibration angles
                for prep_state in calib_order:
                    counts = calib_prog_dict[prep_state].collect_counts(
                        angle=angles_q, threshold=thresholds_q, amplitude_mode=full_mux_expt
                    )
                    data["counts_calib"].append(counts)

                if debug:
                    print(f"thresholds={thresholds_q},")
                    print(f"angles={angles_q},")
                    print(f"ge_avgs={ge_avgs_q},")
                    print(f"counts_calib={np.array(data['counts_calib']).tolist()}")

        data["thresholds"] = thresholds_q
        data["angles"] = angles_q
        data["ge_avgs"] = ge_avgs_q
        data["counts_calib"] = np.array(data["counts_calib"])

        # ================= #
        # Begin actual experiment
        # ================= #

        cycle_sweep = self.cfg.expt["start_N"] + self.cfg.expt["step_N"] * np.arange(self.cfg.expt["expts_N"])
        time_sweep = self.cfg.expt["start_t"] + self.cfg.expt["step_t"] * np.arange(self.cfg.expt["expts_t"])
        if "loops" not in self.cfg.expt:
            self.cfg.expt.loops = 1

        self.cfg.expt.error_amp = True

        data.update(
            {
                "cycle_sweep": cycle_sweep,
                "time_sweep": time_sweep,
                "avgi": np.zeros((self.cfg.expt.loops, len(cycle_sweep), len(time_sweep))),
                "avgq": np.zeros((self.cfg.expt.loops, len(cycle_sweep), len(time_sweep))),
                "amps": np.zeros((self.cfg.expt.loops, len(cycle_sweep), len(time_sweep))),
                "phases": np.zeros((self.cfg.expt.loops, len(cycle_sweep), len(time_sweep))),
                "ishots_raw": np.zeros(
                    (
                        self.cfg.expt.loops,
                        len(cycle_sweep),
                        len(time_sweep),
                        self.num_qubits_sample,
                        self.cfg.expt.n_init_readout + 1,
                        self.cfg.expt.reps,
                    )
                ),
                "qshots_raw": np.zeros(
                    (
                        self.cfg.expt.loops,
                        len(cycle_sweep),
                        len(time_sweep),
                        self.num_qubits_sample,
                        self.cfg.expt.n_init_readout + 1,
                        self.cfg.expt.reps,
                    )
                ),
            }
        )

        # define as the length for the pi pulse ** this is still the specification even when calibrating the pi/2 pulse
        # length of pulse whose error we are trying to find
        if self.checkEF:
            length = np.reshape(self.cfg.device.qubit.pulses.pi_ef.sigma, (4, 4))[qTest, qZZ]
        else:
            length = np.reshape(self.cfg.device.qubit.pulses.pi_ge.sigma, (4, 4))[qTest, qZZ]

        self.cfg.expt.sigma_test = float(length)

        cfg = deepcopy(self.cfg)

        for loop in tqdm(range(self.cfg.expt.loops), disable=not progress or self.cfg.expt.loops == 1):
            for icycle, n_cycle in enumerate(tqdm(cycle_sweep, disable=not progress or self.cfg.expt.loops > 1)):
                for itau, tau in enumerate(time_sweep):
                    cfg.expt.n_pulses = n_cycle
                    cfg.expt.init_read_wait_us = tau

                    # print('n cycle', n_cycle)
                    lengthrabi = LengthRabiProgram(soccfg=self.soccfg, cfg=cfg)
                    self.prog = lengthrabi
                    avgi, avgq = lengthrabi.acquire_rotated(
                        self.im[self.cfg.aliases.soc],
                        angle=angles_q,
                        threshold=thresholds_q,
                        ge_avgs=ge_avgs_q,
                        post_process=self.cfg.expt.post_process,
                        amplitude_mode=full_mux_expt,
                        progress=False,
                        verbose=False,
                    )
                    ishots, qshots = lengthrabi.get_multireadout_shots()
                    data["ishots_raw"][loop, icycle, itau, :, :, :] = ishots
                    data["qshots_raw"][loop, icycle, itau, :, :, :] = qshots

                    self.cfg.expt.gain = self.prog.cfg.expt.gain
                    avgi = avgi[qTest]
                    avgq = avgq[qTest]
                    amp = np.average(np.abs(ishots + 1j * qshots))
                    phase = np.average(np.angle(ishots + 1j * qshots))
                    data["avgi"][loop, icycle, itau] = avgi
                    data["avgq"][loop, icycle, itau] = avgq
                    data["amps"][loop, icycle, itau] = amp
                    data["phases"][loop, icycle, itau] = phase

        for k, a in data.items():
            data[k] = np.array(a)

        data["avgi"] = np.average(data["avgi"], axis=0)
        data["avgq"] = np.average(data["avgq"], axis=0)
        data["amps"] = np.average(data["amps"], axis=0)
        data["phases"] = np.average(data["phases"], axis=0)

        self.data = data

        return data

    def analyze(
        self,
        data=None,
        fit=True,
        amplitude_mode=True,
        preselect=True,
        post_process="threshold",
        ps_qubits=None,
        ps_adjust=None,
        verbose=True,
    ):

        if not fit:
            return
        if data is None:
            data = self.data

        thresholds = data["thresholds"]
        angles = data["angles"]
        ge_avgs = data["ge_avgs"]
        counts_calib = data["counts_calib"]

        if not preselect:
            ps_qubits = []
        data["popln"] = np.zeros((self.cfg.expt.loops, len(data["cycle_sweep"]), len(data["time_sweep"])))
        tomo_analysis = TomoAnalysis(nb_qubits=1)
        for loop in range(self.cfg.expt.loops):
            for icycle, n_cycle in enumerate(data["cycle_sweep"]):
                for itau, tau in enumerate(data["time_sweep"]):
                    if ps_adjust is None:
                        ps_thresholds = thresholds
                    else:
                        ps_thresholds = ps_threshold_adjust(
                            ps_thresholds_init=thresholds,
                            adjust=ps_adjust,
                            ge_avgs=ge_avgs,
                            angles=angles,
                            amplitude_mode=amplitude_mode,
                        )
                    # print(ps_thresholds)
                    shots_ps = post_select_shots(
                        final_qubit=self.cfg.expt.qTest,
                        all_ishots_raw_q=data[f"ishots_raw"][loop, icycle, itau, :, :, :],
                        all_qshots_raw_q=data[f"qshots_raw"][loop, icycle, itau, :, :, :],
                        angles=angles,
                        thresholds=thresholds,
                        ps_thresholds=ps_thresholds,
                        ps_qubits=ps_qubits,
                        n_init_readout=self.cfg.expt.n_init_readout,
                        post_process=post_process,
                        verbose=False if not preselect else verbose,
                        amplitude_mode=amplitude_mode,
                    )

                    counts_raw = tomo_analysis.sort_counts([shots_ps])
                    counts_ge_corrected = tomo_analysis.correct_readout_err([counts_raw], counts_calib)
                    print(counts_calib, counts_raw, counts_ge_corrected)
                    data["popln"][loop, icycle, itau] = counts_ge_corrected[0][1]
        data["popln"] = np.average(data["popln"], axis=0)

    def display(self, data=None, preselect=True, fit=True, scale=None, post_process=None):
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

        title = (
            f"Pre Selection on Q{qTest}"
            + (f" ZZ Q{qZZ}" if (self.checkZZ and qZZ != qTest) else "")
            + (" EF" if self.checkEF else "")
            + (" $\pi/2$" if self.cfg.expt.test_pi_half else " $\pi$")
        )

        data = deepcopy(data)
        inner_sweep = data["time_sweep"]
        outer_sweep = data["cycle_sweep"]

        y_sweep = outer_sweep
        x_sweep = inner_sweep

        label = "($X_{\pi}, X_{-\pi})^N$"
        if self.cfg.expt.test_pi_half:
            label = "($X_{\pi/2}, X_{-\pi/2})^N$"

        rows = 1
        cols = 1
        index = rows * 100 + cols * 10
        plt.figure(figsize=(7 * cols, 6))
        plt.suptitle(title)

        if post_process is None: post_process = self.cfg.expt.post_process
        if post_process == "threshold":
            data_name = "popln"
        else:
            data_name = "amps"

        ax = plt.gca()
        ax.set_ylabel(f"N {label}", fontsize=18)
        ax.set_xlabel("Delay Between Readouts [us]", fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=16)
        plt.pcolormesh(x_sweep, y_sweep, data[data_name], cmap="viridis", shading="auto")

        plt.clim(0, 1.0)
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname


# ====================================================== #


class CDistortPiMinusPiExperiment(Experiment):
    """
    Play (pi, tau, minus pi)^N+Y/2 sweeping different times tau to check for C-distortion

    Experimental Config
    expt = dict(
        start_N: start N sequences of pi, minus pi
        step_N
        expts_N
        start_t: start delay time in clock cycles
        step_t
        expts_t
        reps: number of reps,
        gain: gain to use for the calibration pulse (uses config value by default, calculated based on flags)
        pulse_type: 'gauss' or 'const' (uses config value by default)
        checkZZ: True/False for putting another qubit in e (specify as qZZ)
        checkEF: does ramsey on the EF transition instead of ge
        qubits: if not checkZZ, just specify [1 qubit]. if checkZZ: [qZZ in e , qB sweeps length rabi]
        use_pi2_for_pi: plays 2x pi/2 instead of the pi pulse
    )

    See https://arxiv.org/pdf/2402.17757
    """

    def __init__(
        self,
        soccfg=None,
        path="",
        prefix="CDistortPiMinusPiExpt",
        config_file=None,
        progress=None,
    ):
        super().__init__(
            path=path,
            soccfg=soccfg,
            prefix=prefix,
            config_file=config_file,
            progress=progress,
        )

    def acquire(self, progress=False, debug=True):
        # expand entries in config that are length 1 to fill all qubits
        self.num_qubits_sample = len(self.cfg.device.readout.frequency)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not (isinstance(value3, list)):
                                value2.update({key3: [value3] * self.num_qubits_sample})
                elif not (isinstance(value, list)):
                    subcfg.update({key: [value] * self.num_qubits_sample})

        self.checkEF = self.cfg.expt.checkEF
        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        if qZZ is None:
            qZZ = qTest

        data = dict()

        # ================= #
        # Get single shot calibration for 1 qubit
        # ================= #
        data["thresholds"] = None
        data["angles"] = None
        data["ge_avgs"] = None
        data["counts_calib"] = []

        thresholds_q = ge_avgs_q = angles_q = fids_q = None
        if "post_process" not in self.cfg.expt.keys():  # threshold or scale
            self.cfg.expt.post_process = None

        if self.cfg.expt.post_process is not None:
            if (
                "angles" in self.cfg.expt
                and "thresholds" in self.cfg.expt
                and "ge_avgs" in self.cfg.expt
                and "counts_calib" in self.cfg.expt
                and self.cfg.expt.angles is not None
                and self.cfg.expt.thresholds is not None
                and self.cfg.expt.ge_avgs is not None
                and self.cfg.expt.counts_calib is not None
            ):
                angles_q = self.cfg.expt.angles
                thresholds_q = self.cfg.expt.thresholds
                ge_avgs_q = np.asarray(self.cfg.expt.ge_avgs)
                data["counts_calib"] = self.cfg.expt.counts_calib
                print("Re-using provided angles, thresholds, ge_avgs")
            else:
                thresholds_q = [0] * 4
                ge_avgs_q = [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]
                angles_q = [0] * 4
                fids_q = [0] * 4

                # We really just need the single shot plots here, but convenient to use the ErrorMitigation tomo to do it
                sscfg = AttrDict(deepcopy(self.cfg))
                sscfg.expt.reps = sscfg.expt.singleshot_reps
                sscfg.expt.qubit = qTest

                calib_prog_dict = dict()
                calib_order = ["g", "e"]
                for prep_state in tqdm(calib_order):
                    # print(prep_state)
                    sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state, apply_q1_pi2=False)
                    err_tomo = ErrorMitigationStateTomo1QProgram(soccfg=self.soccfg, cfg=sscfg)
                    err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=False)
                    calib_prog_dict.update({prep_state: err_tomo})

                g_prog = calib_prog_dict["g"]
                Ig, Qg = g_prog.get_shots(verbose=False)

                # Get readout angle + threshold for qubits
                e_prog = calib_prog_dict["e"]
                Ie, Qe = e_prog.get_shots(verbose=False)
                shot_data = dict(Ig=Ig[qTest], Qg=Qg[qTest], Ie=Ie[qTest], Qe=Qe[qTest])
                print(f"Qubit  ({qTest})")
                fid, threshold, angle = hist(data=shot_data, plot=debug, verbose=False)
                thresholds_q[qTest] = threshold[0]
                ge_avgs_q[qTest] = [
                    np.average(Ig[qTest]),
                    np.average(Qg[qTest]),
                    np.average(Ie[qTest]),
                    np.average(Qe[qTest]),
                ]
                angles_q[qTest] = angle
                fids_q[qTest] = fid[0]
                print(
                    f"ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[qTest]} \t threshold ge: {thresholds_q[qTest]}"
                )

                # Process the shots taken for the confusion matrix with the calibration angles
                for prep_state in calib_order:
                    counts = calib_prog_dict[prep_state].collect_counts(angle=angles_q, threshold=thresholds_q)
                    data["counts_calib"].append(counts)

                if debug:
                    print(f"thresholds={thresholds_q},")
                    print(f"angles={angles_q},")
                    print(f"ge_avgs={ge_avgs_q},")
                    print(f"counts_calib={np.array(data['counts_calib']).tolist()}")

        data["thresholds"] = thresholds_q
        data["angles"] = angles_q
        data["ge_avgs"] = ge_avgs_q
        data["counts_calib"] = np.array(data["counts_calib"])

        # ================= #
        # Begin actual experiment
        # ================= #

        cycle_sweep = self.cfg.expt["start_N"] + self.cfg.expt["step_N"] * np.arange(self.cfg.expt["expts_N"])
        time_sweep = self.cfg.expt["start_t"] + self.cfg.expt["step_t"] * np.arange(self.cfg.expt["expts_t"])
        if "loops" not in self.cfg.expt:
            self.cfg.expt.loops = 1

        self.cfg.expt.error_amp = True

        data.update(
            {
                "cycle_sweep": cycle_sweep,
                "time_sweep": time_sweep,
                "avgi": np.zeros((self.cfg.expt.loops, len(cycle_sweep), len(time_sweep))),
                "avgq": np.zeros((self.cfg.expt.loops, len(cycle_sweep), len(time_sweep))),
                "amps": np.zeros((self.cfg.expt.loops, len(cycle_sweep), len(time_sweep))),
                "phases": np.zeros((self.cfg.expt.loops, len(cycle_sweep), len(time_sweep))),
            }
        )

        # define as the length for the pi pulse ** this is still the specification even when calibrating the pi/2 pulse
        # length of pulse whose error we are trying to find
        if self.checkEF:
            length = np.reshape(self.cfg.device.qubit.pulses.pi_ef.sigma, (4, 4))[qTest, qZZ]
        else:
            length = np.reshape(self.cfg.device.qubit.pulses.pi_ge.sigma, (4, 4))[qTest, qZZ]

        self.cfg.expt.sigma_test = float(length)

        self.cfg.expt.pi_minuspi = True
        self.cfg.expt.check_C_distort = True

        if self.cfg.expt.use_pi2_for_pi:
            # takes the pi/2 pulse params for the pi_test and plays it twice for every "pi" pulse
            self.cfg.expt.test_pi_half = True
        else:
            # takes the pi pulse params for the pi_test
            self.cfg.expt.test_pi_half = False

        cfg = deepcopy(self.cfg)

        for loop in tqdm(range(self.cfg.expt.loops), disable=not progress or self.cfg.expt.loops == 1):
            for icycle, n_cycle in enumerate(tqdm(cycle_sweep, disable=not progress or self.cfg.expt.loops > 1)):
                for itau, tau in enumerate(time_sweep):
                    cfg.expt.n_pulses = n_cycle
                    cfg.expt.delay_error_amp = tau

                    # print('n cycle', n_cycle)
                    lengthrabi = LengthRabiProgram(soccfg=self.soccfg, cfg=cfg)
                    self.prog = lengthrabi
                    avgi, avgq = lengthrabi.acquire_rotated(
                        self.im[self.cfg.aliases.soc],
                        angle=angles_q,
                        threshold=thresholds_q,
                        ge_avgs=ge_avgs_q,
                        post_process=self.cfg.expt.post_process,
                        progress=False,
                        verbose=False,
                    )
                    self.cfg.expt.gain = self.prog.cfg.expt.gain
                    avgi = avgi[qTest]
                    avgq = avgq[qTest]
                    amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
                    phase = np.angle(avgi + 1j * avgq)  # Calculating the phase
                    data["avgi"][loop, icycle, itau] = avgi
                    data["avgq"][loop, icycle, itau] = avgq
                    data["amps"][loop, icycle, itau] = amp
                    data["phases"][loop, icycle, itau] = phase

        for k, a in data.items():
            data[k] = np.array(a)

        data["avgi"] = np.average(data["avgi"], axis=0)
        data["avgq"] = np.average(data["avgq"], axis=0)
        data["amps"] = np.average(data["amps"], axis=0)
        data["phases"] = np.average(data["phases"], axis=0)

        self.data = data

        return data

    def analyze(self, data=None, fit=True, scale=None):
        # scale should be [Ig, Qg, Ie, Qe] single shot experiment

        self.checkEF = self.cfg.expt.checkEF
        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        if qZZ is None:
            qZZ = qTest

        if data is None:
            data = self.data

        if self.cfg.expt.post_process == "threshold":
            tomo_analysis = TomoAnalysis(nb_qubits=1)
            old_shape = data["avgi"].shape
            n_raw = np.zeros((np.prod(old_shape), 2))
            n_raw[:, 1] = data["avgi"].flatten()
            n_raw[:, 0] = 1 - data["avgi"].flatten()
            data["avgi"] = np.reshape(tomo_analysis.correct_readout_err(n_raw, data["counts_calib"])[:, 1], old_shape)

        prods = []
        for col in range(len(data["time_sweep"])):
            col_data = data["amps"][:, col]
            prod = np.prod(1 - col_data)
            prods.append(np.sqrt(prod))

        plt.figure(figsize=(8, 4))
        time_sweep_us = np.zeros_like(data["time_sweep"], dtype=np.float64)
        for it, t_cycles in enumerate(data["time_sweep"]):
            time_sweep_us[it] = self.soccfg.cycles2us(t_cycles)
        plt.plot(1e3 * time_sweep_us, prods, ".-")
        plt.xlabel("Delay Time [ns]")
        plt.ylabel("$\sqrt{\Pi_n (1-P(e))}$")

        # if not fit:
        #     plt.show()
        #     return data

        # popt, pcov = fitter.fit_gauss(data["time_sweep"], np.array(prods))
        # fit_time = popt[1]
        # data["best_time"] = fit_freq
        # if self.checkEF:
        #     old_freq = self.cfg.device.qubit.f_ef[qTest * self.num_qubits_sample + qZZ]
        # else:
        #     old_freq = self.cfg.device.qubit.f_ge[qTest * self.num_qubits_sample + qZZ]
        # print("Fit best freq", fit_freq, "which is", fit_freq - old_freq, "away from old freq", old_freq)

        # plt.plot(data["time_sweep"], fitter.gaussian(data["time_sweep"], *popt))
        # plt.show()

        return data

    def display(self, data=None, fit=True, scale=None):
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

        title = (
            f"C-Distortion Q{qTest}"
            + (f" ZZ Q{qZZ}" if (self.checkZZ and qZZ != qTest) else "")
            + (" EF" if self.checkEF else "")
        )

        data = deepcopy(data)
        inner_sweep = np.zeros_like(data["time_sweep"], dtype=np.float64)
        for it, t_cycles in enumerate(data["time_sweep"]):
            inner_sweep[it] = self.soccfg.cycles2us(t_cycles)
        outer_sweep = data["cycle_sweep"]

        y_sweep = outer_sweep
        x_sweep = inner_sweep

        label = "($X, t, -X)^N + Y/2$"

        rows = 1
        cols = 1
        index = rows * 100 + cols * 10
        plt.figure(figsize=(7 * cols, 6))
        plt.suptitle(title)

        data_name = "avgi"

        ax = plt.gca()
        ax.set_ylabel(f"N {label}", fontsize=18)
        ax.set_xlabel("Delay [ns]", fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=16)
        # print(data[data_name])
        plt.pcolormesh(x_sweep * 1e3, y_sweep, data[data_name], cmap="viridis", shading="auto")

        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname


# ====================================================== #


class IDistortDelayExperiment(Experiment):
    """
    Play X/2 + (X + tau + Y(pi+phi))^N + Y/2 sweeping different times tau to check for I-distortion

    Experimental Config
    expt = dict(
        start_N: start N sequences of pi, minus pi
        step_N
        expts_N
        start_t: start delay time in clock cycles
        step_t
        expts_t
        check_I_phase: phase to set for the Y pulse in the error amplification
        reps: number of reps,
        gain: gain to use for the calibration pulse (uses config value by default, calculated based on flags)
        pulse_type: 'gauss' or 'const' (uses config value by default)
        checkZZ: True/False for putting another qubit in e (specify as qZZ)
        checkEF: does ramsey on the EF transition instead of ge
        qubits: if not checkZZ, just specify [1 qubit]. if checkZZ: [qZZ in e , qB sweeps length rabi]
        use_pi2_for_pi: plays 2x pi/2 instead of the pi pulse
    )

    See https://arxiv.org/pdf/2402.17757
    """

    def __init__(
        self,
        soccfg=None,
        path="",
        prefix="IDistortDelayExpt",
        config_file=None,
        progress=None,
    ):
        super().__init__(
            path=path,
            soccfg=soccfg,
            prefix=prefix,
            config_file=config_file,
            progress=progress,
        )

    def acquire(self, progress=False, debug=True):
        # expand entries in config that are length 1 to fill all qubits
        self.num_qubits_sample = len(self.cfg.device.readout.frequency)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not (isinstance(value3, list)):
                                value2.update({key3: [value3] * self.num_qubits_sample})
                elif not (isinstance(value, list)):
                    subcfg.update({key: [value] * self.num_qubits_sample})

        self.checkEF = self.cfg.expt.checkEF
        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        if qZZ is None:
            qZZ = qTest

        data = dict()

        # ================= #
        # Get single shot calibration for 1 qubit
        # ================= #
        data["thresholds"] = None
        data["angles"] = None
        data["ge_avgs"] = None
        data["counts_calib"] = []

        thresholds_q = ge_avgs_q = angles_q = fids_q = None
        if "post_process" not in self.cfg.expt.keys():  # threshold or scale
            self.cfg.expt.post_process = None

        if self.cfg.expt.post_process is not None:
            if (
                "angles" in self.cfg.expt
                and "thresholds" in self.cfg.expt
                and "ge_avgs" in self.cfg.expt
                and "counts_calib" in self.cfg.expt
                and self.cfg.expt.angles is not None
                and self.cfg.expt.thresholds is not None
                and self.cfg.expt.ge_avgs is not None
                and self.cfg.expt.counts_calib is not None
            ):
                angles_q = self.cfg.expt.angles
                thresholds_q = self.cfg.expt.thresholds
                ge_avgs_q = np.asarray(self.cfg.expt.ge_avgs)
                data["counts_calib"] = self.cfg.expt.counts_calib
                print("Re-using provided angles, thresholds, ge_avgs")
            else:
                thresholds_q = [0] * 4
                ge_avgs_q = [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]
                angles_q = [0] * 4
                fids_q = [0] * 4

                # We really just need the single shot plots here, but convenient to use the ErrorMitigation tomo to do it
                sscfg = AttrDict(deepcopy(self.cfg))
                sscfg.expt.reps = sscfg.expt.singleshot_reps
                sscfg.expt.qubit = qTest

                calib_prog_dict = dict()
                calib_order = ["g", "e"]
                for prep_state in tqdm(calib_order):
                    # print(prep_state)
                    sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state, apply_q1_pi2=False)
                    err_tomo = ErrorMitigationStateTomo1QProgram(soccfg=self.soccfg, cfg=sscfg)
                    err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=False)
                    calib_prog_dict.update({prep_state: err_tomo})

                g_prog = calib_prog_dict["g"]
                Ig, Qg = g_prog.get_shots(verbose=False)

                # Get readout angle + threshold for qubits
                e_prog = calib_prog_dict["e"]
                Ie, Qe = e_prog.get_shots(verbose=False)
                shot_data = dict(Ig=Ig[qTest], Qg=Qg[qTest], Ie=Ie[qTest], Qe=Qe[qTest])
                print(f"Qubit  ({qTest})")
                fid, threshold, angle = hist(data=shot_data, plot=debug, verbose=False)
                thresholds_q[qTest] = threshold[0]
                ge_avgs_q[qTest] = [
                    np.average(Ig[qTest]),
                    np.average(Qg[qTest]),
                    np.average(Ie[qTest]),
                    np.average(Qe[qTest]),
                ]
                angles_q[qTest] = angle
                fids_q[qTest] = fid[0]
                print(
                    f"ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[qTest]} \t threshold ge: {thresholds_q[qTest]}"
                )

                # Process the shots taken for the confusion matrix with the calibration angles
                for prep_state in calib_order:
                    counts = calib_prog_dict[prep_state].collect_counts(angle=angles_q, threshold=thresholds_q)
                    data["counts_calib"].append(counts)

                if debug:
                    print(f"thresholds={thresholds_q},")
                    print(f"angles={angles_q},")
                    print(f"ge_avgs={ge_avgs_q},")
                    print(f"counts_calib={np.array(data['counts_calib']).tolist()}")

        data["thresholds"] = thresholds_q
        data["angles"] = angles_q
        data["ge_avgs"] = ge_avgs_q
        data["counts_calib"] = np.array(data["counts_calib"])

        # ================= #
        # Begin actual experiment
        # ================= #

        cycle_sweep = self.cfg.expt["start_N"] + self.cfg.expt["step_N"] * np.arange(self.cfg.expt["expts_N"])
        time_sweep = self.cfg.expt["start_t"] + self.cfg.expt["step_t"] * np.arange(self.cfg.expt["expts_t"])
        if "loops" not in self.cfg.expt:
            self.cfg.expt.loops = 1

        self.cfg.expt.error_amp = True

        data.update(
            {
                "cycle_sweep": cycle_sweep,
                "time_sweep": time_sweep,
                "avgi": np.zeros((self.cfg.expt.loops, len(cycle_sweep), len(time_sweep))),
                "avgq": np.zeros((self.cfg.expt.loops, len(cycle_sweep), len(time_sweep))),
                "amps": np.zeros((self.cfg.expt.loops, len(cycle_sweep), len(time_sweep))),
                "phases": np.zeros((self.cfg.expt.loops, len(cycle_sweep), len(time_sweep))),
            }
        )

        # define as the length for the pi pulse ** this is still the specification even when calibrating the pi/2 pulse
        # length of pulse whose error we are trying to find
        if self.checkEF:
            length = np.reshape(self.cfg.device.qubit.pulses.pi_ef.sigma, (4, 4))[qTest, qZZ]
        else:
            length = np.reshape(self.cfg.device.qubit.pulses.pi_ge.sigma, (4, 4))[qTest, qZZ]

        self.cfg.expt.sigma_test = float(length)

        self.cfg.expt.pi_minuspi = False
        self.cfg.expt.check_I_distort = True
        self.cfg.expt.check_C_distort = False

        if self.cfg.expt.use_pi2_for_pi:
            # takes the pi/2 pulse params for the pi_test and plays it twice for every "pi" pulse
            self.cfg.expt.test_pi_half = True
        else:
            # takes the pi pulse params for the pi_test
            self.cfg.expt.test_pi_half = False

        cfg = deepcopy(self.cfg)

        for loop in tqdm(range(self.cfg.expt.loops), disable=not progress or self.cfg.expt.loops == 1):
            for icycle, n_cycle in enumerate(tqdm(cycle_sweep, disable=not progress or self.cfg.expt.loops > 1)):
                for itau, tau in enumerate(time_sweep):
                    cfg.expt.n_pulses = n_cycle
                    cfg.expt.delay_error_amp = tau

                    # print('n cycle', n_cycle)
                    lengthrabi = LengthRabiProgram(soccfg=self.soccfg, cfg=cfg)
                    self.prog = lengthrabi
                    avgi, avgq = lengthrabi.acquire_rotated(
                        self.im[self.cfg.aliases.soc],
                        angle=angles_q,
                        threshold=thresholds_q,
                        ge_avgs=ge_avgs_q,
                        post_process=self.cfg.expt.post_process,
                        progress=False,
                        verbose=False,
                    )
                    self.cfg.expt.gain = self.prog.cfg.expt.gain
                    avgi = avgi[qTest]
                    avgq = avgq[qTest]
                    amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
                    phase = np.angle(avgi + 1j * avgq)  # Calculating the phase
                    data["avgi"][loop, icycle, itau] = avgi
                    data["avgq"][loop, icycle, itau] = avgq
                    data["amps"][loop, icycle, itau] = amp
                    data["phases"][loop, icycle, itau] = phase

        for k, a in data.items():
            data[k] = np.array(a)

        data["avgi"] = np.average(data["avgi"], axis=0)
        data["avgq"] = np.average(data["avgq"], axis=0)
        data["amps"] = np.average(data["amps"], axis=0)
        data["phases"] = np.average(data["phases"], axis=0)

        self.data = data

        return data

    def analyze(self, data=None, fit=True, scale=None):
        # scale should be [Ig, Qg, Ie, Qe] single shot experiment

        self.checkEF = self.cfg.expt.checkEF
        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        if qZZ is None:
            qZZ = qTest

        if data is None:
            data = self.data

        if self.cfg.expt.post_process == "threshold":
            tomo_analysis = TomoAnalysis(nb_qubits=1)
            old_shape = data["avgi"].shape
            n_raw = np.zeros((np.prod(old_shape), 2))
            n_raw[:, 1] = data["avgi"].flatten()
            n_raw[:, 0] = 1 - data["avgi"].flatten()
            data["avgi"] = np.reshape(tomo_analysis.correct_readout_err(n_raw, data["counts_calib"])[:, 1], old_shape)

        prods = []
        for col in range(len(data["time_sweep"])):
            col_data = data["avgi"][:, col]
            prod = np.prod(1 - col_data)
            prods.append(np.sqrt(prod))

        plt.figure(figsize=(8, 4))
        time_sweep_us = np.zeros_like(data["time_sweep"], dtype=np.float64)
        for it, t_cycles in enumerate(data["time_sweep"]):
            time_sweep_us[it] = self.soccfg.cycles2us(t_cycles)
        plt.plot(1e3 * time_sweep_us, prods, ".-")
        plt.xlabel("Delay Time [ns]")
        plt.ylabel("$\sqrt{\Pi_n (1-P(e))}$")

        # if not fit:
        #     plt.show()
        #     return data

        # popt, pcov = fitter.fit_gauss(data["time_sweep"], np.array(prods))
        # fit_time = popt[1]
        # data["best_time"] = fit_freq
        # if self.checkEF:
        #     old_freq = self.cfg.device.qubit.f_ef[qTest * self.num_qubits_sample + qZZ]
        # else:
        #     old_freq = self.cfg.device.qubit.f_ge[qTest * self.num_qubits_sample + qZZ]
        # print("Fit best freq", fit_freq, "which is", fit_freq - old_freq, "away from old freq", old_freq)

        # plt.plot(data["time_sweep"], fitter.gaussian(data["time_sweep"], *popt))
        # plt.show()

        return data

    def display(self, data=None, fit=True, scale=None):
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

        title = (
            f"I-Distortion Delay Sweep ($\phi={self.cfg.expt.check_I_phase:.1f}$) Q{qTest}"
            + (f" ZZ Q{qZZ}" if (self.checkZZ and qZZ != qTest) else "")
            + (" EF" if self.checkEF else "")
        )

        data = deepcopy(data)
        inner_sweep = np.zeros_like(data["time_sweep"], dtype=np.float64)
        for it, t_cycles in enumerate(data["time_sweep"]):
            inner_sweep[it] = self.soccfg.cycles2us(t_cycles)
        outer_sweep = data["cycle_sweep"]

        y_sweep = outer_sweep
        x_sweep = inner_sweep

        label = "X/2 + ($X + t + Y_{\pi + \phi})^N + Y/2$"

        rows = 1
        cols = 1
        index = rows * 100 + cols * 10
        plt.figure(figsize=(7 * cols, 6))
        plt.suptitle(title)

        data_name = "avgi"

        ax = plt.gca()
        ax.set_ylabel(f"N for {label}", fontsize=18)
        ax.set_xlabel("Delay [ns]", fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=16)
        # print(data[data_name])
        plt.pcolormesh(x_sweep * 1e3, y_sweep, data[data_name], cmap="viridis", shading="auto")

        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname
