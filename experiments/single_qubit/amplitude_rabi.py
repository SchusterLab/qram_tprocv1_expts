from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from qick import *
from qick.helpers import gauss
from slab import AttrDict, Experiment, dsfit
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter
from experiments.clifford_averager_program import QutritRAveragerProgram
from experiments.single_qubit.single_shot import hist
from experiments.two_qubit.twoQ_state_tomography import (
    AbstractStateTomo2QProgram,
    ErrorMitigationStateTomo1QProgram,
    ErrorMitigationStateTomo2QProgram,
    infer_gef_popln_2readout,
)

# ====================================================== #


class AmplitudeRabiProgram(QutritRAveragerProgram):
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
        if self.checkEF:
            if "pulse_ge" not in self.cfg.expt:
                self.pulse_ge = True
            else:
                self.pulse_ge = self.cfg.expt.pulse_ge

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

        # initialize registers
        if self.qubit_ch_types[qTest] == "int4":
            self.r_gain = self.sreg(self.qubit_chs[qTest], "addr")  # get gain register for qubit_ch
        else:
            self.r_gain = self.sreg(self.qubit_chs[qTest], "gain")  # get gain register for qubit_ch
        self.r_gain2 = 4
        self.safe_regwi(self.q_rps[qTest], self.r_gain2, self.cfg.expt.start)

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

        # deal with setting up a robust pulse either by flagging pulse_type or use_robust_pulses
        if "use_robust_pulses" not in self.cfg.expt:
            self.cfg.expt.use_robust_pulses = False
        if self.cfg.expt.pulse_type == "robust":
            assert self.cfg.expt.use_robust_pulses
        if self.cfg.expt.use_robust_pulses:
            if "pulse_type" in self.cfg.expt:
                assert self.cfg.expt.pulse_type == "robust"
            self.cfg.expt.pulse_type = "robust"
            special = "robust"
        if "pulse_type" in self.cfg.expt:
            if "pulse_type" != "gauss" and "pulse_type" != "const":
                special = self.cfg.expt.pulse_type

        # play the test pulse
        # setup pulse regs, gain set by update
        if self.checkEF:
            self.Xef_pulse(
                q=qTest,
                ZZ_qubit=qZZ,
                pihalf=self.test_pi_half or self.use_pi2_for_pi,
                divide_len=self.divide_len,
                play=False,
                set_reg=True,
                special=None,
            )
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
        self.mathi(self.q_rps[qTest], self.r_gain, self.r_gain2, "+", 0)

        n_pulses = 1
        if "n_pulses" in self.cfg.expt:
            n_pulses = self.cfg.expt.n_pulses
        if self.test_pi_half:
            n_pulses *= 2
        for i in range(n_pulses):
            self.pulse(ch=self.qubit_chs[qTest])
            self.sync_all()

        if self.checkEF:  # map excited back to qubit ground state for measurement
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

    def update(self):
        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        if qZZ is None:
            qZZ = qTest

        step = self.cfg.expt.step
        if self.qubit_ch_types[qTest] == "int4":
            step = step << 16
        self.mathi(self.q_rps[qTest], self.r_gain2, self.r_gain2, "+", step)  # update test gain


# ====================================================== #


class AmplitudeRabiExperiment(Experiment):
    """
    Amplitude Rabi Experiment
    Experimental Config:
    expt = dict(
        start: qubit gain [dac level]
        step: gain step [dac level]
        expts: number steps
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        sigma_test: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        checkEF
        test_pi_half
        divide_len (for test_pi_half only)
        qTest: qubit on which to do the test pulse
        qZZ: None if not checkZZ, else specify other qubit to pi pulse
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(self, soccfg=None, path="", prefix="AmplitudeRabi", config_file=None, progress=None):
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

        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        if qZZ is None:
            qZZ = qTest

        print(
            f'Running amp rabi {"EF " if self.cfg.expt.checkEF else ""}on Q{qTest} {"with ZZ Q" + str(qZZ) if qZZ != qTest else ""}'
        )

        readout_cool = False
        if "readout_cool" in self.cfg.expt:
            readout_cool = self.cfg.expt.readout_cool

            self.use_gf_readout = None
            if "use_gf_readout" in self.cfg.expt and self.cfg.expt.use_gf_readout is not None:
                self.use_gf_readout = self.cfg.expt.use_gf_readout
                self.cfg.expt.n_trig *= 2
        # if self.use_gf_readout:
        #     self.cfg.device.readout.readout_length = 2*np.array(self.cfg.device.readout.readout_length)
        #     print('readout params', self.cfg.device.readout)

        full_mux_expt = False
        if "full_mux_expt" in self.cfg.expt:
            full_mux_expt = self.cfg.expt.full_mux_expt

        # ================= #
        # Get single shot calibration for test qubit
        # ================= #
        data = dict()

        if readout_cool:
            self.meas_order = ["Z", "X", "Y"]
            self.calib_order = ["g", "e"]  # should match with order of counts for each tomography measurement
            data.update({"counts_calib": []})
            self.pulse_dict = dict()

            # Error mitigation measurements: prep in g, e to recalibrate measurement angle and measure confusion matrix
            calib_prog_dict = dict()
            for prep_state in tqdm(self.calib_order):
                # print(prep_state)
                cfg = AttrDict(deepcopy(self.cfg))
                cfg.expt.qubit = qTest
                cfg.expt.reps = self.cfg.expt.singleshot_reps
                cfg.expt.rounds = 1
                cfg.expt.state_prep_kwargs = dict(prep_state=prep_state)
                err_tomo = ErrorMitigationStateTomo1QProgram(soccfg=self.soccfg, cfg=cfg)
                err_tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                calib_prog_dict.update({prep_state: err_tomo})

            g_prog = calib_prog_dict["g"]
            Ig, Qg = g_prog.get_shots(verbose=False)
            threshold = [0] * num_qubits_sample
            angle = [0] * num_qubits_sample

            # Get readout angle + threshold for qubit
            e_prog = calib_prog_dict["e"]
            Ie, Qe = e_prog.get_shots(verbose=False)
            shot_data = dict(Ig=Ig[qTest], Qg=Qg[qTest], Ie=Ie[qTest], Qe=Qe[qTest])
            fid, thresholdq, angleq = hist(data=shot_data, plot=progress, verbose=False, amplitude_mode=full_mux_expt)
            threshold[qTest] = thresholdq[0]
            angle[qTest] = angleq

            if progress:
                print("thresholds", threshold)
                print("angles", angle)

            # Process the shots taken for the confusion matrix with the calibration angles
            for prep_state in self.calib_order:
                counts = calib_prog_dict[prep_state].collect_counts(
                    angle=angle, threshold=threshold, amplitude_mode=full_mux_expt
                )
                data["counts_calib"].append(counts)
            data["thresholds"] = threshold
            data["angles"] = angle

        amprabi = AmplitudeRabiProgram(soccfg=self.soccfg, cfg=self.cfg)
        # print(amprabi)
        # from qick.helpers import progs2json
        # print(progs2json([amprabi.dump_prog()]))

        xpts, idata, qdata = amprabi.acquire(
            self.im[self.cfg.aliases.soc], load_pulses=True, progress=progress
        )  # , get_shots=readout_cool
        # )
        # print(amprabi)
        _, ishots, qshots = amprabi.get_shots()
        print(ishots.shape)

        if not readout_cool:
            avgi = idata[qTest][0]
            avgq = qdata[qTest][0]
            amps = np.average(np.abs(ishots + 1j * qshots), axis=2)[qTest]  # Calculating the magnitude
            phases = np.average(np.angle(ishots + 1j * qshots), axis=2)[qTest]  # Calculating the phase
            data.update({"xpts": xpts, "avgi": avgi, "avgq": avgq, "amps": amps, "phases": phases})
        else:
            data.update({"xpts": xpts, "idata": ishots, "qdata": qshots})

        self.data = data
        return data

    def rot_iq_data(self, idata, qdata, angle):
        idata_rot = idata * np.cos(np.pi / 180 * angle) - qdata * np.sin(np.pi / 180 * angle)
        qdata_rot = idata * np.sin(np.pi / 180 * angle) + qdata * np.cos(np.pi / 180 * angle)
        return idata_rot, qdata_rot

    def analyze(self, data=None, fit=True, post_select=False, ps_threshold=None, fitparams=None):
        if data is None:
            data = self.data

        qTest = self.cfg.expt.qTest

        readout_cool = False
        if "readout_cool" in self.cfg.expt:
            readout_cool = self.cfg.expt.readout_cool

        if readout_cool:
            # shots data shape: (len(self.ro_chs), self.cfg.expt.expts, 1+n_init_readout, self.cfg.expt.reps)
            # final_iqdata: data from the normal readout post experiment, post selected if necessary
            final_idata = data["idata"][qTest, :, -1, :]
            final_qdata = data["qdata"][qTest, :, -1, :]
            data["avgi"] = np.average(final_idata, axis=1)
            data["avgq"] = np.average(final_qdata, axis=1)
            keep_prev = np.ones_like(final_idata, dtype="bool")
            if post_select:
                if ps_threshold is None:
                    ps_threshold = data["thresholds"][qTest][0]
                    data["ps_threshold"] = ps_threshold
                for i_readout in range(self.cfg.expt.n_init_readout):
                    # iqdata_readout: data from the i_readout-th readout
                    idata_readout = data["idata"][qTest, :, i_readout, :]
                    qdata_readout = data["qdata"][qTest, :, i_readout, :]
                    idata_readout_rot, qdata_readout_rot = self.rot_iq_data(
                        idata_readout, qdata_readout, data["angles"][qTest]
                    )
                    keep_prev = np.logical_and(keep_prev, idata_readout_rot < ps_threshold)
                    print(
                        "i_readout",
                        i_readout,
                        ": keep",
                        np.sum(keep_prev),
                        "of",
                        self.cfg.expt.expts * self.cfg.expt.reps,
                        f"shots ({np.sum(keep_prev)/(self.cfg.expt.expts * self.cfg.expt.reps)} %)",
                    )
                    print("threshold", ps_threshold)
                    # print(idata_readout_rot)
                    # print()
                    # print(keep_prev)

                    data[f"avgi_select{i_readout}"] = np.zeros(self.cfg.expt.expts)
                    data[f"avgq_select{i_readout}"] = np.zeros(self.cfg.expt.expts)
                    for i_expt in range(self.cfg.expt.expts):
                        data[f"avgi_select{i_readout}"][i_expt] = np.average(final_idata[i_expt][keep_prev[i_expt]])
                        data[f"avgq_select{i_readout}"][i_expt] = np.average(final_qdata[i_expt][keep_prev[i_expt]])

                    data["avgi"] = data[f"avgi_select{i_readout}"]
                    data["avgq"] = data[f"avgq_select{i_readout}"]
            data["amps"] = np.abs(data["avgi"] + 1j * data["avgq"])
            data["phases"] = np.angle(data["avgi"] + 1j * data["avgq"])

        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset]
            # fitparams=[yscale, freq, phase_deg, y0]
            # Remove the first and last point from fit in case weird edge measurements
            xdata = data["xpts"]
            if fitparams is None:
                fitparams = [None] * 4
                n_pulses = 1
                if "n_pulses" in self.cfg.expt:
                    n_pulses = self.cfg.expt.n_pulses
                fitparams[1] = n_pulses / xdata[-1]
                # print(fitparams[1])

            ydata = data["amps"]
            # print(abs(xdata[np.argwhere(ydata==max(ydata))[0,0]] - xdata[np.argwhere(ydata==min(ydata))[0,0]]))
            # fitparams=[max(ydata)-min(ydata), 1/2 / abs(xdata[np.argwhere(ydata==max(ydata))[0,0]] - xdata[np.argwhere(ydata==min(ydata))[0,0]]), None, None, None]
            # fitparams=[max(ydata)-min(ydata), 1/2 / (max(xdata) - min(xdata)), 0, None, None]

            p_avgi, pCov_avgi = fitter.fitsin(data["xpts"][:-1], data["avgi"][:-1], fitparams=fitparams)
            p_avgq, pCov_avgq = fitter.fitsin(data["xpts"][:-1], data["avgq"][:-1], fitparams=fitparams)
            p_amps, pCov_amps = fitter.fitsin(data["xpts"][:-1], data["amps"][:-1], fitparams=fitparams)
            data["fit_avgi"] = p_avgi
            data["fit_avgq"] = p_avgq
            data["fit_amps"] = p_amps
            data["fit_err_avgi"] = pCov_avgi
            data["fit_err_avgq"] = pCov_avgq
            data["fit_err_amps"] = pCov_amps

        return data

    def display(self, data=None, fit=False, **kwargs):
        if data is None:
            data = self.data

        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        self.checkZZ = False
        if qZZ is not None and qZZ != qTest:
            self.checkZZ = True
        else:
            qZZ = qTest

        plt.figure(figsize=(10, 6))
        n_pulses = 1
        if "n_pulses" in self.cfg.expt:
            n_pulses = self.cfg.expt.n_pulses
        title = f"Amplitude Rabi {'EF ' if self.cfg.expt.checkEF else ''}on Q{qTest} (Pulse Length {self.cfg.expt.sigma_test}{(', ZZ Q'+str(qZZ)) if self.checkZZ else ''}, {n_pulses} pulse)"
        plt.subplot(111, title=title, xlabel="Gain [DAC units]", ylabel="Amplitude [ADC units]")
        plt.plot(data["xpts"], data["amps"], ".-")
        if fit:
            p = data["fit_amps"]
            plt.plot(data["xpts"], fitter.sinfunc(data["xpts"], *p))
            if p[2] > 180:
                p[2] = p[2] - 360
            elif p[2] < -180:
                p[2] = p[2] + 360
            if p[2] < 0:
                pi_gain = (1 / 2 - p[2] / 180) / 2 / p[1]
            else:
                pi_gain = (3 / 2 - p[2] / 180) / 2 / p[1]
            pi_gain += (n_pulses - 1) * 1 / 2 / p[1]
            pi2_gain = pi_gain / 2
            print(f"Pi gain from amps data [dac units]: {int(pi_gain)}")
            print(f"\tPi/2 gain from amps data [dac units]: {int(pi2_gain)}")
            plt.axvline(pi_gain, color="0.2", linestyle="--")
            plt.axvline(pi2_gain, color="0.2", linestyle="--")

        plt.figure(figsize=(10, 10))
        plt.subplot(211, title=title, ylabel="I [ADC units]")
        plt.plot(data["xpts"], data["avgi"], ".-")
        # plt.axhline(390)
        # plt.axhline(473)
        # plt.axvline(2114)
        # plt.axvline(3150)
        if fit:
            p = data["fit_avgi"]
            plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
            if p[2] > 180:
                p[2] = p[2] - 360
            elif p[2] < -180:
                p[2] = p[2] + 360
            if p[2] < 0:
                pi_gain = (1 / 2 - p[2] / 180) / 2 / p[1]
            else:
                pi_gain = (3 / 2 - p[2] / 180) / 2 / p[1]
            pi_gain += (n_pulses - 1) * 1 / 2 / p[1]
            pi2_gain = pi_gain / 2
            print(f"Pi gain from avgi data [dac units]: {int(pi_gain)}")
            print(f"\tPi/2 gain from avgi data [dac units]: {int(pi2_gain)}")
            plt.axvline(pi_gain, color="0.2", linestyle="--")
            plt.axvline(pi2_gain, color="0.2", linestyle="--")
        plt.subplot(212, xlabel="Gain [DAC units]", ylabel="Q [ADC units]")
        plt.plot(data["xpts"], data["avgq"], ".-")
        if fit:
            p = data["fit_avgq"]
            plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
            if p[2] > 180:
                p[2] = p[2] - 360
            elif p[2] < -180:
                p[2] = p[2] + 360
            if p[2] < 0:
                pi_gain = (1 / 2 - p[2] / 180) / 2 / p[1]
            else:
                pi_gain = (3 / 2 - p[2] / 180) / 2 / p[1]
            pi_gain += (n_pulses - 1) * 1 / 2 / p[1]
            pi2_gain = pi_gain / 2
            print(f"Pi gain from avgq data [dac units]: {int(pi_gain)}")
            print(f"\tPi/2 gain from avgq data [dac units]: {int(pi2_gain)}")
            plt.axvline(pi_gain, color="0.2", linestyle="--")
            plt.axvline(pi2_gain, color="0.2", linestyle="--")

        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname


# ====================================================== #


class AmplitudeRabiChevronExperiment(Experiment):
    """
    Amplitude Rabi Experiment
    Experimental Config:
    expt = dict(
        start_f: start qubit frequency (MHz),
        step_f: frequency step (MHz),
        expts_f: number of experiments in frequency,
        start_gain: qubit gain [dac level]
        step_gain: gain step [dac level]
        expts_gain: number steps
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        sigma_test: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(self, soccfg=None, path="", prefix="AmplitudeRabiChevron", config_file=None, progress=None):
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

        if self.cfg.expt.checkZZ:
            assert len(self.cfg.expt.qubits) == 2
            qZZ, qTest = self.cfg.expt.qubits
            assert qZZ != 1
            assert qTest == 1
        else:
            qTest = self.cfg.expt.qubits[0]

        freqpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"] * np.arange(self.cfg.expt["expts_f"])
        data = {"xpts": [], "freqpts": [], "avgi": [], "avgq": [], "amps": [], "phases": []}
        adc_ch = self.cfg.hw.soc.adcs.readout.ch

        self.cfg.expt.start = self.cfg.expt.start_gain
        self.cfg.expt.step = self.cfg.expt.step_gain
        self.cfg.expt.expts = self.cfg.expt.expts_gain
        for freq in tqdm(freqpts):
            self.cfg.expt.f_pi_test = freq
            amprabi = AmplitudeRabiProgram(soccfg=self.soccfg, cfg=self.cfg)

            xpts, avgi, avgq = amprabi.acquire(
                self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False
            )
            _, ishots, qshots = amprabi.get_shots(verbose=False)

            avgi = avgi[qTest][0]
            avgq = avgq[qTest][0]
            amps = np.average(np.abs(ishots + 1j * qshots), axis=2)[qTest]  # Calculating the magnitude
            phases = np.average(np.angle(ishots + 1j * qshots), axis=2)[qTest]  # Calculating the phase

            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amps)
            data["phases"].append(phases)

        data["xpts"] = xpts
        data["freqpts"] = freqpts
        for k, a in data.items():
            data[k] = np.array(a)
        self.data = data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data
        pass

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        x_sweep = data["xpts"]
        y_sweep = data["freqpts"]
        avgi = data["avgi"]
        avgq = data["avgq"]

        plt.figure(figsize=(10, 8))
        plt.subplot(211, title="Amplitude Rabi", ylabel="Frequency [MHz]")
        plt.imshow(
            np.flip(avgi, 0), cmap="viridis", extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]], aspect="auto"
        )
        plt.colorbar(label="I [ADC level]")
        plt.clim(vmin=None, vmax=None)
        # plt.axvline(1684.92, color='k')
        # plt.axvline(1684.85, color='r')

        plt.subplot(212, xlabel="Gain [dac units]", ylabel="Frequency [MHz]")
        plt.imshow(
            np.flip(avgq, 0), cmap="viridis", extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]], aspect="auto"
        )
        plt.colorbar(label="Q [ADC level]")
        plt.clim(vmin=None, vmax=None)

        if fit:
            pass

        plt.tight_layout()
        plt.show()

        # plt.plot(y_sweep, data['amps'][:,-1])
        # plt.title(f'Gain {x_sweep[-1]}')
        # plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname
