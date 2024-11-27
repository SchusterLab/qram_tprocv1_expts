import json
import math
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from slab import AttrDict, Experiment, NpEncoder
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter
from experiments.clifford_averager_program import (
    CliffordAveragerProgram,
    post_select_shots,
    ps_threshold_adjust,
)
from experiments.four_qubit.fourQ_state_tomo import (
    AbstractStateTomo4QProgram,
    ErrorMitigationStateTomo4QProgram,
    make_4q_calib_order,
    make_4q_meas_order,
)
from experiments.single_qubit.single_shot import get_ge_avgs, hist, multihist
from experiments.three_qubit.threeQ_state_tomo import (
    AbstractStateTomo3QProgram,
    ErrorMitigationStateTomo3QProgram,
    make_3q_calib_order,
    make_3q_meas_order,
)
from experiments.two_qubit.twoQ_state_tomography import (
    AbstractStateTomo1QProgram,
    AbstractStateTomo2QProgram,
    ErrorMitigationStateTomo1QProgram,
    ErrorMitigationStateTomo2QProgram,
    infer_gef_popln,
)
from TomoAnalysis import TomoAnalysis

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
linestyle_cycle = ["solid", "dashed", "dotted", "dashdot"]
marker_cycle = ["o", "*", "s", "^"]


class QramProtocolProgram(AbstractStateTomo2QProgram):
    def initialize(self):
        super().initialize()
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        self.cfg.expt.state_prep_kwargs = None
        self.all_qubits = self.cfg.all_qubits

        self.swap_chs = self.cfg.hw.soc.dacs.swap.ch
        self.swap_ch_types = self.cfg.hw.soc.dacs.swap.type
        self.f_EgGf_regs = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(self.cfg.device.qubit.f_EgGf, self.swap_chs)]

        self.swap_Q_chs = self.cfg.hw.soc.dacs.swap_Q.ch
        self.swap_Q_ch_types = self.cfg.hw.soc.dacs.swap_Q.type
        self.f_EgGf_Q_regs = [
            self.freq2reg(f, gen_ch=ch) for f, ch in zip(self.cfg.device.qubit.f_EgGf_Q, self.swap_chs)
        ]

        # declare swap dac indexed by qSort
        for qSort in self.all_qubits:
            if qSort == 1:
                continue
            mixer_freq = None
            if self.swap_ch_types[qSort] == "int4":
                mixer_freq = self.cfg.hw.soc.dacs.swap.mixer_freq[qSort]
            if self.swap_chs[qSort] not in self.gen_chs:
                self.declare_gen(
                    ch=self.swap_chs[qSort], nqz=self.cfg.hw.soc.dacs.swap.nyquist[qSort], mixer_freq=mixer_freq
                )

            mixer_freq = None
            if self.swap_Q_ch_types[qSort] == "int4":
                mixer_freq = self.cfg.hw.soc.dacs.swap_Q.mixer_freq[qSort]
            if self.swap_Q_chs[qSort] not in self.gen_chs:
                self.declare_gen(
                    ch=self.swap_Q_chs[qSort], nqz=self.cfg.hw.soc.dacs.swap_Q.nyquist[qSort], mixer_freq=mixer_freq
                )

        # get aliases for the sigmas we need in clock cycles
        self.pi_EgGf_types = self.cfg.device.qubit.pulses.pi_EgGf.type
        assert all(type == "flat_top" for type in self.pi_EgGf_types)
        self.pi_EgGf_sigmas_us = self.cfg.device.qubit.pulses.pi_EgGf.sigma

        self.pi_EgGf_Q_types = self.cfg.device.qubit.pulses.pi_EgGf_Q.type
        assert all(type == "flat_top" for type in self.pi_EgGf_Q_types)
        self.pi_EgGf_Q_sigmas_us = self.cfg.device.qubit.pulses.pi_EgGf_Q.sigma

        # update timestep in outer loop over averager program
        self.timestep_us = np.inf
        if "timestep" in cfg.expt:
            self.timestep_us = cfg.expt.timestep

        # add 2Q pulses to respective channels
        for q in self.all_qubits:
            if q != 1:
                if self.pi_EgGf_types[q] == "gauss":
                    pi_EgGf_sigma_cycles = self.us2cycles(self.pi_EgGf_sigmas_us[q], gen_ch=self.swap_chs[1])
                    self.add_gauss(
                        ch=self.swap_chs[q],
                        name=f"pi_EgGf_swap{q}",
                        sigma=pi_EgGf_sigma_cycles,
                        length=pi_EgGf_sigma_cycles * 4,
                    )
                elif self.pi_EgGf_types[q] == "flat_top":
                    sigma_ramp_cycles = 3
                    self.add_gauss(
                        ch=self.swap_chs[q],
                        name=f"pi_EgGf_swap{q}_ramp",
                        sigma=sigma_ramp_cycles,
                        length=sigma_ramp_cycles * 4,
                    )

                if self.pi_EgGf_Q_types[q] == "flat_top":
                    sigma_ramp_cycles = 3
                    self.add_gauss(
                        ch=self.swap_Q_chs[q],
                        name=f"pi_EgGf_Q_swap{q}_ramp",
                        sigma=sigma_ramp_cycles,
                        length=sigma_ramp_cycles * 4,
                    )

        # self.X_pulse(q=1, adiabatic=True, reload=True, play=False) # initialize adiabatic pulse waveform

        init_state = self.cfg.expt.init_state
        if init_state in ["|0+1>|0+1>", "|0+1>|1>", "|1>|0+1>"] and self.cfg.expt.use_IQ_pulse:
            if "plot_IQ" not in self.cfg.expt or self.cfg.expt.plot_IQ == None:
                self.cfg.expt.plot_IQ = False

            if init_state == "|0+1>|0+1>":
                pulse_cfg = self.cfg.device.qubit.pulses.pulse_pp
                pulse_name = "pulse_pp"
            elif init_state == "|0+1>|1>":
                pulse_cfg = self.cfg.device.qubit.pulses.pulse_p1
                pulse_name = "pulse_p1"
            elif init_state == "|1>|0+1>":
                pulse_cfg = self.cfg.device.qubit.pulses.pulse_1p
                pulse_name = "pulse_1p"
            pulse_filename = pulse_cfg.filename[0]
            pulse_gains = pulse_cfg.gain  # one entry for each qubit for each pulse

            # pulse_filepath = os.path.join(os.getcwd(), pulse_filename + '.npz')
            pulse_filepath = os.path.join("S:\\QRAM\\qram_4QR2\\optctrl_pulses", pulse_filename + ".npz")

            pulse_params_dict = dict()  # open file
            with np.load(pulse_filepath) as npzfile:
                for key in npzfile.keys():
                    pulse_params_dict.update({key: npzfile[key]})
            times = pulse_params_dict["times"]
            I_0 = pulse_params_dict["I_0"]
            Q_0 = pulse_params_dict["Q_0"]
            I_1 = pulse_params_dict["I_1"]
            Q_1 = pulse_params_dict["Q_1"]
            IQ_qubits = [0, 1]
            I_values_MHz = np.array([I_0, I_1]) * 1e-6
            Q_values_MHz = np.array([Q_0, Q_1]) * 1e-6
            times_us = times * 1e6

            for iq, q in enumerate(IQ_qubits):
                self.handle_IQ_pulse(
                    name=f"{pulse_name}_Q{q}",
                    ch=self.qubit_chs[q],
                    I_mhz_vs_us=I_values_MHz[iq],
                    Q_mhz_vs_us=Q_values_MHz[iq],
                    times_us=times_us,
                    freq_MHz=self.f_ges[q, q],
                    phase_deg=0,
                    gain=pulse_gains[q],
                    reload=True,
                    play=False,
                    plot_IQ=self.cfg.expt.plot_IQ,
                )

        self.sync_all(200)

    def handle_next_pulse(self, count_us, ch, freq_reg, type, phase, gain, sigma_us, waveform):
        if type == "gauss":
            new_count_us = count_us + 4 * sigma_us
        else:  # const or flat_top
            new_count_us = count_us + sigma_us

        if new_count_us <= self.timestep_us:  # fit entire pulse
            # print('full pulse')
            if type == "gauss":
                self.setup_and_pulse(ch=ch, style="arb", freq=freq_reg, phase=phase, gain=gain, waveform=waveform)
            elif type == "flat_top":
                sigma_ramp_cycles = 3
                flat_length_cycles = self.us2cycles(sigma_us, gen_ch=ch) - sigma_ramp_cycles * 4
                self.setup_and_pulse(
                    ch=ch,
                    style="flat_top",
                    freq=freq_reg,
                    phase=phase,
                    gain=gain,
                    length=flat_length_cycles,
                    waveform=f"{waveform}_ramp",
                )
            elif type == "const":
                self.setup_and_pulse(
                    ch=ch,
                    style="const",
                    freq=freq_reg,
                    phase=phase,
                    gain=gain,
                    length=self.us2cycles(sigma_us, gen_ch=ch),
                )
            elif type == "adiabatic":
                assert False, "have not implemented time stepping for adiabatic pulses"

        elif count_us < self.timestep_us:  # fit part of pulse
            cut_length_us = self.timestep_us - count_us
            # print(waveform, 'cut length', cut_length_us)
            if type == "gauss":
                sigma_cycles = self.us2cycles(cut_length_us / 4, gen_ch=ch)
                if sigma_cycles > 0:
                    # print(1e3*self.timestep_us, 'sigma cycles', sigma_cycles)
                    self.add_gauss(ch=ch, name=f"{waveform}_cut", sigma=sigma_cycles, length=4 * sigma_cycles)
                    self.setup_and_pulse(
                        ch=ch, style="arb", freq=freq_reg, phase=phase, gain=gain, waveform=f"{waveform}_cut"
                    )
            elif type == "const":
                cut_length_cycles = self.us2cycles(cut_length_us, gen_ch=ch)
                if cut_length_cycles > 3:  # pulse length needs to be at least 2 cycles, add another cycle for buffer
                    self.setup_and_pulse(
                        ch=ch, style="const", freq=freq_reg, phase=phase, gain=gain, length=cut_length_cycles
                    )
            elif type == "flat_top":
                sigma_ramp_cycles = 3
                flat_length_cycles = self.us2cycles(cut_length_us, gen_ch=ch) - sigma_ramp_cycles * 4
                if flat_length_cycles > 0:
                    self.setup_and_pulse(
                        ch=ch,
                        style="flat_top",
                        freq=freq_reg,
                        phase=phase,
                        gain=gain,
                        length=flat_length_cycles,
                        waveform=f"{waveform}_ramp",
                    )
            elif type == "adiabatic":
                assert False, "have not implemented time stepping for adiabatic pulses"

        # else: already done with protocol for this timestep
        return new_count_us

    def collect_counts_post_select(
        self, angle=None, threshold=None, postselect=True, postselect_q=1, amplitude_mode=False
    ):
        if not postselect:
            return self.collect_counts(angle, threshold, amplitude_mode=amplitude_mode)

        avgi, avgq = self.get_shots(angle=angle)
        # collect shots for all adcs, then sorts into e, g based on >/< threshold and angle rotation
        shots = np.array([np.heaviside(avgi[i] - threshold[i], 0) for i in range(len(self.adc_chs))])

        assert self.cfg.expt.tomo_qubits is not None
        qA, qB = self.cfg.expt.tomo_qubits
        shots_psq0 = [[], []]  # postselect qubit measured as 0
        shots_psq1 = [[], []]  # postselect qubit measured as 1
        for i_shot, shot_psq in enumerate(shots[postselect_q]):
            if shot_psq:
                shots_psq1[0].append(shots[qA][i_shot])
                shots_psq1[1].append(shots[qB][i_shot])
            else:
                shots_psq0[0].append(shots[qA][i_shot])
                shots_psq0[1].append(shots[qB][i_shot])

        tomo_analysis = TomoAnalysis(nb_qubits=2)
        counts_psq0 = tomo_analysis.sort_counts([shots_psq0[0], shots_psq0[1]])
        counts_psq1 = tomo_analysis.sort_counts([shots_psq1[0], shots_psq1[1]])
        return (counts_psq0, counts_psq1)

    """
    Protocol pulses
    """

    def gegg_ggfg(self, count_us, add_phase=False, pihalf=False, sync_after=True):
        # 2Q on Q2
        phase_deg = self.overall_phase[2]
        phase = self.deg2reg(phase_deg, gen_ch=self.swap_Q_chs[2])
        sigma_us = self.pi_EgGf_Q_sigmas_us[2]
        waveform = "pi_EgGf_Q_swap2"
        if pihalf:
            sigma_us = self.cfg.device.qubit.pulses.pi_EgGf_Q.half_sigma[2]
        count_us = self.handle_next_pulse(
            count_us=count_us,
            ch=self.swap_Q_chs[2],
            freq_reg=self.f_EgGf_Q_regs[2],
            type=self.pi_EgGf_Q_types[2],
            phase=phase,
            gain=self.cfg.device.qubit.pulses.pi_EgGf_Q.gain[2],
            sigma_us=sigma_us,
            waveform=waveform,
        )
        if add_phase:  # virtual Z
            virtual_Z = self.cfg.device.qubit.pulses.pi_EgGf_Q.phase[2]
            if pihalf:
                virtual_Z = self.cfg.device.qubit.pulses.pi_EgGf_Q.half_phase[2]
            self.overall_phase[2] += virtual_Z
            # print('virtual Z', virtual_Z)
        if sync_after:
            self.sync_all()
        return count_us

    def eegg_eggf(self, count_us, add_phase=False, pihalf=False, sync_after=True):
        # 2Q on Q3
        phase_deg = self.overall_phase[3]
        phase = self.deg2reg(phase_deg, gen_ch=self.swap_Q_chs[3])
        sigma_us = self.pi_EgGf_Q_sigmas_us[3]
        waveform = "pi_EgGf_Q_swap3"
        if pihalf:
            sigma_us = self.cfg.device.qubit.pulses.pi_EgGf_Q.half_sigma[3]
        count_us = self.handle_next_pulse(
            count_us=count_us,
            ch=self.swap_Q_chs[3],
            freq_reg=self.f_EgGf_Q_regs[3],
            type=self.pi_EgGf_Q_types[3],
            phase=phase,
            gain=self.cfg.device.qubit.pulses.pi_EgGf_Q.gain[3],
            sigma_us=sigma_us,
            waveform=waveform,
        )
        if add_phase:  # virtual Z
            virtual_Z = self.cfg.device.qubit.pulses.pi_EgGf_Q.phase[3]
            if pihalf:
                virtual_Z = self.cfg.device.qubit.pulses.pi_EgGf_Q.half_phase[3]
            self.overall_phase[3] += virtual_Z
        if sync_after:
            self.sync_all()
        return count_us

    def q2_ef(self, count_us, pihalf=False, ZZ_qubit=None, sync_after=True):
        self.Xef_pulse(q=2, play=True, ZZ_qubit=ZZ_qubit, pihalf=pihalf)
        if self.timestep_us < np.inf:
            assert False, "no time stepping on the ef pulse right now"
        return 0

    def q3_ef(self, count_us, pihalf=False, ZZ_qubit=None, sync_after=True):
        self.Xef_pulse(q=3, play=True, ZZ_qubit=ZZ_qubit, pihalf=pihalf)
        if self.timestep_us < np.inf:
            assert False, "no time stepping on the ef pulse right now"
        return 0

    def state_prep_pulse(self, qubits=None, **kwargs):
        cfg = AttrDict(self.cfg)

        # ================= #
        # Initial states
        # ================= #

        init_state = self.cfg.expt.init_state

        if init_state == "|0>|0>":
            pass

        elif init_state == "|0>|1>":
            # self.X_pulse(q=1, play=True, special="gauss")
            self.X_pulse(q=1, play=True, special="gauss")

        elif init_state == "|0>|0+1>":
            # self.Y_pulse(q=1, play=True, pihalf=True, special="gauss")
            # print("WARNING, USING ROBUST PULSE ON Q1 PREP")
            self.Y_pulse(q=1, play=True, pihalf=True, special="gauss")

        elif init_state == "|0>|2>":
            self.X_pulse(q=1, play=True, special="gauss")
            self.Xef_pulse(q=1, play=True)

        elif init_state == "|1>|0>":
            self.X_pulse(q=0, play=True)

        elif init_state == "|1>|0+1>":
            if not self.cfg.expt.use_IQ_pulse:
                assert self.use_robust_pulses
                self.Y_pulse(q=1, pihalf=True, play=True, special="gauss")
                # print("WARNING, USING ROBUST PULSE ON Q1 PREP")
                # self.Y_pulse(q=1, play=True, pihalf=True)
                self.X_pulse(q=0, play=True)
            else:
                IQ_qubits = [0, 1]
                for q in IQ_qubits:
                    # play the I + Q component for each qubit in the IQ pulse
                    self.handle_IQ_pulse(name=f"pulse_1p_Q{q}", ch=self.qubit_chs[q], sync_after=False, play=True)
                self.sync_all()

        elif init_state == "|1>|1>":
            # print("WARNING, USING ROBUST PULSE ON Q1 PREP")
            self.X_pulse(q=1, play=True, special="gauss")
            self.X_pulse(q=0, play=True, ZZ_qubit=1)
            # self.X_pulse(q=1, play=True, special="gauss")
            # self.X_pulse(q=0, ZZ_qubit=1, play=True)

        elif init_state == "|0+1>|0+1>":
            if not self.cfg.expt.use_IQ_pulse:
                assert self.use_robust_pulses
                # self.Y_pulse(q=1, pihalf=True, play=True, special="gauss")
                # self.Y_pulse(q=1, pihalf=True, play=True)
                self.Y_pulse(q=1, pihalf=True, play=True, special="gauss")
                self.Y_pulse(q=0, pihalf=True, play=True)

            else:
                IQ_qubits = [0, 1]
                for q in IQ_qubits:
                    # play the I + Q component for each qubit in the IQ pulse
                    self.handle_IQ_pulse(name=f"pulse_pp_Q{q}", ch=self.qubit_chs[q], sync_after=False, play=True)
                self.sync_all()

        elif init_state == "|0+1>|0>":
            self.Y_pulse(q=0, play=True, pihalf=True)  # -> 0+1
            self.sync_all()

        elif init_state == "|0+1>|1>":
            if not self.cfg.expt.use_IQ_pulse:
                assert self.use_robust_pulses
                # self.Y_pulse(q=1, play=True, special="gauss")  # -> 1
                # self.Y_pulse(q=1, play=True)  # -> 1
                # print('Carefull: q0 is played first')
                self.Y_pulse(q=1, play=True, special="gauss")  # -> 1
                self.Y_pulse(q=0, pihalf=True, play=True, ZZ_qubit=1)

            else:
                IQ_qubits = [0, 1]
                for q in IQ_qubits:
                    # play the I + Q component for each qubit in the IQ pulse
                    self.handle_IQ_pulse(name=f"pulse_p1_Q{q}", ch=self.qubit_chs[q], sync_after=False, play=True)
                self.sync_all()

        elif init_state == "|0+1>|0+i>":
            self.X_pulse(q=1, pihalf=True, play=True, special="gauss", neg=True)
            self.Y_pulse(q=0, pihalf=True, play=True)

        elif init_state == "|0+i>|0>":
            self.X_pulse(q=0, play=True, pihalf=True, neg=True)

        elif init_state == "|0+i>|1>":
            self.Y_pulse(q=1, play=True, special="gauss")
            self.X_pulse(q=0, play=True, neg=True, ZZ_qubit=1, pihalf=True)

        elif init_state == "|0+i>|0+1>":
            self.Y_pulse(q=1, play=True, special="gauss")
            self.X_pulse(q=0, pihalf=True, play=True, neg=True)
        elif init_state == "|0+i>|0+i>":
            self.X_pulse(q=1, play=True, pihalf=True, neg=True, special="gauss")
            self.X_pulse(q=0, play=True, pihalf=True, neg=True)
        elif init_state == "|0>|0+i>":
            self.X_pulse(q=1, play=True, pihalf=True, neg=True)
        elif init_state == "|1>|0+i>":
            self.Y_pulse(q=0, play=True, special="gauss")
            self.X_pulse(q=1, play=True, pihalf=True, neg=True, ZZ_qubit=0)

        elif init_state == "|0+i1>|0+1>":
            if not self.cfg.expt.use_IQ_pulse:
                assert self.use_robust_pulses
                assert False, "not implemented!"
            else:
                assert False, "no i+ state prep implemented!"

            self.X_pulse(q=0, play=True, pihalf=True, neg=True)  # -> 0+i1
            self.sync_all()

            freq_reg = self.freq2reg(np.average([self.f_ges[1, 1], self.f_ges[1, 0]]))
            gain = int(np.average([self.pi_ge_gains[1, 1], self.pi_ge_gains[1, 0]]))
            sigma_us = np.average([self.pi_ge_sigmas[1, 1] / 2, self.pi_ge_sigmas[1, 0] / 2])
            pi2_sigma_cycles = self.us2cycles(sigma_us, gen_ch=self.qubit_chs[1])
            phase = self.deg2reg(-90, gen_ch=self.qubit_chs[1])  # +Y/2 -> 0+1
            self.add_gauss(
                ch=self.qubit_chs[1], name="qubit1_semiZZ0_half", sigma=pi2_sigma_cycles, length=4 * pi2_sigma_cycles
            )
            self.setup_and_pulse(
                ch=self.qubit_chs[1],
                style="arb",
                freq=freq_reg,
                phase=phase,
                gain=gain,
                waveform="qubit1_semiZZ0_half",
            )
            self.sync_all()

        elif init_state == "|1002>":
            self.X_pulse(q=0, play=True)
            self.X_pulse(q=3, ZZ_qubit=0, play=True)
            self.Xef_pulse(q=3, ZZ_qubit=0, play=True)

        elif init_state == "|test>":
            self.X_pulse(q=0, pihalf=True, play=True, sync_after=False)
            self.X_pulse(q=1, pihalf=True, play=True, sync_after=False)
            self.X_pulse(q=2, pihalf=True, play=True, sync_after=False)
            self.X_pulse(q=3, pihalf=True, play=True)

        elif (
            "Q" in init_state
        ):  # specify other qubits to prepare. it will always be specified as QaQb_regular_state_name, with regular state name specified as |Qa>|Qb>
            assert init_state[0] == "Q"
            assert init_state[2] == "Q"
            qa = int(init_state[1])
            qb = int(init_state[3])
            init_state_other = init_state[5:]

            if init_state_other == "|0>|0>":
                pass

            elif init_state_other == "|0>|1>":
                self.Y_pulse(q=qb, play=True)

            elif init_state_other == "|0>|2>":
                self.Y_pulse(q=qb, play=True)
                self.Yef_pulse(q=qb, play=True)

            elif init_state_other == "|0>|0+1>":
                self.Y_pulse(q=qb, play=True, pihalf=True)

            elif init_state_other == "|1>|0>":
                self.Y_pulse(q=qa, play=True, pihalf=False)

            elif init_state_other == "|2>|0>":
                self.Y_pulse(q=qa, play=True, pihalf=False)
                self.Yef_pulse(q=qa, play=True)

            elif init_state_other == "|2>|1>":
                self.Y_pulse(q=qb, play=True)
                self.Y_pulse(q=qa, ZZ_qubit=1, play=True)
                self.Yef_pulse(q=qa, ZZ_qubit=1, play=True)

            elif init_state_other == "|2T>|0>":  # no ge pulse (check thermal population)
                self.Yef_pulse(q=qa, play=True)

            elif init_state_other == "|1>|0+1>":
                self.Y_pulse(q=qa, play=True)
                self.Y_pulse(q=qb, ZZ_qubit=qa, pihalf=True, play=True)

            elif init_state_other == "|1>|1>":
                self.Y_pulse(q=qa, play=True)
                self.Y_pulse(q=qb, ZZ_qubit=qa, play=True)

            elif init_state_other == "|0+1>|0>":
                self.Y_pulse(q=qa, play=True, pihalf=True)  # -> 0+1

            elif init_state_other == "|0+1>|1>":
                self.Y_pulse(q=qb, play=True)  # -> 1
                self.Y_pulse(q=qa, ZZ_qubit=qb, pihalf=True, play=True)

            else:
                assert False, f"Init state {init_state} not valid"

        else:
            assert False, f"Init state {init_state} not valid"

        count_us = 0
        self.end_times_us = []

        # ================= #
        # BELL STATE TESTING!!
        # ================= #
        # # 1. apply Eg-Gf/2 with qDrive=2: gegg -> ggfg
        # count_us = self.handle_next_pulse(count_us=count_us, ch=self.swap_Q_chs[2], freq_reg=self.f_EgGf_Q_regs[2], type=self.pi_EgGf_Q_types[2], phase=0, gain=cfg.device.qubit.pulses.pi_EgGf_Q.gain[2], sigma_us=self.pi_EgGf_Q_sigmas_us[2]/2, waveform='pi_EgGf_Q_swap2') # waveform is name of ramp which is same regardless of length
        # if count_us < self.timestep_us: self.end_times_us.append(count_us)
        # self.sync_all() # do simultaneously?

        # # 2. apply Eg-Gf/2 with qDrive=3: eegg -> eggf
        # count_us = self.handle_next_pulse(count_us=count_us, ch=self.swap_Q_chs[3], freq_reg=self.f_EgGf_Q_regs[3], type=self.pi_EgGf_Q_types[3], phase=self.deg2reg(0, gen_ch=self.swap_Q_chs[3]), gain=cfg.device.qubit.pulses.pi_EgGf_Q.gain[3], sigma_us=self.pi_EgGf_Q_sigmas_us[3]/2, waveform='pi_EgGf_Q_swap3')
        # if count_us < self.timestep_us: self.end_times_us.append(count_us)
        # self.sync_all()

        # # 3. apply ef pulse on Q2 (at this point guaranteed no excitation in Q1) [path 1]
        # count_us = self.handle_next_pulse(count_us=count_us, ch=self.qubit_chs[2], freq_reg=self.f_ef_regs[2], type=self.pi_ef_types[2], phase=0, gain=self.cfg.device.qubit.pulses.pi_ef.gain[2], sigma_us=self.pi_ef_sigmas_us[2], waveform='pi_ef_q2')
        # if count_us < self.timestep_us: self.end_times_us.append(count_us)
        # self.sync_all()

        # # 4. apply ef pulse on Q3 (at this point guaranteed no excitation in Q1) [path 2]
        # count_us = self.handle_next_pulse(count_us=count_us, ch=self.qubit_chs[3], freq_reg=self.f_ef_regs[3], type=self.pi_ef_types[3], phase=0, gain=self.cfg.device.qubit.pulses.pi_ef.gain[3], sigma_us=self.pi_ef_sigmas_us[3], waveform='pi_ef_q3')
        # if count_us < self.timestep_us: self.end_times_us.append(count_us)
        # self.sync_all()

        # ================= #
        # Begin protocol v2
        # ================= #
        # print('WARNING, INITIALIZING IN SOMETHING RANDOM')
        # self.X_pulse(q=0, play=True, pihalf=True)
        # self.X_pulse(q=0, play=True, pihalf=True)

        add_phase = False
        if "add_phase" in self.cfg.expt:
            add_phase = self.cfg.expt.add_phase

        play_pulses = [0]  # play_pulses: only those specified pulses are played, in the specified order
        if "run_protocol" in self.cfg.expt and self.cfg.expt.run_protocol:
            play_pulses = [0, 1, 2, 3, 4]
        elif "play_pulses" in self.cfg.expt:
            play_pulses = self.cfg.expt.play_pulses

        pi_half_swaps = [False, False]  # Q2/Q1, Q3/Q1
        if "pi_half_swaps" in self.cfg.expt and self.cfg.expt.pi_half_swaps is not None:
            pi_half_swaps = self.cfg.expt.pi_half_swaps
            assert len(pi_half_swaps) == 2

        if "sync_between_swaps" not in self.cfg.expt:
            self.cfg.expt.sync_between_swaps = False
        if "sync_between_efs" not in self.cfg.expt:
            self.cfg.expt.sync_between_efs = True

        prev_pulse = 0
        for i_pulse, pulse_num in enumerate(play_pulses):
            # 1. apply Eg-Gf with qDrive=2: gegg -> ggfg [path 1]
            if pulse_num == 1:
                count_us = self.gegg_ggfg(count_us, add_phase=add_phase, pihalf=pi_half_swaps[0], sync_after=True)
                if count_us < self.timestep_us:
                    self.end_times_us.append(count_us)

            # 2. apply Eg-Gf with qDrive=3: eegg -> eggf [path 2]
            if pulse_num == 2:
                count_us = self.eegg_eggf(
                    count_us, add_phase=add_phase, pihalf=pi_half_swaps[1], sync_after=self.cfg.expt.sync_between_swaps
                )
                if count_us < self.timestep_us:
                    self.end_times_us.append(count_us)

            # 3. apply ef pulse on Q2 (at this point guaranteed no excitation in Q1) [path 1]
            if pulse_num == 3:
                if prev_pulse == 1 or prev_pulse == 2:
                    self.sync_all()
                count_us = self.q2_ef(count_us, pihalf=False, sync_after=True)
                if count_us < self.timestep_us:
                    self.end_times_us.append(count_us)
                # self.sync_all(5)

            # 4. apply ef pulse on Q3 (at this point guaranteed no excitation in Q1) [path 2]
            # This one should be run before 3 because the ZZ from q2 on q3 EF is very large, while the ZZ from q3 on q2 EF is quite small
            # But actually q2 and q3 should never simultaneously be populated! So technically you should be able to just run the 2 ef pulses simultaneously
            if pulse_num == 4:
                if prev_pulse == 1 or prev_pulse == 2:
                    self.sync_all()
                count_us = self.q3_ef(count_us, pihalf=False, ZZ_qubit=0, sync_after=self.cfg.expt.sync_between_efs)
                if count_us < self.timestep_us:
                    self.end_times_us.append(count_us)
                # self.sync_all(5)

            prev_pulse = pulse_num

        # if 'post_select' in self.cfg.expt and self.cfg.expt.post_select:
        #     self.setup_measure(qubit=1, basis='X', play=True, flag=None)

        # wait any remaining time
        # print('us left', self.timestep_us-count_us)
        if not math.isinf(self.timestep_us) and count_us < self.timestep_us:
            self.sync_all(self.us2cycles(self.timestep_us - count_us))
        self.sync_all()


class QramProtocolExperiment(Experiment):
    """
    Qram protocol over time sweep
    Experimental Config
    expt = dict(
        start: start protocol time [us],
        step: time step,
        expts: number of different time experiments,
        reps: number of reps per time step,
        tomo_2q: True/False whether to do 2q state tomography on state at last time step
        tomo_qubits: the qubits on which to do the 2q state tomo
        singleshot_reps: reps per state for singleshot calibration
        post_process: 'threshold' (uses single shot binning), 'scale' (scale by ge_avgs), or None
        calib_apply_q1_pi2: initialize Q1 in 0+1 for all calibrations
        thresholds: (optional) don't rerun singleshot and instead use this
        ge_avgs: (optional) don't rerun singleshot and instead use this
        angles: (optional) don't rerun singleshot and instead use this

        init_state
        use_IQ_pulse
        play_pulses
        plot_IQ
    )
    """

    def __init__(self, soccfg=None, path="", prefix="qram_protocol", config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=True):
        # expand entries in config that are length 1 to fill all qubits
        self.num_qubits_sample = len(self.cfg.device.readout.frequency)
        qubits = self.cfg.expt.tomo_qubits

        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not (isinstance(value3, list)):
                                value2.update({key3: [value3] * self.num_qubits_sample})
                elif not (isinstance(value, list)):
                    subcfg.update({key: [value] * self.num_qubits_sample})

        timesteps = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        print("timesteps", timesteps)

        full_mux_expt = False
        if "full_mux_expt" in self.cfg.expt:
            full_mux_expt = self.cfg.expt.full_mux_expt

        data = {
            "xpts": [],
            "avgi": [[], [], [], []],
            "avgq": [[], [], [], []],
            "avgi_err": [[], [], [], []],
            "avgq_err": [[], [], [], []],
            "amps": [[], [], [], []],
            "phases": [[], [], [], []],
        }

        self.meas_order = ["ZZ", "ZX", "ZY", "XZ", "XX", "XY", "YZ", "YX", "YY"]
        self.calib_order = [
            "gg",
            "ge",
            "eg",
            "ee",
        ]  # should match with order of counts for each tomography measurement
        self.tomo_qubits = self.cfg.expt.tomo_qubits
        if "post_select" in self.cfg.expt and self.cfg.expt.post_select:
            data.update({"counts_tomo_ps0": [], "counts_tomo_ps1": [], "counts_calib": []})
        else:
            data.update({"counts_tomo": [], "counts_calib": []})
        if "post_select" not in self.cfg.expt:
            self.cfg.expt.post_select = False
        if "calib_apply_q1_pi2" not in self.cfg.expt:
            self.cfg.expt.calib_apply_q1_pi2 = False

        self.readout_cool = False
        if "readout_cool" in self.cfg.expt and self.cfg.expt.readout_cool:
            self.readout_cool = self.cfg.expt.readout_cool
        if not self.readout_cool:
            self.cfg.expt.n_init_readout = 0

        if "n_init_readout" not in self.cfg.expt:
            self.cfg.expt.n_init_readout = 0
        data[f"ishots_raw"] = np.zeros(
            shape=(len(self.meas_order), self.num_qubits_sample, self.cfg.expt.n_init_readout + 1, self.cfg.expt.reps)
        )  # raw data tomo experiments
        data[f"qshots_raw"] = np.zeros(
            shape=(len(self.meas_order), self.num_qubits_sample, self.cfg.expt.n_init_readout + 1, self.cfg.expt.reps)
        )  # raw data tomo experiments

        self.ncalib = len(self.calib_order) + (
            2 * (self.num_qubits_sample - len(self.cfg.expt.tomo_qubits)) if self.readout_cool else 0
        )
        data["calib_ishots_raw"] = np.zeros(
            shape=(
                self.ncalib,
                self.num_qubits_sample,
                self.cfg.expt.n_init_readout + 1,
                self.cfg.expt.singleshot_reps,
            )
        )  # raw rotated g data for the calibration histograms for each of the measure experiments
        data["calib_qshots_raw"] = np.zeros(
            shape=(
                self.ncalib,
                self.num_qubits_sample,
                self.cfg.expt.n_init_readout + 1,
                self.cfg.expt.singleshot_reps,
            )
        )  # raw rotated g data for the calibration histograms for each of the measure experiments

        # ================= #
        # Get single shot calibration for qubits
        # ================= #

        post_process = self.cfg.expt.post_process
        thresholds_q = ge_avgs_q = angles_q = fids_q = None

        if (
            "angles" in self.cfg.expt
            and "thresholds" in self.cfg.expt
            and "ge_avgs" in self.cfg.expt
            and "counts_calib" in self.cfg.expt
            and None
            not in (self.cfg.expt.angles, self.cfg.expt.thresholds, self.cfg.expt.ge_avgs, self.cfg.expt.counts_calib)
        ):
            angles_q = self.cfg.expt.angles
            thresholds_q = self.cfg.expt.thresholds
            ge_avgs_q = self.cfg.expt.ge_avgs
            for q in range(self.num_qubits_sample):
                if ge_avgs_q[q] is None:
                    ge_avgs_q[q] = np.zeros_like(
                        ge_avgs_q[qubits[0]]
                    )  # just get the shape of the arrays correct by picking the old ge_avgs_q of a q that was definitely measured
            ge_avgs_q = np.array(ge_avgs_q)
            data["counts_calib"] = self.cfg.expt.counts_calib
            if progress:
                print("Re-using provided angles, thresholds, ge_avgs, counts_calib")

        else:
            thresholds_q = [0] * 4
            ge_avgs_q = [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]
            angles_q = [0] * 4
            fids_q = [0] * 4

            sscfg = AttrDict(deepcopy(self.cfg))
            sscfg.expt.reps = sscfg.expt.singleshot_reps

            # Error mitigation measurements: prep in gg, ge, eg, ee to recalibrate measurement angle and measure confusion matrix
            calib_prog_dict = dict()
            for prep_state in tqdm(self.calib_order, disable=not progress):
                # print(prep_state)
                sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state, apply_q1_pi2=sscfg.expt.calib_apply_q1_pi2)
                err_tomo = ErrorMitigationStateTomo2QProgram(soccfg=self.soccfg, cfg=sscfg)
                err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=False)
                calib_prog_dict.update({prep_state: err_tomo})

            g_prog = calib_prog_dict["gg"]
            Ig, Qg = g_prog.get_shots(verbose=False)
            threshold = [0] * self.num_qubits_sample
            angle = [0] * self.num_qubits_sample

            # Get readout angle + threshold for qubits
            for qi, q in enumerate(sscfg.expt.tomo_qubits):
                calib_e_state = "gg"
                calib_e_state = calib_e_state[:qi] + "e" + calib_e_state[qi + 1 :]
                e_prog = calib_prog_dict[calib_e_state]
                Ie, Qe = e_prog.get_shots(verbose=False)
                shot_data = dict(Ig=Ig[q], Qg=Qg[q], Ie=Ie[q], Qe=Qe[q])
                print(f"Qubit ({q})")
                fid, threshold, angle = hist(data=shot_data, plot=debug, verbose=False, amplitude_mode=full_mux_expt)
                thresholds_q[q] = threshold[0]
                ge_avgs_q[q] = [np.average(Ig[q]), np.average(Qg[q]), np.average(Ie[q]), np.average(Qe[q])]
                angles_q[q] = angle
                fids_q[q] = fid[0]
                print(
                    f"ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[q]} \t threshold ge: {thresholds_q[q]}"
                )

            # Process the shots taken for the confusion matrix with the calibration angles (for tomography)
            for iprep, prep_state in enumerate(self.calib_order):
                counts = calib_prog_dict[prep_state].collect_counts(
                    angle=angles_q, threshold=thresholds_q, amplitude_mode=full_mux_expt
                )
                data["counts_calib"].append(counts)
                data[f"calib_ishots_raw"][iprep, :, :, :], data[f"calib_qshots_raw"][iprep, :, :, :] = calib_prog_dict[
                    prep_state
                ].get_multireadout_shots()

            # Do the calibration for the remaining qubits in case you want to do post selection
            if (
                self.readout_cool
                or self.cfg.expt.expts > 1
                or (self.cfg.expt.post_select and 1 not in self.cfg.expt.tomo_qubits)
            ):
                sscfg = AttrDict(deepcopy(self.cfg))
                sscfg.expt.reps = self.cfg.expt.singleshot_reps
                ps_calib_prog_dict = dict()
                iprep_temp = len(self.calib_order)
                for q in range(self.num_qubits_sample):
                    if q in self.cfg.expt.tomo_qubits:
                        continue  # already did these
                    sscfg.expt.qubit = q
                    for prep_state in tqdm(["g", "e"], disable=not progress):
                        # print(prep_state)
                        sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state)
                        err_tomo = ErrorMitigationStateTomo1QProgram(soccfg=self.soccfg, cfg=sscfg)
                        err_tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                        ps_calib_prog_dict.update({prep_state: err_tomo})

                        # Save the full counts from the calibration experiments using the measured angles for all qubits in case you want to do post selection on the calibration
                        (
                            data[f"calib_ishots_raw"][iprep_temp, :, :, :],
                            data[f"calib_qshots_raw"][iprep_temp, :, :, :],
                        ) = err_tomo.get_multireadout_shots()
                        iprep_temp += 1

                    g_prog = ps_calib_prog_dict["g"]
                    Ig, Qg = g_prog.get_shots(verbose=False)

                    # Get readout angle + threshold for qubit
                    e_prog = ps_calib_prog_dict["e"]
                    Ie, Qe = e_prog.get_shots(verbose=False)
                    shot_data = dict(Ig=Ig[q], Qg=Qg[q], Ie=Ie[q], Qe=Qe[q])
                    print(f"Qubit ({q}) ge")
                    fid, threshold, angle = hist(
                        data=shot_data, plot=debug, verbose=False, amplitude_mode=full_mux_expt
                    )
                    thresholds_q[q] = threshold[0]
                    angles_q[q] = angle
                    ge_avgs_q[q] = [np.average(Ig[q]), np.average(Qg[q]), np.average(Ie[q]), np.average(Qe[q])]
                    print(
                        f"ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[q]} \t threshold ge: {thresholds_q[q]}"
                    )

        data.update(dict(thresholds=thresholds_q, angles=angles_q, ge_avgs=ge_avgs_q))

        print(f"thresholds={thresholds_q},")
        print(f"angles={angles_q},")
        print(f"ge_avgs={ge_avgs_q},")
        print(f"counts_calib={np.array(data['counts_calib']).tolist()}")

        # ================= #
        # Begin protocol stepping
        # ================= #

        adc_chs = self.cfg.hw.soc.adcs.readout.ch
        self.pulse_dict = dict()
        if "post_select" not in self.cfg.expt:
            self.cfg.expt.post_select = False

        for time_i, timestep in enumerate(tqdm(timesteps, disable=not progress)):
            self.cfg.expt.timestep = float(timestep)
            self.cfg.all_qubits = [0, 1, 2, 3]

            # Perform 2q state tomo only on last timestep
            if self.cfg.expt.tomo_2q and time_i == len(timesteps) - 1:
                for ibasis, basis in enumerate(tqdm(self.meas_order)):

                    # print('WARNING, ONLY MEASURING 1 BASIS')
                    # basis = 'ZX'
                    # print(basis)

                    cfg = AttrDict(deepcopy(self.cfg))
                    cfg.expt.basis = basis
                    tomo_prog = QramProtocolProgram(soccfg=self.soccfg, cfg=cfg)

                    # from qick.helpers import progs2json
                    # print('basis', basis)
                    # print(progs2json([tomo_prog.dump_prog()]))
                    # print()

                    avgi, avgq = tomo_prog.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                    counts = tomo_prog.collect_counts_post_select(
                        angle=angles_q,
                        threshold=thresholds_q,
                        postselect=self.cfg.expt.post_select,
                        postselect_q=1,
                        amplitude_mode=full_mux_expt,
                    )
                    if cfg.expt.post_select:
                        data["counts_tomo_ps0"].append(counts[0])
                        data["counts_tomo_ps1"].append(counts[1])
                    else:
                        data["counts_tomo"].append(counts)
                        print(basis, counts)
                    (
                        data[f"ishots_raw"][ibasis, :, :, :],
                        data[f"qshots_raw"][ibasis, :, :, :],
                    ) = tomo_prog.get_multireadout_shots()
                    self.pulse_dict.update({basis: tomo_prog.pulse_dict})

            else:
                self.cfg.expt.basis = "ZZ"
                protocol_prog = QramProtocolProgram(soccfg=self.soccfg, cfg=self.cfg)
                avgi, avgq = protocol_prog.acquire_rotated(
                    soc=self.im[self.cfg.aliases.soc],
                    progress=False,
                    angle=angles_q,
                    threshold=thresholds_q,
                    ge_avgs=ge_avgs_q,
                    post_process=post_process,
                    amplitude_mode=full_mux_expt,
                )

                for q in range(4):
                    data["avgi"][q].append(avgi[adc_chs[q]])
                    data["avgq"][q].append(avgq[adc_chs[q]])
                    # data['avgi_err'][q].append(avgi_err[adc_chs[q]])
                    # data['avgq_err'][q].append(avgq_err[adc_chs[q]])
                    data["amps"][q].append(np.abs(avgi[adc_chs[q]] + 1j * avgi[adc_chs[q]]))
                    data["phases"][q].append(np.angle(avgi[adc_chs[q]] + 1j * avgi[adc_chs[q]]))

                data["xpts"].append(float(timestep))

        if self.cfg.expt.expts > 1:
            data["end_times"] = protocol_prog.end_times_us
            print("end times", protocol_prog.end_times_us)

        for k, a in data.items():
            data[k] = np.array(a)

        self.data = data

        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data
        return data

    def display(self, data=None, err=True, saveplot=False, **kwargs):
        if data is None:
            data = self.data

        xpts_ns = data["xpts"] * 1e3

        if self.cfg.expt.post_process == "threshold" or self.cfg.expt.post_process == "scale":
            plt.figure(figsize=(14, 8))
            if saveplot:
                plt.style.use("dark_background")
            plt_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            # plt.title(f"Qram Protocol", fontsize=20)

            if err:
                plt.errorbar(xpts_ns, data["avgi"][0], fmt="o-", yerr=data["avgi_err"][0], label="Q0")
                plt.errorbar(xpts_ns, data["avgi"][1], fmt="o-", yerr=data["avgi_err"][1], label="Q1")
                plt.errorbar(xpts_ns, data["avgi"][2], fmt="o-", yerr=data["avgi_err"][2], label="Q2")
                plt.errorbar(xpts_ns, data["avgi"][3], fmt="o-", yerr=data["avgi_err"][3], label="Q3")

            else:
                plt.plot(xpts_ns, data["avgi"][0], "-", marker="o", markersize=8, label="Q0")
                plt.plot(xpts_ns, data["avgi"][1], "-", marker="*", markersize=12, label="Q1")
                plt.plot(xpts_ns, data["avgi"][2], "-", marker="s", markersize=7, label="Q2")
                plt.plot(xpts_ns, data["avgi"][3], "-", marker="^", markersize=8, label="Q3")

            # plt.fill_between(xpts_ns, data["avgi"][0] - data["avgi_err"][0], data["avgi"][0] + data["avgi_err"][0], color=plt_colors[0], alpha=0.4, linestyle='-', edgecolor=plt_colors[0])
            # plt.fill_between(xpts_ns, data["avgi"][1] - data["avgi_err"][1], data["avgi"][1] + data["avgi_err"][1], color=plt_colors[1], alpha=0.4, linestyle='-', edgecolor=plt_colors[1])
            # plt.fill_between(xpts_ns, data["avgi"][2] - data["avgi_err"][2], data["avgi"][2] + data["avgi_err"][2], color=plt_colors[2], alpha=0.4, linestyle='-', edgecolor=plt_colors[2])
            # plt.fill_between(xpts_ns, data["avgi"][3] - data["avgi_err"][3], data["avgi"][3] + data["avgi_err"][3], color=plt_colors[3], alpha=0.4, linestyle='-', edgecolor=plt_colors[3])

            end_times = data["end_times"]
            for end_time in end_times:
                plt.axvline(1e3 * end_time, color="0.4", linestyle="--")

            if self.cfg.expt.post_process == "threshold":
                plt.ylim(-0.02, 1.02)
            plt.legend(fontsize=26)
            plt.xlabel("Time [ns]", fontsize=26)
            plt.ylabel("Population", fontsize=26)
            plt.tick_params(labelsize=24)
            # plt.grid(linewidth=0.3)

        else:
            plt.figure(figsize=(14, 20))
            plt.subplot(421, title=f"Qubit 0", ylabel="I [adc level]")
            plt.plot(xpts_ns, data["avgi"][0], "o-")
            plt.subplot(422, title=f"Qubit 0", ylabel="Q [adc level]")
            plt.plot(xpts_ns, data["avgq"][0], "o-")

            plt.subplot(423, title=f"Qubit 1", ylabel="I [adc level]")
            plt.plot(xpts_ns, data["avgi"][1], "o-")
            plt.subplot(424, title=f"Qubit 1", ylabel="Q [adc level]")
            plt.plot(xpts_ns, data["avgq"][1], "o-")

            plt.subplot(425, title=f"Qubit 2", ylabel="I [adc level]")
            plt.plot(xpts_ns, data["avgi"][2], "o-")
            plt.subplot(426, title=f"Qubit 2", ylabel="Q [adc level]")
            plt.plot(xpts_ns, data["avgq"][2], "o-")

            plt.subplot(427, title=f"Qubit 3", xlabel="Time [ns]", ylabel="I [adc level]")
            plt.plot(xpts_ns, data["avgi"][3], "o-")
            plt.subplot(428, title=f"Qubit 3", xlabel="Time [ns]", ylabel="Q [adc level]")
            plt.plot(xpts_ns, data["avgq"][3], "o-")

        plt.tight_layout()

        if saveplot:
            plot_filename = "qram_protocol.png"
            plt.savefig(plot_filename, format="png", bbox_inches="tight", transparent=True)
            print("Saved", plot_filename)

        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        self.cfg.all_qubits = [1, 2, 3, 4]
        super().save_data(data=data)
        # print(self.pulse_dict)
        with self.datafile() as f:
            f.attrs["pulse_dict"] = json.dumps(self.pulse_dict, cls=NpEncoder)
            f.attrs["meas_order"] = json.dumps(self.meas_order, cls=NpEncoder)
            f.attrs["calib_order"] = json.dumps(self.calib_order, cls=NpEncoder)
        return self.fname


# ===================================================================== #


class QramProtocolSingleShotExperiment(Experiment):
    """
    Basically just a histogram experiment with the qram protocol code already built in
    expt = dict(
        reps: number of shots per expt
        check_states: an array of strs of the init_state specifying each configuration to plot a histogram for
        play_pulses: see code for play_pulses
    )
    """

    def __init__(self, soccfg=None, path="", prefix="QramSingleShotHist", config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=True):
        # expand entries in config that are length 1 to fill all qubits
        self.num_qubits_sample = len(self.cfg.device.readout.frequency)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        if isinstance(value2, dict):
                            for key3, value3 in value2.items():
                                if not (isinstance(value3, list)):
                                    value2.update({key3: [value3] * self.num_qubits_sample})
                elif not (isinstance(value, list)):
                    subcfg.update({key: [value] * self.num_qubits_sample})
        self.cfg.expt.tomo_qubits = [0, 1, 2, 3]  # this is just to make the super class happy

        check_states = self.cfg.expt.check_states
        play_pulses_list = self.cfg.expt.play_pulses_list
        data = dict(iqshots=[])

        adc_chs = self.cfg.hw.soc.adcs.readout.ch
        self.pulse_dict = dict()
        if "post_select" not in self.cfg.expt:
            self.cfg.expt.post_select = False

        timestep = np.inf
        self.cfg.expt.timestep = float(timestep)
        self.cfg.expt.basis = "ZZ"
        self.cfg.all_qubits = [0, 1, 2, 3]

        for check_state, play_pulses in zip(check_states, play_pulses_list):
            cfg_i = deepcopy(self.cfg)
            cfg_i.expt.init_state = check_state
            cfg_i.expt.play_pulses = play_pulses
            protocol_prog = QramProtocolProgram(soccfg=self.soccfg, cfg=cfg_i)
            avgi, avgq = protocol_prog.acquire(soc=self.im[self.cfg.aliases.soc], load_pulses=True, progress=progress)
            idata, qdata = protocol_prog.get_multireadout_shots()
            # each idata, qdata is [ro_chs, n_init_readout+1, reps)
            data["iqshots"].append((idata, qdata))

        for k, a in data.items():
            data[k] = np.array(a)

        self.data = data

        return data

    def analyze(
        self,
        data=None,
        check_qubit=None,
        qubits=None,
        theta=None,
        check_states=None,
        play_pulses_list=None,
        g_states=None,
        e_states=None,
        verbose=True,
        apply_ps=False,
        ps_qubits=None,
        ps_adjust=None,
        ps_check_states_list=None,
    ):
        """
        ps_check_states_list should be a list of indices of check_states that you want to use for post selection
        """
        if data is None:
            data = self.data

        full_mux_expt = False
        if "full_mux_expt" in self.cfg.expt:
            full_mux_expt = self.cfg.expt.full_mux_expt

        data_process = dict(iqshots=[])
        # if "readout_cool" in self.cfg.expt and self.cfg.expt.readout_cool:
        for i_check, check_data in enumerate(data["iqshots"]):
            idata, qdata = check_data
            data_process["iqshots"].append((idata[:, -1, :], qdata[:, -1, :]))
        fids, thresholds, angle = multihist(
            data=data_process,
            check_qubit=check_qubit,
            check_states=check_states,
            play_pulses_list=play_pulses_list,
            g_states=g_states,
            e_states=e_states,
            theta=theta,
            plot=False,
            verbose=verbose,
            amplitude_mode=full_mux_expt,
        )
        print(fids, thresholds, angle)
        data["angles"] = np.zeros(4)
        data["angles"][check_qubit] = angle
        data["thresholds"] = np.zeros(4)
        data["thresholds"][check_qubit] = thresholds[0]
        data["ge_avgs"] = [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]
        Igs, Qgs = data["iqshots"][g_states[0]]
        Ies, Qes = data["iqshots"][e_states[0]]
        data["ge_avgs"][check_qubit] = get_ge_avgs(Igs=Igs, Qgs=Qgs, Ies=Ies, Qes=Qes, amplitude_mode=full_mux_expt)

        if apply_ps:
            output_data = []
            if ps_adjust is None:
                ps_thresholds = data["thresholds"]
            else:
                ps_thresholds = ps_threshold_adjust(
                    ps_thresholds_init=data["thresholds"],
                    adjust=ps_adjust,
                    ge_avgs=data["ge_avgs"],
                    angles=data["angles"],
                    amplitude_mode=full_mux_expt,
                )

            for ps_check_state_i in ps_check_states_list:
                _, keep_indices = post_select_shots(
                    final_qubit=check_qubit,
                    all_ishots_raw_q=data["iqshots"][ps_check_state_i][0],
                    all_qshots_raw_q=data["iqshots"][ps_check_state_i][1],
                    angles=data["angles"],
                    ps_thresholds=ps_thresholds,
                    ps_qubits=ps_qubits,
                    n_init_readout=self.cfg.expt.n_init_readout,
                    post_process=None,
                    verbose=False,
                    return_keep_indices=True,
                    amplitude_mode=full_mux_expt,
                )

                ishots_unrot = data["iqshots"][ps_check_state_i][0][:, -1, :]
                qshots_unrot = data["iqshots"][ps_check_state_i][1][:, -1, :]
                ishots_ps_unrot = ishots_unrot[:, keep_indices]
                qshots_ps_unrot = qshots_unrot[:, keep_indices]
                print("keep", 100 * np.sum(keep_indices) / len(ishots_unrot[check_qubit]), "%")
                output_data.append((ishots_ps_unrot, qshots_ps_unrot))
                check_states.append(f"{check_states[ps_check_state_i]}_ps")
                play_pulses_list.append(play_pulses_list[ps_check_state_i])
            data["iqshots_ps"] = output_data
        return play_pulses_list, check_states, data

    def display(
        self,
        data=None,
        check_qubit=None,
        qubits=None,
        theta=None,
        check_states=None,
        play_pulses_list=None,
        g_states=None,
        e_states=None,
        verbose=True,
        export=False,
        apply_ps=False,
    ):
        if data is None:
            data = self.data
        check_states = np.copy(check_states)

        # if 0 not in qubits:
        #     for i in range(len(check_states)):
        #         check_states[i] = check_states[i][5:]

        full_mux_expt = False
        if "full_mux_expt" in self.cfg.expt:
            full_mux_expt = self.cfg.expt.full_mux_expt

        if apply_ps:
            assert "iqshots_ps" in data

        data_process = dict(iqshots=[])
        # if "readout_cool" in self.cfg.expt and self.cfg.expt.readout_cool:
        for check_data in data["iqshots"]:
            idata, qdata = check_data
            data_process["iqshots"].append((idata[:, -1, :], qdata[:, -1, :]))
        if apply_ps:
            for check_data in data["iqshots_ps"]:
                idata, qdata = check_data
                data_process["iqshots"].append((idata, qdata))

        fids, thresholds, angle = multihist(
            data=data_process,
            check_qubit=check_qubit,
            check_states=check_states,
            play_pulses_list=play_pulses_list,
            g_states=g_states,
            e_states=e_states,
            theta=theta,
            plot=True,
            verbose=verbose,
            export=export,
            amplitude_mode=full_mux_expt,
        )

        print(f"average ge fidelity (%): {100*fids[0]}")
        print(f"rotation angle (deg): {angle}")
        print(f"threshold ge: {thresholds[0]}")

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        with self.datafile() as f:
            f.attrs["check_states"] = json.dumps(self.cfg.expt.check_states, cls=NpEncoder)


# ===================================================================== #
class QramVariantsProgram(QramProtocolProgram):
    def state_prep_pulse(self, qubits=None, **kwargs):
        self.Y_pulse(q=0, play=True)
        # self.Y_pulse(q=2, play=True)
        self.Y_pulse(q=3, play=True)

        cfg = AttrDict(self.cfg)
        count_us = 0

        # 2. apply Eg-Gf with qDrive=3: eegg -> eggf [path 2]
        self.eegg_eggf(count_us, pihalf=False)

        # # 1. apply Eg-Gf with qDrive=2: gegg -> ggfg [path 1]
        # self.gegg_ggfg(count_us, pihalf=False)

        # 4. apply ef pulse on Q3 (at this point guaranteed no excitation in Q1) [path 2]
        self.q3_ef(count_us, pihalf=False)

        # # 3. apply ef pulse on Q2 (at this point guaranteed no excitation in Q1) [path 1]
        # self.q2_ef(count_us, pihalf=False)

        self.sync_all(self.us2cycles(self.cfg.expt.wait_time))


class QramVariantsT1Experiment(Experiment):
    """
    T1 on the qram protocol
    expt = dict(
        init_state
        play_pulses: see code for play_pulses
        start: wait time sweep start [us]
        step: wait time sweep step
        expts: number steps in sweep
        reps: number averages per experiment
    )
    """

    def __init__(self, soccfg=None, path="", prefix="QramT1", config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        # expand entries in config that are length 1 to fill all qubits
        self.num_qubits_sample = len(self.cfg.device.readout.frequency)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        if isinstance(value2, dict):
                            for key3, value3 in value2.items():
                                if not (isinstance(value3, list)):
                                    value2.update({key3: [value3] * self.num_qubits_sample})
                elif not (isinstance(value, list)):
                    subcfg.update({key: [value] * self.num_qubits_sample})
        self.cfg.expt.tomo_qubits = [0, 1, 2, 3]  # this is just to make the super class happy

        adc_chs = self.cfg.hw.soc.adcs.readout.ch
        self.pulse_dict = dict()
        if "post_select" not in self.cfg.expt:
            self.cfg.expt.post_select = False

        self.cfg.expt.basis = "ZZ"
        self.cfg.all_qubits = [0, 1, 2, 3]
        times = self.cfg.expt.start + self.cfg.expt.step * np.arange(self.cfg.expt.expts)

        data = {"times": times, "avgi": [], "avgq": [], "amps": [], "phases": []}
        for t in tqdm(times):
            self.cfg.expt.wait_time = float(t)
            cfg_i = deepcopy(self.cfg)
            protocol_prog = QramVariantsProgram(soccfg=self.soccfg, cfg=cfg_i)
            avgi, avgq = protocol_prog.acquire(soc=self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
            avgi = np.array(avgi)
            avgq = np.array(avgq)
            # each idata, qdata has 4 readouts, each with the number of shots
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(np.abs(avgi + 1j * avgq))
            data["phases"].append(np.angle(avgi + 1j * avgq))

        for k, a in data.items():
            data[k] = np.array(a)

        self.data = data

        return data

    def analyze(self, qubit, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        xpts = data["times"]
        amps = data["amps"][:, qubit].flatten()
        avgi = data["avgi"][:, qubit].flatten()
        avgq = data["avgq"][:, qubit].flatten()

        data[f"fit_amps_{qubit}"], data[f"fit_err_amps_{qubit}"] = fitter.fitexp(xpts, amps, fitparams=None)
        data[f"fit_avgi_{qubit}"], data[f"fit_err_avgi_{qubit}"] = fitter.fitexp(xpts, avgi, fitparams=None)
        data[f"fit_avgq_{qubit}"], data[f"fit_err_avgq_{qubit}"] = fitter.fitexp(xpts, avgq, fitparams=None)
        return data

    def display(self, qubit, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data
        xpts = data["times"]
        amps = data["amps"][:, qubit].flatten()
        avgi = data["avgi"][:, qubit].flatten()
        avgq = data["avgq"][:, qubit].flatten()

        plt.figure(figsize=(10, 10))
        plt.subplot(211, title=f"$T_1 Q{qubit}$", ylabel="I [ADC units]")
        plt.plot(xpts, avgi, "o-")
        if fit:
            p = data[f"fit_avgi_{qubit}"]
            pCov = data[f"fit_err_avgi_{qubit}"]
            captionStr = f"$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}"
            plt.plot(xpts, fitter.expfunc(xpts, *data[f"fit_avgi_{qubit}"]), label=captionStr)
            plt.legend()
            print(f"Fit T1 avgi [us]: {p[3]}")
        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC units]")
        plt.plot(xpts, avgq, "o-")
        if fit:
            p = data[f"fit_avgq_{qubit}"]
            pCov = data[f"fit_err_avgq_{qubit}"]
            captionStr = f"$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}"
            plt.plot(xpts, fitter.expfunc(xpts, *data[f"fit_avgq_{qubit}"]), label=captionStr)
            plt.legend()
            print(f"Fit T1 avgq [us]: {p[3]}")

        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname


# ===================================================================== #


class QramProtocol1QTomoProgram(QramProtocolProgram, AbstractStateTomo1QProgram):
    def initialize(self):
        super().initialize()

    def body(self):
        AbstractStateTomo1QProgram.body(self)

    def collect_counts(self, angle=None, threshold=None, amplitude_mode=False):
        return AbstractStateTomo1QProgram.collect_counts(self, angle, threshold, amplitude_mode)


# --------------------------------------------------------------------- #


class QramProtocol1QTomoExperiment(Experiment):
    # outer loop over measurement bases
    # set the state prep pulse to be preparing the gg, ge, eg, ee states for confusion matrix
    """
    Perform state tomography on 1Q state with error mitigation.
    Experimental Config:
    expt = dict(
        reps: number averages per measurement basis iteration
        singleshot_reps: number averages in single shot calibration
        qubit
    )
    """

    def __init__(self, soccfg=None, path="", prefix="QramProtocol1QTomo", config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)
        self.cfg.all_qubits = [0, 1, 2, 3]

    def acquire(self, progress=False, debug=True):
        # expand entries in config that are length 1 to fill all qubits
        self.num_qubits_sample = len(self.cfg.device.readout.frequency)
        self.qubit = self.cfg.expt.qubit

        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not (isinstance(value3, list)):
                                value2.update({key3: [value3] * self.num_qubits_sample})
                elif not (isinstance(value, list)):
                    subcfg.update({key: [value] * self.num_qubits_sample})

        if "meas_order" not in self.cfg.expt or self.cfg.expt.meas_order is None:
            self.meas_order = ["Z", "X", "Y"]
        else:
            self.meas_order = self.cfg.expt.meas_order
        self.calib_order = ["g", "e"]  # should match with order of counts for each tomography measurement
        data = {"counts_tomo": [], "counts_calib": []}
        self.pulse_dict = dict()

        full_mux_expt = False
        if "full_mux_expt" in self.cfg.expt:
            full_mux_expt = self.cfg.expt.full_mux_expt

        if "n_init_readout" not in self.cfg.expt:
            self.cfg.expt.n_init_readout = 0
        data[f"ishots_raw"] = np.zeros(
            shape=(len(self.meas_order), self.num_qubits_sample, self.cfg.expt.n_init_readout + 1, self.cfg.expt.reps)
        )  # raw data tomo experiments
        data[f"qshots_raw"] = np.zeros(
            shape=(len(self.meas_order), self.num_qubits_sample, self.cfg.expt.n_init_readout + 1, self.cfg.expt.reps)
        )  # raw data tomo experiments

        self.readout_cool = False
        if "readout_cool" in self.cfg.expt and self.cfg.expt.readout_cool:
            self.readout_cool = self.cfg.expt.readout_cool
        if not self.readout_cool:
            self.cfg.expt.n_init_readout = 0

        self.ncalib = len(self.calib_order) + (2 * (self.num_qubits_sample - 1) if self.readout_cool else 0)
        data["calib_ishots_raw"] = np.zeros(
            shape=(
                self.ncalib,
                self.num_qubits_sample,
                self.cfg.expt.n_init_readout + 1,
                self.cfg.expt.singleshot_reps,
            )
        )  # raw rotated g data for the calibration histograms for each of the measure experiments
        data["calib_qshots_raw"] = np.zeros(
            shape=(
                self.ncalib,
                self.num_qubits_sample,
                self.cfg.expt.n_init_readout + 1,
                self.cfg.expt.singleshot_reps,
            )
        )  # raw rotated g data for the calibration histograms for each of the measure experiments

        # ================= #
        # Get single shot calibration for qubits
        # ================= #

        if (
            "angles" in self.cfg.expt
            and "thresholds" in self.cfg.expt
            and "ge_avgs" in self.cfg.expt
            and "counts_calib" in self.cfg.expt
            and None
            not in (self.cfg.expt.angles, self.cfg.expt.thresholds, self.cfg.expt.ge_avgs, self.cfg.expt.counts_calib)
        ):
            angles_q = self.cfg.expt.angles
            thresholds_q = self.cfg.expt.thresholds
            ge_avgs_q = self.cfg.expt.ge_avgs
            for q in range(self.num_qubits_sample):
                if ge_avgs_q[q] is None:
                    ge_avgs_q[q] = np.zeros_like(
                        ge_avgs_q[self.cfg.expt.tomo_qubits[0]]
                    )  # just get the shape of the arrays correct by picking the old ge_avgs_q of a q that was definitely measured
            ge_avgs_q = np.array(ge_avgs_q)
            data["counts_calib"] = self.cfg.expt.counts_calib
            if progress:
                print("Re-using provided angles, thresholds, ge_avgs, counts_calib")

        else:
            # Error mitigation measurements: prep in g, e to recalibrate measurement angle and measure confusion matrix
            calib_prog_dict = dict()
            for prep_state in tqdm(self.calib_order, disable=not progress):
                # print(prep_state)
                cfg = AttrDict(deepcopy(self.cfg))
                cfg.expt.reps = self.cfg.expt.singleshot_reps
                cfg.expt.state_prep_kwargs = dict(prep_state=prep_state)
                err_tomo = ErrorMitigationStateTomo1QProgram(soccfg=self.soccfg, cfg=cfg)
                err_tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                calib_prog_dict.update({prep_state: err_tomo})

            g_prog = calib_prog_dict["g"]
            Ig, Qg = g_prog.get_shots(verbose=False)
            thresholds_q = [0] * self.num_qubits_sample
            angles_q = [0] * self.num_qubits_sample
            ge_avgs_q = [[0] * 4] * self.num_qubits_sample

            # Get readout angle + threshold for qubit
            e_prog = calib_prog_dict["e"]
            Ie, Qe = e_prog.get_shots(verbose=False)
            shot_data = dict(Ig=Ig[self.qubit], Qg=Qg[self.qubit], Ie=Ie[self.qubit], Qe=Qe[self.qubit])
            fid, threshold, angle = hist(data=shot_data, plot=debug, verbose=False, amplitude_mode=full_mux_expt)
            thresholds_q[self.qubit] = threshold[0]
            angles_q[self.qubit] = angle
            ge_avgs_q[self.qubit] = [
                np.average(Ig[self.qubit]),
                np.average(Qg[self.qubit]),
                np.average(Ie[self.qubit]),
                np.average(Qe[self.qubit]),
            ]
            print(f"Qubit ({self.qubit}) ge")
            print(
                f"ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[self.qubit]} \t threshold ge: {thresholds_q[self.qubit]}"
            )

            # Process the shots taken for the confusion matrix with the calibration angles
            for iprep, prep_state in enumerate(self.calib_order):
                counts = calib_prog_dict[prep_state].collect_counts(
                    angle=angles_q, threshold=thresholds_q, amplitude_mode=full_mux_expt
                )
                data["counts_calib"].append(counts)
                data[f"calib_ishots_raw"][iprep, :, :, :], data[f"calib_qshots_raw"][iprep, :, :, :] = calib_prog_dict[
                    prep_state
                ].get_multireadout_shots()

            # Do the calibration for the remaining qubits in case you want to do post selection
            if self.readout_cool:
                sscfg = AttrDict(deepcopy(self.cfg))
                sscfg.expt.reps = self.cfg.expt.singleshot_reps
                ps_calib_prog_dict = dict()
                iprep_temp = len(self.calib_order)
                for q in range(self.num_qubits_sample):
                    if q == self.cfg.expt.qubit:
                        continue  # already did these
                    sscfg.expt.qubit = q
                    for prep_state in tqdm(["g", "e"], disable=not progress):
                        # print(prep_state)
                        sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state)
                        err_tomo = ErrorMitigationStateTomo1QProgram(soccfg=self.soccfg, cfg=sscfg)
                        err_tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                        ps_calib_prog_dict.update({prep_state: err_tomo})

                        # Save the full counts from the calibration experiments using the measured angles for all qubits in case you want to do post selection on the calibration
                        (
                            data[f"calib_ishots_raw"][iprep_temp, :, :, :],
                            data[f"calib_qshots_raw"][iprep_temp, :, :, :],
                        ) = err_tomo.get_multireadout_shots()
                        iprep_temp += 1

                    g_prog = ps_calib_prog_dict["g"]
                    Ig, Qg = g_prog.get_shots(verbose=False)

                    # Get readout angle + threshold for qubit
                    e_prog = ps_calib_prog_dict["e"]
                    Ie, Qe = e_prog.get_shots(verbose=False)
                    shot_data = dict(Ig=Ig[q], Qg=Qg[q], Ie=Ie[q], Qe=Qe[q])
                    print(f"Qubit ({q}) ge")
                    fid, threshold, angle = hist(
                        data=shot_data, plot=debug, verbose=False, amplitude_mode=full_mux_expt
                    )
                    thresholds_q[q] = threshold[0]
                    angles_q[q] = angle
                    ge_avgs_q[q] = [np.average(Ig[q]), np.average(Qg[q]), np.average(Ie[q]), np.average(Qe[q])]
                    print(
                        f"ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[q]} \t threshold ge: {thresholds_q[q]}"
                    )

            data.update(dict(thresholds=thresholds_q, angles=angles_q, ge_avgs=ge_avgs_q))

            if progress:
                print(f"thresholds={thresholds_q},")
                print(f"angles={angles_q},")
                print(f"ge_avgs={ge_avgs_q}", ",")

            if progress:
                print(f"counts_calib={np.array(data['counts_calib']).tolist()}")

        data.update(dict(thresholds=thresholds_q, angles=angles_q, ge_avgs=ge_avgs_q))

        # ================= #
        # Begin protocol stepping
        # ================= #

        # Tomography measurements
        for ibasis, basis in enumerate(tqdm(self.meas_order, disable=not progress)):
            # print(basis)
            cfg = AttrDict(deepcopy(self.cfg))
            # print('WARNING CHANGING PI PULSE FREQUENCY')
            # cfg.device.qubit.f_ge[0] = cfg.device.qubit.f_ge[0] + 5.3
            # print(cfg.device.qubit.f_ge)

            cfg.expt.basis = basis
            cfg.expt.timestep = np.inf
            tomo = QramProtocol1QTomoProgram(soccfg=self.soccfg, cfg=cfg)
            # print(tomo)
            tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
            counts = tomo.collect_counts(angle=angles_q, threshold=thresholds_q, amplitude_mode=full_mux_expt)
            data["counts_tomo"].append(counts)
            data[f"ishots_raw"][ibasis, :, :, :], data[f"qshots_raw"][ibasis, :, :, :] = tomo.get_multireadout_shots()
            self.pulse_dict.update({basis: tomo.pulse_dict})

        self.data = data
        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data
        print("Analyze function does nothing, use the analysis notebook.")
        return data

    def display(self, qubit, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data
        print("Display function does nothing, use the analysis notebook.")

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        # print(self.pulse_dict)
        with self.datafile() as f:
            f.attrs["pulse_dict"] = json.dumps(self.pulse_dict, cls=NpEncoder)
            f.attrs["meas_order"] = json.dumps(self.meas_order, cls=NpEncoder)
            f.attrs["calib_order"] = json.dumps(self.calib_order, cls=NpEncoder)
        return self.fname


# ===================================================================== #


class QramProtocol3QTomoProgram(QramProtocolProgram, AbstractStateTomo3QProgram):
    def initialize(self):
        super().initialize()

    def body(self):
        AbstractStateTomo3QProgram.body(self)

    def collect_counts(self, angle=None, threshold=None):
        return AbstractStateTomo3QProgram.collect_counts(self, angle, threshold)


class QramProtocol3QTomoExperiment(Experiment):
    def __init__(self, soccfg=None, path="", prefix="QramProtocol3QTomo", config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)
        self.cfg.all_qubits = [0, 1, 2, 3]

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
        self.num_qubits_sample = len(self.cfg.device.readout.frequency)
        qubits = self.cfg.expt.tomo_qubits
        qA, qB, qC = qubits

        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not (isinstance(value3, list)):
                                value2.update({key3: [value3] * self.num_qubits_sample})
                elif not (isinstance(value, list)):
                    subcfg.update({key: [value] * self.num_qubits_sample})

        self.meas_order = make_3q_meas_order()
        # self.meas_order = ['XXY']
        self.calib_order = make_3q_calib_order()  # should match with order of counts for each tomography measurement

        timesteps = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        print("timesteps", timesteps)
        measure_f_qubits = None
        if "measure_f" in self.cfg.expt:
            measure_f_qubits = self.cfg.expt.measure_f

        data = dict()
        data.update(
            {
                "xpts": [],
                "gpop": [[], [], [], []],
                "epop": [[], [], [], []],
                "fpop": [[], [], [], []],
                "counts_calib": [],
            }
        )
        if self.cfg.expt.post_process == "threshold" and measure_f_qubits is not None:
            data.update({"counts_raw": [[], []]})
        else:
            data.update({"counts_raw": [[]]})
        if self.cfg.expt.tomo_3q:
            data.update({"counts_tomo": []})
        self.pulse_dict = dict()

        self.readout_cool = False
        if "readout_cool" in self.cfg.expt and self.cfg.expt.readout_cool:
            self.readout_cool = self.cfg.expt.readout_cool
        if not self.readout_cool:
            self.cfg.expt.n_init_readout = 0

        if "n_init_readout" not in self.cfg.expt:
            self.cfg.expt.n_init_readout = 0
        data[f"ishots_raw"] = np.zeros(
            shape=(len(self.meas_order), self.num_qubits_sample, self.cfg.expt.n_init_readout + 1, self.cfg.expt.reps)
        )  # raw data tomo experiments
        data[f"qshots_raw"] = np.zeros(
            shape=(len(self.meas_order), self.num_qubits_sample, self.cfg.expt.n_init_readout + 1, self.cfg.expt.reps)
        )  # raw data tomo experiments

        self.ncalib = len(self.calib_order) + (
            2 * (self.num_qubits_sample - len(self.cfg.expt.tomo_qubits)) if self.readout_cool else 0
        )
        data["calib_ishots_raw"] = np.zeros(
            shape=(
                self.ncalib,
                self.num_qubits_sample,
                self.cfg.expt.n_init_readout + 1,
                self.cfg.expt.singleshot_reps,
            )
        )  # raw rotated g data for the calibration histograms for each of the measure experiments
        data["calib_qshots_raw"] = np.zeros(
            shape=(
                self.ncalib,
                self.num_qubits_sample,
                self.cfg.expt.n_init_readout + 1,
                self.cfg.expt.singleshot_reps,
            )
        )  # raw rotated g data for the calibration histograms for each of the measure experiments

        # ================= #
        # Get single shot calibration for qubits
        # ================= #

        # Error mitigation measurements: prep in gg, ge, eg, ee to recalibrate measurement angle and measure confusion matrix
        if (
            "angles" in self.cfg.expt
            and "thresholds" in self.cfg.expt
            and "ge_avgs" in self.cfg.expt
            and "counts_calib" in self.cfg.expt
            and None
            not in (self.cfg.expt.angles, self.cfg.expt.thresholds, self.cfg.expt.ge_avgs, self.cfg.expt.counts_calib)
        ):
            angles_q = self.cfg.expt.angles
            thresholds_q = self.cfg.expt.thresholds
            ge_avgs_q = self.cfg.expt.ge_avgs
            q_not_in_tomo = 1
            for q in range(self.num_qubits_sample):
                if ge_avgs_q[q] is None:
                    ge_avgs_q[q] = np.zeros_like(
                        ge_avgs_q[qubits[0]]
                    )  # just get the shape of the arrays correct by picking the old ge_avgs_q of a q that was definitely measured
            ge_avgs_q = np.array(ge_avgs_q)
            data["counts_calib"] = self.cfg.expt.counts_calib
            if progress:
                print("Re-using provided angles, thresholds, ge_avgs, counts_calib")

        else:
            # Initialize Q1 in e for calibration matrix
            setup_q1_e = False
            if "setup_q1_e" in self.cfg.expt:
                setup_q1_e = self.cfg.expt.setup_q1_e

            calib_prog_dict = dict()
            for prep_state in tqdm(self.calib_order, disable=not progress):
                # print(prep_state)
                cfg = AttrDict(deepcopy(self.cfg))
                cfg.expt.reps = self.cfg.expt.singleshot_reps
                cfg.expt.state_prep_kwargs = dict(prep_state=prep_state, setup_q1_e=setup_q1_e)
                err_tomo = ErrorMitigationStateTomo3QProgram(soccfg=self.soccfg, cfg=cfg)
                err_tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                calib_prog_dict.update({prep_state: err_tomo})

            g_prog = calib_prog_dict["ggg"]
            Ig, Qg = g_prog.get_shots(verbose=False)
            thresholds_q = [0] * self.num_qubits_sample
            angles_q = [0] * self.num_qubits_sample
            ge_avgs_q = [[0] * 4] * self.num_qubits_sample

            for iq, q in enumerate(qubits):
                state = "ggg"
                state = state[:iq] + "e" + state[iq + 1 :]
                e_prog = calib_prog_dict[state]
                Ie, Qe = e_prog.get_shots(verbose=False)
                shot_data = dict(Ig=Ig[q], Qg=Qg[q], Ie=Ie[q], Qe=Qe[q])
                print(f"Qubit  ({q})")
                fid, threshold, angle = hist(data=shot_data, plot=debug, verbose=False)
                thresholds_q[q] = threshold[0]
                angles_q[q] = angle
                ge_avgs_q[q] = [np.average(Ig[q]), np.average(Qg[q]), np.average(Ie[q]), np.average(Qe[q])]
                print(
                    f"ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[q]} \t threshold ge: {thresholds_q[q]}"
                )

            # Process the shots taken for the confusion matrix with the calibration angles
            for iprep, prep_state in enumerate(self.calib_order):
                counts = calib_prog_dict[prep_state].collect_counts(angle=angles_q, threshold=thresholds_q)
                data["counts_calib"].append(counts)
                data[f"calib_ishots_raw"][iprep, :, :, :], data[f"calib_qshots_raw"][iprep, :, :, :] = calib_prog_dict[
                    prep_state
                ].get_multireadout_shots()

            # Do the calibration for the remaining qubits in case you want to do post selection
            if (
                self.readout_cool
                or self.cfg.expt.expts > 1
                or (
                    "post_select" in self.cfg.expt and self.cfg.expt.post_select and 1 not in self.cfg.expt.tomo_qubits
                )
            ):
                sscfg = AttrDict(deepcopy(self.cfg))
                sscfg.expt.reps = self.cfg.expt.singleshot_reps
                ps_calib_prog_dict = dict()
                iprep_temp = len(self.calib_order)
                for q in range(self.num_qubits_sample):
                    if q in self.cfg.expt.tomo_qubits:
                        continue  # already did these
                    sscfg.expt.qubit = q
                    for prep_state in tqdm(["g", "e"], disable=not progress):
                        # print(prep_state)
                        sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state)
                        err_tomo = ErrorMitigationStateTomo1QProgram(soccfg=self.soccfg, cfg=sscfg)
                        err_tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                        ps_calib_prog_dict.update({prep_state: err_tomo})

                        # Save the full counts from the calibration experiments using the measured angles for all qubits in case you want to do post selection on the calibration
                        (
                            data[f"calib_ishots_raw"][iprep_temp, :, :, :],
                            data[f"calib_qshots_raw"][iprep_temp, :, :, :],
                        ) = err_tomo.get_multireadout_shots()
                        iprep_temp += 1

                    g_prog = ps_calib_prog_dict["g"]
                    Ig, Qg = g_prog.get_shots(verbose=False)

                    # Get readout angle + threshold for qubit
                    e_prog = ps_calib_prog_dict["e"]
                    Ie, Qe = e_prog.get_shots(verbose=False)
                    shot_data = dict(Ig=Ig[q], Qg=Qg[q], Ie=Ie[q], Qe=Qe[q])
                    print(f"Qubit ({q}) ge")
                    fid, threshold, angle = hist(data=shot_data, plot=debug, verbose=False)
                    thresholds_q[q] = threshold[0]
                    angles_q[q] = angle
                    ge_avgs_q[q] = [np.average(Ig[q]), np.average(Qg[q]), np.average(Ie[q]), np.average(Qe[q])]
                    print(
                        f"ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[q]} \t threshold ge: {thresholds_q[q]}"
                    )

            print(f"thresholds={thresholds_q},")
            print(f"angles={angles_q},")
            print(f"ge_avgs={ge_avgs_q}", ",")
            print(f"counts_calib={np.array(data['counts_calib']).tolist()}")

        data.update(dict(thresholds=thresholds_q, angles=angles_q, ge_avgs=ge_avgs_q))
        print()

        # ================= #
        # Begin protocol stepping
        # ================= #
        adc_chs = self.cfg.hw.soc.adcs.readout.ch

        gpop_q_times = np.zeros(shape=(4, len(timesteps)))
        epop_q_times = np.zeros(shape=(4, len(timesteps)))
        fpop_q_times = np.zeros(shape=(4, len(timesteps)))
        self.cfg.expt.basis = "ZZZ"
        for time_i, timestep in enumerate(tqdm(timesteps, disable=not progress)):
            if len(timesteps) > 1:
                self.cfg.expt.timestep = float(timestep)

                # With standard pulse setup, measure the g population
                cfg = deepcopy(self.cfg)
                cfg.expt.ge_pulse = None
                protocol_prog_g = QramProtocol3QTomoProgram(soccfg=self.soccfg, cfg=cfg)
                popln_g, avgq = protocol_prog_g.acquire_rotated(
                    soc=self.im[self.cfg.aliases.soc],
                    progress=False,
                    angle=angles_q,
                    threshold=thresholds_q,
                    ge_avgs=ge_avgs_q,
                    post_process=self.cfg.expt.post_process,
                )
                if measure_f_qubits is not None:
                    # Add a ge pulse, measure the g population, which is really measuring the e population
                    cfg.expt.ge_pulse = (
                        measure_f_qubits  # assumes the qubit(s) in measure_f_qubits have minimal ZZ to each other
                    )
                    protocol_prog_e = QramProtocol3QTomoProgram(soccfg=self.soccfg, cfg=cfg)
                    popln_e, avgq = protocol_prog_e.acquire_rotated(
                        soc=self.im[self.cfg.aliases.soc],
                        progress=False,
                        angle=angles_q,
                        threshold=thresholds_q,
                        ge_avgs=ge_avgs_q,
                        post_process=self.cfg.expt.post_process,
                    )

                gpop_q = gpop_q_times[:, time_i]
                epop_q = epop_q_times[:, time_i]
                fpop_q = fpop_q_times[:, time_i]

                # Scaling post process
                if self.cfg.expt.post_process == "scale":
                    for q in range(4):
                        gpop_q[q] = 1 - popln_g[adc_chs[q]]
                        epop_q[q] = popln_g[
                            adc_chs[q]
                        ]  # this is the final answer if we don't care about distinguishing e/f
                    if measure_f_qubits is not None:
                        # if we care about distinguishing e/f, the "g" popln of the 2nd experiment is the real e popln, and the real f popln is whatever is left
                        for q in measure_f_qubits:
                            epop_q[q] = 1 - popln_e[adc_chs[q]]  # e population shows up as g population
                            fpop_q[q] = 1 - epop_q[q] - gpop_q[q]
                            # print(q, gpop_q[q], epop_q[q], fpop_q[q])

                # Threshold post process
                elif self.cfg.expt.post_process == "threshold":
                    # Need to re-do population counts using the readout matrix correction

                    counts_g = protocol_prog_g.collect_counts(angle=angles_q, threshold=thresholds_q)
                    data["counts_raw"][0].append(counts_g)

                    counts_e = None
                    if measure_f_qubits is not None:
                        # if we care about distinguishing e/f, the "g" popln of the 2nd experiment is the real e popln, and the real f popln is whatever is left
                        for q in qubits:
                            epop_q[q] = 0  # reset this to recalculate e population

                        counts_e = protocol_prog_e.collect_counts(angle=angles_q, threshold=thresholds_q)
                        data["counts_raw"][1].append(counts_e)

                    gpop_q, epop_q, fpop_q = infer_gef_popln(
                        counts1=counts_g,
                        counts2=counts_e,
                        qubits=qubits,
                        calib_order=self.calib_order,
                        measure_f_qubits=measure_f_qubits,
                        counts_calib=data["counts_calib"],
                        fix_neg_counts_flag=True,
                    )

                    gpop_q_times[:, time_i] = gpop_q
                    epop_q_times[:, time_i] = epop_q
                    fpop_q_times[:, time_i] = fpop_q

                data["gpop"] = gpop_q_times
                data["epop"] = epop_q_times
                data["fpop"] = fpop_q_times

                data["xpts"].append(float(timestep))

            # -------------- #
            # Perform 3q state tomo only on last timestep
            # -------------- #
            if self.cfg.expt.tomo_3q and time_i == len(timesteps) - 1:
                # Tomography measurements
                for ibasis, basis in enumerate(tqdm(self.meas_order, disable=not progress)):
                    # print(basis)
                    cfg = AttrDict(deepcopy(self.cfg))
                    cfg.expt.basis = basis
                    cfg.expt.timestep = np.inf

                    tomo = QramProtocol3QTomoProgram(soccfg=self.soccfg, cfg=cfg)

                    # from qick.helpers import progs2json
                    # print('basis', basis)
                    # print(progs2json([tomo.dump_prog()]))
                    # print()

                    # print(tomo)
                    # from qick.helpers import progs2json
                    # print(progs2json([tomo.dump_prog()]))
                    # xpts, avgi, avgq = tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                    avgi, avgq = tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)

                    # print(basis)
                    adc_chs = self.cfg.hw.soc.adcs.readout.ch
                    # avgi, avgq = tomo.get_shots(angle=None, avg_shots=True)
                    # for q in self.cfg.expt.tomo_qubits:
                    #     print('q', q, 'avgi', avgi[adc_chs[q]])
                    #     print('q', q, 'avgq', avgq[adc_chs[q]])
                    #     print('q', q, 'amps', np.abs(avgi[adc_chs[q]]+1j*avgi[adc_chs[q]]))

                    counts = tomo.collect_counts(angle=angles_q, threshold=thresholds_q)
                    data["counts_tomo"].append(counts)
                    (
                        data[f"ishots_raw"][ibasis, :, :, :],
                        data[f"qshots_raw"][ibasis, :, :, :],
                    ) = tomo.get_multireadout_shots()
                    self.pulse_dict.update({basis: tomo.pulse_dict})

        if self.cfg.expt.expts > 1:
            data["end_times"] = protocol_prog_g.end_times_us
            print("end times", protocol_prog_g.end_times_us)

        for k, a in data.items():
            # print(k)
            # print(a)
            data[k] = np.array(a)

        self.data = data

        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data

        if self.cfg.expt.post_process == "threshold":
            timesteps = data["xpts"]
            gpop_q_times = np.zeros(shape=(4, len(timesteps)))
            epop_q_times = np.zeros(shape=(4, len(timesteps)))
            fpop_q_times = np.zeros(shape=(4, len(timesteps)))
            for time_i in range(len(timesteps)):
                counts_g = data["counts_raw"][0][time_i]
                counts_e = None
                if np.shape(data["counts_raw"])[0] == 2:
                    counts_e = data["counts_raw"][1][time_i]

                gpop_q, epop_q, fpop_q = infer_gef_popln(
                    counts1=counts_g,
                    counts2=counts_e,
                    qubits=self.cfg.expt.tomo_qubits,
                    calib_order=self.calib_order,
                    measure_f_qubits=self.cfg.expt.measure_f,
                    counts_calib=data["counts_calib"],
                    fix_neg_counts_flag=True,
                )

                gpop_q_times[:, time_i] = gpop_q
                epop_q_times[:, time_i] = epop_q
                fpop_q_times[:, time_i] = fpop_q

            data["gpop"] = gpop_q_times
            data["epop"] = epop_q_times
            data["fpop"] = fpop_q_times
        return data

    def display(self, data=None, saveplot=False, **kwargs):
        if data is None:
            data = self.data
        qubits = self.cfg.expt.tomo_qubits

        if self.cfg.expt.tomo_3q and len(data["xpts"]) < 2:
            print("Display function does nothing, use the analysis notebook.")
            return

        xpts_ns = np.array(data["xpts"]) * 1e3

        if self.cfg.expt.post_process == "threshold" or self.cfg.expt.post_process == "scale":
            plt.figure(figsize=(14, 8))
            if saveplot:
                plt.style.use("dark_background")
            plt_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            # plt.title(f"Qram Protocol", fontsize=20)

            for i_q, q in enumerate(qubits):
                plt.plot(
                    xpts_ns,
                    data["epop"][q],
                    "-",
                    marker=marker_cycle[q],
                    markersize=8,
                    color=default_colors[q],
                    label=f"Q{q} e",
                )
                if self.cfg.expt.measure_f is not None and q in self.cfg.expt.measure_f:
                    plt.plot(
                        xpts_ns,
                        data["fpop"][q],
                        "--",
                        marker=marker_cycle[q],
                        markersize=8,
                        color=default_colors[q],
                        label=f"Q{q} f",
                    )

            if "end_times" in data:
                end_times = data["end_times"]
                for end_time in end_times:
                    plt.axvline(1e3 * end_time, color="0.4", linestyle="--")

            # if self.cfg.expt.post_process == 'threshold':
            # plt.ylim(-0.1, 1.1)
            plt.legend(fontsize=26)
            plt.xlabel("Time [ns]", fontsize=26)
            plt.ylabel("Population", fontsize=26)
            plt.tick_params(labelsize=24)
            # plt.grid(linewidth=0.3)

        else:
            # plt.figure(figsize=(14,20))
            # plt.subplot(421, title=f'Qubit 0', ylabel="I [adc level]")
            # plt.plot(xpts_ns, data["avgi"][0],'o-')
            # plt.subplot(422, title=f'Qubit 0', ylabel="Q [adc level]")
            # plt.plot(xpts_ns, data["avgq"][0],'o-')

            # plt.subplot(423, title=f'Qubit 1', ylabel="I [adc level]")
            # plt.plot(xpts_ns, data["avgi"][1],'o-')
            # plt.subplot(424, title=f'Qubit 1', ylabel="Q [adc level]")
            # plt.plot(xpts_ns, data["avgq"][1],'o-')

            # plt.subplot(425, title=f'Qubit 2', ylabel="I [adc level]")
            # plt.plot(xpts_ns, data["avgi"][2],'o-')
            # plt.subplot(426, title=f'Qubit 2', ylabel="Q [adc level]")
            # plt.plot(xpts_ns, data["avgq"][2],'o-')

            # plt.subplot(427, title=f'Qubit 3', xlabel='Time [ns]', ylabel="I [adc level]")
            # plt.plot(xpts_ns, data["avgi"][3],'o-')
            # plt.subplot(428, title=f'Qubit 3', xlabel='Time [ns]', ylabel="Q [adc level]")
            # plt.plot(xpts_ns, data["avgq"][3],'o-')
            print("Not implemented!")

        plt.tight_layout()

        if saveplot:
            plot_filename = "qram_protocol.png"
            plt.savefig(plot_filename, format="png", bbox_inches="tight", transparent=True)
            print("Saved", plot_filename)

        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        # print(self.pulse_dict)
        with self.datafile() as f:
            f.attrs["pulse_dict"] = json.dumps(self.pulse_dict, cls=NpEncoder)
            f.attrs["meas_order"] = json.dumps(self.meas_order, cls=NpEncoder)
            f.attrs["calib_order"] = json.dumps(self.calib_order, cls=NpEncoder)
        return self.fname


# ===================================================================== #


class QramProtocol4QProgram(QramProtocolProgram, AbstractStateTomo4QProgram):
    def initialize(self):
        super().initialize()

    def body(self):
        AbstractStateTomo4QProgram.body(self)

    def collect_counts(self, angle=None, threshold=None):
        return AbstractStateTomo4QProgram.collect_counts(self, angle, threshold)


class QramProtocolExperiment4Q(Experiment):
    """
    Qram protocol over time sweep
    Experimental Config
    expt = dict(
        start: start protocol time [us],
        step: time step,
        expts: number of different time experiments,
        reps: number of reps per time step,
        # tomo: True/False whether to do state tomography on state at last time step
        # tomo_qubits: the qubits on which to do the 2q state tomo
        measure_f: None or [] of qubits to measure the f state on - should only be either Q2/Q3 or Q1
        singleshot_reps: reps per state for singleshot calibration
        post_process: 'threshold' (uses single shot binning), 'scale' (scale by ge_avgs), or None
        thresholds: (optional) don't rerun singleshot and instead use this
        ge_avgs: (optional) don't rerun singleshot and instead use this
        angles: (optional) don't rerun singleshot and instead use this
        counts_calib: (optional) don't rerun singleshot and instead use this
    )
    """

    def __init__(self, soccfg=None, path="", prefix="QramProtocol4QTomo", config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)
        self.cfg.all_qubits = [0, 1, 2, 3]

    def acquire(self, progress=False, debug=True):
        # expand entries in config that are length 1 to fill all qubits
        self.num_qubits_sample = len(self.cfg.device.readout.frequency)
        qubits = [0, 1, 2, 3]
        # qubits = self.cfg.expt.tomo_qubits
        # qA, qB, qC, qD = qubits
        self.cfg.expt.tomo_qubits = qubits

        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not (isinstance(value3, list)):
                                value2.update({key3: [value3] * self.num_qubits_sample})
                elif not (isinstance(value, list)):
                    subcfg.update({key: [value] * self.num_qubits_sample})

        timesteps = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        print("timesteps", timesteps)

        data = {
            "xpts": [],
            "gpop": [[], [], [], []],
            "epop": [[], [], [], []],
            "fpop": [[], [], [], []],
        }

        self.meas_order = make_4q_meas_order()
        self.calib_order = make_4q_calib_order()  # should match with order of counts for each tomography measurement
        measure_f_qubits = self.cfg.expt.measure_f

        # self.tomo_qubits = self.cfg.expt.tomo_qubits
        # if 'post_select' in self.cfg.expt and self.cfg.expt.post_select: data.update({'counts_tomo_ps0':[], 'counts_tomo_ps1':[],'counts_calib':[]})
        # else: data.update({'counts_tomo':[], 'counts_calib':[]})
        if self.cfg.expt.post_process == "threshold" and measure_f_qubits is not None:
            data.update({"counts_calib": [], "counts_raw": [[], []]})
        else:
            data.update({"counts_calib": [], "counts_raw": [[]]})

        # ================= #
        # Get single shot calibration for qubits
        # ================= #

        # Error mitigation measurements: prep in gg, ge, eg, ee to recalibrate measurement angle and measure confusion matrix
        if (
            "angles" in self.cfg.expt
            and "thresholds" in self.cfg.expt
            and "ge_avgs" in self.cfg.expt
            and "counts_calib" in self.cfg.expt
        ):
            angles_q = self.cfg.expt.angles
            thresholds_q = self.cfg.expt.thresholds
            ge_avgs_q = np.asarray(self.cfg.expt.ge_avgs)
            data["counts_calib"] = self.cfg.expt.counts_calib
            if progress:
                print("Re-using provided angles, thresholds, ge_avgs")
        else:
            calib_prog_dict = dict()
            for prep_state in tqdm(self.calib_order):
                # print(prep_state)
                cfg = AttrDict(deepcopy(self.cfg))
                cfg.expt.reps = self.cfg.expt.singleshot_reps
                cfg.expt.state_prep_kwargs = dict(prep_state=prep_state)
                err_tomo = ErrorMitigationStateTomo4QProgram(soccfg=self.soccfg, cfg=cfg)
                err_tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                calib_prog_dict.update({prep_state: err_tomo})

            g_prog = calib_prog_dict["gggg"]
            Ig, Qg = g_prog.get_shots(verbose=False)
            thresholds_q = [0] * self.num_qubits_sample
            angles_q = [0] * self.num_qubits_sample
            ge_avgs_q = [0] * self.num_qubits_sample
            for iq, q in enumerate(qubits):
                state = "gggg"
                state = state[:iq] + "e" + state[iq + 1 :]
                e_prog = calib_prog_dict[state]
                Ie, Qe = e_prog.get_shots(verbose=False)
                shot_data = dict(Ig=Ig[q], Qg=Qg[q], Ie=Ie[q], Qe=Qe[q])
                print(f"Qubit  ({q})")
                fid, threshold, angle = hist(data=shot_data, plot=debug, verbose=False)
                thresholds_q[q] = threshold[0]
                angles_q[q] = angle
                ge_avgs_q[q] = [np.average(Ig[q]), np.average(Qg[q]), np.average(Ie[q]), np.average(Qe[q])]

            print("thresholds", thresholds_q)
            print("angles", angles_q)
            print(f"ge_avgs={ge_avgs_q}")

            # Process the shots taken for the confusion matrix with the calibration angles
            for prep_state in self.calib_order:
                counts = calib_prog_dict[prep_state].collect_counts(angle=angles_q, threshold=thresholds_q)
                data["counts_calib"].append(counts)
            print(f"counts_calib={np.array(data['counts_calib']).tolist()}")

        # ================= #
        # Begin protocol stepping
        # ================= #

        adc_chs = self.cfg.hw.soc.adcs.readout.ch
        self.pulse_dict = dict()
        if "post_select" not in self.cfg.expt:
            self.cfg.expt.post_select = False

        # NO TOMO SUPPORTED
        self.cfg.expt.basis = "ZZZZ"

        gpop_q_times = np.zeros(shape=(4, len(timesteps)))
        epop_q_times = np.zeros(shape=(4, len(timesteps)))
        fpop_q_times = np.zeros(shape=(4, len(timesteps)))
        for time_i, timestep in enumerate(tqdm(timesteps, disable=not progress)):
            self.cfg.expt.timestep = float(timestep)

            # With standard pulse setup, measure the g population
            cfg = deepcopy(self.cfg)
            cfg.expt.ge_pulse = None
            protocol_prog_g = QramProtocol4QProgram(soccfg=self.soccfg, cfg=cfg)
            popln_g, avgq = protocol_prog_g.acquire_rotated(
                soc=self.im[self.cfg.aliases.soc],
                progress=False,
                angle=angles_q,
                threshold=thresholds_q,
                ge_avgs=ge_avgs_q,
                post_process=self.cfg.expt.post_process,
            )
            if measure_f_qubits is not None:
                # Add a ge pulse, measure the g population, which is really measuring the e population
                cfg.expt.ge_pulse = (
                    measure_f_qubits  # assumes the qubit(s) in measure_f_qubits have minimal ZZ to each other
                )
                protocol_prog_e = QramProtocol4QProgram(soccfg=self.soccfg, cfg=cfg)
                popln_e, avgq = protocol_prog_e.acquire_rotated(
                    soc=self.im[self.cfg.aliases.soc],
                    progress=False,
                    angle=angles_q,
                    threshold=thresholds_q,
                    ge_avgs=ge_avgs_q,
                    post_process=self.cfg.expt.post_process,
                )

            gpop_q = gpop_q_times[:, time_i]
            epop_q = epop_q_times[:, time_i]
            fpop_q = fpop_q_times[:, time_i]

            # Scaling post process
            if self.cfg.expt.post_process == "scale":
                for q in range(4):
                    gpop_q[q] = 1 - popln_g[adc_chs[q]]
                    epop_q[q] = popln_g[
                        adc_chs[q]
                    ]  # this is the final answer if we don't care about distinguishing e/f
                if measure_f_qubits is not None:
                    # if we care about distinguishing e/f, the "g" popln of the 2nd experiment is the real e popln, and the real f popln is whatever is left
                    for q in measure_f_qubits:
                        epop_q[q] = 1 - popln_e[adc_chs[q]]  # e population shows up as g population
                        fpop_q[q] = 1 - epop_q[q] - gpop_q[q]
                        # print(q, gpop_q[q], epop_q[q], fpop_q[q])

            # Threshold post process
            elif self.cfg.expt.post_process == "threshold":
                # Need to re-do population counts using the readout matrix correction

                counts_g = protocol_prog_g.collect_counts(angle=angles_q, threshold=thresholds_q)
                data["counts_raw"][0].append(counts_g)

                counts_e = None
                if measure_f_qubits is not None:
                    # if we care about distinguishing e/f, the "g" popln of the 2nd experiment is the real e popln, and the real f popln is whatever is left
                    for q in qubits:
                        epop_q[q] = 0  # reset this to recalculate e population

                    counts_e = protocol_prog_e.collect_counts(angle=angles_q, threshold=thresholds_q)
                    data["counts_raw"][1].append(counts_e)

                gpop_q, epop_q, fpop_q = infer_gef_popln(
                    counts1=counts_g,
                    counts2=counts_e,
                    qubits=range(4),
                    calib_order=self.calib_order,
                    measure_f_qubits=measure_f_qubits,
                    counts_calib=data["counts_calib"],
                    fix_neg_counts_flag=True,
                )

                gpop_q_times[:, time_i] = gpop_q
                epop_q_times[:, time_i] = epop_q
                fpop_q_times[:, time_i] = fpop_q

            data["gpop"] = gpop_q_times
            data["epop"] = epop_q_times
            data["fpop"] = fpop_q_times

            data["xpts"].append(float(timestep))

        if self.cfg.expt.expts > 1:
            data["end_times"] = protocol_prog_g.end_times_us
            print("end times", protocol_prog_g.end_times_us)

        for k, a in data.items():
            # print(k)
            # print(a)
            data[k] = np.array(a)

        self.data = data

        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        if self.cfg.expt.post_process == "threshold":
            timesteps = data["xpts"]
            gpop_q_times = np.zeros(shape=(4, len(timesteps)))
            epop_q_times = np.zeros(shape=(4, len(timesteps)))
            fpop_q_times = np.zeros(shape=(4, len(timesteps)))
            for time_i in range(len(timesteps)):
                counts_g = data["counts_raw"][0][time_i]
                counts_e = None
                if np.shape(data["counts_raw"])[0] == 2:
                    counts_e = data["counts_raw"][1][time_i]

                gpop_q, epop_q, fpop_q = infer_gef_popln(
                    counts1=counts_g,
                    counts2=counts_e,
                    qubits=range(4),
                    calib_order=self.calib_order,
                    measure_f_qubits=self.cfg.expt.measure_f,
                    counts_calib=data["counts_calib"],
                    fix_neg_counts_flag=True,
                )

                gpop_q_times[:, time_i] = gpop_q
                epop_q_times[:, time_i] = epop_q
                fpop_q_times[:, time_i] = fpop_q

            data["gpop"] = gpop_q_times
            data["epop"] = epop_q_times
            data["fpop"] = fpop_q_times
        return data

    def display(self, data=None, err=True, saveplot=False, **kwargs):
        if data is None:
            data = self.data

        xpts_ns = data["xpts"] * 1e3

        if self.cfg.expt.post_process == "threshold" or self.cfg.expt.post_process == "scale":
            plt.figure(figsize=(14, 8))
            if saveplot:
                plt.style.use("dark_background")
            plt_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            # plt.title(f"Qram Protocol", fontsize=20)

            for q in range(4):
                plt.plot(
                    xpts_ns,
                    data["epop"][q],
                    "-",
                    marker=marker_cycle[q],
                    markersize=8,
                    color=default_colors[q],
                    label=f"Q{q} e",
                )
                if self.cfg.expt.measure_f is not None and q in self.cfg.expt.measure_f:
                    plt.plot(
                        xpts_ns,
                        data["fpop"][q],
                        "--",
                        marker=marker_cycle[q],
                        markersize=8,
                        color=default_colors[q],
                        label=f"Q{q} f",
                    )

            if "end_times" in data:
                end_times = data["end_times"]
                for end_time in end_times:
                    plt.axvline(1e3 * end_time, color="0.4", linestyle="--")

            if self.cfg.expt.post_process == "threshold":
                plt.ylim(-0.02, 1.02)
            plt.legend(fontsize=26)
            plt.xlabel("Time [ns]", fontsize=26)
            plt.ylabel("Population", fontsize=26)
            plt.tick_params(labelsize=24)
            # plt.grid(linewidth=0.3)

        else:
            plt.figure(figsize=(14, 20))
            plt.subplot(421, title=f"Qubit 0", ylabel="I [adc level]")
            plt.plot(xpts_ns, data["avgi"][0], "o-")
            plt.subplot(422, title=f"Qubit 0", ylabel="Q [adc level]")
            plt.plot(xpts_ns, data["avgq"][0], "o-")

            plt.subplot(423, title=f"Qubit 1", ylabel="I [adc level]")
            plt.plot(xpts_ns, data["avgi"][1], "o-")
            plt.subplot(424, title=f"Qubit 1", ylabel="Q [adc level]")
            plt.plot(xpts_ns, data["avgq"][1], "o-")

            plt.subplot(425, title=f"Qubit 2", ylabel="I [adc level]")
            plt.plot(xpts_ns, data["avgi"][2], "o-")
            plt.subplot(426, title=f"Qubit 2", ylabel="Q [adc level]")
            plt.plot(xpts_ns, data["avgq"][2], "o-")

            plt.subplot(427, title=f"Qubit 3", xlabel="Time [ns]", ylabel="I [adc level]")
            plt.plot(xpts_ns, data["avgi"][3], "o-")
            plt.subplot(428, title=f"Qubit 3", xlabel="Time [ns]", ylabel="Q [adc level]")
            plt.plot(xpts_ns, data["avgq"][3], "o-")

        plt.tight_layout()

        if saveplot:
            plot_filename = "qram_protocol.png"
            plt.savefig(plot_filename, format="png", bbox_inches="tight", transparent=True)
            print("Saved", plot_filename)

        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        self.cfg.all_qubits = [1, 2, 3, 4]
        super().save_data(data=data)
        # print(self.pulse_dict)
        with self.datafile() as f:
            f.attrs["pulse_dict"] = json.dumps(self.pulse_dict, cls=NpEncoder)
            f.attrs["meas_order"] = json.dumps(self.meas_order, cls=NpEncoder)
            f.attrs["calib_order"] = json.dumps(self.calib_order, cls=NpEncoder)
        # return self.fname
