from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from slab import AttrDict, Experiment, dsfit
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter
from experiments.clifford_averager_program import QutritRAveragerProgram, flat_top_pulse


class AmplitudeRabiEgGfProgram(QutritRAveragerProgram):
    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        self.num_qubits_sample = len(self.cfg.device.readout.frequency)
        self.qubits = self.cfg.expt.qubits
        qA, qB = self.qubits

        qSort = qA
        if qA == 1:
            qSort = qB
        qDrive = 1
        if "qDrive" in self.cfg.expt and self.cfg.expt.qDrive is not None:
            qDrive = self.cfg.expt.qDrive
        qNotDrive = -1
        if qA == qDrive:
            qNotDrive = qB
        else:
            qNotDrive = qA

        self.qDrive = qDrive
        self.qNotDrive = qNotDrive
        self.qSort = qSort
        # print('qDrive', qDrive)
        # print('qNotDrive', qNotDrive)
        # print('qSort', qSort)

        self.test_pi_half = "test_pi_half" in self.cfg.expt and self.cfg.expt.test_pi_half
        self.tot_length_test_us = self.cfg.expt.pi_EgGf_sigma

        super().initialize()

        # self.swap_chs = self.cfg.hw.soc.dacs.swap.ch
        # self.swap_ch_types = self.cfg.hw.soc.dacs.swap.type
        self.swap_Q_chs = self.cfg.hw.soc.dacs.swap_Q.ch
        self.swap_Q_ch_types = self.cfg.hw.soc.dacs.swap_Q.type

        name = "X_EgGf"
        waveformname = "pi_EgGf_Q"
        assert qDrive != 1
        if self.test_pi_half:
            f_EgGf_MHz = self.cfg.device.qubit.f_EgGf_Q_half[qDrive]
            gain = self.cfg.device.qubit.pulses.pi_EgGf_Q.half_gain[qDrive]
            name += "_half"
            waveformname = "_half"
        else:
            f_EgGf_MHz = self.cfg.device.qubit.f_EgGf_Q[qDrive]
            gain = self.cfg.device.qubit.pulses.pi_EgGf_Q.gain[qDrive]
        phase_deg = 0
        ch = self.swap_Q_chs[qDrive]

        name = f"{name}_{qNotDrive}{qDrive}"
        # t_rise_us = self.cycles2us(3, gen_ch=ch)
        # t_rise_us = 0.003
        self.handle_flat_top_pulse(
            name=name,
            ch=ch,
            waveformname=f"{waveformname}_{qNotDrive}{qDrive}",
            # t_rise_us=t_rise_us,
            # sigma_n=sigma_n,
            tot_length_us=self.tot_length_test_us,
            freq_MHz=f_EgGf_MHz,
            phase_deg=phase_deg,
            gain=gain,
            play=False,
            set_reg=True,
            # plot_IQ=True,
        )
        self.swap_name = name

        # get gain register for swap ch
        assert self.swap_Q_ch_types[qSort] == "full"
        self.r_gain_swap = self.sreg(self.swap_Q_chs[qSort], "gain")
        # register to hold the current sweep gain
        self.r_gain_swap_update = 4
        # initialize gain
        self.safe_regwi(self.ch_page(self.swap_Q_chs[qSort]), self.r_gain_swap_update, self.cfg.expt.start)
        # if (
        #     cfg.expt.pulse_type.lower() == "flat_top"
        # ):  # the gain for the const section of a flat top pulse needs to be set to 2*the ramp gain
        #     self.r_gain_swap_const = self.sreg(self.swap_Q_chs[qSort], "gain2")
        #     self.r_gain_swap_update_const = 5
        #     self.safe_regwi(
        #         self.ch_page(self.swap_Q_chs[qSort]), self.r_gain_swap_update_const, self.cfg.expt.start // 2
        #     )

        self.sync_all(200)

    def body(self):
        cfg = AttrDict(self.cfg)
        qDrive = self.qDrive
        qNotDrive = self.qNotDrive
        qSort = self.qSort

        self.reset_and_sync()

        if "cool_qubits" in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            cool_idle = [self.cfg.device.qubit.pulses.pi_f0g1.idle[q] for q in self.cfg.expt.cool_qubits]
            if "cool_idle" in self.cfg.expt and self.cfg.expt.cool_idle is not None:
                cool_idle = self.cfg.expt.cool_idle
            self.active_cool(cool_qubits=self.cfg.expt.cool_qubits, cool_idle=cool_idle)

        self.use_gf_readout = None
        if "use_gf_readout" in self.cfg.expt and self.cfg.expt.use_gf_readout:
            self.use_gf_readout = self.cfg.expt.use_gf_readout

        if self.readout_cool:
            self.measure_readout_cool()

        setup_ZZ = self.cfg.expt.setup_ZZ
        assert setup_ZZ in [None, 0]
        if setup_ZZ is None:
            setup_ZZ = qNotDrive

        if setup_ZZ != qNotDrive:
            self.X_pulse(q=setup_ZZ, play=True)
        self.X_pulse(q=qNotDrive, ZZ_qubit=setup_ZZ, play=True)
        self.sync_all()

        # apply Eg -> Gf pulse on B: expect to end in Gf
        if self.tot_length_test_us > 0:
            self.handle_flat_top_pulse(name=self.swap_name, play=False, set_reg=True)
            self.mathi(self.ch_page(self.swap_Q_chs[qSort]), self.r_gain_swap, self.r_gain_swap_update, "+", 0)
            n_pulse = 1
            if "test_pi_half" in self.cfg.expt and self.cfg.expt.test_pi_half:
                n_pulse = 2
            for i in range(n_pulse):
                self.pulse(ch=self.swap_Q_chs[qSort])
        self.sync_all()

        # take qubit B f->e: expect to end in Ge (or Eg if incomplete Eg-Gf)
        self.Xef_pulse(q=qDrive, play=True)

        self.sync_all()
        self.measure(
            pulse_ch=self.measure_chs,
            adcs=self.adc_chs,
            adc_trig_offset=cfg.device.readout.trig_offset[0],
            wait=True,
            syncdelay=self.us2cycles(max([cfg.device.readout.relax_delay[q] for q in self.qubits])),
        )

    def update(self):
        qA, qB = self.qubits
        qDrive = self.qDrive
        qNotDrive = self.qNotDrive
        qSort = self.qSort
        step = self.cfg.expt.step
        if self.swap_Q_ch_types[qSort] == "int4":
            step = step << 16
        self.mathi(
            self.ch_page(self.swap_Q_chs[qSort]), self.r_gain_swap_update, self.r_gain_swap_update, "+", step
        )  # update test gain
        # if (
        #     self.cfg.expt.pulse_type.lower() == "flat_top"
        # ):  # the gain for the const section of a flat top pulse needs to be set to 2*the ramp gain
        #     self.mathi(
        #         self.ch_page(self.swap_Q_chs[qSort]),
        #         self.r_gain_swap_update_const,
        #         self.r_gain_swap_update_const,
        #         "+",
        #         step // 2,
        #     )  # update test gain


class AmplitudeRabiEgGfExperiment(Experiment):
    """
    Amplitude Rabi Eg<->Gf Experiment
    Experimental Config:
    expt = dict(
        start: qubit gain [dac units]
        step: gain step [dac units]
        expts: number steps
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        pi_EgGf_sigma: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'const'
        qubits: qubit 0 goes E->G, apply drive on qubit 1 (g->f)
        singleshot: (optional) if true, uses threshold
    )
    """

    def __init__(self, soccfg=None, path="", prefix="AmplitudeRabiEgGf", config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        qA, qB = self.cfg.expt.qubits

        qSort = qA
        if qA == 1:  # convention is to reorder the indices so qA is the differentiating index, qB is 1
            qSort = qB
        self.qDrive = 1
        if "qDrive" in self.cfg.expt and self.cfg.expt.qDrive is not None:
            self.qDrive = self.cfg.expt.qDrive
        qDrive = self.qDrive

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

        adcA_ch = self.cfg.hw.soc.adcs.readout.ch[qA]
        adcB_ch = self.cfg.hw.soc.adcs.readout.ch[qB]

        test_pi_half = False
        if "test_pi_half" in self.cfg.expt and self.cfg.expt.test_pi_half:
            test_pi_half = self.cfg.expt.test_pi_half
        assert qDrive != 1

        if "pi_EgGf_sigma" not in self.cfg.expt:
            if test_pi_half:
                self.cfg.expt.pi_EgGf_sigma = self.cfg.device.qubit.pulses.pi_EgGf_Q_half.sigma[qSort]
            else:
                self.cfg.expt.pi_EgGf_sigma = self.cfg.device.qubit.pulses.pi_EgGf_Q.sigma[qSort]

        threshold = None
        angle = None
        if "singleshot" in self.cfg.expt.keys():
            if self.cfg.expt.singleshot:
                threshold = self.cfg.device.readout.threshold
                # angle = self.cfg.device.readout.phase

        amprabi = AmplitudeRabiEgGfProgram(soccfg=self.soccfg, cfg=self.cfg)
        x_pts, avgi, avgq = amprabi.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=progress)
        self.prog = amprabi

        data = dict(
            xpts=x_pts,
            avgi=(avgi[adcA_ch][0], avgi[adcB_ch][0]),
            avgq=(avgq[adcA_ch][0], avgq[adcB_ch][0]),
            amps=(np.abs(avgi[adcA_ch][0] + 1j * avgq[adcA_ch][0]), np.abs(avgi[adcB_ch][0] + 1j * avgq[adcB_ch][0])),
            phases=(
                np.angle(avgi[adcA_ch][0] + 1j * avgq[adcA_ch][0]),
                np.angle(avgi[adcB_ch][0] + 1j * avgq[adcB_ch][0]),
            ),
        )
        self.data = data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), amp offset]
            # Remove the first and last point from fit in case weird edge measurements
            # fitparams = [None, 1/max(data['xpts']), None, None]
            fitparams = None
            pA_avgi, pCovA_avgi = fitter.fitsin(data["xpts"][:-1], data["avgi"][0][:-1], fitparams=fitparams)
            pA_avgq, pCovA_avgq = fitter.fitsin(data["xpts"][:-1], data["avgq"][0][:-1], fitparams=fitparams)
            pA_amps, pCovA_amps = fitter.fitsin(data["xpts"][:-1], data["amps"][0][:-1], fitparams=fitparams)
            data["fitA_avgi"] = pA_avgi
            data["fitA_avgq"] = pA_avgq
            data["fitA_amps"] = pA_amps
            data["fitA_err_avgi"] = pCovA_avgi
            data["fitA_err_avgq"] = pCovA_avgq
            data["fitA_err_amps"] = pCovA_amps

            pB_avgi, pCovB_avgi = fitter.fitsin(data["xpts"][:-1], data["avgi"][1][:-1], fitparams=fitparams)
            pB_avgq, pCovB_avgq = fitter.fitsin(data["xpts"][:-1], data["avgq"][1][:-1], fitparams=fitparams)
            pB_amps, pCovB_amps = fitter.fitsin(data["xpts"][:-1], data["amps"][1][:-1], fitparams=fitparams)
            data["fitB_avgi"] = pB_avgi
            data["fitB_avgq"] = pB_avgq
            data["fitB_amps"] = pB_amps
            data["fitB_err_avgi"] = pCovB_avgi
            data["fitB_err_avgq"] = pCovB_avgq
            data["fitB_err_amps"] = pCovB_amps
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        # plt.figure(figsize=(20,6))
        # plt.suptitle(f"Amplitude Rabi Eg-Gf (Drive Length {self.cfg.expt.pi_EgGf_sigma} us)")
        # plt.subplot(121, title=f'Qubit A ({self.cfg.expt.qubits[0]})', ylabel="Amplitude [adc units]", xlabel='Gain [DAC units]')
        # plt.plot(data["xpts"][0:-1], data["amps"][0][0:-1],'.-')
        # if fit:
        #     p = data['fitA_amps']
        #     plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
        #     if p[2] > 180: p[2] = p[2] - 360
        #     elif p[2] < -180: p[2] = p[2] + 360
        #     if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
        #     else: pi_gain= (3/2 - p[2]/180)/2/p[1]
        #     pi2_gain = pi_gain/2
        #     print(f'Pi gain from amps data (qubit A) [dac units]: {int(pi_gain)}')
        #     print(f'\tPi/2 gain from amps data (qubit A) [dac units]: {int(pi2_gain)}')
        #     plt.axvline(pi_gain, color='0.2', linestyle='--')
        #     plt.axvline(pi2_gain, color='0.2', linestyle='--')
        # plt.subplot(122, title=f'Qubit B ({self.cfg.expt.qubits[1]})', xlabel='Gain [DAC units]')
        # plt.plot(data["xpts"][0:-1], data["amps"][1][0:-1],'.-')
        # if fit:
        #     p = data['fitB_amps']
        #     plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
        #     if p[2] > 180: p[2] = p[2] - 360
        #     elif p[2] < -180: p[2] = p[2] + 360
        #     if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
        #     else: pi_gain= (3/2 - p[2]/180)/2/p[1]
        #     pi2_gain = pi_gain/2
        #     print(f'Pi gain from amps data (qubit B) [dac units]: {int(pi_gain)}')
        #     print(f'\tPi/2 gain from amps data (qubit B) [dac units]: {int(pi2_gain)}')
        #     plt.axvline(pi_gain, color='0.2', linestyle='--')
        #     plt.axvline(pi2_gain, color='0.2', linestyle='--')

        plt.figure(figsize=(9, 7))
        plt.suptitle(f"Amplitude Rabi Eg-Gf (Drive Length {self.cfg.expt.pi_EgGf_sigma} us)")
        if self.cfg.expt.singleshot:
            plt.subplot(221, title=f"Qubit A ({self.cfg.expt.qubits[0]})", ylabel=r"Probability of $|e\rangle$")
        else:
            plt.subplot(221, title=f"Qubit A ({self.cfg.expt.qubits[0]})", ylabel="I [ADC units]")
        plt.plot(data["xpts"][0:-1], data["avgi"][0][0:-1], ".-")
        if fit:
            p = data["fitA_avgi"]
            plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
            if p[2] > 180:
                p[2] = p[2] - 360
            elif p[2] < -180:
                p[2] = p[2] + 360
            if p[2] < 0:
                pi_gain = (1 / 2 - p[2] / 180) / 2 / p[1]
            else:
                pi_gain = (3 / 2 - p[2] / 180) / 2 / p[1]
            pi2_gain = pi_gain / 2
            print(f"Pi gain from avgi data (qubit A) [dac units]: {int(pi_gain)}")
            print(f"\tPi/2 gain from avgi data (qubit A) [dac units]: {int(pi2_gain)}")
            plt.axvline(pi_gain, color="0.2", linestyle="--")
            plt.axvline(pi2_gain, color="0.2", linestyle="--")
        plt.subplot(223, xlabel="Gain [DAC units]", ylabel="Q [ADC units]")
        plt.plot(data["xpts"][0:-1], data["avgq"][0][0:-1], ".-")
        if fit:
            p = data["fitA_avgq"]
            plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
            if p[2] > 180:
                p[2] = p[2] - 360
            elif p[2] < -180:
                p[2] = p[2] + 360
            if p[2] < 0:
                pi_gain = (1 / 2 - p[2] / 180) / 2 / p[1]
            else:
                pi_gain = (3 / 2 - p[2] / 180) / 2 / p[1]
            pi2_gain = pi_gain / 2
            print(f"Pi gain from avgq data (qubit A) [dac units]: {int(pi_gain)}")
            print(f"\tPi/2 gain from avgq data (qubit A) [dac units]: {int(pi2_gain)}")
            plt.axvline(pi_gain, color="0.2", linestyle="--")
            plt.axvline(pi2_gain, color="0.2", linestyle="--")

        plt.subplot(222, title=f"Qubit B ({self.cfg.expt.qubits[1]})")
        plt.plot(data["xpts"][0:-1], data["avgi"][1][0:-1], ".-")
        if fit:
            p = data["fitB_avgi"]
            plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
            if p[2] > 180:
                p[2] = p[2] - 360
            elif p[2] < -180:
                p[2] = p[2] + 360
            if p[2] < 0:
                pi_gain = (1 / 2 - p[2] / 180) / 2 / p[1]
            else:
                pi_gain = (3 / 2 - p[2] / 180) / 2 / p[1]
            pi2_gain = pi_gain / 2
            print(f"Pi gain from avgi data (qubit B) [dac units]: {int(pi_gain)}")
            print(f"\tPi/2 gain from avgi data (qubit B) [dac units]: {int(pi2_gain)}")
            plt.axvline(pi_gain, color="0.2", linestyle="--")
            plt.axvline(pi2_gain, color="0.2", linestyle="--")
        plt.subplot(224, xlabel="Gain [DAC units]")
        plt.plot(data["xpts"][0:-1], data["avgq"][1][0:-1], ".-")
        if fit:
            p = data["fitB_avgq"]
            plt.plot(data["xpts"][0:-1], fitter.sinfunc(data["xpts"][0:-1], *p))
            if p[2] > 180:
                p[2] = p[2] - 360
            elif p[2] < -180:
                p[2] = p[2] + 360
            if p[2] < 0:
                pi_gain = (1 / 2 - p[2] / 180) / 2 / p[1]
            else:
                pi_gain = (3 / 2 - p[2] / 180) / 2 / p[1]
            pi2_gain = pi_gain / 2
            print(f"Pi gain from avgq data (qubit B) [dac units]: {int(pi_gain)}")
            print(f"\tPi/2 gain from avgq data (qubit B) [dac units]: {int(pi2_gain)}")
            plt.axvline(pi_gain, color="0.2", linestyle="--")
            plt.axvline(pi2_gain, color="0.2", linestyle="--")
        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)


# ===================================================================== #


class EgGfLenGainChevronExperiment(Experiment):
    """
    Rabi Eg<->Gf Experiment Chevron sweeping length vs. gain
    Experimental Config:
    expt = dict(
        start_gain: qubit gain [dac units]
        step_gain: gain step [dac units]
        expts_gain: number steps
        start_len: start length [us],
        step_len: length step,
        expts_len: number of different length experiments,
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(self, soccfg=None, path="", prefix="RabiEgGfLenGainChevron", config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        qA, qB = self.cfg.expt.qubits

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

        adc_chs = self.cfg.hw.soc.adcs.readout.ch

        lenpts = self.cfg.expt.start_len + self.cfg.expt.step_len * np.arange(self.cfg.expt.expts_len)
        gainpts = self.cfg.expt.start_gain + self.cfg.expt.step_gain * np.arange(self.cfg.expt.expts_gain)

        data = {
            "gainpts": gainpts,
            "lenpts": lenpts,
            "avgi": [[], []],
            "avgq": [[], []],
            "amps": [[], []],
            "phases": [[], []],
        }

        self.cfg.expt.start = self.cfg.expt.start_gain
        self.cfg.expt.step = self.cfg.expt.step_gain
        self.cfg.expt.expts = self.cfg.expt.expts_gain

        threshold = None
        angle = None

        for length in tqdm(lenpts, disable=not progress):
            self.cfg.expt.pi_EgGf_sigma = float(length)

            amprabi = AmplitudeRabiEgGfProgram(soccfg=self.soccfg, cfg=self.cfg)
            gainpts, avgi, avgq = amprabi.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)

            for q_ind, q in enumerate(self.cfg.expt.qubits):
                data["avgi"][q_ind].append(avgi[adc_chs[q], 0])
                data["avgq"][q_ind].append(avgq[adc_chs[q], 0])
                data["amps"][q_ind].append(np.abs(avgi[adc_chs[q], 0] + 1j * avgi[adc_chs[q], 0]))
                data["phases"][q_ind].append(np.angle(avgi[adc_chs[q], 0] + 1j * avgi[adc_chs[q], 0]))

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

        inner_sweep = data["gainpts"]
        outer_sweep = data["lenpts"]

        y_sweep = outer_sweep
        x_sweep = inner_sweep

        plt.figure(figsize=(9, 7))
        plt.suptitle(f"Eg-Gf Chevron Length vs. Gain")

        plt.subplot(221, title=f"Qubit A ({self.cfg.expt.qubits[0]})", ylabel="Pulse Length [us]")
        plt.pcolormesh(x_sweep, y_sweep, data["avgi"][0], cmap="viridis", shading="auto")
        plt.colorbar(label="I [ADC level]")

        plt.subplot(223, xlabel="Gain [DAC units]", ylabel="Pulse Length [us]")
        plt.pcolormesh(x_sweep, y_sweep, data["avgq"][0], cmap="viridis", shading="auto")
        plt.colorbar(label="Q [ADC level]")

        plt.subplot(222, title=f"Qubit B ({self.cfg.expt.qubits[1]})")
        plt.pcolormesh(x_sweep, y_sweep, data["avgi"][1], cmap="viridis", shading="auto")
        plt.colorbar(label="I [ADC level]")

        plt.subplot(224, xlabel="Gain [DAC units]")
        plt.pcolormesh(x_sweep, y_sweep, data["avgq"][1], cmap="viridis", shading="auto")
        plt.colorbar(label="Q [ADC level]")

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)


# ===================================================================== #


class EgGfFreqGainChevronExperiment(Experiment):
    """
    Rabi Eg<->Gf Experiment Chevron sweeping freq vs. gain
    Experimental Config:
    expt = dict(
        start_gain: qubit gain [dac units]
        step_gain: gain step [dac units]
        expts_gain: number steps
        start_f: start freq [MHz],
        step_f: freq step,
        expts_f: number of different freq experiments,
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(self, soccfg=None, path="", prefix="RabiEgGfFreqGainChevron", config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        qA, qB = self.cfg.expt.qubits

        qSort = qA
        if qA == 1:  # convention is to reorder the indices so qA is the differentiating index, qB is 1
            qSort = qB
        self.qDrive = 1
        if "qDrive" in self.cfg.expt and self.cfg.expt.qDrive is not None:
            self.qDrive = self.cfg.expt.qDrive
        qDrive = self.qDrive

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

        adc_chs = self.cfg.hw.soc.adcs.readout.ch

        freqpts = self.cfg.expt.start_f + self.cfg.expt.step_f * np.arange(self.cfg.expt.expts_f)
        gainpts = self.cfg.expt.start_gain + self.cfg.expt.step_gain * np.arange(self.cfg.expt.expts_gain)

        data = {
            "gainpts": gainpts,
            "freqpts": freqpts,
            "avgi": [[], []],
            "avgq": [[], []],
            "amps": [[], []],
            "phases": [[], []],
        }

        self.cfg.expt.start = self.cfg.expt.start_gain
        self.cfg.expt.step = self.cfg.expt.step_gain
        self.cfg.expt.expts = self.cfg.expt.expts_gain

        test_pi_half = False
        if "test_pi_half" in self.cfg.expt and self.cfg.expt.test_pi_half:
            test_pi_half = self.cfg.expt.test_pi_half
        assert qDrive != 1

        if "pi_EgGf_sigma" not in self.cfg.expt:
            if test_pi_half:
                self.cfg.expt.pi_EgGf_sigma = self.cfg.device.qubit.pulses.pi_EgGf_Q_half.sigma[qSort]
            else:
                self.cfg.expt.pi_EgGf_sigma = self.cfg.device.qubit.pulses.pi_EgGf_Q.sigma[qSort]

        for freq in tqdm(freqpts, disable=not progress):
            if test_pi_half:
                self.cfg.device.qubit.f_EgGf_Q_half[qSort] = float(freq)
            else:
                self.cfg.device.qubit.f_EgGf_Q[qSort] = float(freq)

            amprabi = AmplitudeRabiEgGfProgram(soccfg=self.soccfg, cfg=self.cfg)
            gainpts, avgi, avgq = amprabi.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)

            for q_ind, q in enumerate(self.cfg.expt.qubits):
                data["avgi"][q_ind].append(avgi[adc_chs[q]][0])
                data["avgq"][q_ind].append(avgq[adc_chs[q]][0])
                data["amps"][q_ind].append(np.abs(avgi[adc_chs[q]][0] + 1j * avgq[adc_chs[q]][0]))
                data["phases"][q_ind].append(np.angle(avgi[adc_chs[q]][0] + 1j * avgq[adc_chs[q]][0]))

        for k, a in data.items():
            data[k] = np.array(a)
        self.data = data
        return data

    def analyze(self, data=None, fit=True, fitparams=None):
        if data is None:
            data = self.data

        data = deepcopy(data)
        inner_sweep = data["gainpts"]
        outer_sweep = data["freqpts"]

        y_sweep = outer_sweep  # index 0
        x_sweep = inner_sweep  # index 1

        length = self.cfg.expt.pi_EgGf_sigma

        # fitparams = [yscale, freq, phase_deg, y0]
        # fitparams=[None, 2/x_sweep[-1], None, None]
        for data_name in ["avgi", "avgq", "amps"]:
            data.update({f"fit{data_name}": [None] * len(self.cfg.expt.measure_qubits)})
            data.update({f"fit{data_name}_err": [None] * len(self.cfg.expt.measure_qubits)})
            data.update({f"data_fit{data_name}": [None] * len(self.cfg.expt.measure_qubits)})

            # Scale the amplitude and add them together to increase SNR
            fit_data = np.zeros_like(data[data_name][0])
            for q_index, q in enumerate(self.cfg.expt.measure_qubits):
                this_data = data[data_name][q_index]
                # fit_data += np.abs(this_data - np.median(this_data))  # / (np.max(this_data) - np.min(this_data))

                if q == self.cfg.expt.qDrive:
                    sign = 1
                else:
                    sign = -1
                fit_data += sign * (this_data - np.min(this_data)) / (np.max(this_data) - np.min(this_data))

                # fit = [None] * len(x_sweep)
                # fit_err = [None] * len(x_sweep)
                # data_fit = [None] * len(x_sweep)

                # for i_gain, gain in enumerate(x_sweep):
                #     p, pCov = fitter.fitrabi_gainslice(
                #         y_sweep, this_data[:, i_gain], length=length, fitparams=fitparams
                #     )
                #     fit[i_gain] = p
                #     fit_err[i_gain] = pCov
                #     data_fit[i_gain] = fitter.rabifunc(y_sweep, *p)

                # data[f"fit{data_name}"][q_index] = fit
                # data[f"fit{data_name}_err"][q_index] = fit_err
                # data[f"data_fit{data_name}"][q_index] = data_fit

            # Find opt point for each gain directly
            freq_opt = np.zeros(len(x_sweep))
            amp_opt = np.zeros(len(x_sweep))

            for idx, g in enumerate(x_sweep):
                idx_max = np.argmax(fit_data[:, idx])
                freq_opt[idx] = y_sweep[idx_max]
                amp_opt[idx] = fit_data[idx_max, idx]

            data["fit_data"] = fit_data
            data["fit_freq"] = freq_opt
            data["metric_at_fit_freq"] = amp_opt
            idx_max_metric = np.argmax(amp_opt)
            data["best_freq"] = freq_opt[idx_max_metric]
            data["best_gain"] = x_sweep[idx_max_metric]

        return data

    def display(
        self,
        data=None,
        fit=True,
        plot_freq=None,
        plot_gain=None,
        saveplot=False,
        range_start=0,
        range_end=-1,
        **kwargs,
    ):
        """
        range_start, range_end are gain indices to fit the chevron
        """
        if data is None:
            data = self.data

        inner_sweep = data["gainpts"]
        outer_sweep = data["freqpts"]

        y_sweep = outer_sweep
        x_sweep = inner_sweep

        if saveplot:
            plt.style.use("dark_background")
        plt.figure(figsize=(9, 7))
        plt.suptitle(f"Eg-Gf Chevron Frequency vs. Gain (Length {self.cfg.expt.pi_EgGf_sigma} us)")

        max_freq_i, max_gain_i = np.unravel_index(np.argmax(data["amps"][0], axis=None), data["amps"][0].shape)
        max_gain = data["gainpts"][max_gain_i]
        max_freq = data["freqpts"][max_freq_i]
        # print("QA: max at gain", data["gainpts"][max_gain_i], "freq", data["freqpts"][max_freq_i])

        if saveplot:
            plt.subplot(221, title=f"Qubit A ({self.cfg.expt.qubits[0]})")
            ax = plt.gca()
            ax.set_ylabel("Pulse Frequency [MHz]", fontsize=18)
            ax.tick_params(axis="both", which="major", labelsize=16)
        else:
            plt.subplot(221, title=f"Qubit A ({self.cfg.expt.qubits[0]})", ylabel="Pulse Frequency [MHz]")
        plt.pcolormesh(x_sweep, y_sweep, data["avgi"][0], cmap="viridis", shading="auto")
        if plot_freq is not None:
            plt.axhline(plot_freq, color="r")
        if plot_gain is not None:
            plt.axvline(plot_gain, color="r")
        if fit:
            plt.axhline(max_freq, color="k", linestyle="--")
            plt.axvline(max_gain, color="k", linestyle="--")
            print("QA: max at gain", max_gain, "freq", max_freq)
        if saveplot:
            plt.colorbar().set_label(label="$S_{21}$ [arb. units]", size=15)
        else:
            plt.colorbar(label="I [ADC level]")

        if saveplot:
            plt.subplot(223)
            ax = plt.gca()
            ax.set_ylabel("Pulse Frequency [MHz]", fontsize=18)
            ax.set_xlabel("Pulse Amplitude [arb. units]", fontsize=18)
            ax.tick_params(axis="both", which="major", labelsize=16)
        else:
            plt.subplot(223, xlabel="Gain [DAC units]", ylabel="Pulse Frequency [MHz]")
        plt.pcolormesh(x_sweep, y_sweep, data["avgq"][0], cmap="viridis", shading="auto")
        if plot_freq is not None:
            plt.axhline(plot_freq, color="r")
        if plot_gain is not None:
            plt.axvline(plot_gain, color="r")
        if fit:
            plt.axhline(max_freq, color="k", linestyle="--")
            plt.axvline(max_gain, color="k", linestyle="--")
            print("QA: max at gain", max_gain, "freq", max_freq)
        if saveplot:
            plt.colorbar().set_label(label="$S_{21}$ [arb. units]", size=15)
        else:
            plt.colorbar(label="Q [ADC level]")

        min_freq_i, min_gain_i = np.unravel_index(np.argmin(data["amps"][1], axis=None), data["amps"][1].shape)
        min_gain = data["gainpts"][min_gain_i]
        min_freq = data["freqpts"][min_freq_i]
        # print("QB: min at gain", data["gainpts"][min_gain_i], "freq", data["freqpts"][min_freq_i])

        plt.subplot(222, title=f"Qubit B ({self.cfg.expt.qubits[1]})")
        if saveplot:
            ax = plt.gca()
            ax.tick_params(axis="both", which="major", labelsize=16)
        plt.pcolormesh(x_sweep, y_sweep, data["avgi"][1], cmap="viridis", shading="auto")
        if plot_freq is not None:
            plt.axhline(plot_freq, color="r")
        if plot_gain is not None:
            plt.axvline(plot_gain, color="r")
        if fit:
            plt.axhline(min_freq, color="k", linestyle="--")
            plt.axvline(min_gain, color="k", linestyle="--")
            print("QB: min at gain", min_gain, "freq", min_freq)
        if saveplot:
            plt.colorbar().set_label(label="$S_{21}$ [arb. units]", size=15)
        else:
            plt.colorbar(label="I [ADC level]")

        if saveplot:
            plt.subplot(224)
            ax = plt.gca()
            ax.set_xlabel("Pulse Amplitude [arb. units]", fontsize=18)
            ax.tick_params(axis="both", which="major", labelsize=16)
        else:
            plt.subplot(224, xlabel="Gain [DAC units]")
        plt.pcolormesh(x_sweep, y_sweep, data["avgq"][1], cmap="viridis", shading="auto")
        if plot_freq is not None:
            plt.axhline(plot_freq, color="r")
        if plot_gain is not None:
            plt.axvline(plot_gain, color="r")
        if fit:
            plt.axhline(max_freq, color="k", linestyle="--")
            plt.axvline(max_gain, color="k", linestyle="--")
            print("QB: max at gain", max_gain, "freq", max_freq)
        if saveplot:
            plt.colorbar().set_label(label="$S_{21}$ [arb. units]", size=15)
        else:
            plt.colorbar(label="Q [ADC level]")

        plt.tight_layout()

        if saveplot:
            plot_filename = f"gain_freq_chevron_EgGf{self.cfg.expt.qubits[0]}{self.cfg.expt.qubits[1]}.png"
            plt.savefig(plot_filename, format="png", bbox_inches="tight", transparent=True)
            print("Saved", plot_filename)

        plt.show()

        if not fit:
            return

        """
        Plot fit chevron
        """
        fit_xsweep_set = x_sweep[range_start:range_end]

        # Output data: fit{data_name}: (len(measure_qubits), len(x_sweep), len(fitparams))
        # fit_freqs = np.array([f[2] if f is not None else np.nan for f in data["fitamps"][0]])
        fit_freqs = data["fit_freq"]
        fit_freqs_set = fit_freqs[range_start:range_end]
        print("fit gains set", fit_xsweep_set)
        print("fit freqs set", fit_freqs_set)
        fitparams = [
            fit_xsweep_set[0],
            fit_freqs_set[0],
            (fit_freqs_set[-1] - fit_freqs_set[0]) / (fit_xsweep_set[-1] - fit_xsweep_set[0]) ** 2,
        ]
        # fitparams = None
        p, pCov = fitter.fitquadratic(fit_xsweep_set, fit_freqs_set, fitparams=fitparams)
        fit_freqs_fit = fitter.quadraticfunc(x_sweep, *p)
        print(f"fit_gain_sweep =", x_sweep.tolist())
        print(f"fit_freqs =", fit_freqs_fit.tolist())

        plt.figure(figsize=(9, 7))
        plt.suptitle(f"Fit Eg-Gf Chevron Frequency vs. Gain (Length {self.cfg.expt.pi_EgGf_sigma} us)")

        plt.subplot(
            221,
            title=f"Qubit A ({self.cfg.expt.qubits[0]})",
            xlabel="Gain [DAC units]",
            ylabel="Pulse Frequency [MHz]",
        )
        plt.pcolormesh(x_sweep, y_sweep, data["amps"][0], cmap="viridis", shading="auto")
        # print("WARNING PLOTTING SOMETHIGN DIFFERENT HERE")
        # plt.pcolormesh(x_sweep, y_sweep, data["fit_data"], cmap="viridis", shading="auto")
        if fit:
            plt.plot(x_sweep, fit_freqs, color="k", linestyle="-.")
            plt.plot(x_sweep, fit_freqs_fit, color="r", linestyle="--")
            plt.plot(data["best_gain"], data["best_freq"], "o", markersize=8, markeredgecolor="k", markerfacecolor="r")
        plt.colorbar(label="[ADC level]")

        plt.subplot(222, title=f"Qubit B ({self.cfg.expt.qubits[1]})", xlabel="Gain [DAC units]")
        plt.pcolormesh(x_sweep, y_sweep, data["amps"][1], cmap="viridis", shading="auto")
        if fit:
            plt.plot(x_sweep, fit_freqs, color="k", linestyle="-.")
            plt.plot(x_sweep, fit_freqs_fit, color="r", linestyle="--")
            plt.plot(data["best_gain"], data["best_freq"], "o", markersize=8, markeredgecolor="k", markerfacecolor="r")
        plt.colorbar(label="[ADC level]")

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname
