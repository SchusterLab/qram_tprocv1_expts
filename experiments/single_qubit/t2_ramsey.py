import experiments.fitting as fitter
import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from slab import AttrDict, Experiment, dsfit
from tqdm import tqdm_notebook as tqdm


class RamseyProgram(RAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        self.gen_delays = [0] * len(soccfg["gens"])  # need to calibrate via oscilloscope

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds

        super().__init__(soccfg, self.cfg)

    def reset_and_sync(self):
        # Phase reset all channels except readout DACs (since mux ADCs can't be phase reset)
        for ch in self.gen_chs.keys():
            if ch not in self.measure_chs:  # doesn't work for the mux ADCs
                # print('resetting', ch)
                self.setup_and_pulse(ch=ch, style="const", freq=100, phase=0, gain=100, length=10, phrst=1)
        self.sync_all(10)

    def set_gen_delays(self):
        for ch in self.gen_chs:
            delay_ns = self.cfg.hw.soc.dacs.delay_chs.delay_ns[
                np.argwhere(np.array(self.cfg.hw.soc.dacs.delay_chs.ch) == ch)[0][0]
            ]
            delay_cycles = self.us2cycles(delay_ns * 1e-3, gen_ch=ch)
            self.gen_delays[ch] = delay_cycles

    def sync_all(self, t=0):
        super().sync_all(t=t, gen_t0=self.gen_delays)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)

        qTest = self.cfg.expt.qTest
        qZZ = self.cfg.expt.qZZ
        self.checkZZ = False
        if qZZ is not None:
            self.checkZZ = True
        else:
            qZZ = qTest
        self.checkEF = self.cfg.expt.checkEF

        self.num_qubits_sample = len(self.cfg.device.readout.frequency)

        self.adc_chs = cfg.hw.soc.adcs.readout.ch
        self.res_chs = cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = cfg.hw.soc.dacs.readout.type
        self.qubit_chs = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = cfg.hw.soc.dacs.qubit.type
        if "cool_qubits" in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            self.swap_f0g1_chs = self.cfg.hw.soc.dacs.swap_f0g1.ch
            self.swap_f0g1_ch_types = self.cfg.hw.soc.dacs.swap_f0g1.type
            mixer_freqs = self.cfg.hw.soc.dacs.swap_f0g1.mixer_freq

        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs]  # get register page for qubit_chs

        self.f_ges = np.reshape(self.cfg.device.qubit.f_ge, (4, 4))
        self.f_efs = np.reshape(self.cfg.device.qubit.f_ef, (4, 4))
        self.pi_ge_gains = np.reshape(self.cfg.device.qubit.pulses.pi_ge.gain, (4, 4))
        self.pi_ge_sigmas = np.reshape(self.cfg.device.qubit.pulses.pi_ge.sigma, (4, 4))
        self.pi_ge_half_gains = np.reshape(self.cfg.device.qubit.pulses.pi_ge.half_gain, (4, 4))
        self.pi_ge_half_gain_pi_sigmas = np.reshape(self.cfg.device.qubit.pulses.pi_ge.half_gain_pi_sigma, (4, 4))
        self.pi_ef_gains = np.reshape(self.cfg.device.qubit.pulses.pi_ef.gain, (4, 4))
        self.pi_ef_sigmas = np.reshape(self.cfg.device.qubit.pulses.pi_ef.sigma, (4, 4))
        self.pi_ef_half_gains = np.reshape(self.cfg.device.qubit.pulses.pi_ef.half_gain, (4, 4))
        self.pi_ef_half_gain_pi_sigmas = np.reshape(self.cfg.device.qubit.pulses.pi_ef.half_gain_pi_sigma, (4, 4))

        self.f_res_regs = [
            self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch)
            for f, gen_ch, adc_ch in zip(cfg.device.readout.frequency, self.res_chs, self.adc_chs)
        ]
        if "cool_qubits" in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            self.f_f0g1_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_f0g1, self.qubit_chs)]
        self.readout_lengths_dac = [
            self.us2cycles(length, gen_ch=gen_ch)
            for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)
        ]
        self.readout_lengths_adc = [
            1 + self.us2cycles(length, ro_ch=ro_ch)
            for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)
        ]

        # declare all res dacs
        self.measure_chs = []
        mask = []  # indices of mux_freqs, mux_gains list to play
        mux_mixer_freq = None
        mux_freqs = [0] * 4  # MHz
        mux_gains = [0] * 4
        mux_ro_ch = None
        mux_nqz = None
        for q in range(self.num_qubits_sample):
            assert self.res_ch_types[q] in ["full", "mux4"]
            if self.res_ch_types[q] == "full":
                if self.res_chs[q] not in self.measure_chs:
                    self.declare_gen(ch=self.res_chs[q], nqz=cfg.hw.soc.dacs.readout.nyquist[q])
                    self.measure_chs.append(self.res_chs[q])

            elif self.res_ch_types[q] == "mux4":
                assert self.res_chs[q] == 6
                mask.append(q)
                if mux_mixer_freq is None:
                    mux_mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[q]
                else:
                    assert (
                        mux_mixer_freq == cfg.hw.soc.dacs.readout.mixer_freq[q]
                    )  # ensure all mux channels have specified the same mixer freq
                mux_freqs[q] = cfg.device.readout.frequency[q]
                mux_gains[q] = cfg.device.readout.gain[q]
                mux_ro_ch = self.adc_chs[q]
                mux_nqz = cfg.hw.soc.dacs.readout.nyquist[q]
                if self.res_chs[q] not in self.measure_chs:
                    self.measure_chs.append(self.res_chs[q])
        if "mux4" in self.res_ch_types:  # declare mux4 channel
            # print('mux params', mux_mixer_freq, mux_freqs, mux_gains, mux_ro_ch, mask)
            self.declare_gen(
                ch=6, nqz=mux_nqz, mixer_freq=mux_mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=mux_ro_ch
            )

        # declare adcs - readout for all qubits everytime, defines number of buffers returned regardless of number of adcs triggered
        for q in range(self.num_qubits_sample):
            if self.adc_chs[q] not in self.ro_chs:
                self.declare_readout(
                    ch=self.adc_chs[q],
                    length=self.readout_lengths_adc[q],
                    freq=self.cfg.device.readout.frequency[q],
                    gen_ch=self.res_chs[q],
                )

        # declare qubit dacs
        for q in range(self.num_qubits_sample):
            mixer_freq = None
            if self.qubit_ch_types[q] == "int4":
                mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in self.gen_chs:
                self.declare_gen(ch=self.qubit_chs[q], nqz=cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)

        if "cool_qubits" in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            mixer_freq = None
            for q in self.cfg.expt.cool_qubits:
                if self.swap_f0g1_ch_types[q] == "int4":
                    mixer_freq = mixer_freqs[q]
                if self.swap_f0g1_chs[q] not in self.gen_chs:
                    self.declare_gen(
                        ch=self.swap_f0g1_chs[q], nqz=self.cfg.hw.soc.dacs.swap_f0g1.nyquist[q], mixer_freq=mixer_freq
                    )

        # declare registers for phase incrementing
        # self.r_wait = 3
        # self.r_phase2 = 4
        self.r_wait = 5
        self.r_phase2 = 6
        if self.qubit_ch_types[qTest] == "int4":
            self.r_phase = self.sreg(self.qubit_chs[qTest], "freq")
            self.r_phase3 = 5  # for storing the left shifted value
        else:
            self.r_phase = self.sreg(self.qubit_chs[qTest], "phase")

        # define pisigma_ge as the ge pulse for the qubit that we are calibrating the pulse on (mostly for use for preparation if we need to calibrate ef)
        self.pisigma_ge = self.us2cycles(
            self.pi_ge_sigmas[qTest, qZZ], gen_ch=self.qubit_chs[qTest]
        )  # default pi_ge value
        self.f_ge_init_reg = self.freq2reg(self.f_ges[qTest, qZZ], gen_ch=self.qubit_chs[qTest])
        self.gain_ge_init = (
            self.pi_ge_gains[qTest, qZZ] if self.pi_ge_gains[qTest, qZZ] > 0 else self.pi_ge_gains[qTest, qTest]
        )  # this contingency is possible if the ge pulse is not calibrated but we want to calibrate the EF pulse for a specific ZZ configuration
        if self.checkZZ:
            self.pisigma_ge_qZZ = self.us2cycles(self.pi_ge_sigmas[qZZ, qZZ], gen_ch=self.qubit_chs[qZZ])

        # parameters for test pulse that we are trying to calibrate
        self.pi_test_sigma = self.us2cycles(
            self.pi_ge_sigmas[qTest, qZZ], gen_ch=self.qubit_chs[qTest]
        )  # default pi_ge value
        self.pi2_test_sigma = self.us2cycles(
            self.pi_ge_sigmas[qTest, qZZ] / 2, gen_ch=self.qubit_chs[qTest]
        )  # pi/2 for test pulse
        self.gain_pi_test = (
            self.pi_ge_gains[qTest, qZZ] if self.pi_ge_gains[qTest, qZZ] > 0 else self.pi_ge_gains[qTest, qTest]
        )
        self.f_pi_test_reg = self.freq2reg(self.f_ges[qTest, qZZ])
        if self.checkEF:
            self.pi_test_sigma = self.us2cycles(
                self.pi_ef_sigmas[qTest, qZZ], gen_ch=self.qubit_chs[qTest]
            )  # default pi_ge value
            self.pi2_test_sigma = self.us2cycles(
                self.pi_ef_sigmas[qTest, qZZ] / 2, gen_ch=self.qubit_chs[qTest]
            )  # pi/2 for test pulse
            self.gain_pi_test = (
                self.pi_ef_gains[qTest, qZZ] if self.pi_ef_gains[qTest, qZZ] > 0 else self.pi_ef_gains[qTest, qTest]
            )
            self.f_pi_test_reg = self.freq2reg(self.f_efs[qTest, qZZ])
        assert self.f_pi_test_reg > 0
        assert self.gain_pi_test > 0
        self.cfg.expt.gain = self.gain_pi_test

        if "cool_qubits" in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            for q in self.cfg.expt.cool_qubits:
                self.pisigma_ef = self.us2cycles(
                    self.pi_ef_sigmas[q, q], gen_ch=self.qubit_chs[q]
                )  # default pi_ef value
                self.add_gauss(
                    ch=self.qubit_chs[q], name=f"pi_ef_qubit{q}", sigma=self.pisigma_ef, length=self.pisigma_ef * 4
                )
                if self.cfg.device.qubit.pulses.pi_f0g1.type[q] == "flat_top":
                    self.add_gauss(ch=self.swap_f0g1_chs[q], name=f"pi_f0g1_{q}", sigma=3, length=3 * 4)
                else:
                    assert False, "not implemented"

        # add qubit pulses to respective channels
        self.add_gauss(
            ch=self.qubit_chs[qTest], name="pi2_test", sigma=self.pi2_test_sigma, length=self.pi2_test_sigma * 4
        )
        self.add_gauss(
            ch=self.qubit_chs[qTest], name="pi_test", sigma=self.pi_test_sigma, length=self.pi_test_sigma * 4
        )
        if self.checkZZ:
            self.add_gauss(
                ch=self.qubit_chs[qZZ], name="pi_qubitZZ", sigma=self.pisigma_ge_qZZ, length=self.pisigma_ge_qZZ * 4
            )
        if self.checkEF:
            self.add_gauss(
                ch=self.qubit_chs[qTest], name="pi_qubit_ge", sigma=self.pisigma_ge, length=self.pisigma_ge * 4
            )

        # add readout pulses to respective channels
        if "mux4" in self.res_ch_types:
            self.set_pulse_registers(ch=6, style="const", length=max(self.readout_lengths_dac), mask=mask)
        for q in range(self.num_qubits_sample):
            if self.res_ch_types[q] != "mux4":
                if cfg.device.readout.gain[q] < 1:
                    gain = int(cfg.device.readout.gain[q] * 2**15)
                self.set_pulse_registers(
                    ch=self.res_chs[q],
                    style="const",
                    freq=self.f_res_regs[q],
                    phase=0,
                    gain=gain,
                    length=max(self.readout_lengths_dac),
                )

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

        # print('warning, debug pulses')
        # cool_qubit = self.cfg.expt.cool_qubits[0]
        # self.setup_and_pulse(ch=self.qubit_chs[cool_qubit], style="arb", phase=0, freq=self.f_ge_reg[cool_qubit], gain=cfg.device.qubit.pulses.pi_ge.gain[cool_qubit], waveform="pi_qubit_ge")
        # self.sync_all()

        if "cool_qubits" in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            cool_idle = [self.cfg.device.qubit.pulses.pi_f0g1.idle[q] for q in self.cfg.expt.cool_qubits]
            cool_qubits = self.cfg.expt.cool_qubits
            if "cool_idle" in self.cfg.expt and self.cfg.expt.cool_idle is not None:
                cool_idle = self.cfg.expt.cool_idle
            sorted_indices = np.argsort(cool_idle)[::-1]  # sort cooling times longest first
            cool_qubits = np.array(cool_qubits)
            cool_idle = np.array(cool_idle)
            sorted_cool_qubits = cool_qubits[sorted_indices]
            sorted_cool_idle = cool_idle[sorted_indices]
            max_idle = sorted_cool_idle[0]

            last_pulse_len = 0
            remaining_idle = max_idle
            for q, idle in zip(sorted_cool_qubits, sorted_cool_idle):
                remaining_idle -= last_pulse_len

                last_pulse_len = 0
                self.setup_and_pulse(
                    ch=self.qubit_chs[q],
                    style="arb",
                    phase=0,
                    freq=self.freq2reg(self.f_efs[q, q], gen_ch=self.qubit_chs[q]),
                    gain=self.pi_ef_gains[q, q],
                    waveform=f"pi_ef_qubit{q}",
                )
                self.sync_all()
                last_pulse_len += self.pi_ef_sigmas[q, q] * 4

                pulse_type = self.cfg.device.qubit.pulses.pi_f0g1.type[q]
                pisigma_f0g1 = self.us2cycles(
                    self.cfg.device.qubit.pulses.pi_f0g1.sigma[q], gen_ch=self.swap_f0g1_chs[q]
                )
                if pulse_type == "flat_top":
                    sigma_ramp_cycles = 3
                    flat_length_cycles = pisigma_f0g1 - sigma_ramp_cycles * 4
                    self.setup_and_pulse(
                        ch=self.swap_f0g1_chs[q],
                        style="flat_top",
                        freq=self.f_f0g1_regs[q],
                        phase=0,
                        gain=self.cfg.device.qubit.pulses.pi_f0g1.gain[q],
                        length=flat_length_cycles,
                        waveform=f"pi_f0g1_{q}",
                    )
                else:
                    assert False, "not implemented"
                self.sync_all()
                last_pulse_len += self.cfg.device.qubit.pulses.pi_f0g1.sigma[q]

            remaining_idle -= last_pulse_len
            last_idle = max((remaining_idle, sorted_cool_idle[-1]))
            self.sync_all(self.us2cycles(last_idle))

        # initializations as necessary
        if self.checkZZ:
            assert self.pi_ge_gains[qZZ, qZZ] > 0
            self.setup_and_pulse(
                ch=self.qubit_chs[qZZ],
                style="arb",
                phase=0,
                freq=self.freq2reg(self.f_ges[qZZ, qZZ], gen_ch=self.qubit_chs[qZZ]),
                gain=self.pi_ge_gains[qZZ, qZZ],
                waveform="pi_qubitZZ",
            )
            self.sync_all()
        if self.checkEF:
            assert self.gain_ge_init > 0
            assert self.f_ge_init_reg > 0
            self.setup_and_pulse(
                ch=self.qubit_chs[qTest],
                style="arb",
                freq=self.f_ge_init_reg,
                phase=0,
                gain=self.gain_ge_init,
                waveform="pi_qubit_ge",
            )
            self.sync_all()

        # play pi/2 pulse with the freq that we want to calibrate (phase = 0)
        self.setup_and_pulse(
            ch=self.qubit_chs[qTest],
            style="arb",
            freq=self.f_pi_test_reg,
            phase=0,
            gain=self.gain_pi_test,
            waveform="pi2_test",
        )
        self.sync_all()

        # handle echo
        num_pi = 0
        if "num_pi" in self.cfg.expt:
            num_pi = self.cfg.expt.num_pi
        if num_pi >= 1:
            assert "echo_type" in self.cfg.expt
            assert self.cfg.expt.echo_type in ["cp", "cpmg"]
            echo_type = self.cfg.expt.echo_type
        for i in range(num_pi):
            self.sync(self.q_rps[qTest], self.r_wait)
            if echo_type == "cp":
                phase = 0
            elif echo_type == "cpmg":
                phase = 90
            self.setup_and_pulse(
                ch=self.qubit_chs[qTest],
                style="arb",
                freq=self.f_pi_test_reg,
                phase=self.deg2reg(phase, gen_ch=self.qubit_chs[qTest]),
                gain=self.gain_pi_test,
                waveform="pi_test",
            )
            self.sync(self.q_rps[qTest], self.r_wait)

        # wait advanced wait time (ramsey wait time)
        if num_pi == 0:
            self.sync(self.q_rps[qTest], self.r_wait)

        # play pi/2 pulse with advanced phase
        self.set_pulse_registers(
            ch=self.qubit_chs[qTest],
            style="arb",
            freq=self.f_pi_test_reg,
            phase=0,
            gain=self.gain_pi_test,
            waveform="pi2_test",
        )
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
                self.setup_and_pulse(
                    ch=self.qubit_chs[qTest],
                    style="arb",
                    freq=self.f_ge_init_reg,
                    phase=0,
                    gain=self.gain_ge_init,
                    waveform="pi_qubit_ge",
                )

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

        x_pts, avgi, avgq = ramsey.acquire(
            self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress
        )

        avgi = avgi[qTest][0]
        avgq = avgq[qTest][0]
        amps = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
        phases = np.angle(avgi + 1j * avgq)  # Calculating the phase

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
        if qZZ is not None:
            self.checkZZ = True
        else:
            qZZ = qTest
        self.checkEF = self.cfg.expt.checkEF

        f_pi_test = np.reshape(self.cfg.device.qubit.f_ge, (4, 4))[qTest, qZZ]
        if self.checkEF:
            f_pi_test = np.reshape(self.cfg.device.qubit.f_ef, (4, 4))[qTest, qZZ]

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
