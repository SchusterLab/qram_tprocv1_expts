import matplotlib.pyplot as plt
import numpy as np
from qick import *
from slab import AttrDict, Experiment, dsfit
from tqdm import tqdm_notebook as tqdm

"""
Run this calibration when the wiring of the setup is changed.

This calibration measures the time of flight of measurement pulse so we only start capturing data from this point in time onwards. Time of flight (tof) is stored in parameter cfg.device.readout.trig_offset.
"""


class ToFCalibrationProgram(AveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        self.gen_delays = [0] * len(soccfg["gens"])  # need to calibrate via oscilloscope

        # copy over parameters for the acquire method
        self.cfg.soft_avgs = cfg.expt.reps  # same as reps
        self.cfg.reps = 1
        super().__init__(soccfg, self.cfg)

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
        cfg = self.cfg

        self.adc_ch = cfg.hw.soc.adcs.readout.ch[self.cfg.expt.qubit]
        self.dac_ch = cfg.hw.soc.dacs.readout.ch[self.cfg.expt.qubit]
        self.dac_ch_type = cfg.hw.soc.dacs.readout.type[self.cfg.expt.qubit]

        self.frequency = cfg.expt.frequency
        self.freqreg = self.freq2reg(
            self.frequency, gen_ch=self.dac_ch, ro_ch=self.adc_ch
        )  # convert frequency to dac frequency (ensuring it is an available adc frequency)
        self.gain = cfg.expt.gain
        self.pulse_length = self.us2cycles(cfg.expt.pulse_length, gen_ch=self.dac_ch)
        self.readout_length = self.us2cycles(cfg.expt.readout_length, ro_ch=self.adc_ch)
        print(self.pulse_length, self.readout_length)

        mask = None
        mixer_freq = 0  # MHz
        mux_freqs = None  # MHz
        mux_gains = None
        ro_ch = None
        if self.dac_ch_type == "int4":
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[self.cfg.expt.qubit]
        elif self.dac_ch_type == "mux4":
            assert self.dac_ch == 6
            ro_ch = self.adc_ch
            mask = [0, 1, 2, 3]  # indices of mux_freqs, mux_gains list to play
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[self.cfg.expt.qubit]
            mux_freqs = [0] * 4
            mux_freqs[cfg.expt.qubit] = cfg.expt.frequency
            mux_gains = [0] * 4
            mux_gains[cfg.expt.qubit] = cfg.expt.gain
        self.declare_gen(
            ch=self.dac_ch,
            nqz=cfg.hw.soc.dacs.readout.nyquist[self.cfg.expt.qubit],
            mixer_freq=mixer_freq,
            mux_freqs=mux_freqs,
            mux_gains=mux_gains,
            ro_ch=ro_ch,
        )
        print(f"readout freq {mixer_freq} + {cfg.expt.frequency}")

        self.declare_readout(
            ch=self.adc_ch, length=self.readout_length, freq=self.frequency, gen_ch=self.dac_ch
        )  # gen_ch links to the mixer_freq being used on the mux

        if self.dac_ch_type == "mux4":
            self.set_pulse_registers(ch=self.dac_ch, style="const", length=self.pulse_length, mask=mask)
        else:
            if self.gain < 1:
                self.gain = int(self.gain * 2**15)
            self.set_pulse_registers(
                ch=self.dac_ch,
                style="const",
                freq=self.freqreg,
                phase=0,
                gain=int(self.gain),
                length=self.pulse_length,
            )

        self.set_gen_delays()
        self.sync_all(200)

    def body(self):
        cfg = AttrDict(self.cfg)
        self.measure(
            pulse_ch=self.dac_ch,
            adcs=[self.adc_ch],
            adc_trig_offset=cfg.expt.trig_offset,
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay),
        )


# ====================================================== #


class ToFCalibrationExperiment(Experiment):
    """
    Time of flight experiment
    Experimental Config
    expt_cfg = dict(
        pulse_length [us]
        readout_length [us]
        gain [DAC units]
        frequency [MHz]
        adc_trig_offset [Clock ticks]
    }
    """

    def __init__(self, soccfg=None, path="", prefix="ToFCalibration", config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        q_ind = self.cfg.expt.qubit

        self.num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, list) and len(value) == self.num_qubits_sample:
                    subcfg.update({key: value[q_ind]})
                elif isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if isinstance(value3, list) and len(value3) == self.num_qubits_sample:
                                value2.update({key3: value3[q_ind]})

        data = {"i": [], "q": [], "amps": [], "phases": []}
        tof = ToFCalibrationProgram(soccfg=self.soccfg, cfg=self.cfg)
        # from qick.helpers import progs2json
        # print(progs2json([tof.dump_prog()]))
        # print(self.im)
        iq = tof.acquire_decimated(self.im[self.cfg.aliases.soc], load_pulses=True, progress=True)
        i, q = iq[0]
        amp = np.abs(i + 1j * q)  # Calculating the magnitude
        phase = np.angle(i + 1j * q)  # Calculating the phase

        data = dict(i=i, q=q, amps=amp, phases=phase)

        for k, a in data.items():
            data[k] = np.array(a)

        self.data = data
        return data

    def analyze(self, data=None, fit=False, findpeaks=False, **kwargs):
        if data is None:
            data = self.data
        return data

    def display(self, data=None, adc_trig_offset=0, **kwargs):
        if data is None:
            data = self.data

        q_ind = self.cfg.expt.qubit
        adc_ch = self.cfg.hw.soc.adcs.readout.ch[q_ind]
        dac_ch = self.cfg.hw.soc.dacs.readout.ch[q_ind]
        plt.figure()
        plt.subplot(
            111,
            title=f"Time of flight calibration: dac ch {dac_ch} to adc ch {adc_ch}",
            xlabel="Clock ticks",
            ylabel="Transmission [ADC units]",
        )

        plt.plot(data["i"], label="I")
        plt.plot(data["q"], label="Q")
        plt.axvline(adc_trig_offset, c="k", ls="--")
        # plt.ylim(-100, 100)
        plt.legend()
        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)


# ====================================================== #
"""
Play a tone at the readout frequency from a non-mux DAC, read it out from the mux ADC, uses the default readout frequency.
"""


class PhotonPumpLoopCalibrationProgram(AveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = self.cfg

        self.adc_ch = cfg.hw.soc.adcs.readout.ch
        self.res_meas_ch = cfg.hw.soc.dacs.readout.ch
        self.res_meas_ch_type = cfg.hw.soc.dacs.readout.type
        self.dac_ch = cfg.hw.soc.dacs.res_pump.ch
        self.dac_ch_type = cfg.hw.soc.dacs.res_pump.type

        self.f_mux = self.cfg.device.readout.frequency
        # self.f_res_pump = float(self.cfg.hw.lo.readout.frequency)*1e-6 + self.cfg.device.readout.lo_sideband*(self.cfg.hw.soc.dacs.readout.mixer_freq + self.f_mux)
        self.f_res_pump = float(self.cfg.hw.lo.readout.frequency) * 1e-6 + self.cfg.device.readout.lo_sideband * (
            self.cfg.hw.soc.dacs.readout.mixer_freq + self.f_mux
        )
        print("pump freq", self.f_res_pump)
        self.f_res_pump_reg = self.freq2reg(self.f_res_pump, gen_ch=self.dac_ch, ro_ch=self.adc_ch)
        # self.f_res_meas_reg = self.freq2reg(self.f_mux, gen_ch=self.res_meas_ch, ro_ch=self.adc_ch)

        self.gain = cfg.expt.gain
        self.pulse_length = self.us2cycles(cfg.expt.pulse_length, gen_ch=self.dac_ch)
        # self.readout_length_dac = self.us2cycles(cfg.expt.readout_length, gen_ch=self.dac_ch)
        self.readout_length_adc = self.us2cycles(cfg.expt.readout_length, ro_ch=self.adc_ch)
        print(self.pulse_length, self.readout_length_adc)

        ro_ch = None
        mixer_freq = None
        if self.res_meas_ch_type == "full":
            ro_ch = self.adc_ch
            mixer_freq = cfg.hw.soc.dacs.res_pump.mixer_freq

        self.declare_gen(
            ch=self.dac_ch,
            nqz=cfg.hw.soc.dacs.res_pump.nyquist[self.cfg.expt.qubit],
            mixer_freq=mixer_freq,
            mux_freqs=None,
            mux_gains=None,
            ro_ch=ro_ch,
        )

        # mask = None
        # mixer_freq = 0 # MHz
        # mux_freqs = None # MHz
        # mux_gains = None
        # if self.res_meas_ch_type == 'int4':
        #     mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
        # elif self.res_meas_ch_type == 'mux4':
        #     assert self.res_meas_ch == 6
        #     mask = [0, 1, 2, 3] # indices of mux_freqs, mux_gains list to play
        #     mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
        #     mux_freqs = [0]*4
        #     mux_freqs[cfg.expt.qubit] = self.f_mux
        #     mux_gains = [0]*4
        #     mux_gains[cfg.expt.qubit] = 0
        # self.declare_gen(ch=self.res_meas_ch, nqz=cfg.hw.soc.dacs.readout.nyquist, mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)
        # print(f'readout freq {mixer_freq} +/- {self.f_mux}')

        self.declare_readout(
            ch=self.adc_ch, length=self.readout_length_adc, freq=self.f_res_pump, gen_ch=self.dac_ch
        )  # gen_ch is used for downconversion on the mux ADC

        assert self.dac_ch_type != "mux4"
        self.set_pulse_registers(
            ch=self.dac_ch, style="const", freq=self.f_res_pump_reg, phase=0, gain=self.gain, length=self.pulse_length
        )
        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        cfg = AttrDict(self.cfg)
        self.measure(
            pulse_ch=self.dac_ch,
            adcs=[self.adc_ch],
            adc_trig_offset=cfg.expt.trig_offset,
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay),
        )

    def collect_shots(self):
        # collect shots for the relevant adc and I and Q channels
        cfg = AttrDict(self.cfg)
        # print(np.average(self.di_buf[0]))
        shots_i0 = self.di_buf[0] / self.readout_length_adc
        shots_q0 = self.dq_buf[0] / self.readout_length_adc
        return shots_i0, shots_q0
        # return shots_i0[:5000], shots_q0[:5000]


# ====================================================== #


class PhotonPumpLoopCalibrationExperiment(Experiment):
    """
    Time of flight experiment
    Experimental Config
    expt_cfg = dict(
        pulse_length [us]
        readout_length [us]
        gain [DAC units]
        adc_trig_offset [Clock ticks]
    }
    """

    def __init__(self, soccfg=None, path="", prefix="PhotonPumpLoopCalibration", config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        q_ind = self.cfg.expt.qubit
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, list):
                    subcfg.update({key: value[q_ind]})
                elif isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if isinstance(value3, list):
                                value2.update({key3: value3[q_ind]})

        data = {"i": [], "q": [], "amps": [], "phases": []}
        prog = PhotonPumpLoopCalibrationProgram(soccfg=self.soccfg, cfg=self.cfg)
        self.prog = prog
        avgi, avgq = prog.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress)
        i, q = prog.collect_shots()
        amp = np.abs(i + 1j * q)  # Calculating the magnitude
        phase = np.angle(i + 1j * q)  # Calculating the phase

        data = dict(i=i, q=q, amps=amp, phases=phase)

        for k, a in data.items():
            data[k] = np.array(a)

        self.data = data
        return data

    def analyze(self, data=None, fit=False, findpeaks=False, **kwargs):
        if data is None:
            data = self.data
        return data

    def display(self, data=None, adc_trig_offset=0, **kwargs):
        if data is None:
            data = self.data

        q_ind = self.cfg.expt.qubit
        adc_ch = self.cfg.hw.soc.adcs.readout.ch[q_ind]
        dac_ch = self.cfg.hw.soc.dacs.res_pump.ch[q_ind]

        plt.figure()
        plt.subplot(
            111,
            title=f"Time of flight calibration: dac ch {dac_ch} to adc ch {adc_ch}",
            xlabel="Shot Number",
            ylabel="Transmission [ADC units]",
        )

        plt.plot(data["amps"], ".", label="Amps")
        plt.plot(data["i"], ".", label="I")
        plt.plot(data["q"], ".", label="Q")
        # plt.ylim(-100, 100)
        plt.legend()
        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
