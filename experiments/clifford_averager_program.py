import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from qick import *
from qick.helpers import ch2list
from slab import AttrDict, Experiment
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter

logger = logging.getLogger("qick.qick_asm")
logger.setLevel(logging.ERROR)


def rotate_and_threshold(ishots_1q, qshots_1q=None, angle=None, threshold=None, amplitude_mode=False, avg_shots=False):
    """
    Takes ishots, qshots, angle, threshold all specified for 1 qubit only (so angle, threshold are both numbers)
    If angle is not None, applies the rotation to ishots, qshots
    If threshold is not None, bins shots into 0/1 if less than/greater than threshold
    If avg shots is True, returns the average value of ishots (qshots if specified)
    """
    if amplitude_mode:
        assert qshots_1q is not None
    ishots_1q = np.array(ishots_1q)
    qshots_1q = np.array(qshots_1q)
    assert len(np.array(ishots_1q).shape) == 1  # 1d array, for 1q only
    if qshots_1q is not None:
        assert len(np.array(ishots_1q).shape) == 1  # 1d array, for 1q only
    if angle is not None:
        assert qshots_1q is not None
        assert len(np.array(angle).shape) == 0  # number for 1q only
    if threshold is not None:
        assert len(np.array(threshold).shape) == 0  # number for 1q only

    ifinal = ishots_1q
    qfinal = qshots_1q if qshots_1q is not None else np.zeros_like(ifinal)

    if angle is not None:
        ifinal = ishots_1q * np.cos(np.pi / 180 * angle) - qshots_1q * np.sin(np.pi / 180 * angle)
        qfinal = ishots_1q * np.sin(np.pi / 180 * angle) + qshots_1q * np.cos(np.pi / 180 * angle)

    if threshold is not None:
        if amplitude_mode:
            ifinal = np.heaviside(np.abs(ifinal + 1j * qfinal) - threshold, 0)
        else:
            ifinal = np.heaviside(ifinal - threshold, 0)
        qfinal = np.zeros_like(ifinal)

    if avg_shots:
        ifinal = np.average(ifinal)
        qfinal = np.average(qfinal)

    return ifinal, qfinal


def post_select_shots(
    final_qubit,
    all_ishots_raw_q,
    ps_thresholds,
    ps_qubits,
    n_init_readout,
    all_qshots_raw_q=None,
    angles=None,
    post_process="threshold",
    amplitude_mode=False,
    thresholds=None,
    verbose=False,
    return_keep_indices=False,
):

    """
    final_qubit: qubit whose final, post selected shots should be returned
    all_ishots_raw_q: ishots for each qubit, shape should be (adc_chs, n_init_readout + 1, reps)
    all_qshots_raw_q: optional qshots if ishots was not already rotated
    angles: if specified, combines ishots_raw and qshots_raw to get the rotated shots
    ps_thresholds: post selection thresholds for all qubits. Make sure these have been calibrated for each of ps_qubits!
    ps_qubits: qubits to do post selection on
    post_process: post processing on the final readout, 'threshold' or 'scale'
    thresholds: thresholding for all qubits. Only uses the value for the final readout, and only does so if post_process='threshold (1 qubit only)

    returns: shots_final for final_qubit only, post processed as requested
    """

    assert len(all_ishots_raw_q.shape) == 3
    if angles is not None:
        assert all_qshots_raw_q is not None
        assert len(all_qshots_raw_q.shape) == 3
    if angles is not None:
        ishots_final, qshots_final = rotate_and_threshold(
            ishots_1q=all_ishots_raw_q[final_qubit, -1, :],
            qshots_1q=all_qshots_raw_q[final_qubit, -1, :],
            angle=angles[final_qubit],
            threshold=None,
            amplitude_mode=amplitude_mode,
            avg_shots=False,
        )
    else:
        ishots_final = all_ishots_raw_q[final_qubit, -1, :]
    reps_orig = len(ishots_final)

    keep_prev = np.ones_like(ishots_final, dtype="bool")
    for i_readout in range(n_init_readout):
        for ps_qubit in ps_qubits:
            # For initialization readouts, shots_raw is the rotated i value
            if angles is not None:
                shots_readout, _ = rotate_and_threshold(
                    ishots_1q=all_ishots_raw_q[ps_qubit, i_readout, :],
                    qshots_1q=all_qshots_raw_q[ps_qubit, i_readout, :],
                    angle=angles[ps_qubit],
                    threshold=None,
                    amplitude_mode=amplitude_mode,
                    avg_shots=False,
                )
            else:
                shots_readout = all_ishots_raw_q[ps_qubit, i_readout, :]
            # print(ps_qubit, np.average(shots_readout))

            keep_prev = np.logical_and(keep_prev, shots_readout < ps_thresholds[ps_qubit])

            # if verbose:
            #     print('i_readout', i_readout, 'ps_qubit', ps_qubit, 'keep', np.sum(keep_prev), 'of', reps_orig, f'shots ({100*np.sum(keep_prev)/reps_orig} %)')
    if verbose:
        print("keep", np.sum(keep_prev), "of", reps_orig, f"shots ({100*np.sum(keep_prev)/reps_orig} %)")

    # Apply thresholding if necessary
    assert post_process in [None, "threshold"]
    if post_process == "threshold" and thresholds is not None:
        assert len(np.array(thresholds).shape) == 1  # array
    if post_process is None:
        thresholds = [None] * 4
    if thresholds is None:
        assert post_process is None
    shots_final, _ = rotate_and_threshold(
        ishots_1q=ishots_final,
        qshots_1q=qshots_final,
        threshold=thresholds[final_qubit],
        avg_shots=False,
        amplitude_mode=amplitude_mode,
    )

    assert shots_final.shape == keep_prev.shape
    if return_keep_indices:
        return shots_final[keep_prev], keep_prev
    return shots_final[keep_prev]


def ps_threshold_adjust(ps_thresholds_init, adjust, ge_avgs, angles, amplitude_mode=False):
    """
    Given a set of ps_thresholds_init for all qubits, adjust by ratio. The ratio is defined relative to the
    ge_avgs provided as [Ig, Qg, Ie, Qe]*num_qubits, which will be rotated by angles for each of the qubits;
    Adjust should be specified for each qubit
    Adjust = 0 indicates keep the ps_threshold as the default threshold point
    Adjust < 0: adjust = -1 indicates to set the ps_threshold to the (rotated) g avg value, linear scaling between 0 and -1
    Adjust > 0: adjust = +1 indicates to set the ps_threshold to the (rotated) e avg value, linear scaling between 0 and 1
    """

    num_qubits_sample = 4
    ps_thresholds_init = np.array(ps_thresholds_init)
    ge_avgs = np.array(ge_avgs)
    angles = np.array(angles)
    adjust = np.array(adjust)
    assert ps_thresholds_init.shape == (num_qubits_sample,)
    assert ge_avgs.shape == (num_qubits_sample, 4)
    assert adjust.shape == (num_qubits_sample,)
    g_avgs = np.array(
        [
            rotate_and_threshold(
                ishots_1q=[ge_avgs[q, 0]],
                qshots_1q=[ge_avgs[q, 1]],
                angle=angles[q],
                threshold=None,
                amplitude_mode=amplitude_mode,
                avg_shots=False,
            )[0][0]
            for q in range(num_qubits_sample)
        ]
    )
    e_avgs = np.array(
        [
            rotate_and_threshold(
                ishots_1q=[ge_avgs[q, 2]],
                qshots_1q=[ge_avgs[q, 3]],
                angle=angles[q],
                threshold=None,
                amplitude_mode=amplitude_mode,
                avg_shots=False,
            )[0][0]
            for q in range(num_qubits_sample)
        ]
    )
    # print('new g avgs', g_avgs)
    # print('new e avgs', e_avgs)

    ps_thresholds = ps_thresholds_init.copy()
    # print('old ps threshold', ps_thresholds)
    for q in range(len(adjust)):
        if adjust[q] < 0:
            ps_thresholds[q] += adjust[q] * (ps_thresholds_init[q] - g_avgs[q])
        elif adjust[q] > 0:
            ps_thresholds[q] += adjust[q] * (e_avgs[q] - ps_thresholds_init[q])
    # print('new ps threshold', ps_thresholds)
    return ps_thresholds


class CliffordAveragerProgram(AveragerProgram):
    """
    Averager program that takes care of the standard pulse loading for basic X, Y, Z +/- pi and pi/2
    """

    # def update(self):
    #     pass

    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        self.gen_delays = [0] * len(soccfg["gens"])  # need to calibrate via oscilloscope

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps

        super().__init__(soccfg, self.cfg)

    """
    Wrappers to load and play pulses.
    If play is false, must specify all parameters and all params will be saved (load params).

    If play is true, uses the default values saved from the load call, temporarily overriding freq, phase, or gain if specified to not be None. Sets the pulse registers with these settings and plays the pulse. If you want to set freq, phase, or gain via registers/update,
    be sure to set the default value to be None at loading time.

    If play is True, registers will automatically be set regardless of set_reg flag.
    If play is False, registers will b set based on value of set_reg flag, but pulse will not be played.
    """

    def handle_const_pulse(
        self,
        name,
        waveformname=None,
        ch=None,
        length=None,
        freq_MHz=None,
        phase_deg=None,
        gain=None,
        reload=False,
        play=False,
        set_reg=False,
        ro_ch=None,
        flag=None,
        phrst=0,
        sync_after=True,
    ):
        """
        Load/play a constant pulse of given length.
        """
        if name is not None and (name not in self.pulse_dict.keys() or reload):
            assert ch is not None
            self.pulse_dict.update(
                {
                    name: dict(
                        ch=ch,
                        name=name,
                        type="const",
                        length=length,
                        freq_MHz=freq_MHz,
                        phase_deg=phase_deg,
                        gain=gain,
                        ro_ch=ro_ch,
                        flag=flag,
                    )
                }
            )
        if play or set_reg:
            assert name in self.pulse_dict.keys()
            # if not (ch == None):
            #     print('Warning: you have specified a pulse parameter that can only be changed when loading.')
            params = self.pulse_dict[name].copy()
            if freq_MHz is not None:
                params["freq_MHz"] = freq_MHz
            if phase_deg is not None:
                params["phase_deg"] = phase_deg
            if gain is not None:
                params["gain"] = gain
            if ro_ch is not None:
                params["ro_ch"] = ro_ch
            self.set_pulse_registers(
                ch=params["ch"],
                style="const",
                freq=self.freq2reg(params["freq_MHz"], gen_ch=params["ch"], ro_ch=params["ro_ch"]),
                phase=self.deg2reg(params["phase_deg"], gen_ch=params["ch"]),
                gain=params["gain"],
                length=params["length"],
                phrst=phrst,
            )
            if play:
                self.pulse(ch=params["ch"])
                if sync_after:
                    self.sync_all()

    def handle_gauss_pulse(
        self,
        name,
        waveformname=None,
        ch=None,
        sigma=None,
        freq_MHz=None,
        phase_deg=None,
        gain=None,
        reload=False,
        play=False,
        set_reg=False,
        flag=None,
        phrst=0,
        sync_after=True,
    ):
        """
        Load/play a gaussian pulse of length 4 sigma on channel ch
        If either play, registers are always set, or can just set the regs and not play with set_reg=True, play=False
        """
        if name not in self.pulse_dict.keys() or reload:
            assert None not in [ch, sigma]
            if waveformname is None:
                waveformname = name
            self.pulse_dict.update(
                {
                    name: dict(
                        ch=ch,
                        name=name,
                        waveformname=waveformname,
                        type="gauss",
                        sigma=sigma,
                        freq_MHz=freq_MHz,
                        phase_deg=phase_deg,
                        gain=gain,
                        flag=flag,
                    )
                }
            )
            if reload or waveformname not in self.envelopes[ch].keys():
                self.add_gauss(ch=ch, name=waveformname, sigma=sigma, length=sigma * 4)
        if play or set_reg:
            # if not (ch == sigma == None):
            #     print('Warning: you have specified a pulse parameter that can only be changed when loading.')
            params = self.pulse_dict[name].copy()
            if freq_MHz is not None:
                params["freq_MHz"] = freq_MHz
            if phase_deg is not None:
                params["phase_deg"] = phase_deg
            if gain is not None:
                params["gain"] = gain
            self.set_pulse_registers(
                ch=params["ch"],
                style="arb",
                freq=self.freq2reg(params["freq_MHz"], gen_ch=params["ch"]),
                phase=self.deg2reg(params["phase_deg"], gen_ch=params["ch"]),
                gain=params["gain"],
                waveform=params["waveformname"],
                phrst=phrst,
            )
            if play:
                # print('playing gauss pulse', params['name'], 'on ch', params['ch'])
                self.pulse(ch=params["ch"])
                if sync_after:
                    self.sync_all()

    def handle_flat_top_pulse(
        self,
        name,
        waveformname=None,
        ch=None,
        sigma=3,
        flat_length=None,
        freq_MHz=None,
        phase_deg=None,
        gain=None,
        reload=False,
        play=False,
        set_reg=False,
        flag=None,
        phrst=0,
        sync_after=True,
    ):
        """
        Plays a gaussian ramp up (2*sigma), a constant pulse of length flat_length+4*sigma,
        plus a gaussian ramp down (2*sigma) on channel ch.
        By default: sigma=3 clock cycles
        """
        if name not in self.pulse_dict.keys() or reload:
            assert None not in [ch, sigma, flat_length]
            if waveformname is None:
                waveformname = name
            self.pulse_dict.update(
                {
                    name: dict(
                        ch=ch,
                        name=name,
                        waveformname=waveformname,
                        type="flat_top",
                        sigma=sigma,
                        flat_length=flat_length,
                        freq_MHz=freq_MHz,
                        phase_deg=phase_deg,
                        gain=gain,
                        flag=flag,
                    )
                }
            )
            if reload or waveformname not in self.envelopes[ch].keys():
                # print('all waveforms')
                # for i_ch in range(len(self.envelopes)):
                #     print(self.envelopes[i_ch].keys())
                self.add_gauss(ch=ch, name=waveformname, sigma=sigma, length=sigma * 4)
                # print('added', waveformname, 'ch', ch)
                # print(self.gen_chs.keys())
        if play or set_reg:
            # if not (ch == name == sigma == length == None):
            #     print('Warning: you have specified a pulse parameter that can only be changed when loading.')
            assert name in self.pulse_dict.keys()
            params = self.pulse_dict[name].copy()
            if freq_MHz is not None:
                params["freq_MHz"] = freq_MHz
            if phase_deg is not None:
                params["phase_deg"] = phase_deg
            if gain is not None:
                params["gain"] = gain
            self.set_pulse_registers(
                ch=params["ch"],
                style="flat_top",
                freq=self.freq2reg(params["freq_MHz"], gen_ch=params["ch"]),
                phase=self.deg2reg(params["phase_deg"], gen_ch=params["ch"]),
                gain=params["gain"],
                waveform=params["waveformname"],
                length=params["flat_length"],
                phrst=phrst,
            )
            if play:
                self.pulse(ch=params["ch"])
                if sync_after:
                    self.sync_all()

    def handle_mux4_pulse(
        self, name, ch=None, mask=None, length=None, reload=False, play=False, set_reg=False, flag=None
    ):
        """
        Load/play a constant pulse of given length on the mux4 channel.
        """
        # if name is not None or reload: # and name not in self.pulse_dict.keys():
        if name not in self.pulse_dict.keys() or reload:
            assert ch is not None
            assert ch == 6, "Only ch 6 on q3diamond supports mux4 currently!"
            self.pulse_dict.update({name: dict(ch=ch, name=name, type="mux4", mask=mask, length=length, flag=flag)})
        if play or set_reg:
            assert name in self.pulse_dict.keys()
            params = self.pulse_dict[name].copy()
            if mask is not None:
                params["mask"] = mask
            if length is not None:
                params["length"] = length
            self.set_pulse_registers(ch=params["ch"], style="const", length=params["length"], mask=params["mask"])
            if play:
                self.pulse(ch=params["ch"])
                self.sync_all()

    def add_IQ(self, ch, name, I_mhz_vs_us, Q_mhz_vs_us, times_us, plot_IQ=True):
        """
        I_mhz_vs_us, Q_mhz_vs_us = functions of time in us, in units of MHz
        times_us = times at which I_mhz_vs_us and Q_mhz_vs_us are defined
        """
        gencfg = self.soccfg["gens"][ch]
        maxv = gencfg["maxv"] * gencfg["maxv_scale"] - 1
        samps_per_clk = gencfg["samps_per_clk"]

        num_samps_tot = samps_per_clk * self.us2cycles(times_us[-1], gen_ch=ch)
        times_samps = np.arange(0, int(num_samps_tot))
        times_samps_interp = np.linspace(0, num_samps_tot, len(times_us))
        # print("num samps", num_samps_tot, times_us[-1])

        IQ_scale = max((np.max(np.abs(I_mhz_vs_us)), np.max(np.abs(Q_mhz_vs_us))))
        I_func = sp.interpolate.interp1d(times_samps_interp, I_mhz_vs_us / IQ_scale, kind="linear", fill_value=0)
        Q_func = sp.interpolate.interp1d(times_samps_interp, -Q_mhz_vs_us / IQ_scale, kind="linear", fill_value=0)
        iamps = I_func(times_samps)
        qamps = Q_func(times_samps)

        # plot_IQ = True
        if plot_IQ:
            plt.figure()
            plt.title(f"Pulse on ch{ch}, waveform {name}")
            # plt.plot(iamps, '.-')
            plt.plot(times_samps, I_func(times_samps), ".-", label="I")
            # plt.plot(qamps, '.-')
            plt.plot(times_samps, Q_func(times_samps), ".-", label="Q")
            # print(times_samps.shape, I_func(times_samps).shape)
            plt.ylabel("Amplitude [a.u.]")
            plt.xlabel("Sample Index")
            plt.legend()
            plt.show()

        self.add_pulse(ch=ch, name=name, idata=maxv * iamps, qdata=maxv * qamps)

    def add_IQ_ILC(self, ch, name, I_mhz_vs_us, Q_mhz_vs_us, times_us, plot_IQ=True):

        gencfg = self.soccfg["gens"][ch]
        maxv = gencfg["maxv"] * gencfg["maxv_scale"] - 1
        samps_per_clk = gencfg["samps_per_clk"]
        times_cycles = np.linspace(0, self.us2cycles(times_us[-1], gen_ch=ch), len(times_us))
        times_samps = samps_per_clk * times_cycles
        I_func = sp.interpolate.interp1d(times_samps, I_mhz_vs_us, kind="linear", fill_value=0)
        Q_func = sp.interpolate.interp1d(times_samps, -Q_mhz_vs_us, kind="linear", fill_value=0)
        t = np.arange(0, np.round(times_samps[-1]))
        iamps = I_func(t)
        qamps = Q_func(t)

        if plot_IQ:
            plt.figure()
            plt.title(f"Pulse on ch{ch}, waveform {name}")
            # plt.plot(iamps, '.-')
            plt.plot(times_samps, I_func(times_samps), ".-", label="I")
            # plt.plot(qamps, '.-')
            plt.plot(times_samps, Q_func(times_samps), ".-", label="Q")
            plt.ylabel("Amplitude [a.u.]")
            plt.xlabel("Sample Index")
            plt.legend()
            plt.show()

        # rescale with the voltage
        v_scale = maxv // 2
        _i = v_scale * iamps
        _q = v_scale * qamps

        # check that the IQ values are not larger than the max
        assert np.all(np.abs(_i) <= maxv)
        assert np.all(np.abs(_i) <= maxv)

        self.add_pulse(ch=ch, name=name, idata=_i, qdata=_q)

    def handle_IQ_pulse(
        self,
        name,
        waveformname=None,
        ch=None,
        I_mhz_vs_us=None,
        Q_mhz_vs_us=None,
        times_us=None,
        freq_MHz=None,
        phase_deg=None,
        gain=None,
        reload=False,
        ro_ch=None,
        play=False,
        set_reg=False,
        flag=None,
        phrst=0,
        sync_after=True,
        plot_IQ=True,
        ILC=False,
    ):
        """
        Load/play an arbitrary IQ pulse on channel ch
        """
        if name not in self.pulse_dict.keys() or reload:
            assert ch is not None and I_mhz_vs_us is not None and Q_mhz_vs_us is not None and times_us is not None
            if waveformname is None:
                waveformname = name
            self.pulse_dict.update(
                {
                    name: dict(
                        ch=ch,
                        name=name,
                        waveformname=waveformname,
                        type="IQpulse",
                        I_mhz_vs_us=I_mhz_vs_us,
                        Q_mhz_vs_us=Q_mhz_vs_us,
                        times_us=times_us,
                        freq_MHz=freq_MHz,
                        phase_deg=phase_deg,
                        gain=gain,
                        flag=flag,
                        ro_ch=ro_ch,
                    )
                }
            )
            if reload or waveformname not in self.envelopes[ch].keys():
                if ILC:
                    self.add_IQ_ILC(
                        ch=ch,
                        name=waveformname,
                        I_mhz_vs_us=I_mhz_vs_us,
                        Q_mhz_vs_us=Q_mhz_vs_us,
                        times_us=times_us,
                        plot_IQ=plot_IQ,
                    )
                else:
                    self.add_IQ(
                        ch=ch,
                        name=waveformname,
                        I_mhz_vs_us=I_mhz_vs_us,
                        Q_mhz_vs_us=Q_mhz_vs_us,
                        times_us=times_us,
                        plot_IQ=plot_IQ,
                    )
        if play or set_reg:
            # if not (ch == sigma == None):
            #     print('Warning: you have specified a pulse parameter that can only be changed when loading.')
            params = self.pulse_dict[name].copy()
            if freq_MHz is not None:
                params["freq_MHz"] = freq_MHz
            if phase_deg is not None:
                params["phase_deg"] = phase_deg
            if gain is not None:
                params["gain"] = gain
            if ro_ch is not None:
                params["ro_ch"] = ro_ch
                
            assert params["freq_MHz"] > 0, "IQ pulse may not be calibrated for frequency"
            assert params["gain"] > 0, "IQ pulse may not be calibrated for gain"
            self.set_pulse_registers(
                ch=params["ch"],
                style="arb",
                freq=self.freq2reg(params["freq_MHz"], gen_ch=params["ch"], ro_ch=params["ro_ch"]),
                phase=self.deg2reg(params["phase_deg"], gen_ch=params["ch"]),
                gain=int(params["gain"]),
                waveform=params["waveformname"],
                phrst=phrst,
            )
            if play:
                self.pulse(ch=params["ch"])
                if sync_after:
                    self.sync_all()

    def setup_robust_pulse(
        self,
        q,
        play,
        name,
        ZZ_qubit=None,
        phase_deg=0,
        set_reg=True,
        flag=None,
        reload=False,
        plot_IQ=False,
        sync_after=True,
    ):
        I_values_MHz = None
        Q_values_MHz = None
        times_us = None
        freq_MHz = None
        gain = None

        # print(f"Setting up robust pulse for qubit {q}, ZZ_qubit {ZZ_qubit}")
        if ZZ_qubit is None:
            ZZ_qubit = q

        pulse_cfg = self.cfg.device.qubit.pulses.pihalf_ge_robust
        if name not in self.pulse_dict.keys() or reload:
            pulse_filename = pulse_cfg.filename[q]
            pulse_filepath = os.path.join("S:\\QRAM\\qram_4QR2\\optctrl_pulses", pulse_filename + ".npz")
            pulse_params_dict = dict()  # open file
            with np.load(pulse_filepath) as npzfile:
                for key in npzfile.keys():
                    pulse_params_dict.update({key: npzfile[key]})
            times = pulse_params_dict["times"]

            I_values_MHz = np.array(pulse_params_dict[f"I_{q}"]) * 1e-6
            Q_values_MHz = np.array(pulse_params_dict[f"Q_{q}"]) * 1e-6
            times_us = times * 1e6
            freq_MHz = self.f_ges_robust[q, ZZ_qubit]
            gain = self.pihalf_gain_robust[q, ZZ_qubit]
            # print(name, f"freq_MHz: {freq_MHz}, gain: {gain}")

        self.handle_IQ_pulse(
            name=name,
            ch=self.qubit_chs[q],
            I_mhz_vs_us=I_values_MHz,
            Q_mhz_vs_us=Q_values_MHz,
            times_us=times_us,
            freq_MHz=freq_MHz,
            phase_deg=phase_deg,
            gain=gain,
            set_reg=set_reg,
            flag=flag,
            play=play,
            plot_IQ=plot_IQ,
            reload=reload,
            sync_after=sync_after,
        )

    def add_adiabatic(self, ch, name, mu, beta, period_us):
        """mu, beta are dimensionless"""
        period = self.us2cycles(period_us, gen_ch=ch)
        gencfg = self.soccfg["gens"][ch]
        maxv = gencfg["maxv"] * gencfg["maxv_scale"]
        samps_per_clk = gencfg["samps_per_clk"]
        length = np.round(period) * samps_per_clk
        period *= samps_per_clk
        t = np.arange(0, length)
        iamp, qamp = fitter.adiabatic_iqamp(t, amp_max=1, mu=mu, beta=beta, period=period)
        self.add_pulse(ch=ch, name=name, idata=maxv * iamp, qdata=maxv * qamp)

    def handle_adiabatic_pulse(
        self,
        name,
        waveformname=None,
        ch=None,
        mu=None,
        beta=None,
        period_us=None,
        freq_MHz=None,
        phase_deg=None,
        gain=None,
        reload=False,
        play=False,
        set_reg=False,
        flag=None,
        phrst=0,
    ):
        """
        Load/play an adiabatic pi pulse on channel ch
        """
        if name not in self.pulse_dict.keys() or reload:
            assert None not in [ch, mu, beta, period_us]
            if waveformname is None:
                waveformname = name
            self.pulse_dict.update(
                {
                    name: dict(
                        ch=ch,
                        name=name,
                        waveformname=waveformname,
                        type="adiabatic",
                        mu=mu,
                        beta=beta,
                        period_us=period_us,
                        freq_MHz=freq_MHz,
                        phase_deg=phase_deg,
                        gain=gain,
                        flag=flag,
                    )
                }
            )
            if reload or waveformname not in self.envelopes[ch].keys():
                self.add_adiabatic(ch=ch, name=waveformname, mu=mu, beta=beta, period_us=period_us)
                # print('added gauss pulse', name, 'on ch', ch)
        if play or set_reg:
            # if not (ch == sigma == None):
            #     print('Warning: you have specified a pulse parameter that can only be changed when loading.')
            params = self.pulse_dict[name].copy()
            if freq_MHz is not None:
                params["freq_MHz"] = freq_MHz
            if phase_deg is not None:
                params["phase_deg"] = phase_deg
            if gain is not None:
                params["gain"] = gain
            self.set_pulse_registers(
                ch=params["ch"],
                style="arb",
                freq=self.freq2reg(params["freq_MHz"], gen_ch=params["ch"]),
                phase=self.deg2reg(params["phase_deg"], gen_ch=params["ch"]),
                gain=params["gain"],
                waveform=params["waveformname"],
                phrst=phrst,
            )
            if play:
                # print('playing gauss pulse', params['name'], 'on ch', params['ch'])
                self.pulse(ch=params["ch"])
                self.sync_all()

    def build_full_mux_pulse(
        self,
        mask,
        mux_freqs,
        mixer_freq=None,
        relative_amps=None,
        lengths=None,
        pulse_I_shapes=None,
        pulse_Q_shapes=None,
        times_us=None,
        plot_IQ=True,
        dt_us=0.01e-3,
    ):
        """
        Generates the I/Q modulated pulse on 1 full ch which generates each of the mux_freqs on top of a mixer_freq
        that allows for pulse shaping on each of the frequencies.

        mask: list of which indices in lengths, mux_freqs, pulse_I_shapes, pulse_Q_shapes to play
        mux_freqs: list of frequencies to play in MHz
        mixer_freq: carrier frequency in MHz
        relative_amps: additional scaling factor applied on top of the IQ pulse waveforms to scale the amplitude of each frequency component individually

        Specify either:
        lengths: length of each frequency to play in us; for indices that don't use the max length, nothing is played for that frequency for the rest of the time. If this is specified, a constant pulse is played.

        Or:
        pulse_I_shapes, pulse_Q_shapes: intended pulse shaping on each frequency as if they were generated individually. Just need the I/Q to have right relative amplitude vs us; amplitude will be rescaled to 1 and then by the DAC gain. If pulse shapes are specified, they must be lists of shape (len(mux_freqs), num_time_steps) where num_time_steps is determined by times_us

        returns: tot_I_vs_us, tot_Q_vs_us, the total I/Q waveforms to be played and times_us the times at which these waveforms are sampled
        """
        use_const_lengths = lengths is not None
        use_pulse_shapes = pulse_I_shapes is not None and pulse_Q_shapes is not None and times_us is not None
        assert use_const_lengths or use_pulse_shapes
        assert not (use_const_lengths and use_pulse_shapes), "Specify either const lengths or pulse shapes, not both"

        if use_const_lengths:
            assert len(lengths) == len(mux_freqs)
            tot_length_us = max(lengths)
            times_us = np.linspace(0, tot_length_us, int(tot_length_us / dt_us))
            pulse_I_shapes = np.ones((len(mux_freqs), len(times_us)))
            pulse_Q_shapes = np.zeros((len(mux_freqs), len(times_us)))
        
        
        print(pulse_I_shapes.shape)
        print(pulse_Q_shapes.shape)
        print(times_us.shape)   
            


        modulated_times_us = np.linspace(0, times_us[-1], int(times_us[-1] / dt_us))

        tot_I_vs_us = np.zeros_like(modulated_times_us)
        tot_Q_vs_us = np.zeros_like(modulated_times_us)
        modulated_Is = np.zeros((len(mux_freqs), len(modulated_times_us)))
        modulated_Qs = np.zeros((len(mux_freqs), len(modulated_times_us)))
        if relative_amps is None:
            relative_amps = np.ones(len(mux_freqs))
        relative_amps = np.array(relative_amps)
        bool_mask = np.array([q in mask for q in range(len(relative_amps))], dtype=int)
        relative_amps *= bool_mask
        for q in mask:
            if use_const_lengths:
                mask_func = relative_amps[q] * np.heaviside(lengths[q] - modulated_times_us, 0)
                # print(relative_amps[q])
            else:
                mask_func = relative_amps[q] * np.ones_like(modulated_times_us)
            mask_func[0] = 0
            mask_func[-1] = 0

            I_func = sp.interpolate.interp1d(times_us, pulse_I_shapes[q], kind="linear", fill_value=0)
            Q_func = sp.interpolate.interp1d(times_us, pulse_Q_shapes[q], kind="linear", fill_value=0)
            pulse_I_shape = mask_func * I_func(modulated_times_us)
            pulse_Q_shape = mask_func * Q_func(modulated_times_us)

            modulated_I = pulse_I_shape * np.cos(
                mux_freqs[q] * 2 * np.pi * modulated_times_us
            ) + pulse_Q_shape * np.sin(mux_freqs[q] * 2 * np.pi * modulated_times_us)
            modulated_Is[q] = modulated_I
            tot_I_vs_us += modulated_I

            modulated_Q = -pulse_I_shape * np.sin(
                mux_freqs[q] * 2 * np.pi * modulated_times_us
            ) + pulse_Q_shape * np.cos(mux_freqs[q] * 2 * np.pi * modulated_times_us)
            modulated_Qs[q] = modulated_Q
            tot_Q_vs_us += modulated_Q

        # plot_IQ = True
        if plot_IQ:
            assert mixer_freq is not None
            plt.figure()
            xpts = modulated_times_us
            for q in mask:
                fourier = np.fft.fftshift(
                    np.abs(
                        np.fft.fft(
                            modulated_Is[q] * np.cos(mixer_freq * 2 * np.pi * modulated_times_us)
                            + modulated_Qs[q] * np.sin(mixer_freq * 2 * np.pi * modulated_times_us)
                        )
                    )
                )
                freqs = np.fft.fftshift(np.fft.fftfreq(len(fourier), d=(xpts[1] - xpts[0])))
                plt.plot(freqs, fourier, label=f"Q{q}")
            plt.xlabel("Frequency [MHz]")
            # plt.xlim(0, 2000)
            # plt.ylim(0, 30000)
            plt.legend()
            plt.title(f"Fourier Transform of Modulated Pulse")
            plt.show()

            plt.figure()
            for q in mask:
                plt.plot(modulated_times_us, modulated_Is[q], label=f"I{q}")
                plt.plot(modulated_times_us, modulated_Qs[q], label=f"Q{q}")
            plt.xlabel("Time [us]")
            plt.ylabel("Amplitude [a.u.]")
            plt.legend()
            plt.title(f"Pulse Envelopes")
            plt.show()

        return tot_I_vs_us, tot_Q_vs_us, modulated_times_us

    def handle_full_mux_pulse(
        self,
        name,
        ch,
        relative_amps,
        mask=None,
        mux_freqs=None,
        mixer_freq=None,
        lengths=None,
        pulse_I_shapes=None,
        pulse_Q_shapes=None,
        times_us=None,
        phase_deg=None,
        plot_IQ=True,
        dt_us=0.01e-3,
        ro_ch=None,
        reload=False,
        play=False,
        set_reg=False,
        sync_after=True,
        flag=None,
    ):
        relative_amps = np.array(relative_amps)
        bool_mask = np.array([q in mask for q in range(len(relative_amps))], dtype=int)
        relative_amps *= bool_mask
        assert np.all((relative_amps <= 1) & (relative_amps >= 0)), "Relative amplitudes must be between 0 and 1"

        if name not in self.pulse_dict.keys() or reload:
            assert ch is not None
            assert mask is not None
            assert mux_freqs is not None
            assert mixer_freq is not None
            # assert lengths is not None
            tot_I_vs_us, tot_Q_vs_us, times_us = self.build_full_mux_pulse(
                mask=mask,
                mux_freqs=mux_freqs,
                mixer_freq=mixer_freq,
                lengths=lengths,
                relative_amps=relative_amps,
                pulse_I_shapes=pulse_I_shapes,
                pulse_Q_shapes=pulse_Q_shapes,
                times_us=times_us,
                plot_IQ=plot_IQ,
                dt_us=dt_us,
            )
        else:
            tot_I_vs_us, tot_Q_vs_us, times_us = [None] * 3

        # Scale so the same relative amp always gives the same amount of power regardless of what the other amplitudes are specified as
        gencfg = self.soccfg["gens"][ch]
        maxv = gencfg["maxv"] * gencfg["maxv_scale"] - 1
        gain = maxv * np.sum(relative_amps) / len(mux_freqs)
        assert times_us[-1] > 0, f"Full mux pulse length on gen ch {ch} must be greater than 0"
        # print("gain", gain, mask, mux_freqs, mixer_freq)

        self.handle_IQ_pulse(
            name=name,
            ch=ch,
            I_mhz_vs_us=tot_I_vs_us,
            Q_mhz_vs_us=tot_Q_vs_us,
            times_us=times_us,
            freq_MHz=mixer_freq,
            ro_ch=ro_ch,
            phase_deg=phase_deg,
            gain=gain,
            reload=reload,
            play=play,
            set_reg=set_reg,
            flag=flag,
            sync_after=sync_after,
            plot_IQ=plot_IQ,
        )

    """
    Clifford pulse defns. extra_phase is given in deg. flag can be used to identify certain pulses.
    If play=False, just loads pulse.
    special: adiabatic, pulseiq
    General drive: Omega cos((wt+phi)X) -> Delta/2 Z + Omega/2 (cos(phi) X + sin(phi) Y)
    """

    def X_half_pulse(
        self,
        q,
        divide_len=True,
        ZZ_qubit=None,
        neg=False,
        extra_phase=0,
        play=False,
        set_reg=False,
        name="X",
        flag=None,
        special=None,
        phrst=0,
        reload=False,
        sync_after=True,
        pihalf=True,
        **kwargs,
    ):

        # q: qubit number in config
        if ZZ_qubit is None:
            ZZ_qubit = q

        if special is None and self.use_robust_pulses:  # use robust pulses as the default X/2
            special = "robust"
            
        print('qubit', q)
        print('special', special)
        
        
        
        

        # Get the freq, phase, length, type (assumes using default ge pulse)
        assert self.f_ges.shape == (self.num_qubits_sample, self.num_qubits_sample)
        f_ge_MHz = self.f_ges[q, ZZ_qubit]
        # gain = self.pi_ge_gains[q, ZZ_qubit]
        correction_phase = self.cfg.device.qubit.pulses.pi_ge.half_correction_phase[
            q * self.num_qubits_sample + ZZ_qubit
        ]
        # print("correction phase", correction_phase)
        phase_deg = self.overall_phase[q] + extra_phase + correction_phase
        if neg:
            phase_deg -= 180
        sigma_cycles = self.us2cycles(self.pi_ge_sigmas[q, ZZ_qubit], gen_ch=self.qubit_chs[q])
        type = self.cfg.device.qubit.pulses.pi_ge.type[q]

        waveformname = "pi_ge"
        if special:
            if special == "adiabatic":
                gain = self.cfg.device.qubit.pulses.pi_ge_adiabatic.gain[q]
                period_us = self.cfg.device.qubit.pulses.pi_ge_adiabatic.period[q]
                mu = self.cfg.device.qubit.pulses.pi_ge_adiabatic.mu[q]
                beta = self.cfg.device.qubit.pulses.pi_ge_adiabatic.beta[q]
                if "adiabatic" not in name:
                    name = name + "_adiabatic"
                waveformname = "pi_ge_adiabatic"
                type = "adiabatic"
            elif special == "pulseiq":
                type = "pulseiq"
                waveformname = "pi_ge_IQ"
                assert (
                    "I_mhz_vs_us" in kwargs.keys() and "Q_mhz_vs_us" in kwargs.keys() and "times_us" in kwargs.keys()
                )
                I_mhz_vs_us = kwargs["I_mhz_vs_us"]
                Q_mhz_vs_us = kwargs["Q_mhz_vs_us"]
                times_us = kwargs["times_us"]

            elif special == "robust":
                type = "robust"
                name += "_robust"
            elif special == "gauss":
                type = "gauss"
                name += "_gauss"

        if ZZ_qubit != q:
            waveformname += f"_ZZ{ZZ_qubit}"
            name += f"_ZZ{ZZ_qubit}"

        if divide_len:
            sigma_cycles = sigma_cycles // 2
            waveformname += "_half"
            gain = self.pi_ge_half_gains[q, ZZ_qubit]
        else:
            gain = self.pi_ge_half_gain_pi_sigmas[q, ZZ_qubit]

        name += "_half"

        if special == "robust":
            # gain, freq are retrieved within setup_robust_pulse
            plot_IQ = False
            if "plot_IQ" in self.cfg.expt and self.cfg.expt.plot_IQ:
                plot_IQ = True
            self.setup_robust_pulse(
                q,
                name=f"{name}_q{q}",
                ZZ_qubit=ZZ_qubit,
                phase_deg=phase_deg,
                play=play,
                set_reg=set_reg,
                reload=reload,
                plot_IQ=plot_IQ,
                sync_after=sync_after,
                flag=flag,
            )
            return

        assert f_ge_MHz > 0, f'pulse on {q} {"ZZ "+str(ZZ_qubit) if ZZ_qubit != q else ""}freq may not be calibrated'
        assert (
            gain > 0
        ), f'pihalf pulse on {q} {"ZZ "+str(ZZ_qubit) if ZZ_qubit != q else ""}gain may not be calibrated'
        assert (
            sigma_cycles > 0
        ), f'pulse on {q} {"ZZ "+str(ZZ_qubit) if ZZ_qubit != q else ""}sigma may not be calibrated'

        if type == "const":
            self.handle_const_pulse(
                name=f"{name}_q{q}",
                ch=self.qubit_chs[q],
                waveformname=f"{waveformname}_q{q}",
                length=sigma_cycles,
                freq_MHz=f_ge_MHz,
                phase_deg=phase_deg,
                gain=gain,
                play=play,
                set_reg=set_reg,
                flag=flag,
                phrst=phrst,
                reload=reload,
                sync_after=sync_after,
            )
        elif type == "gauss":
            self.handle_gauss_pulse(
                name=f"{name}_q{q}",
                ch=self.qubit_chs[q],
                waveformname=f"{waveformname}_q{q}",
                sigma=sigma_cycles,
                freq_MHz=f_ge_MHz,
                phase_deg=phase_deg,
                gain=gain,
                play=play,
                set_reg=set_reg,
                flag=flag,
                phrst=phrst,
                reload=reload,
                sync_after=sync_after,
            )
            # if play: print(f"playing phase {phase_deg}, waveform {waveformname}_q{q}, gain {gain}, ch {self.qubit_chs[q]}")
        elif type == "adiabatic":
            assert not pihalf, "Cannot do pihalf pulse with adiabatic"
            self.handle_adiabatic_pulse(
                name=f"{name}_q{q}",
                ch=self.qubit_chs[q],
                waveformname=f"{waveformname}_q{q}",
                mu=mu,
                beta=beta,
                period_us=period_us,
                freq_MHz=f_ge_MHz,
                phase_deg=phase_deg,
                gain=gain,
                play=play,
                set_reg=set_reg,
                flag=flag,
                phrst=phrst,
                reload=reload,
                sync_after=sync_after,
            )
        elif type == "pulseiq":
            assert not pihalf, "Cannot do pihalf pulse with pulseiq"
            self.handle_IQ_pulse(
                name=f"{name}_q{q}",
                ch=self.qubit_chs[q],
                waveformname=f"{waveformname}_q{q}",
                I_mhz_vs_us=I_mhz_vs_us,
                Q_mhz_vs_us=Q_mhz_vs_us,
                times_us=times_us,
                freq_MHz=f_ge_MHz,
                phase_deg=phase_deg,
                gain=gain,
                play=play,
                set_reg=set_reg,
                flag=flag,
                phrst=phrst,
                reload=reload,
                sync_after=sync_after,
            )

        elif type == "flat_top":
            assert False, "flat top not checked yet"
            flat_length = (
                self.us2cycles(self.cfg.device.qubit.pulses.pi_ge.length[q], gen_ch=self.qubit_chs[q]) - 3 * 4
            )
            self.handle_flat_top_pulse(
                name=f"{name}_q{q}",
                ch=self.qubit_chs[q],
                waveformname=f"{waveformname}_q{q}",
                sigma=sigma_cycles,
                flat_length=flat_length,
                freq_MHz=f_ge_MHz,
                phase_deg=phase_deg,
                gain=gain,
                play=play,
                set_reg=set_reg,
                flag=flag,
                phrst=phrst,
                reload=reload,
            )
        else:
            assert False, f"Pulse type {type} not supported."

    def X_pulse(
        self,
        q,
        pihalf=False,
        divide_len=True,
        ZZ_qubit=None,
        neg=False,
        extra_phase=0,
        play=False,
        set_reg=False,
        name="X",
        flag=None,
        special=None,
        phrst=0,
        reload=False,
        sync_after=True,
        **kwargs,
    ):
        
        # if q==1:
        #     print('ENFORCING GAUSSIAN PULSE for qubit 1')
        #     special="gauss"
            
        # print('using X_pulse')
        # print('qubit', q)
        # print('special', special)
            
            
        n_pulse = 1
        if not pihalf:
            n_pulse = 2

        for i in range(n_pulse):
            self.X_half_pulse(
                q=q,
                divide_len=divide_len,
                ZZ_qubit=ZZ_qubit,
                neg=neg,
                extra_phase=extra_phase,
                play=play,
                set_reg=set_reg,
                name=name,
                flag=flag,
                special=special,
                phrst=phrst,
                reload=reload,
                sync_after=sync_after,
                pihalf=pihalf,
                **kwargs,
            )

    def Y_pulse(
        self,
        q,
        pihalf=False,
        divide_len=True,
        ZZ_qubit=None,
        neg=False,
        extra_phase=0,
        adiabatic=False,
        play=False,
        set_reg=False,
        flag=None,
        phrst=0,
        special=None,
        reload=False,
        sync_after=True,
    ):
        # The sign of the 180 does not matter, but the sign of the pihalf does!
        self.X_pulse(
            q,
            pihalf=pihalf,
            divide_len=divide_len,
            ZZ_qubit=ZZ_qubit,
            neg=not neg,
            extra_phase=90 + extra_phase,
            play=play,
            set_reg=set_reg,
            name="Y",
            flag=flag,
            adiabatic=adiabatic,
            special=special,
            phrst=phrst,
            reload=reload,
            sync_after=sync_after,
        )

    def Z_pulse(self, q, pihalf=False, neg=False, extra_phase=0, play=False, **kwargs):
        dac_type = self.qubit_ch_types[q]
        assert not dac_type == "mux4", "Currently cannot set phase for mux4!"
        phase_adjust = 180
        if pihalf:
            phase_adjust = 90  # the sign of the 180 does not matter, but the sign of the pihalf does!
        if neg:
            phase_adjust *= -1
        if play:
            self.overall_phase[q] += phase_adjust + extra_phase

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

    def setup_readout(self):
        """
        Declare resonator generators, declare ADCs, add readout pulses

        For a fullmux expt:
        full_mux_expt: whether to use the full_mux to do the readout or standard mux setup
        If True, specify:
        full_mux_chs: list of full mux channels to use for all possible qubits
        mask: list of qubits to play the readout
        Specify either (see build_full_mux_pulse for more details):
        lengths
        OR
        pulse_I_shapes
        pulse_Q_shapes
        times_us (should be just a 1d array that is the same for all channels in full_mux_chs)

        To automatically set up readout reset using the cfg params, the minimal you need to specify is
        full_mux_expt: True
        resonator_reset: list of qubits to generate a reset pulse on the full_mux_chs
        (other qubits will get a constant pulse)
        """
        full_mux_expt = False
        if "full_mux_expt" in self.cfg.expt and self.cfg.expt.full_mux_expt:
            full_mux_expt = self.cfg.expt.full_mux_expt

        if "resonator_reset" in self.cfg.expt and self.cfg.expt.resonator_reset is not None:
            assert (
                full_mux_expt
            ), "Warning: you are trying to do resonator reset but have not specified cfg.full_mux_expt = True"

        if not full_mux_expt:
            self.setup_mux_gen_readout()
        else:
            override_reset_pulse_shape = "pulse_I_shapes" in self.cfg.expt and self.cfg.expt.pulse_I_shapes is not None
            if (
                "resonator_reset" in self.cfg.expt
                and self.cfg.expt.resonator_reset is not None
                and not override_reset_pulse_shape
            ):
                assert (
                    len(np.array(self.cfg.expt.resonator_reset)) > 0
                ), "resonator_reset must be a list of qubits to apply reset pulse to; other qubits will receive a constant pulse of the same length as the reset pulse"

                kappa_ext = self.cfg.device.readout.kappa_ext
                kerr = self.cfg.device.readout.kerr
                t_rise = self.cfg.device.readout.t_rise_reset
                readout_length = self.cfg.device.readout.readout_length
                self.cfg.expt.full_mux_chs = self.cfg.hw.soc.dacs.readout.full_mux_chs
                self.cfg.expt.mask = [0, 1, 2, 3]

                # check that all readout lengths are the same
                assert len(set(readout_length)) == 1, "All readout lengths must be the same"
                Tp = readout_length[0]
                nstep = 500
                times_us = np.linspace(0, Tp, nstep)
                assert Tp > 0, "Readout pulse length must be greater than 0"

                pulse_I_shapes = []
                pulse_Q_shapes = []

                if "plot_IQ" in self.cfg.expt and self.cfg.expt.plot_IQ:
                    plt.figure(figsize=(5, 4))
                    plt.suptitle(f"Readout Reset Pulses Masking {self.cfg.expt.resonator_reset}")

                for q in range(self.num_qubits_sample):
                    if q in self.cfg.expt.resonator_reset:
                        readout_pulser = ReadoutResetPulser(
                            kappa_ext_MHz=2 * np.pi * kappa_ext[q],
                            kappa_int_MHz=0,
                            chi_MHz=0,
                            kerr_MHz=2 * np.pi * kerr[q],
                        )
                        _, (I_pulse, Q_pulse) = readout_pulser.flat_top_kerr_drive(
                            t_rise=t_rise[q], Tp=Tp, num_filter=1, nstep=nstep
                        )
                    else:
                        I_pulse, Q_pulse = np.ones((2, nstep))
                    pulse_I_shapes.append(I_pulse)
                    pulse_Q_shapes.append(Q_pulse)

                    if "plot_IQ" in self.cfg.expt and self.cfg.expt.plot_IQ:
                        plt.plot(times_us, I_pulse, label=f"I{q}")
                        plt.plot(times_us, Q_pulse, label=f"Q{q}")

                pulse_I_shapes = np.array(pulse_I_shapes)
                pulse_Q_shapes = np.array(pulse_Q_shapes)

                if "plot_IQ" in self.cfg.expt and self.cfg.expt.plot_IQ:
                    plt.legend()
                    plt.xlabel("Time [us]")
                    plt.tight_layout()
                    plt.show()

                self.cfg.expt.pulse_I_shapes = pulse_I_shapes
                self.cfg.expt.pulse_Q_shapes = pulse_Q_shapes
                self.cfg.expt.times_us = times_us
                self.cfg.expt.lengths = None

            assert "full_mux_chs" in self.cfg.expt and self.cfg.expt.full_mux_chs is not None
            full_mux_chs = self.cfg.expt.full_mux_chs
            assert "mask" in self.cfg.expt and self.cfg.expt.mask is not None
            mask = self.cfg.expt.mask
            assert mask == [
                0,
                1,
                2,
                3,
            ], "Most programs assume all ADC chs are triggered so the overall mask should be all qubits!"

            lengths = self.cfg.expt.lengths if "lengths" in self.cfg.expt else None
            pulse_I_shapes = self.cfg.expt.pulse_I_shapes if "pulse_I_shapes" in self.cfg.expt else None
            pulse_Q_shapes = self.cfg.expt.pulse_Q_shapes if "pulse_Q_shapes" in self.cfg.expt else None
            times_us = self.cfg.expt.times_us if "times_us" in self.cfg.expt else None
            if lengths is not None:
                self.readout_lengths_adc = [
                    self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(lengths, self.adc_chs)
                ]
            else:
                self.readout_lengths_adc = [self.us2cycles(times_us[-1], ro_ch=ro_ch) for ro_ch in self.adc_chs]
                
            if "len_readout_adc" in self.cfg.expt and self.cfg.expt.len_readout_adc is not None:
                self.readout_lengths_adc = [
                    self.us2cycles(length, ro_ch=ro_ch)
                    for length, ro_ch in zip(self.cfg.expt.len_readout_adc, self.adc_chs)
                ]

            if "len_readout_adc" in self.cfg.expt and self.cfg.expt.len_readout_adc is not None:
                self.readout_lengths_adc = [
                    self.us2cycles(length, ro_ch=ro_ch)
                    for length, ro_ch in zip(self.cfg.expt.len_readout_adc, self.adc_chs)
                ]

            mixer_freqs = np.array(self.cfg.hw.soc.dacs.readout.mixer_freq)
            # assert np.all(mixer_freqs == mixer_freqs[0])
            if "plot_IQ" in self.cfg.expt:
                plot_IQ = self.cfg.expt.plot_IQ
            else:
                plot_IQ = False
            mux_freqs = self.cfg.device.readout.frequency
            mux_gains = self.cfg.device.readout.gain
            self.setup_fullmux_readout(
                full_mux_chs=full_mux_chs,
                mask=mask,
                mixer_freqs=mixer_freqs,
                mux_freqs=mux_freqs,
                mux_gains=mux_gains,
                lengths=lengths,
                pulse_I_shapes=pulse_I_shapes,
                pulse_Q_shapes=pulse_Q_shapes,
                times_us=times_us,
                plot_IQ=plot_IQ,
            )

    def setup_mux_gen_readout(self):
        self.f_res_regs = [
            self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch)
            for f, gen_ch, adc_ch in zip(self.cfg.device.readout.frequency, self.res_chs, self.adc_chs)
        ]

        mux_mixer_freq = None
        mux_freqs = [0] * 4  # MHz
        mux_gains = [0] * 4
        mux_ro_ch = None
        mux_nqz = None
        for q in range(self.num_qubits_sample):
            assert self.res_ch_types[q] in ["full", "mux4"]
            if self.res_ch_types[q] == "full":
                if self.res_chs[q] not in self.measure_chs:
                    self.declare_gen(
                        ch=self.res_chs[q], nqz=self.cfg.hw.soc.dacs.readout.nyquist[q]
                    )  # , ro_ch=self.adc_chs[q])

                    if self.cfg.device.readout.gain[q] < 1:
                        gain = int(self.cfg.device.readout.gain[q] * 2**15)
                    self.handle_const_pulse(
                        name=f"measure{q}",
                        ch=self.res_chs[q],
                        ro_ch=self.adc_chs[q],
                        length=max(self.readout_lengths_dac),
                        freq_MHz=self.cfg.device.readout.frequency[q],
                        phase_deg=0,
                        gain=gain,
                        play=False,
                        set_reg=True,
                    )
                    self.measure_chs.append(self.res_chs[q])
                    self.meas_ch_types.append(self.res_ch_types[q])
                    self.meas_ch_qs.append(q)

            elif self.res_ch_types[q] == "mux4":
                assert self.res_chs[q] == 6
                self.mask.append(q)
                if mux_mixer_freq is None:
                    mux_mixer_freq = self.cfg.hw.soc.dacs.readout.mixer_freq[q]
                else:
                    assert (
                        mux_mixer_freq == self.cfg.hw.soc.dacs.readout.mixer_freq[q]
                    )  # ensure all mux channels have specified the same mixer freq
                mux_freqs[q] = self.cfg.device.readout.frequency[q]
                mux_gains[q] = self.cfg.device.readout.gain[q]
                mux_ro_ch = self.adc_chs[q]
                mux_nqz = self.cfg.hw.soc.dacs.readout.nyquist[q]
                if self.res_chs[q] not in self.measure_chs:
                    self.measure_chs.append(self.res_chs[q])
                    self.meas_ch_types.append("mux4")
                    self.meas_ch_qs.append(-1)

        # declare mux4 channel
        if "mux4" in self.res_ch_types:
            self.declare_gen(
                ch=6, nqz=mux_nqz, mixer_freq=mux_mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=mux_ro_ch
            )
            self.handle_mux4_pulse(
                name=f"measure", ch=6, length=max(self.readout_lengths_dac), mask=self.mask, play=False, set_reg=True
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

    def setup_fullmux_readout(
        self,
        full_mux_chs,
        mask,
        mixer_freqs,
        mux_freqs,
        mux_gains,
        mux_nqz=2,
        lengths=None,
        pulse_I_shapes=None,
        pulse_Q_shapes=None,
        times_us=None,
        plot_IQ=False,
    ):
        mask_dict_unique_chs = dict()
        mixer_dict_unique_chs = dict()

        for i_ch, ch in enumerate(full_mux_chs):
            if ch not in mask_dict_unique_chs.keys():
                mask_dict_unique_chs[ch] = []
                mixer_dict_unique_chs[ch] = mixer_freqs[i_ch]
            if i_ch in mask:
                mask_dict_unique_chs[ch].append(i_ch)
            assert (
                mixer_freqs[i_ch] == mixer_dict_unique_chs[ch]
            ), f"All mixer freqs that use the same full_mux_ch should be the same but mixer for qubit {i_ch} is {mixer_freqs[i_ch]} and previous mixer_freq is {mixer_dict_unique_chs[ch]} on full mux ch {ch}"

        self.f_res_regs = [0] * self.num_qubits_sample
        rounded_mux_freqs = [0] * self.num_qubits_sample
        rounded_freqs = [0] * self.num_qubits_sample
        rounded_mixer_freq_dict_unique_chs = dict()
        for q in range(self.num_qubits_sample):
            # Need mixer_mux_rounded + mux_rounded = adc_rounded = mixer_full_rounded + full_rounded
            full_mux_ch = full_mux_chs[q]
            orig_mixer_freq = mixer_freqs[q]
            chs_to_round = [self.soccfg["gens"][full_mux_ch]]

            mask_ch = mask_dict_unique_chs[full_mux_ch]
            adc_chs_ch = [self.adc_chs[qq] for qq in mask_ch]
            for ch in adc_chs_ch:
                chs_to_round.append(self.soccfg["readouts"][ch])

            rounded_mixer_freq = self.roundfreq(orig_mixer_freq, chs_to_round)
            rounded_mixer_freq_dict_unique_chs[full_mux_ch] = rounded_mixer_freq
            # rounded_mux_freqs[q] = np.abs(self.roundfreq(mux_freqs[q], chs_to_round))
            rounded_mux_freqs[q] = self.roundfreq(mux_freqs[q], chs_to_round)
            rounded_freqs[q] = rounded_mixer_freq + rounded_mux_freqs[q]

            self.f_res_regs[q] = self.freq2reg(rounded_mux_freqs[q], gen_ch=full_mux_ch, ro_ch=self.adc_chs[q])
            self.res_chs[q] = full_mux_ch

        for full_mux_ch in mask_dict_unique_chs.keys():
            qubits_ch = mask_dict_unique_chs[full_mux_ch]
            mixer_freq_ch = mixer_dict_unique_chs[full_mux_ch]
            adc_chs_ch = [self.adc_chs[q] for q in qubits_ch]
            self.declare_gen(ch=full_mux_ch, nqz=mux_nqz, ro_ch=adc_chs_ch[0])

            mask_ch = [i for i in range(len(qubits_ch))]
            mux_freqs_ch = [rounded_mux_freqs[q] for q in qubits_ch]
            rounded_mixer_freq = rounded_mixer_freq_dict_unique_chs[full_mux_ch]
            mux_gains_ch = [mux_gains[q] for q in qubits_ch]
            lengths_ch = [lengths[q] for q in qubits_ch] if lengths is not None else None
            pulse_I_shapes_ch = [pulse_I_shapes[q] for q in qubits_ch] if pulse_I_shapes is not None else None
            pulse_Q_shapes_ch = [pulse_Q_shapes[q] for q in qubits_ch] if pulse_Q_shapes is not None else None
            

            assert max(mux_gains_ch) > 0, f"All mux gains are 0 for gen ch {full_mux_ch}"
            assert len(mask_ch) > 0, f"Len of mask is 0 for gen_ch {full_mux_ch}"
            self.handle_full_mux_pulse(
                name=f"measure",
                ch=full_mux_ch,
                mask=mask_ch,
                mux_freqs=mux_freqs_ch,
                mixer_freq=rounded_mixer_freq,
                relative_amps=mux_gains_ch,
                lengths=lengths_ch,
                pulse_I_shapes=np.array(pulse_I_shapes_ch),
                pulse_Q_shapes=(-1* np.array(pulse_Q_shapes_ch) if pulse_Q_shapes is not None else None),  # the convention is actually consistent with the rfsoc convention, unlike in the optimal control code
                times_us=times_us,
                phase_deg=0,
                plot_IQ=plot_IQ,
                # ro_ch=self.adc_chs[0],  # don't need this since we already rounded the freq I guess
                reload=True,
                play=False,
                set_reg=True,
            )
            self.measure_chs.append(full_mux_ch)
            self.meas_ch_types.append("full")

        # declare adcs - readout for all qubits everytime, defines number of buffers returned regardless of number of adcs triggered
        for q in range(self.num_qubits_sample):
            print('ADC readout length', self.readout_lengths_adc[q])
            if self.adc_chs[q] not in self.ro_chs:
                self.declare_readout(
                    ch=self.adc_chs[q],
                    length=self.readout_lengths_adc[q],
                    freq=rounded_freqs[q],
                    gen_ch=self.res_chs[q],
                )
                # print(f"adc {self.adc_chs[q]} freq {rounded_freqs[q]}")

    def measure_readout_cool(
        self, n_init_readout=None, n_trig=None, init_read_wait_us=None, extended_readout_delay_cycles=3
    ):
        """
        Pulse and acquire the initial readout pulse for pre-selection
        """
        if n_init_readout is None:
            assert "n_init_readout" in self.cfg.expt
            n_init_readout = self.cfg.expt.n_init_readout
        if n_trig is None:
            assert "n_trig" in self.cfg.expt
            n_trig = self.cfg.expt.n_trig
        if init_read_wait_us is None:
            assert "init_read_wait_us" in self.cfg.expt
            init_read_wait_us = self.cfg.expt.init_read_wait_us
        if "rounds" in self.cfg.expt:
            assert self.cfg.expt.rounds == 1, "shots get averaged in a weird way when rounds != 1"

        if "use_gf_readout" not in self.cfg.expt or not self.cfg.expt.use_gf_readout:
            self.use_gf_readout = None
        else:
            self.use_gf_readout = self.cfg.expt.use_gf_readout
        if self.use_gf_readout is not None:
            # n_trig *= 2
            self.gf_readout_init(qubits=self.cfg.expt.use_gf_readout)
        for i_readout in range(n_init_readout):
            for i_trig in range(n_trig):
                trig_offset = self.cfg.device.readout.trig_offset[0]
                if i_trig == n_trig - 1:
                    syncdelay = self.us2cycles(init_read_wait_us)  # last readout for this trigger stack
                else:
                    syncdelay = (
                        extended_readout_delay_cycles  # only sync to the next readout in the same trigger stack
                    )
                    # trig_offset = 0

                # print('sync delay us', self.cycles2us(syncdelay))
                # Note that by default the mux channel will play the pulse for all frequencies for the max of the pulse times on all channels - but the acquistion may not be happening the entire time.
                self.measure(
                    # pulse_ch=self.readout_cool_measure_chs,
                    pulse_ch=self.measure_chs,
                    adcs=self.adc_chs,
                    adc_trig_offset=trig_offset,
                    wait=True,
                    syncdelay=syncdelay,
                )

    def get_shots(self, angle=None, threshold=None, avg_shots=False, verbose=False, amplitude_mode=False):
        """
        Collect shots for all adcs, rotates by given angle (degrees), separate based on threshold (if not None), and averages over all shots (i.e. returns data[num_chs, 1] as opposed to data[num_chs, num_shots]) if requested.
        Returns avgi (idata), avgq (qdata) which avgi/q are avg over shot_avg
        """

        idata, qdata = self.get_multireadout_shots(
            angle=angle, threshold_final=threshold, amplitude_mode=amplitude_mode
        )

        idata = idata[:, -1, :]
        qdata = qdata[:, -1, :]
        if avg_shots:
            idata = np.average(idata, axis=1)
            qdata = np.average(qdata, axis=1)
        return idata, qdata

    def get_multireadout_shots(self, angle=None, threshold_final=None, amplitude_mode=False, avg_trigs=True):
        """
        For all readouts, angle is applied if None; threshold_final is applied only to the last readout
        threshold_final should be specified for all qubits
        If avg_trigs is False, return as if each trig is a separate readout
        If amplitude_mode is True, does thresholding based on amplitude (and assumes the threshold provided is this threshold)
        """
        n_init_readout = self.cfg.expt.n_init_readout
        n_trig = self.cfg.expt.n_trig
        # print('n_init_readout', n_init_readout, 'n_trig', n_trig)

        # NOTE: this code assumes the number of expts in a single program is 1 (i.e. this must be an Averager not RAverager program!)

        self.num_qubits_sample = len(self.cfg.device.readout.frequency)
        if angle is None:
            angle = [0] * self.num_qubits_sample

        di_buf = np.array([self.di_buf[i] / ro["length"] for i, (ch, ro) in enumerate(self.ro_chs.items())])
        dq_buf = np.array([self.dq_buf[i] / ro["length"] for i, (ch, ro) in enumerate(self.ro_chs.items())])
        for i, ch in enumerate(self.ro_chs):
            idata_ch, qdata_ch = rotate_and_threshold(
                ishots_1q=di_buf[i],
                qshots_1q=dq_buf[i],
                angle=angle[i],
                threshold=None,
                amplitude_mode=amplitude_mode,
                avg_shots=False,
            )
            di_buf[i] = idata_ch
            dq_buf[i] = qdata_ch

        shots_i = di_buf.reshape((len(self.ro_chs), (1 + n_init_readout * n_trig) * self.cfg.expt.reps))
        shots_q = dq_buf.reshape((len(self.ro_chs), (1 + n_init_readout * n_trig) * self.cfg.expt.reps))

        shots_reshaped_shape = (len(self.ro_chs), 1 + n_init_readout, self.cfg.expt.reps)
        if not avg_trigs:
            shots_reshaped_shape = (len(self.ro_chs), 1 + n_init_readout * n_trig, self.cfg.expt.reps)
        shots_i_reshaped = np.zeros(shots_reshaped_shape)
        shots_q_reshaped = np.zeros(shots_reshaped_shape)
        for i in range(len(self.ro_chs)):
            meas_per_expt = 1 + n_init_readout * n_trig

            # reshape + average over n_trig for the init readouts
            if n_init_readout > 0 and n_trig > 0:
                # init reads shape: reps, n_init_readout, n_trig
                if avg_trigs:
                    shots_i_init_reads = np.reshape(
                        np.reshape(shots_i[i], (self.cfg.expt.reps, meas_per_expt))[:, :-1],
                        (self.cfg.expt.reps, n_init_readout, n_trig),
                    )
                    shots_q_init_reads = np.reshape(
                        np.reshape(shots_q[i], (self.cfg.expt.reps, meas_per_expt))[:, :-1],
                        (self.cfg.expt.reps, n_init_readout, n_trig),
                    )
                    shots_i_reshaped[i, :-1, :] = np.average(shots_i_init_reads, axis=2).T
                    shots_q_reshaped[i, :-1, :] = np.average(shots_q_init_reads, axis=2).T
                else:
                    shots_i_init_reads = np.reshape(
                        np.reshape(shots_i[i], (self.cfg.expt.reps, meas_per_expt))[:, :-1],
                        (self.cfg.expt.reps, n_init_readout * n_trig),
                    )
                    shots_q_init_reads = np.reshape(
                        np.reshape(shots_q[i], (self.cfg.expt.reps, meas_per_expt))[:, :-1],
                        (self.cfg.expt.reps, n_init_readout * n_trig),
                    )
                    shots_i_reshaped[i, :-1, :] = shots_i_init_reads.T
                    shots_q_reshaped[i, :-1, :] = shots_q_init_reads.T

            # reshape for the final readout (only 1 n_trig here)
            # final read shape: reps
            shots_i_final_read = np.reshape(shots_i[i], (self.cfg.expt.reps, meas_per_expt))[:, -1]
            shots_q_final_read = np.reshape(shots_q[i], (self.cfg.expt.reps, meas_per_expt))[:, -1]
            shots_i_reshaped[i, -1, :] = shots_i_final_read
            shots_q_reshaped[i, -1, :] = shots_q_final_read

        if threshold_final is not None:
            for ch in range(len(self.ro_chs)):
                shots_i_reshaped[ch, -1, :], _ = rotate_and_threshold(
                    ishots_1q=shots_i_reshaped[ch, -1, :],
                    qshots_1q=shots_q_reshaped[ch, -1, :],
                    threshold=threshold_final[ch],
                    amplitude_mode=amplitude_mode,
                )

        # final shape: (ro_chs, n_init_readout + 1, reps)
        # or if not avg_trigs: (ro_chs, n_init_readout*n_trig + 1, reps)
        return shots_i_reshaped, shots_q_reshaped

    def acquire(self, soc, load_pulses=True, progress=False, save_experiments=None):
        if not self.readout_cool:
            self.cfg.expt.n_trig = 1
            self.cfg.expt.n_init_readout = 0
        return super().acquire(
            soc,
            load_pulses=load_pulses,
            progress=progress,
            readouts_per_experiment=1 + self.cfg.expt.n_trig * self.cfg.expt.n_init_readout,
            save_experiments=save_experiments,
        )

    def acquire_rotated(
        self,
        soc,
        progress,
        angle=None,
        threshold=None,
        ge_avgs=None,
        amplitude_mode=False,
        post_process=None,
        verbose=False,
    ):
        """
        If post_process == 'threshold': uses angle + threshold to categorize shots into 0 or 1 and calculate the population
        If post_process == 'scale': uses angle + ge_avgs to scale the average of all shots on a scale of 0 to 1. ge_avgs should be of shape (num_total_qubits, 4) and should represent the pre-rotation Ig, Qg, Ie, Qe
        If post_process == None: uses angle to rotate the i and q and then returns the avg i and q
        """
        avgi, avgq = self.acquire(soc, load_pulses=True, progress=progress)
        if post_process == None:
            avgi_rot, avgq_rot = self.get_shots(
                angle=angle, avg_shots=True, verbose=verbose, amplitude_mode=amplitude_mode
            )
            return avgi_rot, avgq_rot
        elif post_process == "threshold":
            assert threshold is not None
            popln, avgq_rot = self.get_shots(
                angle=angle, threshold=threshold, avg_shots=True, verbose=verbose, amplitude_mode=amplitude_mode
            )
            return popln, avgq_rot
        elif post_process == "scale":
            assert ge_avgs is not None
            avgi_rot, avgq_rot = self.get_shots(
                angle=angle, avg_shots=True, verbose=verbose, amplitude_mode=amplitude_mode
            )

            ge_avgs_rot = [None] * 4
            for q, angle_q in enumerate(angle):
                if not isinstance(ge_avgs[q], (list, np.ndarray)):
                    continue  # this qubit was not calibrated
                Ig_q, Qg_q, Ie_q, Qe_q = ge_avgs[q]
                ge_avgs_rot[q] = [
                    Ig_q * np.cos(np.pi / 180 * angle_q) - Qg_q * np.sin(np.pi / 180 * angle_q),
                    Ie_q * np.cos(np.pi / 180 * angle_q) - Qe_q * np.sin(np.pi / 180 * angle_q),
                ]
            shape = None
            for q in range(4):
                if ge_avgs_rot[q] is not None:
                    shape = np.shape(ge_avgs_rot[q])
                    break
            for q in range(4):
                if ge_avgs_rot[q] is None:
                    ge_avgs_rot[q] = np.zeros(shape=shape)

            ge_avgs_rot = np.asarray(ge_avgs_rot)
            avgi_rot -= ge_avgs_rot[:, 0]
            avgi_rot /= ge_avgs_rot[:, 1] - ge_avgs_rot[:, 0]
            return avgi_rot, avgq_rot
        else:
            assert False, "Undefined post processing flag, options are None, threshold, scale"

    def initialize(self):
        self.cfg = AttrDict(self.cfg)
        self.cfg.update(self.cfg.expt)
        if "qubits" in self.cfg.expt:
            self.qubits = self.cfg.expt.qubits
        else:
            self.qubits = range(4)
        self.pulse_dict = dict()
        self.num_qubits_sample = len(self.cfg.device.readout.frequency)

        # all of these saved self.whatever instance variables should be indexed by the actual qubit number as opposed to qubits_i. this means that more values are saved as instance variables than is strictly necessary, but this is overall less confusing
        self.adc_chs = self.cfg.hw.soc.adcs.readout.ch
        self.res_chs = self.cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = self.cfg.hw.soc.dacs.readout.type
        self.qubit_chs = self.cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = self.cfg.hw.soc.dacs.qubit.type

        self.cool_qubits = False
        if "cool_qubits" in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            self.cool_qubits = self.cfg.expt.cool_qubits
            self.swap_f0g1_chs = self.cfg.hw.soc.dacs.swap_f0g1.ch
            self.swap_f0g1_ch_types = self.cfg.hw.soc.dacs.swap_f0g1.type
            mixer_freqs = self.cfg.hw.soc.dacs.swap_f0g1.mixer_freq

        self.readout_cool = False
        if "readout_cool" in self.cfg.expt and self.cfg.expt.readout_cool:
            self.readout_cool = self.cfg.expt.readout_cool

        self.overall_phase = [0] * self.num_qubits_sample

        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs]  # get register page for qubit_ch

        self.f_ges = np.reshape(self.cfg.device.qubit.f_ge, (self.num_qubits_sample, self.num_qubits_sample))
        self.f_efs = np.reshape(self.cfg.device.qubit.f_ef, (self.num_qubits_sample, self.num_qubits_sample))
        self.pi_ge_gains = np.reshape(
            self.cfg.device.qubit.pulses.pi_ge.gain, (self.num_qubits_sample, self.num_qubits_sample)
        )
        self.pi_ge_sigmas = np.reshape(
            self.cfg.device.qubit.pulses.pi_ge.sigma, (self.num_qubits_sample, self.num_qubits_sample)
        )
        self.pi_ge_half_gains = np.reshape(
            self.cfg.device.qubit.pulses.pi_ge.half_gain, (self.num_qubits_sample, self.num_qubits_sample)
        )
        self.pi_ge_half_gain_pi_sigmas = np.reshape(
            self.cfg.device.qubit.pulses.pi_ge.half_gain_pi_sigma, (self.num_qubits_sample, self.num_qubits_sample)
        )
        self.pi_ef_gains = np.reshape(
            self.cfg.device.qubit.pulses.pi_ef.gain, (self.num_qubits_sample, self.num_qubits_sample)
        )
        self.pi_ef_sigmas = np.reshape(
            self.cfg.device.qubit.pulses.pi_ef.sigma, (self.num_qubits_sample, self.num_qubits_sample)
        )
        self.pi_ef_half_gains = np.reshape(
            self.cfg.device.qubit.pulses.pi_ef.half_gain, (self.num_qubits_sample, self.num_qubits_sample)
        )
        self.pi_ef_half_gain_pi_sigmas = np.reshape(
            self.cfg.device.qubit.pulses.pi_ef.half_gain_pi_sigma, (self.num_qubits_sample, self.num_qubits_sample)
        )
        self.f_ges_robust = np.reshape(
            self.cfg.device.qubit.f_ge_robust, (self.num_qubits_sample, self.num_qubits_sample)
        )
        self.pihalf_gain_robust = np.reshape(
            self.cfg.device.qubit.pulses.pihalf_ge_robust.gain, (self.num_qubits_sample, self.num_qubits_sample)
        )
        self.pi_ge_types = self.cfg.device.qubit.pulses.pi_ge.type
        self.pi_ef_types = self.cfg.device.qubit.pulses.pi_ef.type

        if self.cool_qubits:
            self.f_f0g1_regs = [
                self.freq2reg(f, gen_ch=ch) for f, ch in zip(self.cfg.device.qubit.f_f0g1, self.qubit_chs)
            ]

        self.readout_lengths_dac = [
            self.us2cycles(length, gen_ch=gen_ch)
            for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)
        ]

        if "len_readout_adc" in self.cfg.expt and self.cfg.expt.len_readout_adc is not None:
            _ro_lengths_adc = self.cfg.expt.len_readout_adc
            print('using custom readout lengths', _ro_lengths_adc)
        else:
            _ro_lengths_adc = self.cfg.device.readout.readout_length

        self.readout_lengths_adc = [
            self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(_ro_lengths_adc, self.adc_chs)
        ]

        # add IQ pulses
        self.use_robust_pulses = False
        if "use_robust_pulses" in self.cfg.expt and self.cfg.expt.use_robust_pulses:
            self.use_robust_pulses = True

        # declare qubit dacs, add qubit pi_ge pulses
        for q in range(self.num_qubits_sample):
            mixer_freq = None
            if self.qubit_ch_types[q] == "int4":
                mixer_freq = self.cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in self.gen_chs:
                self.declare_gen(
                    ch=self.qubit_chs[q], nqz=self.cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq
                )
            self.X_pulse(q=q, play=False, reload=True)

        if self.cool_qubits:
            mixer_freq = None
            for q in self.cfg.expt.cool_qubits:
                if self.swap_f0g1_ch_types[q] == "int4":
                    mixer_freq = mixer_freqs[q]
                if self.swap_f0g1_chs[q] not in self.gen_chs:
                    self.declare_gen(
                        ch=self.swap_f0g1_chs[q], nqz=self.cfg.hw.soc.dacs.swap_f0g1.nyquist[q], mixer_freq=mixer_freq
                    )

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

        # declare res dacs, add readout pulses, declare ADCs
        self.measure_chs = []
        self.meas_ch_types = []
        self.meas_ch_qs = []
        self.mask = []  # indices of mux_freqs, mux_gains list to play
        self.setup_readout()

        self.set_gen_delays()
        self.sync_all(200)


# ===================================================================== #

"""
Take care of extra clifford pulses for qutrits.
"""


class QutritAveragerProgram(CliffordAveragerProgram):
    def Xef_half_pulse(
        self,
        q,
        divide_len=True,
        name="X_ef",
        ZZ_qubit=None,
        neg=False,
        extra_phase=0,
        play=False,
        set_reg=False,
        flag=None,
        phrst=0,
        reload=True,
        sync_after=True,
    ):
        ch = self.qubit_chs[q]
        if ZZ_qubit is None:
            ZZ_qubit = q
        f_ef_MHz = self.f_efs[q, ZZ_qubit]
        # gain = self.pi_ef_gains[q, ZZ_qubit]
        phase_deg = self.overall_phase_ef[q] + extra_phase
        sigma_cycles = self.us2cycles(self.pi_ef_sigmas[q, ZZ_qubit], gen_ch=ch)
        type = self.cfg.device.qubit.pulses.pi_ef.type[q]
        waveformname = "pi_ef"
        if ZZ_qubit != q:
            waveformname += f"_ZZ{ZZ_qubit}"
            name += f"_ZZ{ZZ_qubit}"
        if divide_len:
            sigma_cycles = sigma_cycles // 2
            waveformname += "_half"
            gain = self.pi_ef_half_gains[q, ZZ_qubit]
        else:
            gain = self.pi_ef_half_gain_pi_sigmas[q, ZZ_qubit]
        name += "_half"
        assert (
            f_ef_MHz > 0
        ), f'EF pulse on {q} {"ZZ "+str(ZZ_qubit)+" " if ZZ_qubit != q else ""}freq may not be calibrated'
        assert (
            gain > 0
        ), f'pihalf EF pulse on {q} {"ZZ "+str(ZZ_qubit)+" " if ZZ_qubit != q else ""}gain may not be calibrated'
        assert (
            sigma_cycles > 0
        ), f'pihalf EF pulse on {q} {"ZZ "+str(ZZ_qubit)+" " if ZZ_qubit != q else ""}sigma may not be calibrated'
        if neg:
            phase_deg -= 180
        if type == "const":
            self.handle_const_pulse(
                name=f"{name}_q{q}",
                ch=ch,
                waveformname=f"{waveformname}_q{q}",
                length=sigma_cycles,
                freq_MHz=f_ef_MHz,
                phase_deg=phase_deg,
                gain=gain,
                play=play,
                set_reg=set_reg,
                flag=flag,
                phrst=phrst,
                reload=reload,
                sync_after=sync_after,
            )
        elif type == "gauss":
            self.handle_gauss_pulse(
                name=f"{name}_q{q}",
                ch=ch,
                waveformname=f"{waveformname}_q{q}",
                sigma=sigma_cycles,
                freq_MHz=f_ef_MHz,
                phase_deg=phase_deg,
                gain=gain,
                play=play,
                set_reg=set_reg,
                flag=flag,
                phrst=phrst,
                reload=reload,
                sync_after=sync_after,
            )
        elif type == "flat_top":
            sigma_ramp_cycles = 3
            flat_length_cycles = sigma_cycles - sigma_ramp_cycles * 4
            self.handle_flat_top_pulse(
                name=f"{name}_q{q}",
                ch=ch,
                waveformname=f"{waveformname}_q{q}",
                sigma=sigma_ramp_cycles,
                flat_length=flat_length_cycles,
                freq_MHz=f_ef_MHz,
                phase_deg=phase_deg,
                gain=gain,
                play=play,
                set_reg=set_reg,
                flag=flag,
                phrst=phrst,
                reload=reload,
                sync_after=sync_after,
            )
        else:
            assert False, f"Pulse type {type} not supported."

    def Xef_pulse(
        self,
        q,
        pihalf=False,
        divide_len=True,
        name="X_ef",
        ZZ_qubit=None,
        neg=False,
        extra_phase=0,
        play=False,
        set_reg=False,
        flag=None,
        phrst=0,
        reload=True,
        sync_after=True,
    ):
        n_pulse = 1
        if not pihalf:
            n_pulse = 2
        for i in range(n_pulse):
            self.Xef_half_pulse(
                q=q,
                divide_len=divide_len,
                name=name,
                ZZ_qubit=ZZ_qubit,
                neg=neg,
                extra_phase=extra_phase,
                play=play,
                set_reg=set_reg,
                flag=flag,
                phrst=phrst,
                reload=reload,
                sync_after=sync_after,
            )

    def Yef_pulse(
        self,
        q,
        pihalf=False,
        divide_len=True,
        ZZ_qubit=None,
        neg=False,
        extra_phase=0,
        play=False,
        set_reg=False,
        flag=None,
        phrst=0,
        reload=True,
        sync_after=True,
    ):
        # the sign of the 180 does not matter, but the sign of the pihalf does!
        self.Xef_pulse(
            q,
            pihalf=pihalf,
            divide_len=divide_len,
            ZZ_qubit=ZZ_qubit,
            neg=not neg,
            extra_phase=90 + extra_phase,
            play=play,
            set_reg=set_reg,
            name="Y_ef",
            flag=flag,
            phrst=phrst,
            reload=reload,
            sync_after=sync_after,
        )

    def Zef_pulse(self, q, pihalf=False, neg=False, extra_phase=0, play=False, **kwargs):
        dac_type = self.qubit_ch_types[q]
        assert not dac_type == "mux4", "Currently cannot set phase for mux4!"
        phase_adjust = 180
        if pihalf:
            phase_adjust = 90  # the sign of the 180 does not matter, but the sign of the pihalf does!
        if neg:
            phase_adjust *= -1
        if play:
            self.overall_phase_ef[q] += phase_adjust + extra_phase

    def active_cool(self, cool_qubits, cool_idle):
        """
        cool_idle should be the same length as cool_qubits
        """
        # print('cooling qubits', cool_qubits, 'with idle times', cool_idle)
        assert len(cool_idle) == len(cool_qubits)

        sorted_indices = np.argsort(cool_idle)[::-1]  # sort cooling times longest first
        cool_qubits = np.array(cool_qubits)
        cool_idle = np.array(cool_idle)
        sorted_cool_qubits = cool_qubits[sorted_indices]
        sorted_cool_idle = cool_idle[sorted_indices]
        # print('sorted cool_qubits', sorted_cool_qubits)
        max_idle = sorted_cool_idle[0]

        last_pulse_len = 0
        remaining_idle = max_idle
        for q, idle in zip(sorted_cool_qubits, sorted_cool_idle):
            remaining_idle -= last_pulse_len

            last_pulse_len = 0
            self.Xef_pulse(q=q, play=True)
            last_pulse_len += self.pi_ef_sigmas[q, q] * 4

            pulse_type = self.cfg.device.qubit.pulses.pi_f0g1.type[q]
            pisigma_f0g1 = self.us2cycles(self.cfg.device.qubit.pulses.pi_f0g1.sigma[q], gen_ch=self.swap_f0g1_chs[q])
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
            # print(f'last pulse len q{q}', last_pulse_len)

        remaining_idle -= last_pulse_len
        last_idle = max((remaining_idle, sorted_cool_idle[-1]))
        # print('last idle', last_idle)
        self.sync_all(self.us2cycles(last_idle))

    def gf_readout_init(self, qubits=None, sync_after=False):
        if qubits is None:
            qubits = range(self.num_qubits_sample)
        for q in qubits:
            self.Xef_pulse(q=q, play=True, sync_after=sync_after)
        self.sync_all()

    def initialize(self):
        super().initialize()
        self.overall_phase_ef = [0] * self.num_qubits_sample
        # declare qubit ef pulses
        # print(self.gen_chs)
        for q in range(self.num_qubits_sample):
            self.Xef_pulse(q=q, play=False)
        self.sync_all(200)


# ===================================================================== #
"""
Multiple inheritence testing
"""
# class Clifford():
#     def xpulse(self):
#         print('normal clifford')

#     def ypulse(self):
#         print('y')
#         self.xpulse()

# class CliffordEF(Clifford):
#     def xefpulse(self):
#         print('ef')

# class CliffordEgGf(CliffordEF):
#     def xpulse(self):
#         super().xpulse()
#         print('EgGf')

# class SimRB(Clifford):
#     def clifford(self, flag=None):
#         if flag == 'X': self.xpulse()
#         elif flag == 'Y': self.ypulse()

# class RBEgGf(CliffordEgGf, SimRB):
#     pass

# rbeggf = RBEgGf()
# print(RBEgGf.__mro__)
# rbeggf.clifford(flag='X')
# rbeggf.clifford(flag='Y')

"""
Replace the X/Y/Z pulses with an effective TLS represented by the Eg-Gf pulse.
add_virtual_Z is a flag to determine whether is a virtual Z gate applied on the drive qubit *after* the swap.
extra_phase is applied to the swap itself.
"""


class CliffordEgGfAveragerProgram(QutritAveragerProgram):
    # self.overall_phase keeps track of the EgGf phase insetad of the e-g pulse phase

    def XEgGf_pulse(
        self,
        qDrive,
        qNotDrive,
        pihalf=False,
        divide_len=True,
        name="X_EgGf",
        neg=False,
        extra_phase=0,
        add_virtual_Z=True,
        play=False,
        flag=None,
        phrst=0,
        reload=True,
    ):
        # convention is waveformname is pi_EgGf_qNotDriveqDrive
        if qDrive == 1:
            ch = self.swap_chs[qNotDrive]
            f_EgGf_MHz = self.cfg.device.qubit.f_EgGf[qNotDrive]
            gain = self.cfg.device.qubit.pulses.pi_EgGf.gain[qNotDrive]
            phase_deg = self.overall_phase[qNotDrive] + extra_phase
            sigma_cycles = self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf.sigma[qNotDrive], gen_ch=ch)
            type = self.cfg.device.qubit.pulses.pi_EgGf.type[qNotDrive]
            waveformname = "pi_EgGf"
            if add_virtual_Z:
                virtual_Z = self.cfg.device.qubit.pulses.pi_EgGf.phase[qNotDrive]
        else:
            ch = self.swap_Q_chs[qDrive]
            f_EgGf_MHz = self.cfg.device.qubit.f_EgGf_Q[qDrive]
            gain = self.cfg.device.qubit.pulses.pi_EgGf_Q.gain[qDrive]
            phase_deg = self.overall_phase[qDrive] + extra_phase
            sigma_cycles = self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf_Q.sigma[qDrive], gen_ch=ch)
            type = self.cfg.device.qubit.pulses.pi_EgGf_Q.type[qDrive]
            waveformname = "pi_EgGf"
            if add_virtual_Z:
                virtual_Z = self.cfg.device.qubit.pulses.pi_EgGf_Q.phase[qDrive]
        if pihalf:
            if divide_len:
                # sigma_cycles = sigma_cycles // 2
                if qDrive == 1:
                    sigma_cycles = self.us2cycles(
                        self.cfg.device.qubit.pulses.pi_EgGf.half_sigma[qNotDrive], gen_ch=ch
                    )
                    virtual_Z = self.cfg.device.qubit.pulses.pi_EgGf.half_phase[qNotDrive]
                else:
                    sigma_cycles = self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf_Q.half_sigma[qDrive], gen_ch=ch)
                    virtual_Z = self.cfg.device.qubit.pulses.pi_EgGf_Q.half_phase[qDrive]
                waveformname += "half"
            else:
                assert False, "dividing gain for an eg-gf pi/2 pulse is a bad idea!"
            name += "half"
        if neg:
            phase_deg -= 180
        if type == "const":
            self.handle_const_pulse(
                name=f"{name}_{qNotDrive}{qDrive}",
                ch=ch,
                waveformname=f"{waveformname}_{qNotDrive}{qDrive}",
                length=sigma_cycles,
                freq_MHz=f_EgGf_MHz,
                phase_deg=phase_deg,
                gain=gain,
                play=play,
                flag=flag,
                phrst=phrst,
                reload=reload,
            )
        elif type == "gauss":
            self.handle_gauss_pulse(
                name=f"{name}_{qNotDrive}{qDrive}",
                ch=ch,
                waveformname=f"{waveformname}_{qNotDrive}{qDrive}",
                sigma=sigma_cycles,
                freq_MHz=f_EgGf_MHz,
                phase_deg=phase_deg,
                gain=gain,
                play=play,
                flag=flag,
                phrst=phrst,
                reload=reload,
            )
        elif type == "flat_top":
            sigma_ramp_cycles = 3
            flat_length_cycles = sigma_cycles - sigma_ramp_cycles * 4
            self.handle_flat_top_pulse(
                name=f"{name}_{qNotDrive}{qDrive}",
                ch=ch,
                waveformname=f"{waveformname}_{qNotDrive}{qDrive}",
                sigma=sigma_ramp_cycles,
                flat_length=flat_length_cycles,
                freq_MHz=f_EgGf_MHz,
                phase_deg=phase_deg,
                gain=gain,
                play=play,
                flag=flag,
                phrst=phrst,
                reload=reload,
            )
        else:
            assert False, f"Pulse type {type} not supported."

        if add_virtual_Z:
            self.overall_phase[qDrive] += virtual_Z
        # print('ch keys', self.gen_chs.keys())

    def YEgGf_pulse(
        self,
        qDrive,
        qNotDrive,
        pihalf=False,
        neg=False,
        extra_phase=0,
        add_virtual_Z=False,
        play=False,
        flag=None,
        phrst=0,
        reload=True,
    ):
        # the sign of the 180 does not matter, but the sign of the pihalf does!
        self.XEgGf_pulse(
            qDrive,
            qNotDrive,
            pihalf=pihalf,
            neg=not neg,
            extra_phase=90 + extra_phase,
            add_virtual_Z=add_virtual_Z,
            play=play,
            name="Y_EgGf",
            flag=flag,
            phrst=phrst,
            reload=reload,
        )

    def ZEgGf_pulse(
        self,
        qDrive,
        qNotDrive,
        pihalf=False,
        divide_len=True,
        neg=False,
        extra_phase=0,
        add_virtual_Z=False,
        play=False,
        reload=None,
    ):
        if qDrive == 1:
            sigma_us = self.cfg.device.qubit.pulses.pi_EgGf.sigma[qNotDrive]
            dac_type = self.swap_ch_types[qNotDrive]
        else:
            sigma_us = self.cfg.device.qubit.pulses.pi_EgGf_Q.sigma[qDrive]
            dac_type = self.swap_Q_ch_types[qDrive]
        if pihalf:
            if divide_len:
                sigma_us /= 2
        assert not dac_type == "mux4", "Currently cannot set phase for mux4!"
        phase_adjust = 180
        if pihalf:
            phase_adjust = 90  # the sign of the 180 does not matter, but the sign of the pihalf does!
        if neg:
            phase_adjust *= -1
        if play:
            self.overall_phase[qDrive] += phase_adjust + extra_phase
            # self.sync_all(self.us2cycles(sigma_us))

    def initialize(self):
        self.swap_chs = self.cfg.hw.soc.dacs.swap.ch
        self.swap_ch_types = self.cfg.hw.soc.dacs.swap.type
        self.swap_Q_chs = self.cfg.hw.soc.dacs.swap_Q.ch
        self.swap_Q_ch_types = self.cfg.hw.soc.dacs.swap_Q.type
        super().initialize()
        for q in self.qubits:
            if q == 1:
                continue
            mixer_freq = 0
            if self.swap_ch_types[q] == "int4":
                mixer_freq = self.cfg.hw.soc.dacs.swap.mixer_freq[q]
            if self.swap_chs[q] not in self.gen_chs:
                self.declare_gen(ch=self.swap_chs[q], nqz=self.cfg.hw.soc.dacs.swap.nyquist[q], mixer_freq=mixer_freq)
            # else: print('nqz', self.gen_chs[self.swap_chs[q]]['nqz'])
            mixer_freq = 0
            if self.swap_Q_ch_types[q] == "int4":
                mixer_freq = self.cfg.hw.soc.dacs.swap_Q.mixer_freq[q]
            if self.swap_Q_chs[q] not in self.gen_chs:
                self.declare_gen(
                    ch=self.swap_Q_chs[q], nqz=self.cfg.hw.soc.dacs.swap_Q.nyquist[q], mixer_freq=mixer_freq
                )
        self.sync_all(100)


def sin_sqrd_func(t, Tp):
    # zero if t < 0 or t > Tp else np.sin(np.pi*t/Tp)**2
    return np.where((t < 0) | (t > Tp), 0, np.sin(np.pi * t / Tp) ** 2)


def flat_top(t, Tp, t_rise):
    # zero if t < 0 or t > Tp else 1
    rise = np.where((t > 0) & (t < t_rise), np.sin(np.pi * t / t_rise / 2) ** 2, 0)
    flat = np.where((t >= t_rise) & (t <= Tp - t_rise), 1, 0)
    fall = np.where(t > Tp - t_rise, np.sin(np.pi * (Tp - t) / t_rise / 2) ** 2, 0)

    return np.where((t > 0) & (t < Tp), rise + flat + fall, 0)


class ReadoutResetPulser:
    def __init__(self, kappa_ext_MHz, kappa_int_MHz, chi_MHz=0, kerr_MHz=0):
        """
        All frequencies should be provided as angular frequencies!
        Each ReadoutResetPulser has private variables for resonator paramaters (kappa, chi, kerr)
        """
        self.kappa_ext_MHz = kappa_ext_MHz
        self.kappa_int_MHz = kappa_int_MHz
        self.chi_MHz = chi_MHz
        self.kerr_MHz = kerr_MHz

    def filter(self, times_us, response_func, **kwargs):
        """
        Find the drive function to get a resonator response of response_func,
        given that the response function occurs when the resonator has intrinsic frequency
        that is chi away from the drive.

        times_us can either be a single time or a np array

        kwargs are any additional arguments for the desired response function
        """
        # _f1 = (kappa/2 + 1j*chi)*func(t, **kwargs)
        _f1 = ((self.kappa_ext_MHz + self.kappa_int_MHz) / 2 + 1j * self.chi_MHz) * response_func(times_us, **kwargs)
        _f2 = (response_func(times_us + 1e-7, **kwargs) - response_func(times_us - 1e-7, **kwargs)) / (2e-7)

        a_in = -np.sqrt(2 / self.kappa_ext_MHz) * (_f1 + _f2)
        if self.kerr_MHz != 0:
            _f1 = (
                (self.kappa_ext_MHz + self.kappa_int_MHz) / 2
                + 1j * (self.chi_MHz + 4 * self.kerr_MHz * np.abs(a_in) ** 2)
            ) * response_func(times_us, **kwargs)
            a_in = -np.sqrt(2 / self.kappa_ext_MHz) * (_f1 + _f2)
        return a_in

    def cavity_response(self, times_us, drive_func, **kwargs):
        integrand = (
            lambda t: (np.sqrt(self.kappa_ext_MHz) - self.kappa_MHz_ext / 2) ** 2
            + self.chi_MHz**2 * drive_func(t, **kwargs) ** 2
        )
        return np.exp(-times_us / self.kappa_MHz_ext) * np.array(
            [sp.integrate.quad(integrand, 0, t_i)[0] for t_i in times_us]
        )

    def kerr_drive(self, times_us, response_func, num_filter=1, **kwargs):
        """
        kwargs are any additional arguments for the desired response function
        returns I and Q for the drive that will give the target response_func
        """

        assert num_filter in [1, 2], "Only implemnted filtering 1x or 2x"
        if num_filter == 2:
            assert abs(chi_MHz) > 0, "You are filtering twice but chi is 0, make sure this is right!"

        drive_values = self.filter(times_us, response_func, **kwargs)
        if num_filter > 1:
            drive_values = self.filter(times_us, lambda t, **kwargs: self.filter(t, response_func, **kwargs), **kwargs)

        return np.real(drive_values), np.imag(drive_values)

    def flat_top_kerr_drive(self, t_rise, Tp, nstep=500, num_filter=1):
        times_us = np.linspace(0, Tp, nstep)
        kwargs = dict(Tp=Tp, t_rise=t_rise)
        return times_us, self.kerr_drive(times_us, flat_top, num_filter=num_filter, **kwargs)
