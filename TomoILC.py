import os
import time

import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qick import *
from qick.helpers import gauss
from tqdm import tqdm_notebook as tqdm

expt_path = os.getcwd() + "/data"
import json
import sys

import Pyro4.util
import qutip as qt
import qutip.visualization as qplt
import scipy as sp
import yaml
from slab import AttrDict, get_next_filename
from slab.datamanagement import SlabFile
from slab.experiment import Experiment
from slab.instruments import *

import experiments as meas

sys.path.append(os.getcwd() + "/../../qutip_sims")
from PulseSequence import PulseSequence
from QSwitch import QSwitch
from TomoAnalysis import TomoAnalysis


class TomoILC:
    def __init__(
        self,
        IQ_pulse_seed,
        gains_filename,
        nb_qubits=2,
        n_shot_calib=20000,
        n_shot_tomo=20000,
        qubit_drive=[0, 1],
        qubit_exp=4,
        time_calib=60 * 10,
        gains=None,
        tomo_qubits=None,
        ip_address="10.108.30.56",
        config_file="config_q3diamond_full688and638_reset.yml",
        save_path="data_241007",
        evolv_path="evol_mats/evol_mats",
        debug=False,
        using_LO=False,
    ):

        self.nb_qubits = nb_qubits
        self.n_shot_calib = n_shot_calib
        self.n_shot_tomo = n_shot_tomo
        self.debug = debug
        self.qubit_exp = qubit_exp
        self.evolv_path = evolv_path
        self.evolv_mat = None
        self.time_calib = time_calib
        self.time = time.time()
        # load experiment config and rfsoc config
        self.im = InstrumentManager(ns_address=ip_address)
        config_path = os.path.join(os.getcwd(), config_file)
        with open(config_path, "r") as cfg_file:
            yaml_cfg = yaml.safe_load(cfg_file)
        self.config_file = AttrDict(yaml_cfg)
        if self.debug:
            print("config loaded")
            print(self.config_file)
        self.rfsoc_config = QickConfig(self.im[yaml_cfg["aliases"]["soc"]].get_cfg())
        if self.debug:
            print("rfsoc config loaded")
            print(self.rfsoc_config)

        if using_LO:
            # load lo frequency
            lo1 = self.im[self.config_file.aliases.readout_LO]
            lo1.open_device()

            # turn on LO source if needed
            lo1.set_standby(False)
            lo1.set_output_state(True)
            lo_freq = float(self.config_file.hw.lo.readout.frequency)
            lo_power = float(self.config_file.hw.lo.readout.power)
            lo1.set_frequency(lo_freq)
            lo1.set_power(lo_power)
        self.qubit_drive = qubit_drive
        if tomo_qubits is None:
            self.tomo_qubits = qubit_drive
        else:
            self.tomo_qubits = tomo_qubits

        # set the gains
        if gains is None:
            assert gains_filename is not None, "Please provide a gains filename"
            pulse_dict = self.config_file.device.qubit.pulses
            # take all the entry starting with 'pulse_'
            pulse_keys = [key for key in pulse_dict.keys() if "pulse_" in key]
            # look for the filename of the gains
            for key in pulse_keys:
                if pulse_dict[key].get("filename", None) == gains_filename:
                    gains = pulse_dict[key].get("gain", None)
                    break
            # to be consistent the gains are now define multiplied by 2 to compensate the fact that the IQ pulse is already divided by 2 for ILC
            # gains = [int(gain) for gain in gains]
            gains = [int(gain * 2) for gain in gains]

            self.gains = gains

        # normalise the IQ_seed so that the max amplitude of all pulses is 1

        self.IQ_pulse_seed = IQ_pulse_seed

        expt_path = os.path.join(os.getcwd(), "data", save_path)

        str_tomo_qubits = "".join([str(q) for q in self.tomo_qubits])

        self.tomoExpt = meas.OptimalCtrlTomo2QExperiment(
            soccfg=self.rfsoc_config,
            path=expt_path,
            prefix=f"OptimalCtrlTomo2Q_{str_tomo_qubits}",
            config_file=config_path,
        )

        self.thresholds = None
        self.angles = None
        self.ge_avgs = None
        self.counts_calib = None

    def scale_IQ_pulse(self, IQ_pulse):
        """
        scale the IQ pulse with respect to the seed pulse to
        have the right amplitude
        Inputs:
        - IQ_pulse: dict of IQ pulses to scale
        Outputs:
        - scaled IQ_pulse
        """

        IQ_pulse_seed = self.IQ_pulse_seed
        IQ_scaled = dict()

        for i in self.qubit_drive:

            _i_temps = IQ_pulse["I_{}".format(i)]
            _q_temps = IQ_pulse["Q_{}".format(i)]

            _i_seed = IQ_pulse_seed["I_{}".format(i)]
            _q_seed = IQ_pulse_seed["Q_{}".format(i)]

            scale_factor = max((np.max(np.abs(_i_seed)), np.max(np.abs(_q_seed))))

            _i_temps = _i_temps / scale_factor
            _q_temps = _q_temps / scale_factor

            IQ_scaled["I_{}".format(i)] = _i_temps
            IQ_scaled["Q_{}".format(i)] = _q_temps

        # add the time values in us
        IQ_scaled["times_us"] = IQ_pulse["times"] * 1e6

        return IQ_scaled

    def tomo_experiment(self, IQ_pulse):

        # take the gain that are none zero
        gains = self.gains
        gains = np.array([gain for gain in gains if gain != 0])

        print("gains: ", gains)

        # scale the IQ pulse with respect to the seed pulse
        IQ_pulse_scaled = self.scale_IQ_pulse(IQ_pulse)
        I_values = np.array([IQ_pulse_scaled["I_{}".format(i)] for i in self.qubit_drive])
        Q_values = np.array([IQ_pulse_scaled["Q_{}".format(i)] for i in self.qubit_drive])
        times_us = IQ_pulse_scaled["times_us"]

        # if self.threshold is not None:
        #     self.tomoExpt.cfg.expt['thresholds'] = self.thresholds
        # if self.angles is not None:
        #     self.tomoExpt.cfg.expt['angles'] = self.angles
        # if self.ge_avgs is not None:
        #     self.tomoExpt.cfg.expt['ge_avgs'] = self.ge_avgs
        # if self.counts_calib is not None:
        #     self.tomoExpt.cfg.expt['counts_calib'] = self.counts_calib

        if self.debug:
            print("I_values: ", I_values)
            print("Q_values: ", Q_values)
            print("times_us: ", times_us)
            print("gains: ", gains)
            print("tomo_qubits: ", self.tomo_qubits)
            print("IQ_qubits: ", self.qubit_drive)

        self.tomoExpt.cfg.expt = dict(
            starts=gains.astype(int),  # start gain for each qubit in IQ_qubits
            steps=np.zeros(len(gains)).astype(int),
            expts=np.ones(len(gains)).astype(int),
            reps=self.n_shot_calib,  # reps per measurement basis
            singleshot_reps=self.n_shot_tomo,  # reps for single shot calib
            tomo_qubits=self.tomo_qubits,  # qubits to perform tomography on
            Icontrols=I_values,  # array with array of Icontrols for each of IQ_qubits
            Qcontrols=Q_values,  # array with array of Qcontrols for each of IQ_qubits
            times_us=times_us,
            IQ_qubits=self.qubit_drive,  # qubits to perform IQ on)
            thresholds=self.thresholds,
            angles=self.angles,
            ge_avgs=self.ge_avgs,
            counts_calib=self.counts_calib,
            ILC=True,
            # cool_qubits=self.qubit_drive,
            # use_IQ_pulse=True,
            # plot_IQ=False,
        )

        try:
            self.tomoExpt.go(analyze=False, display=False, progress=self.debug, save=False)
        except Exception:
            print("Pyro traceback:")
            print("".join(Pyro4.util.getPyroTraceback()))

    def get_tomo_results(self, IQ_pulse, ZZ_correction=False, evolv_mat=None):

        _t = time.time()
        print(_t - self.time)
        if np.abs(_t - self.time) > self.time_calib:
            self.thresholds = None
            self.angles = None
            self.ge_avgs = None
            self.counts_calib = None
            self.time = _t

        self.tomo_experiment(IQ_pulse)
        # self.tomoExpt.save_data()

        data = self.tomoExpt.data
        pulse_dict = self.tomoExpt.pulse_dict

        n_tomo_raw = np.array(data["counts_tomo_gains"])[0][0]
        n_calib = np.array(data["counts_calib"])
        if self.debug:
            print("n_tomo_raw: ", n_tomo_raw)
            print("n_calib: ", n_calib)

        if self.thresholds is None:
            self.thresholds = data["thresholds"]
        if self.angles is None:
            self.angles = data["angles"]
        if self.ge_avgs is None:
            self.ge_avgs = data["ge_avgs"]
        if self.counts_calib is None:
            self.counts_calib = data["counts_calib"]

        tomo_analysis = TomoAnalysis(
            nb_qubits=self.nb_qubits,
            rfsoc_config=self.rfsoc_config,
            meas_order=self.tomoExpt.meas_order,
            calib_order=self.tomoExpt.calib_order,
        )

        rho = tomo_analysis.get_rho_from_counts(
            n_tomo_raw=n_tomo_raw,
            n_calib=n_calib,
            pulse_dict=pulse_dict,
            ZZ_correction=ZZ_correction,
            evol_mats=self.evolv_mat,
        )

        return rho
