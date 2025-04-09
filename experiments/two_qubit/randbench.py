# Author: Connie 2022/02/17

import json
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from scipy.optimize import curve_fit
from slab import AttrDict, Experiment, NpEncoder
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter
from experiments.clifford_averager_program import (
    CliffordAveragerProgram,
    CliffordEgGfAveragerProgram,
    QutritAveragerProgram,
)
from experiments.single_qubit.single_shot import get_ge_avgs, hist
from experiments.three_qubit.threeQ_state_tomo import ErrorMitigationStateTomo3QProgram
from experiments.two_qubit.length_rabi_EgGf import LengthRabiEgGfProgram
from experiments.two_qubit.twoQ_state_tomography import (
    AbstractStateTomo2QProgram,
    ErrorMitigationStateTomo1QProgram,
    ErrorMitigationStateTomo2QProgram,
    infer_gef_popln_2readout,
)
from TomoAnalysis import TomoAnalysis

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

"""
Define matrices representing (all) Clifford gates for single
qubit in the basis of Z, X, Y, -Z, -X, -Y, indicating
where on the 6 cardinal points of the Bloch sphere the
+Z, +X, +Y axes go after each gate. Each Clifford gate
can be uniquely identified just by checking where +X and +Y
go.
"""
clifford_1q = dict()
# clifford_1q["Z"] = np.matrix(
#     [
#         [1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0],
#         [0, 0, 0, 0, 0, 1],
#         [0, 0, 0, 1, 0, 0],
#         [0, 1, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0],
#     ]
# )
clifford_1q["X"] = np.matrix(
    [
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0],
    ]
)
clifford_1q["Y"] = np.matrix(
    [
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
    ]
)
# clifford_1q["Z/2"] = np.matrix(
#     [
#         [1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1],
#         [0, 1, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0, 0],
#         [0, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0],
#     ]
# )
clifford_1q["X/2"] = np.matrix(
    [
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
    ]
)
clifford_1q["Y/2"] = np.matrix(
    [
        [0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1],
    ]
)
# clifford_1q["-Z/2"] = np.matrix(
#     [
#         [1, 0, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0],
#         [0, 0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 0, 1],
#         [0, 1, 0, 0, 0, 0],
#     ]
# )
clifford_1q["-X/2"] = np.matrix(
    [
        [0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0],
    ]
)
clifford_1q["-Y/2"] = np.matrix(
    [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
    ]
)
identity = np.diag([1] * 6)
clifford_1q["I"] = identity


# Read pulse as a matrix product acting on state (meaning apply pulses in reverse order of the tuple)
# two_step_pulses = [
#     ("X", "Z/2"),
#     ("X/2", "Z/2"),
#     ("-X/2", "Z/2"),
#     ("Y", "Z/2"),
#     ("Y/2", "Z/2"),
#     ("-Y/2", "Z/2"),
#     ("X", "Z"),
#     ("X/2", "Z"),
#     ("-X/2", "Z"),
#     ("Y", "Z"),
#     ("Y/2", "Z"),
#     ("-Y/2", "Z"),
#     ("X", "-Z/2"),
#     ("X/2", "-Z/2"),
#     ("-X/2", "-Z/2"),
#     ("Y", "-Z/2"),
#     ("Y/2", "-Z/2"),
#     ("-Y/2", "-Z/2"),
# ]

step_pulses = [
    ("Y/2", "X"),
    ("Y/2", "X/2"),
    ("X/2", "-Y/2", "-X/2"),
    ("-X/2", "-Y/2"),
    ("Y/2", "-X/2"),
    ("-X/2", "Y/2", "-X/2"),
    ("X/2", "Y/2"),
    ("-Y/2", "X"),
    ("-X/2", "Y"),
    ("-Y/2", "-X/2"),
    ("X/2", "Y/2", "X/2"),
    ("-X/2", "Y/2"),
    ("X", "Y"),
    ("X/2", "Y"),
    ("-Y/2", "X/2"),
    ("X/2", "Y/2", "-X/2"),
    ("X/2", "-Y/2"),
]

for pulse in step_pulses:
    new_mat = clifford_1q[pulse[0]]
    for p in pulse[1:]:
        new_mat = new_mat @ clifford_1q[p]
    repeat = False
    # Make sure there are no repeats
    for existing_pulse_name, existing_pulse in clifford_1q.items():
        if np.array_equal(new_mat, existing_pulse):
            print("found repeat", pulse, existing_pulse_name)
            repeat = True
    if not repeat:
        clifford_1q[pulse[0] + "," + ",".join(pulse[1:])] = new_mat
clifford_1q_names = list(clifford_1q.keys())
assert (
    len(clifford_1q_names) == 24
), f"you have {len(clifford_1q_names)} elements in your Clifford group instead of 24!"
# print(len(clifford_1q_names), "elements in clifford_1q")
# print(clifford_1q_names)

# Get the average number of X/2 gates per Clifford gate
count = 0
for n in range(len(clifford_1q_names)):  # n is index in clifford_1q_names
    gates = clifford_1q_names[n].split(",")
    for gate in gates:
        # print(gate)
        if gate == "I" or "Z" in gate:
            continue
        if "/2" in gate:
            count += 1
            # print("added 1 to count")
        else:
            count += 2
            # print("added 2 to count")
# print("Average number of X/2 gates per Clifford gate:", count / len(clifford_1q_names))

for name, matrix in clifford_1q.items():
    z_new = np.argmax(matrix[:, 0])  # +Z goes to row where col 0 is 1
    x_new = np.argmax(matrix[:, 1])  # +X goes to row where col 1 is 1
    # print(name, z_new, x_new)
    clifford_1q[name] = (matrix, (z_new, x_new))


def gate_sequence(rb_depth, pulse_n_seq=None, debug=False):
    """
    Generate RB forward gate sequence of length rb_depth as a list of pulse names;
    also return the Clifford gate that is equivalent to the total pulse sequence.
    The effective inverse is pi phase + the total Clifford.
    Optionally, provide pulse_n_seq which is a list of the indices of the Clifford
    gates to apply in the sequence.
    """
    if pulse_n_seq == None:
        pulse_n_seq = (len(clifford_1q_names) * np.random.rand(rb_depth)).astype(int)
    pulse_name_seq = [clifford_1q_names[n] for n in pulse_n_seq]
    if debug:
        print("pulse seq", pulse_name_seq)
    psi_nz = np.matrix([[1, 0, 0, 0, 0, 0]]).transpose()
    psi_nx = np.matrix([[0, 1, 0, 0, 0, 0]]).transpose()
    for n in pulse_n_seq:  # n is index in clifford_1q_names
        gates = clifford_1q_names[n].split(",")
        for gate in reversed(gates):  # Apply matrices from right to left of gates
            psi_nz = clifford_1q[gate][0] @ psi_nz
            psi_nx = clifford_1q[gate][0] @ psi_nx
    psi_nz = psi_nz.flatten()
    psi_nx = psi_nx.flatten()
    if debug:
        print("+Z axis after seq:", psi_nz, "+X axis after seq:", psi_nx)

    total_clifford = None
    if np.argmax(psi_nz) == 0:
        total_clifford = "I"
    else:
        for clifford in clifford_1q_names:  # Get the clifford equivalent to the total seq
            if clifford_1q[clifford][1] == (np.argmax(psi_nz), np.argmax(psi_nx)):
                # z_new, x_new = clifford_1q[clifford][1]
                # if z_new == np.argmax(psi_nz):
                total_clifford = clifford
                break
    assert total_clifford is not None, f"Failed to invert gate sequence! {pulse_name_seq} which brings +Z to {psi_nz}"

    if debug:
        total_clifford_mat = clifford_1q[total_clifford][0]
        print("Total gate matrix:\n", total_clifford_mat)

    return pulse_name_seq, total_clifford


def interleaved_gate_sequence(rb_depth, gate_char: str, debug=False):
    """
    Generate RB gate sequence with rb_depth random gates interleaved with gate_char
    Returns the total gate list (including the interleaved gates) and the total
    Clifford gate equivalent to the total pulse sequence.
    """
    pulse_n_seq_rand = (len(clifford_1q_names) * np.random.rand(rb_depth)).astype(int)
    pulse_n_seq = []
    assert gate_char in clifford_1q_names
    n_gate_char = clifford_1q_names.index(gate_char)
    if debug:
        print("n gate char:", n_gate_char, clifford_1q_names[n_gate_char])
    for n_rand in pulse_n_seq_rand:
        pulse_n_seq.append(n_rand)
        pulse_n_seq.append(n_gate_char)
    return gate_sequence(len(pulse_n_seq), pulse_n_seq=pulse_n_seq, debug=debug)


if __name__ == "__main__":
    print("Clifford gates:", clifford_1q_names)
    print("Total number Clifford gates:", len(clifford_1q_names))
    pulse_name_seq, total_clifford = gate_sequence(2, debug=True)
    print("Pulse sequence:", pulse_name_seq)
    print("Total clifford of seq:", total_clifford)
    gate_char = "X/2"
    print()
    print("Interleaved RB with gate", gate_char)
    pulse_name_seq, total_clifford = interleaved_gate_sequence(2, gate_char=gate_char, debug=True)
    print("Pulse sequence:", pulse_name_seq)
    print("Total clifford of seq:", total_clifford)

# ===================================================================== #


class SimultaneousRBProgram(QutritAveragerProgram):
    """
    RB program for single qubit gates
    """

    def clifford(self, qubit, pulse_name: str, extra_phase=0, inverted=False, play=False):
        """
        Convert a clifford pulse name into the function that performs the pulse.
        If inverted, play the inverse of this gate (the extra phase is added on top of the inversion)
        """
        pulse_name = pulse_name.upper()
        assert pulse_name in clifford_1q_names
        gates = pulse_name.split(",")

        # Normally gates are applied right to left, but if inverted apply them left to right
        gate_order = reversed(gates)
        if inverted:
            gate_order = gates
        for gate in gate_order:
            pulse_func = None
            if gate == "I":
                continue
            if "X" in gate:
                pulse_func = self.X_pulse
            elif "Y" in gate:
                pulse_func = self.Y_pulse
            elif "Z" in gate:
                pulse_func = self.Z_pulse
            else:
                assert False, "Invalid gate"

            neg = "-" in gate
            if inverted:
                neg = not neg
            # pulse_func(qubit, pihalf='/2' in gate, neg=neg, extra_phase=extra_phase, play=play, reload=False) # very important to not reload unless necessary to save memory on the gen
            pulse_func(
                qubit, pihalf="/2" in gate, neg=neg, divide_len=False, extra_phase=extra_phase, play=play, reload=False
            )  # very important to not reload unless necessary to save memory on the gen
            # print(self.overall_phase[qubit])

    def __init__(self, soccfg, cfg, gate_list, qubit_list):
        # gate_list should include the total gate!
        # qubit_list should specify the qubit on which each random gate will be applied
        self.gate_list = gate_list
        self.qubit_list = qubit_list
        super().__init__(soccfg, cfg)

    def body(self):
        # Phase reset all channels except readout DACs (since mux ADCs can't be phase reset)
        self.reset_and_sync()

        if "cool_qubits" in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            cool_idle = [self.cfg.device.qubit.pulses.pi_f0g1.idle[q] for q in self.cfg.expt.cool_qubits]
            if "cool_idle" in self.cfg.expt and self.cfg.expt.cool_idle is not None:
                cool_idle = self.cfg.expt.cool_idle
            self.active_cool(cool_qubits=self.cfg.expt.cool_qubits, cool_idle=cool_idle)

        num_rb_qubits = len(set(self.qubit_list))
        assert num_rb_qubits == 1, "only support 1 qubit in rb right now"

        # Do all the gates given in the initialize except for the total gate, measure
        cfg = AttrDict(self.cfg)
        for i in range(len(self.gate_list) - 1):
            self.clifford(qubit=self.qubit_list[i], pulse_name=self.gate_list[i], play=True)
            self.sync_all()

        # Do the inverse by applying the total gate with pi phase
        # This is actually wrong if there is more than 1 qubit!!! need to apply an inverse total gate for each qubit!!
        self.clifford(qubit=self.qubit_list[-1], pulse_name=self.gate_list[-1], inverted=True, play=True)
        self.sync_all()  # align channels and wait 10ns

        self.measure(
            pulse_ch=self.measure_chs,
            adcs=self.adc_chs,
            adc_trig_offset=cfg.device.readout.trig_offset[0],
            wait=True,
            syncdelay=self.us2cycles(max([cfg.device.readout.relax_delay[q] for q in self.qubits])),
        )


# ===================================================================== #


class SimultaneousRBExperiment(Experiment):
    """
    Simultaneous Randomized Benchmarking Experiment
    Experimental Config:
    expt = dict(
        start: rb depth start - for interleaved RB, depth specifies the number of random gates
        step: step rb depth
        expts: number steps
        reps: number averages per unique sequence
        variations: number different sequences per depth
        gate_char: a single qubit clifford gate (str) to characterize. If not None, runs interleaved RB instead of regular RB.
        qubits: the qubits to perform simultaneous RB on. If using EgGf subspace, specify just qA (where qA, qB represents the Eg->Gf qubits)
        singleshot_reps: reps per state for singleshot calibration
        post_process: 'threshold' (uses single shot binning), 'scale' (scale by ge_avgs), or None
        thresholds: (optional) don't rerun singleshot and instead use this
        ge_avgs: (optional) don't rerun singleshot and instead use this
        angles: (optional) don't rerun singleshot and instead use this
    )
    """

    def __init__(self, soccfg=None, path="", prefix="SimultaneousRB", config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        qubits = self.cfg.expt.qubits
        assert len(qubits) == 1, "only 1 qubit supported for now in RB"
        self.qubit = qubits[0]

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

        if "use_EgGf_subspace" in self.cfg.expt and self.cfg.expt.use_EgGf_subspace:
            assert False, "use the RbEgGfExperiment!"

        # ================= #
        # Get single shot calibration for all qubits
        # ================= #
        data = {"counts_calib": [], "counts_raw": []}

        thresholds_q = ge_avgs_q = angles_q = fids_q = None
        if "post_process" not in self.cfg.expt.keys():  # threshold or scale
            self.cfg.expt.post_process = None

        self.calib_order = ["g", "e"]  # should match with order of counts for each tomography measurement
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
            for q in range(num_qubits_sample):
                if ge_avgs_q[q] is None:
                    ge_avgs_q[q] = np.zeros_like(
                        ge_avgs_q[self.cfg.expt.qubits[0]]
                    )  # just get the shape of the arrays correct by picking the old ge_avgs_q of a q that was definitely measured
            ge_avgs_q = np.array(ge_avgs_q)
            data["counts_calib"] = self.cfg.expt.counts_calib
            print("Re-using provided angles, thresholds, ge_avgs, counts_calib")

        else:
            # Error mitigation measurements: prep in g, e to recalibrate measurement angle and measure confusion matrix
            sscfg = AttrDict(deepcopy(self.cfg))
            sscfg.expt.qubit = self.qubit
            sscfg.expt.reps = self.cfg.expt.singleshot_reps

            calib_prog_dict = dict()
            for prep_state in tqdm(self.calib_order):
                # print(prep_state)
                sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state)
                err_tomo = ErrorMitigationStateTomo1QProgram(soccfg=self.soccfg, cfg=sscfg)
                err_tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                calib_prog_dict.update({prep_state: err_tomo})

            g_prog = calib_prog_dict["g"]
            Ig, Qg = g_prog.get_shots(verbose=False)
            thresholds_q = [0] * num_qubits_sample
            angles_q = [0] * num_qubits_sample
            ge_avgs_q = [[0] * 4] * num_qubits_sample

            # Get readout angle + threshold for qubit
            e_prog = calib_prog_dict["e"]
            Ie, Qe = e_prog.get_shots(verbose=False)
            shot_data = dict(Ig=Ig[self.qubit], Qg=Qg[self.qubit], Ie=Ie[self.qubit], Qe=Qe[self.qubit])
            fid, threshold, angle = hist(data=shot_data, plot=debug, verbose=False)
            thresholds_q[self.qubit] = threshold[0]
            angles_q[self.qubit] = angle
            ge_avgs_q[self.qubit] = [
                np.average(Ig[self.qubit]),
                np.average(Qg[self.qubit]),
                np.average(Ie[self.qubit]),
                np.average(Qe[self.qubit]),
            ]

            if progress:
                print(f"thresholds={thresholds_q},")
                print(f"angles={angles_q},")
                print(f"ge_avgs={ge_avgs_q}", ",")

            # Process the shots taken for the confusion matrix with the calibration angles
            for prep_state in self.calib_order:
                counts = calib_prog_dict[prep_state].collect_counts(angle=angles_q, threshold=thresholds_q)
                data["counts_calib"].append(counts)
            if progress:
                print(f"counts_calib={np.array(data['counts_calib']).tolist()}")

        data.update(dict(thresholds=thresholds_q, angles=angles_q, ge_avgs=ge_avgs_q))

        # ================= #
        # Begin RB
        # ================= #

        if "shot_avg" not in self.cfg.expt:
            self.cfg.expt.shot_avg = 1
        data.update({"xpts": [], "popln": [], "popln_err": []})

        depths = self.cfg.expt.start + self.cfg.expt.step * np.arange(self.cfg.expt.expts)
        tomo_analysis = TomoAnalysis(nb_qubits=1)
        for depth in tqdm(depths):
            # print(f'depth {depth} gate list (last gate is the total gate)')
            data["xpts"].append([])
            data["popln"].append([])
            data["popln_err"].append([])
            for var in range(self.cfg.expt.variations):
                if "gate_char" in self.cfg.expt and self.cfg.expt.gate_char is not None:
                    gate_list, total_gate = interleaved_gate_sequence(depth, gate_char=self.cfg.expt.gate_char)
                else:
                    gate_list, total_gate = gate_sequence(depth)
                gate_list.append(total_gate)  # make sure to do the inverse gate

                # print(gate_list)

                # gate_list = ['X', '-X/2,Z', 'Y/2', '-X/2,-Z/2', '-Y/2,Z', '-Z/2', 'X', 'Y']
                # gate_list = ['X/2,Z/2', 'X/2,Z/2']
                # gate_list = ['I', 'I']
                # gate_list = ['X', 'I']
                # gate_list = ['X', '-X/2,Z', 'X/2']
                # gate_list = ['X', '-X/2,Z', 'Y/2', 'X/2']
                # gate_list = ['X', '-X/2,Z', 'Y/2', '-X/2,-Z/2', '-Y/2']

                # gate_list = ['X/2']*depth
                # if depth % 4 == 0: gate_list.append('I')
                # elif depth % 4 == 1: gate_list.append('X/2')
                # elif depth % 4 == 2: gate_list.append('X')
                # elif depth % 4 == 3: gate_list.append('-X/2')
                # gate_list = ['X/2']*depth
                # if depth % 4 == 0: gate_list.append('I')
                # elif depth % 4 == 1: gate_list.append('X/2')
                # elif depth % 4 == 2: gate_list.append('X')
                # elif depth % 4 == 3: gate_list.append('-X/2')

                # print('variation', var)
                qubit_list = np.random.choice(self.cfg.expt.qubits, size=len(gate_list) - 1)

                randbench = SimultaneousRBProgram(
                    soccfg=self.soccfg, cfg=self.cfg, gate_list=gate_list, qubit_list=qubit_list
                )
                # print(randbench)
                # # from qick.helpers import progs2json
                # # print(progs2json([randbench.dump_prog()]))

                assert self.cfg.expt.post_process is not None, "need post processing for RB to make sense!"
                popln, popln_err = randbench.acquire_rotated(
                    soc=self.im[self.cfg.aliases.soc],
                    progress=False,
                    angle=angles_q,
                    threshold=thresholds_q,
                    ge_avgs=ge_avgs_q,
                    post_process=self.cfg.expt.post_process,
                )

                adc_ch = self.cfg.hw.soc.adcs.readout.ch[qubits[0]]

                if self.cfg.expt.post_process == "threshold":
                    shots, _ = randbench.get_shots(angle=angles_q, threshold=thresholds_q)
                    # 0, 1
                    counts = np.array([tomo_analysis.sort_counts([shots[adc_ch]])])
                    data["counts_raw"].append(counts)
                    tomo_analysis = TomoAnalysis(nb_qubits=1, tomo_qubits=qubits)
                    counts = tomo_analysis.fix_neg_counts(
                        tomo_analysis.correct_readout_err(counts, data["counts_calib"])
                    )
                    counts = counts[0]  # go back to just 1d array
                    data["popln"][-1].append(counts[1] / sum(counts))
                    # print('variation', var, 'gate list', gate_list, 'counts', counts)
                else:
                    data["popln"][-1].append(popln[adc_ch])
                    # print(depth, var, iq, avgi)
                    data["popln_err"][-1].append(popln_err[adc_ch])
                data["xpts"][-1].append(depth)

        for k, a in data.items():
            data[k] = np.array(a)
        # print(np.shape(data['avgi'][iq]))

        self.data = data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        qubits = self.cfg.expt.qubits
        data["probs"] = [None] * len(qubits)
        data["fit"] = [None] * len(qubits)
        data["fit_err"] = [None] * len(qubits)
        data["error"] = [100.0] * len(qubits)
        data["std_dev_probs"] = [None] * len(qubits)
        data["med_probs"] = [None] * len(qubits)
        data["avg_probs"] = [None] * len(qubits)

        probs = np.zeros_like(data["popln"])
        for depth in range(len(data["popln"])):
            probs[depth] = 1 - np.asarray(data["popln"][depth])
        probs = np.asarray(probs)
        data["xpts"] = np.asarray(data["xpts"])
        data["probs"] = probs
        # probs = np.reshape(probs, (self.cfg.expt.expts, self.cfg.expt.variations))
        std_dev_probs = []
        med_probs = []
        avg_probs = []
        working_depths = []
        depths = data["xpts"]
        for depth in range(len(probs)):
            probs_depth = probs[depth]
            if len(probs_depth) > 0:
                std_dev_probs.append(np.std(probs_depth))
                med_probs.append(np.median(probs_depth))
                avg_probs.append(np.average(probs_depth))
                working_depths.append(depths[depth][0])
        std_dev_probs = np.asarray(std_dev_probs)
        med_probs = np.asarray(med_probs)
        avg_probs = np.asarray(avg_probs)
        working_depths = np.asarray(working_depths)
        flat_depths = np.concatenate(depths)
        flat_probs = np.concatenate(data["probs"])
        # depths = self.cfg.expt.start + self.cfg.expt.step * np.arange(self.cfg.expt.expts)
        # popt, pcov = fitter.fitrb(depths[:-4], med_probs[:-4])
        # popt, pcov = fitter.fitrb(depths, med_probs)
        # print(working_depths, avg_probs)
        # popt, pcov = fitter.fitrb(working_depths, avg_probs)
        data["std_dev_probs"] = std_dev_probs
        data["med_probs"] = med_probs
        data["avg_probs"] = avg_probs
        data["working_depths"] = working_depths
        if fit:
            popt, pcov = fitter.fitrb(flat_depths, flat_probs)
            data["fit"] = popt
            data["fit_err"] = pcov
            data["error"] = fitter.rb_error(popt[0], d=2)
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        plt.figure(figsize=(8, 6))
        irb = "gate_char" in self.cfg.expt and self.cfg.expt.gate_char is not None
        title = f'{"Interleaved " + self.cfg.expt.gate_char + " Gate" if irb else ""} RB on Q{self.cfg.expt.qubits[0]}'

        plt.subplot(111, title=title, xlabel="Sequence Depth", ylabel="Population in g")
        depths = data["xpts"]
        flat_depths = np.concatenate(depths)
        flat_probs = np.concatenate(data["probs"])
        plt.plot(flat_depths, flat_probs, "x", color="tab:grey")

        probs_vs_depth = data["probs"]
        std_dev_probs = data["std_dev_probs"]
        med_probs = data["med_probs"]
        avg_probs = data["avg_probs"]
        working_depths = data["working_depths"]
        # plt.errorbar(working_depths, avg_probs, fmt='o', yerr=2*std_dev_probs, color='k', elinewidth=0.75)
        plt.errorbar(working_depths, med_probs, fmt="o", yerr=std_dev_probs, color="k", elinewidth=0.75)

        if fit:
            cov_p = data["fit_err"][0][0]
            fit_plt_xpts = range(working_depths[-1] + 1)
            # plt.plot(depths, avg_probs, 'o-', color='tab:blue')
            plt.plot(fit_plt_xpts, fitter.rb_func(fit_plt_xpts, *data["fit"]))
            print(f'Running {"interleaved " + self.cfg.expt.gate_char + " gate" if irb else "regular"} RB')
            print(f'Depolarizing parameter p from fit: {data["fit"][0]} +/- {np.sqrt(cov_p)}')
            print(
                f'Average RB gate error: {data["error"]} +/- {np.sqrt(fitter.error_fit_err(cov_p, 2**(len(self.cfg.expt.qubits))))}'
            )
            print(
                f'\tFidelity=1-error: {1-data["error"]} +/- {np.sqrt(fitter.error_fit_err(cov_p, 2**(len(self.cfg.expt.qubits))))}'
            )

        plt.grid(linewidth=0.3)
        # if self.cfg.expt.post_process is not None: plt.ylim(-0.1, 1.1)
        plt.ylim(-0.05, 1.05)
        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        with self.datafile() as f:
            f.attrs["calib_order"] = json.dumps(self.calib_order, cls=NpEncoder)
        return self.fname


# ===================================================================== #


class SimultaneousRBEFProgram(QutritAveragerProgram):
    """
    RB program for single qubit gates on the ef subspace
    """

    def clifford(self, qubit, pulse_name: str, extra_phase=0, ZZ_qubit=None, inverted=False, play=False):
        """
        Convert a clifford pulse name into the function that performs the pulse.
        If inverted, play the inverse of this gate (the extra phase is added on top of the inversion)
        """
        pulse_name = pulse_name.upper()
        assert pulse_name in clifford_1q_names
        gates = pulse_name.split(",")

        # Normally gates are applied right to left, but if inverted apply them left to right
        gate_order = reversed(gates)
        if inverted:
            gate_order = gates
        for gate in gate_order:
            pulse_func = None
            if gate == "I":
                continue
            if "X" in gate:
                pulse_func = self.Xef_pulse
            elif "Y" in gate:
                pulse_func = self.Yef_pulse
            elif "Z" in gate:
                pulse_func = self.Zef_pulse
            # if 'X' in gate: pulse_func = self.X_pulse
            # elif 'Y' in gate: pulse_func = self.Y_pulse
            # elif 'Z' in gate: pulse_func = self.Z_pulse
            else:
                assert False, "Invalid gate"

            neg = "-" in gate
            if inverted:
                neg = not neg

            # print('WARNING NOT PLAYING PULSE')
            # pulse_func(qubit, pihalf='/2' in gate, neg=neg, extra_phase=extra_phase, play=play, reload=False) # very important to not reload unless necessary to save memory on the gen
            pulse_func(
                qubit,
                pihalf="/2" in gate,
                ZZ_qubit=ZZ_qubit,
                neg=neg,
                divide_len=False,
                extra_phase=extra_phase,
                play=play,
                reload=False,
            )  # very important to not reload unless necessary to save memory on the gen
            self.sync_all(
                5
            )  # THIS IS NECESSARY IN RB WHEN THERE ARE MORE THAN O(30) PULSES SINCE THE TPROC CAN'T KEEP UP FOR SHORT PULSES

    def __init__(self, soccfg, cfg, gate_list, qubit_list):
        # gate_list should include the total gate!
        # qubit_list should specify the qubit on which each random gate will be applied
        self.gate_list = gate_list
        self.qubit_list = qubit_list
        super().__init__(soccfg, cfg)

    def body(self):
        # Phase reset all channels except readout DACs (since mux ADCs can't be phase reset)
        self.reset_and_sync()

        if "cool_qubits" in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            cool_idle = [self.cfg.device.qubit.pulses.pi_f0g1.idle[q] for q in self.cfg.expt.cool_qubits]
            if "cool_idle" in self.cfg.expt and self.cfg.expt.cool_idle is not None:
                cool_idle = self.cfg.expt.cool_idle
            self.active_cool(cool_qubits=self.cfg.expt.cool_qubits, cool_idle=cool_idle)

        num_rb_qubits = len(set(self.qubit_list))
        assert num_rb_qubits == 1, "only support 1 qubit in rb right now"

        qTest = self.qubit_list[0]
        if "ZZ_qubit" in self.cfg.expt and self.cfg.expt.ZZ_qubit is not None:
            ZZ_qubit = self.cfg.expt.ZZ_qubit
            if ZZ_qubit != qTest:
                self.X_pulse(q=ZZ_qubit, play=True)
        self.X_pulse(q=qTest, ZZ_qubit=ZZ_qubit, play=True)

        test_qZZ = None
        if "test_qZZ" in self.cfg.expt:
            test_qZZ = self.cfg.expt.test_qZZ

        # Do all the gates given in the initialize except for the total gate, measure
        for i in range(len(self.gate_list) - 1):
            self.clifford(qubit=self.qubit_list[i], pulse_name=self.gate_list[i], ZZ_qubit=test_qZZ, play=True)
            self.sync_all()

        # Do the inverse by applying the total gate with pi phase
        # This is actually wrong if there is more than 1 qubit!!! need to apply an inverse total gate for each qubit!!
        self.clifford(
            qubit=self.qubit_list[-1], pulse_name=self.gate_list[-1], ZZ_qubit=test_qZZ, inverted=True, play=True
        )
        self.sync_all()  # align channels and wait 10ns

        # Measure the population of just the e state when e/f are not distinguishable - check the g population
        setup_measure = None
        if "setup_measure" in self.cfg.expt:
            setup_measure = self.cfg.expt.setup_measure
        if setup_measure != None and "qDrive_ge" in setup_measure:
            self.X_pulse(q=self.qDrive, ZZ_qubit=ZZ_qubit, play=True)  # not sure whether needs some phase adjustment
        # Bring Gf to Ge, or stays in same state.
        if setup_measure != None and "qDrive_ef" in setup_measure:
            self.Xef_pulse(self.qDrive, ZZ_qubit=ZZ_qubit, play=True)  # not sure whether needs some phase adjustment

        self.measure(
            pulse_ch=self.measure_chs,
            adcs=self.adc_chs,
            adc_trig_offset=self.cfg.device.readout.trig_offset[0],
            wait=True,
            syncdelay=self.us2cycles(max([self.cfg.device.readout.relax_delay[q] for q in self.qubits])),
        )


# ===================================================================== #


class SimultaneousRBEFExperiment(Experiment):
    """
    Simultaneous Randomized Benchmarking Experiment
    Experimental Config:
    expt = dict(
        start: rb depth start - for interleaved RB, depth specifies the number of random gates
        step: step rb depth
        expts: number steps
        reps: number averages per unique sequence
        reps_f: use a different number of averages per sequence when measuring with the f readout
        variations: number different sequences per depth
        gate_char: a single qubit clifford gate (str) to characterize. If not None, runs interleaved RB instead of regular RB.
        qubits: [qTest]
        singleshot_reps: reps per state for singleshot calibration
        post_process: 'threshold' (uses single shot binning), 'scale' (scale by ge_avgs), or None
        ZZ_qubit: if not None, initializes this qubit in e in addition to the qubit we are doing the EF RB on
        test_qZZ: plays the pulse on qTest that is ZZ shifted by test_qZZ for all clifford gates
        measure_f: qubit: if not None, calibrates the single qubit f state measurement on this qubit and also runs the measurement twice to distinguish e and f states
        thresholds: (optional) don't rerun singleshot and instead use this
        ge_avgs: (optional) don't rerun singleshot and instead use this
        angles: (optional) don't rerun singleshot and instead use this
    )
    """

    def __init__(self, soccfg=None, path="", prefix="SimultaneousRBEgGf", config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.readout.frequency)
        assert len(self.cfg.expt.qubits) == 1, "only 1 qubit supported for now in RB"
        self.qubit = self.cfg.expt.qubits[0]

        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not (isinstance(value3, list)):
                                value2.update({key3: [value3] * num_qubits_sample})
                elif not (isinstance(value, list)):
                    subcfg.update({key: [value] * num_qubits_sample})

        self.measure_f = False
        q_measure_f = None
        if self.cfg.expt.measure_f is not None and len(self.cfg.expt.measure_f) >= 0:
            self.measure_f = True
            q_measure_f = self.cfg.expt.measure_f[0]

        assert self.measure_f, "you really should be running this experiment with a measure_f"

        thresholds_q = ge_avgs_q = angles_q = fids_q = None
        if "post_process" not in self.cfg.expt.keys():  # threshold or scale
            self.cfg.expt.post_process = None

        if "meas_order" not in self.cfg.expt or self.cfg.expt.meas_order is None:
            self.meas_order = ["Z", "X", "Y"]
        else:
            self.meas_order = self.cfg.expt.meas_order
        self.calib_order = ["g", "e"]  # should match with order of counts for each tomography measurement
        if self.measure_f:
            self.calib_order += ["f"]  # assumes the 2nd qubit is the measure_f

        data = {"counts_raw": [[]]}
        if self.cfg.expt.measure_f is not None:
            for i in range(
                len(self.cfg.expt.measure_f)
            ):  # measure g of everybody, second measurement of each measure_f qubit using the g/f readout
                data["counts_raw"].append([])
        if "reps_f" not in self.cfg.expt:
            self.cfg.expt.reps_f = self.cfg.expt.reps

        data["thresholds_loops"] = []
        data["angles_loops"] = []
        data["ge_avgs_loops"] = []
        data["counts_calib_loops"] = []

        if self.measure_f:
            data["thresholds_f_loops"] = []
            data["angles_f_loops"] = []
            data["gf_avgs_loops"] = []
            data["counts_calib_f_loops"] = []
        data["xpts"] = []

        if "depths" not in self.cfg.expt or self.cfg.expt.depths is None:
            print("WARNING: depths not in expt config, calculating depths")
            self.cfg.expt.depths = self.cfg.expt.start + self.cfg.expt.step * np.arange(self.cfg.expt.expts)
        else:
            print("depths", self.cfg.expt.depths)
            depths = self.cfg.expt.depths

        print("depths", depths)

        gate_list_variations = [None] * len(depths)

        if "loops" not in self.cfg.expt:
            self.cfg.expt.loops = 1
        print("running", self.cfg.expt.loops, "loops")

        # print('WARNING doing ge instead of ef pulses! CHECK IN THE CLIFFORD DEFINITION TO FIX THIS')
        for loop in tqdm(range(self.cfg.expt.loops), disable=not progress or self.cfg.expt.loops == 1):

            # ================= #
            # Get single shot calibration for all qubits
            # ================= #

            if (
                "angles" in self.cfg.expt
                and "thresholds" in self.cfg.expt
                and "ge_avgs" in self.cfg.expt
                and "counts_calib" in self.cfg.expt
                and None
                not in (
                    self.cfg.expt.angles,
                    self.cfg.expt.thresholds,
                    self.cfg.expt.ge_avgs,
                    self.cfg.expt.counts_calib,
                )
            ):
                angles_q = self.cfg.expt.angles
                thresholds_q = self.cfg.expt.thresholds
                ge_avgs_q = self.cfg.expt.ge_avgs
                for q in range(num_qubits_sample):
                    if ge_avgs_q[q] is None:
                        ge_avgs_q[q] = np.zeros_like(
                            ge_avgs_q[self.cfg.expt.qubits[0]]
                        )  # just get the shape of the arrays correct by picking the old ge_avgs_q of a q that was definitely measured
                ge_avgs_q = np.array(ge_avgs_q)
                counts_calib = self.cfg.expt.counts_calib
                print("Re-using provided angles, thresholds, ge_avgs, counts_calib")

            else:
                thresholds_q = [0] * 4
                ge_avgs_q = [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]
                angles_q = [0] * 4
                fids_q = [0] * 4
                counts_calib = []

                # Error mitigation measurements: prep in g, e to recalibrate measurement angle and measure confusion matrix
                sscfg = AttrDict(deepcopy(self.cfg))
                sscfg.expt.qubit = self.qubit
                sscfg.expt.reps = self.cfg.expt.singleshot_reps

                calib_prog_dict = dict()
                for prep_state in tqdm(self.calib_order):
                    # print(prep_state)
                    sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state)
                    err_tomo = ErrorMitigationStateTomo1QProgram(soccfg=self.soccfg, cfg=sscfg)
                    err_tomo.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                    calib_prog_dict.update({prep_state: err_tomo})

                g_prog = calib_prog_dict["g"]
                Ig, Qg = g_prog.get_shots(verbose=False)
                thresholds_q = [0] * num_qubits_sample
                angles_q = [0] * num_qubits_sample
                ge_avgs_q = [[0] * 4] * num_qubits_sample

                # Get readout angle + threshold for qubit
                e_prog = calib_prog_dict["e"]
                Ie, Qe = e_prog.get_shots(verbose=False)
                shot_data = dict(Ig=Ig[self.qubit], Qg=Qg[self.qubit], Ie=Ie[self.qubit], Qe=Qe[self.qubit])
                fid, threshold, angle = hist(data=shot_data, plot=debug, verbose=False)
                thresholds_q[self.qubit] = threshold[0]
                angles_q[self.qubit] = angle
                ge_avgs_q[self.qubit] = [
                    np.average(Ig[self.qubit]),
                    np.average(Qg[self.qubit]),
                    np.average(Ie[self.qubit]),
                    np.average(Qe[self.qubit]),
                ]

                if progress:
                    print(f"thresholds={thresholds_q},")
                    print(f"angles={angles_q},")
                    print(f"ge_avgs={ge_avgs_q}", ",")

                # Process the shots taken for the confusion matrix with the calibration angles
                for prep_state in self.calib_order:
                    counts = calib_prog_dict[prep_state].collect_counts(angle=angles_q, threshold=thresholds_q)
                    counts_calib.append(counts)

                if progress:
                    print(f"thresholds={thresholds_q},")
                    print(f"angles={angles_q},")
                    print(f"ge_avgs={ge_avgs_q},")
                    print(f"counts_calib={np.array(counts_calib).tolist()}")

                data["thresholds_loops"].append(thresholds_q)
                data["angles_loops"].append(angles_q)
                data["ge_avgs_loops"].append(ge_avgs_q)
                data["counts_calib_loops"].append(np.array(counts_calib))

            # ================= #
            # Begin RB
            # ================= #

            if "shot_avg" not in self.cfg.expt:
                self.cfg.expt.shot_avg = 1

            tomo_analysis = TomoAnalysis(nb_qubits=1)
            for i_depth, depth in enumerate(tqdm(depths, disable=not progress)):
                # print(f'depth {depth} gate list (last gate is the total gate)')
                if loop == 0:
                    data["xpts"].append([])
                    gate_list_variations[i_depth] = []
                for var in range(self.cfg.expt.variations):
                    if loop == 0:
                        if "gate_char" in self.cfg.expt and self.cfg.expt.gate_char is not None:
                            gate_list, total_gate = interleaved_gate_sequence(depth, gate_char=self.cfg.expt.gate_char)
                        else:
                            gate_list, total_gate = gate_sequence(depth)
                        gate_list.append(total_gate)  # make sure to do the inverse gate
                        # gate_list = ['X', '-X/2,Z', 'Y/2', '-X/2,-Z/2', '-Y/2,Z', '-Z/2', 'X', 'Y']
                        # gate_list = ['X', 'X', 'I']
                        # print('variation', var)
                        # print(gate_list)
                        # gate_list = ['X/2', 'Z/2', '-Y/2', 'I']

                        # gate_list = ['X']*depth
                        # if depth % 2 == 0: gate_list.append('I')
                        # elif depth % 2 == 1: gate_list.append('X')

                        # gate_list = ['X/2']*depth
                        # if depth % 4 == 0: gate_list.append('I')
                        # elif depth % 4 == 1: gate_list.append('X/2')
                        # elif depth % 4 == 2: gate_list.append('X')
                        # elif depth % 4 == 3: gate_list.append('-X/2')

                        gate_list_variations[i_depth].append(gate_list)
                    else:
                        gate_list = gate_list_variations[i_depth][var]

                    qubit_list = [self.qubit] * len(gate_list)

                    randbench = SimultaneousRBEFProgram(
                        soccfg=self.soccfg, cfg=self.cfg, gate_list=gate_list, qubit_list=qubit_list
                    )
                    # print('\n\n ge measurement program')
                    # print(randbench)
                    # # from qick.helpers import progs2json
                    # # print(progs2json([randbench.dump_prog()]))
                    # print('\n\n')

                    assert self.cfg.expt.post_process is not None, "need post processing for RB to make sense!"
                    popln, popln_err = randbench.acquire_rotated(
                        soc=self.im[self.cfg.aliases.soc],
                        progress=False,
                        angle=angles_q,
                        threshold=thresholds_q,
                        ge_avgs=ge_avgs_q,
                        post_process=self.cfg.expt.post_process,
                    )
                    assert self.cfg.expt.post_process == "threshold", "Can only bin EF RB properly using threshold"

                    adc_ch = self.cfg.hw.soc.adcs.readout.ch[self.qubit]

                    if self.cfg.expt.post_process == "threshold":
                        shots, _ = randbench.get_shots(angle=angles_q, threshold=thresholds_q)
                        # 0, 1/2
                        counts = np.array([tomo_analysis.sort_counts([shots[adc_ch]])])
                        data["counts_raw"][0].append(counts)
                        # print('variation', var, 'gate list', gate_list, 'counts', counts)

                    if loop == 0:
                        data["xpts"][-1].append(depth)

            # ================= #
            # Measure the same thing with g/f distinguishing
            # ================= #

            if self.measure_f:
                counts_calib_f = []

                # ================= #
                # Get f state single shot calibration (this must be re-run if you just ran measurement with the standard readout)
                # ================= #

                thresholds_f_q = [0] * 4
                gf_avgs_q = [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]
                angles_f_q = [0] * 4
                fids_f_q = [0] * 4

                # We really just need the single shot plots here, but convenient to use the ErrorMitigation tomo to do it
                sscfg = AttrDict(deepcopy(self.cfg))
                sscfg.expt.qubit = self.qubit
                sscfg.expt.reps = sscfg.expt.singleshot_reps
                sscfg.device.readout.frequency[q_measure_f] = sscfg.device.readout.frequency_ef[q_measure_f]
                sscfg.device.readout.gain = sscfg.device.readout.gain_ef
                sscfg.device.readout.readout_length[q_measure_f] = sscfg.device.readout.readout_length_ef[q_measure_f]

                calib_prog_dict = dict()
                for prep_state in tqdm(self.calib_order):
                    # print(prep_state)
                    sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state, apply_q1_pi2=False)
                    err_tomo = ErrorMitigationStateTomo1QProgram(soccfg=self.soccfg, cfg=sscfg)
                    err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=False)
                    calib_prog_dict.update({prep_state: err_tomo})

                g_prog = calib_prog_dict["g"]
                Ig, Qg = g_prog.get_shots(verbose=False)

                # Get readout angle + threshold for qubits to distinguish g/f on one of the qubits
                f_prog = calib_prog_dict["f"]
                If, Qf = f_prog.get_shots(verbose=False)
                shot_data = dict(Ig=Ig[self.qubit], Qg=Qg[self.qubit], Ie=If[self.qubit], Qe=Qf[self.qubit])
                print(f"Qubit ({self.qubit}) f")
                fid, threshold, angle = hist(data=shot_data, plot=debug, verbose=False)
                thresholds_f_q[self.qubit] = threshold[0]
                gf_avgs_q[self.qubit] = [
                    np.average(Ig[self.qubit]),
                    np.average(Qg[self.qubit]),
                    np.average(If[self.qubit]),
                    np.average(Qf[self.qubit]),
                ]
                angles_f_q[self.qubit] = angle
                fids_f_q[self.qubit] = fid[0]
                print(f"gf fidelity (%): {100*fid[0]}")

                # Process the shots taken for the confusion matrix with the calibration angles
                for prep_state in self.calib_order:
                    counts = calib_prog_dict[prep_state].collect_counts(angle=angles_f_q, threshold=thresholds_f_q)
                    counts_calib_f.append(counts)

                print(f"thresholds_f={thresholds_f_q},")
                print(f"angles_f={angles_f_q},")
                print(f"gf_avgs={gf_avgs_q},")
                print(f"counts_calib_f={np.array(counts_calib_f).tolist()}")

                data["thresholds_f_loops"].append(thresholds_f_q)
                data["angles_f_loops"].append(angles_f_q)
                data["gf_avgs_loops"].append(gf_avgs_q)
                data["counts_calib_f_loops"].append(np.array(counts_calib_f))

                # ================= #
                # Begin RB for measure f, using same gate list as measure with g/e
                # ================= #

                assert q_measure_f == self.qubit, "this code assumes we will be processing to distinguish gf from ge"
                for i_depth, depth in enumerate(tqdm(depths, disable=not progress)):
                    for var in range(self.cfg.expt.variations):
                        gate_list = gate_list_variations[i_depth][var]

                        rbcfg = deepcopy(self.cfg)
                        rbcfg.expt.reps = rbcfg.expt.reps_f
                        rbcfg.device.readout.frequency[q_measure_f] = rbcfg.device.readout.frequency_ef[q_measure_f]
                        rbcfg.device.readout.gain = rbcfg.device.readout.gain_ef
                        rbcfg.device.readout.readout_length[q_measure_f] = rbcfg.device.readout.readout_length_ef[
                            q_measure_f
                        ]

                        qubit_list = [self.qubit] * len(gate_list)
                        randbench = SimultaneousRBEFProgram(
                            soccfg=self.soccfg, cfg=rbcfg, gate_list=gate_list, qubit_list=qubit_list
                        )

                        # print('\n\n gf measurement program')
                        # print(randbench)
                        # # from qick.helpers import progs2json
                        # # print(progs2json([randbench.dump_prog()]))
                        # print('\n\n')

                        assert self.cfg.expt.post_process is not None, "need post processing for RB to make sense!"
                        popln, popln_err = randbench.acquire_rotated(
                            soc=self.im[self.cfg.aliases.soc],
                            progress=False,
                            angle=angles_f_q,
                            threshold=thresholds_f_q,
                            ge_avgs=gf_avgs_q,
                            post_process=self.cfg.expt.post_process,
                        )
                        assert self.cfg.expt.post_process == "threshold", "Can only bin EF RB properly using threshold"

                        if self.cfg.expt.post_process == "threshold":
                            shots, _ = randbench.get_shots(angle=angles_f_q, threshold=thresholds_f_q)
                            # 0/1, 2
                            counts = np.array([tomo_analysis.sort_counts([shots[adc_ch]])])
                            data["counts_raw"][1].append(counts)
                            # print('variation', var, 'gate list', gate_list, 'counts', counts)

        # print('shape', np.shape(data['counts_raw']))
        for icounts in range(len(data["counts_raw"])):
            data["counts_raw"][icounts] = np.reshape(
                data["counts_raw"][icounts], (self.cfg.expt.loops, len(depths), self.cfg.expt.variations, 2)
            )

        for k, a in data.items():
            # print(k)
            # print(a)
            # print(np.shape(a))
            data[k] = np.array(a)

        self.data = data
        return data

    def analyze(self, data=None, fit=True, separate_correction=True, **kwargs):
        if data is None:
            data = self.data

        self.qubit = self.cfg.expt.qubits[0]

        data["xpts"] = np.asarray(data["xpts"])
        unique_depths = np.average(data["xpts"], axis=1)

        assert self.measure_f

        data["counts_calib_total"] = np.concatenate((data["counts_calib_loops"], data["counts_calib_f_loops"]), axis=2)
        data["counts_raw_total"] = np.concatenate(
            (data["counts_raw"][0], data["counts_raw"][1] * self.cfg.expt.reps / self.cfg.expt.reps_f), axis=3
        )
        print("counts calib total shape", np.shape(data["counts_calib_total"]))
        # print(data['counts_calib_total'])
        print("counts raw shape", np.shape(data["counts_raw_total"]), "(loops, depths, variations, 4)")
        print(data["counts_raw_total"][0, 0, 0, :])
        # print('counts raw total', np.shape(data['counts_raw_total']))

        # g, e, f
        data["poplns_2q_loops"] = np.zeros(
            shape=(self.cfg.expt.loops, len(unique_depths), self.cfg.expt.variations, 3)
        )

        if separate_correction:
            g_count_plot = []
            e_count_plot = []
            f_count_plot = []
            g_pop_plot = []
            e_pop_plot = []
            f_pop_plot = []

        for loop in range(self.cfg.expt.loops):
            for idepth, depth in enumerate(unique_depths):
                for ivar in range(self.cfg.expt.variations):
                    # after correcting readout error, counts corrected should correspond to counts in [g, e, f] (the calib_order)
                    # instead of [gA, eA, gB, fB] (the raw counts)

                    tomo_analysis = TomoAnalysis(nb_qubits=1, tomo_qubits=self.cfg.expt.qubits)
                    if separate_correction:
                        counts_raw_total_this = data["counts_raw_total"][loop, idepth, ivar]
                        counts_calib_this = data["counts_calib_total"][loop]

                        # print('ge counts calib', counts_calib_this[:,:2])
                        counts_ge_corrected = tomo_analysis.correct_readout_err(
                            [counts_raw_total_this[:2]], counts_calib_this[:, :2]
                        )
                        # print('ge corrected', counts_ge_corrected)
                        counts_ge_corrected = tomo_analysis.fix_neg_counts(counts_ge_corrected)[0]
                        # print('neg ge corrected', counts_ge_corrected)

                        # print('gf counts calib', counts_calib_this[:,2:])
                        # print('gf raw', counts_raw_total_this[2:])
                        counts_gf_corrected = tomo_analysis.correct_readout_err(
                            [counts_raw_total_this[2:]], counts_calib_this[:, 2:]
                        )
                        # print('gf corrected', counts_gf_corrected)
                        counts_gf_corrected = tomo_analysis.fix_neg_counts(counts_gf_corrected)[0]
                        # print('neg gf corrected', counts_gf_corrected)

                        g = counts_ge_corrected[0] / np.sum(counts_ge_corrected)
                        f = counts_gf_corrected[2] / np.sum(counts_gf_corrected)
                        e = 1 - g - f
                        counts_corrected = [[g, e, f]]
                        # print('counts_corrected', counts_corrected)
                        counts_corrected = tomo_analysis.fix_neg_counts(counts_corrected)
                        # print('neg corrected', counts_corrected)

                        g_count_plot.append(counts_raw_total_this[0])
                        f_count_plot.append(counts_raw_total_this[3])
                        g_pop_plot.append(g * np.sum(counts_ge_corrected))
                        f_pop_plot.append(f * np.sum(counts_gf_corrected))

                    else:
                        counts_corrected = tomo_analysis.correct_readout_err(
                            [data["counts_raw_total"][loop, idepth, ivar]], data["counts_calib_total"][loop]
                        )
                        counts_corrected = tomo_analysis.fix_neg_counts(counts_corrected)

                    data["poplns_2q_loops"][loop, idepth, ivar, :] = counts_corrected / np.sum(counts_corrected)

        if separate_correction:
            plt.figure()
            plt.plot(
                unique_depths,
                np.average(
                    np.average(
                        np.reshape(
                            g_count_plot, newshape=(self.cfg.expt.loops, len(unique_depths), self.cfg.expt.variations)
                        ),
                        axis=0,
                    ),
                    axis=1,
                ),
                ".-",
                label="g_counts",
            )
            plt.plot(
                unique_depths,
                np.average(
                    np.average(
                        np.reshape(
                            f_count_plot, newshape=(self.cfg.expt.loops, len(unique_depths), self.cfg.expt.variations)
                        ),
                        axis=0,
                    ),
                    axis=1,
                ),
                ".-",
                label="f_counts",
            )
            plt.plot(
                unique_depths,
                np.average(
                    np.average(
                        np.reshape(
                            g_pop_plot, newshape=(self.cfg.expt.loops, len(unique_depths), self.cfg.expt.variations)
                        ),
                        axis=0,
                    ),
                    axis=1,
                ),
                ".--",
                label="g pop corrected",
            )
            plt.plot(
                unique_depths,
                np.average(
                    np.average(
                        np.reshape(
                            f_pop_plot, newshape=(self.cfg.expt.loops, len(unique_depths), self.cfg.expt.variations)
                        ),
                        axis=0,
                    ),
                    axis=1,
                ),
                ".--",
                label="f pop corrected",
            )

            plt.xlabel("Sequence Depths")
            plt.legend()
            plt.show()

        # Average over loops
        data["poplns_2q"] = np.average(data["poplns_2q_loops"], axis=0)

        # print('poplns_2q_loops shape', np.shape(data['poplns_2q_loops']))
        # print('poplns_2q shape', np.shape(data['poplns_2q']))

        # [g, e, f]
        probs_g = data["poplns_2q"][:, :, 0]
        probs_e = data["poplns_2q"][:, :, 1]
        probs_f = data["poplns_2q"][:, :, 2]

        data["popln_e_std"] = np.std(probs_e, axis=1)
        data["popln_e_avg"] = np.average(probs_e, axis=1)
        # print('WARNING using probs_g for probs_e subspace')
        # data['popln_e_std'] = np.std(probs_g, axis=1)
        # data['popln_e_avg'] = np.average(probs_g, axis=1)

        sum_prob_subspace = probs_e + probs_f
        # print('WARNING using the ge subspace')
        # sum_prob_subspace = probs_e + probs_g
        data["popln_subspace"] = sum_prob_subspace
        data["popln_subspace_std"] = np.std(sum_prob_subspace, axis=1)
        data["popln_subspace_avg"] = np.average(sum_prob_subspace, axis=1)
        data["popln_e_subspace"] = probs_e / sum_prob_subspace
        data["popln_e_subspace_std"] = np.std(probs_e / sum_prob_subspace, axis=1)
        data["popln_e_subspace_avg"] = np.average(probs_e / sum_prob_subspace, axis=1)
        # print('WARNING using probs_g for probs_e')
        # data['popln_e_subspace'] = probs_g/sum_prob_subspace
        # data['popln_e_subspace_std'] = np.std(probs_g/sum_prob_subspace, axis=1)
        # data['popln_e_subspace_avg'] = np.average(probs_g/sum_prob_subspace, axis=1)

        # print('shape sum prob_e + prob_f', np.shape(sum_prob_subspace))
        # print('shape average sum over each depth', np.shape(data['popln_subspace_avg']), 'should equal', np.shape(unique_depths))

        if not fit:
            return data

        depths = data["xpts"]
        flat_depths = np.concatenate(depths)
        # popt1, pcov1 = fitter.fitrb(unique_depths, data['popln_subspace_avg'])
        popt1, pcov1 = fitter.fitrb(flat_depths, np.concatenate(sum_prob_subspace))
        print("fit1 p1, a, offset", popt1)
        data["fit1"] = popt1
        data["fit1_err"] = pcov1
        p1, a, offset = popt1
        data["l1"] = fitter.leakage_err(p1, offset)
        data["l2"] = fitter.seepage_err(p1, offset)

        # popt2, pcov2 = fitter.fitrb_l1_l2(unique_depths, data['popln_e_avg'], p1=p1, offset=offset)
        popt2, pcov2 = fitter.fitrb_l1_l2(flat_depths, np.concatenate(probs_e), p1=p1, offset=offset)
        print("fit2 a0, b0, c0, p2", popt2)
        data["fit2"] = popt2
        data["fit2_err"] = pcov2
        a0, b0, c0, p2 = popt2

        data["fidelity"], data["fidelity_err"] = fitter.rb_fidelity_l1_l2(
            d=2, p2=p2, l1=data["l1"], p2_err=pcov2[3][3], l1_err=pcov1[0][0]
        )

        # popt3, pcov3 = fitter.fitrb(unique_depths, data['popln_e_subspace_avg'])
        popt3, pcov3 = fitter.fitrb(flat_depths, np.concatenate(probs_e / sum_prob_subspace))
        data["fit3"] = popt3
        data["fit3_err"] = pcov3

        return data

    def display(self, data=None, fit=True, show_all_vars=False):
        if data is None:
            data = self.data

        plt.figure(figsize=(8, 6))
        irb = "gate_char" in self.cfg.expt and self.cfg.expt.gate_char is not None
        title = f'{"Interleaved " + self.cfg.expt.gate_char + " Gate" if irb else ""} EF RB on Q{self.cfg.expt.qubits[0]}{" with ZZ Q"+str(self.cfg.expt.test_qZZ) if "test_qZZ" in self.cfg.expt and self.cfg.expt.test_qZZ is not None else ""}'

        plt.subplot(111, title=title, xlabel="Sequence Depth", ylabel="Population")
        depths = data["xpts"]
        unique_depths = np.average(depths, axis=1)
        flat_depths = np.concatenate(depths)
        # g, e, f
        flat_probs_e = np.concatenate(data["poplns_2q"][:, :, 1])
        # print('WARNING using probs_ge for probs_e')
        # flat_probs_e = np.concatenate(data['poplns_2q'][:, :, 0])
        flat_probs_subspace = np.concatenate(data["popln_subspace"])
        if show_all_vars:
            plt.plot(flat_depths, flat_probs_e, "x", color="tab:grey")
            plt.plot(flat_depths, flat_probs_subspace, "v", color="tab:grey")

        probs_e_avg = data["popln_e_avg"]
        probs_e_std = data["popln_e_std"]
        probs_subspace_avg = data["popln_subspace_avg"]
        probs_subspace_std = data["popln_subspace_std"]
        probs_e_subspace_avg = data["popln_e_subspace_avg"]
        probs_e_subspace_std = data["popln_e_subspace_std"]

        # print('prob_e_avg', probs_e_avg, '+/-', probs_e_std)
        # print('prob_subspace_avg', probs_subspace_avg, '+/-', probs_subspace_std)
        # print('prob_e_subspace_avg', probs_e_subspace_avg, '+/-', probs_e_subspace_std)
        # plt.errorbar(working_depths, avg_probs, fmt='o', yerr=2*std_dev_probs, color='k', elinewidth=0.75)
        plt.errorbar(
            unique_depths,
            probs_e_avg,
            fmt="x",
            yerr=probs_e_std,
            color=default_colors[0],
            elinewidth=0.75,
            label="e probability",
        )
        plt.errorbar(
            unique_depths,
            probs_subspace_avg,
            fmt="v",
            yerr=probs_subspace_std,
            color=default_colors[1],
            elinewidth=0.75,
            label="subspace probability",
        )
        plt.errorbar(
            unique_depths,
            probs_e_subspace_avg,
            fmt="o",
            yerr=probs_e_subspace_std,
            color=default_colors[2],
            elinewidth=0.75,
            label="e/subspace probability",
        )

        if fit:
            pcov1 = data["fit1_err"]
            # plt.plot(depths, avg_probs, 'o-', color='tab:blue')
            plt.plot(unique_depths, fitter.rb_func(unique_depths, *data["fit1"]), color=default_colors[1])
            print(
                f'Running {"interleaved " + self.cfg.expt.gate_char + " gate" if irb else "regular"} RB on EF subspace'
            )
            p1 = data["fit1"][0]
            print(f"Depolarizing parameter p1 from fit: {p1} +/- {np.sqrt(pcov1[0][0])}")
            # print(f'Average RB gate error: {data["error"]} +/- {np.sqrt(fitter.error_fit_err(pcov1, 2**(len(self.cfg.expt.qubits))))}')
            # print(f'\tFidelity=1-error: {1-data["error"]} +/- {np.sqrt(fitter.error_fit_err(pcov1, 2**(len(self.cfg.expt.qubits))))}')

            pcov2 = data["fit2_err"]
            # plt.plot(depths, avg_probs, 'o-', color='tab:blue')
            plt.plot(unique_depths, fitter.rb_decay_l1_l2(unique_depths, p1, *data["fit2"]), color=default_colors[0])
            print(
                f'Running {"interleaved " + self.cfg.expt.gate_char + " gate" if irb else "regular"} RB on EF subspace'
            )
            print(f'Depolarizing parameter p2 from fit: {data["fit2"][3]} +/- {np.sqrt(pcov2[3][3])}')
            print(f'Fidelity: {data["fidelity"]} +/- {data["fidelity_err"]}')
            print(f'Leakage L1: {data["l1"]}')
            print(f'Seepage L2: {data["l2"]}')

            pcov3 = data["fit3_err"][0][0]
            # plt.plot(depths, avg_probs, 'o-', color='tab:blue')
            plt.plot(unique_depths, fitter.rb_func(unique_depths, *data["fit3"]), color=default_colors[2])
            p = data["fit3"][0]
            print(f"Depolarizing parameter p from e/subspace fit: {p} +/- {np.sqrt(pcov3)}")
            err = fitter.rb_error(p, d=2)
            print(f"Average RB gate error on e/subspace: {err} +/- {np.sqrt(fitter.error_fit_err(pcov3, 2))}")
            print(f"\tFidelity of e/subspace=1-error: {1-err} +/- {np.sqrt(fitter.error_fit_err(pcov3, 2))}")

        plt.grid(linewidth=0.3)
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        with self.datafile() as f:
            f.attrs["calib_order"] = json.dumps(self.calib_order, cls=NpEncoder)
        return self.fname


# ===================================================================== #


class RBEgGfProgram(CliffordEgGfAveragerProgram):
    """
    RB program for single qubit gates, treating the Eg/Gf subspace as the TLS
    """

    def __init__(self, soccfg, cfg, gate_list, qubits, qDrive):
        # gate_list should include the total gate!
        # qA should specify the the qubit that is not q1 for the Eg-Gf swap
        self.gate_list = gate_list
        self.cfg = cfg

        qA, qB = qubits
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

        self.wrong_init = False
        if "wrong_init" in self.cfg.expt:
            self.wrong_init = self.cfg.expt.wrong_init

        self.ground_state_init = False
        if "ground_state_init" in self.cfg.expt:
            self.ground_state_init = self.cfg.expt.ground_state_init

        assert qNotDrive == 1, "none of this class will work for driving Q1"
        super().__init__(soccfg, cfg)

    def initialize(self):
        super().initialize()

        self.swap_rphase = self.sreg(self.swap_Q_chs[self.qDrive], "phase")

    def cliffordEgGf(
        self,
        qDrive,
        pulse_name: str,
        add_virtual_Z=False,
        inverted=False,
        play=False,
        sync_after=True,
    ):
        """
        Convert a clifford pulse name (in the Eg-Gf subspace) into the function that performs the pulse.
        swap_phase defines the additional phase needed to calibrate each swap
        If inverted, play the inverse of this gate (the extra phase is added on top of the inversion)
        """
        pulse_name = pulse_name.upper()
        assert pulse_name in clifford_1q_names or pulse_name == "I"
        gates = pulse_name.split(",")

        # Normally gates are applied right to left, but if inverted apply them left to right
        gate_order = reversed(gates)
        if inverted:
            gate_order = gates

        for gate in gate_order:
            # pulse_func = None
            if gate == "I":
                continue

            neg = "-" in gate
            if inverted:
                neg = not neg
            pihalf = "/2" in gate

            # Figure out the phase updates
            # ASSUMES OTHER PULSE REGS HAVE ALREADY BEEN SET!
            phase_deg = self.overall_phase[qDrive]
            if "X" in gate:
                pass
            elif "Y" in gate:
                phase_deg += 90
                neg = not neg
            elif "Z" in gate:
                phase_adjust = 180
                if pihalf:
                    phase_adjust = 90
                if neg:
                    phase_adjust = -phase_adjust
                if play:
                    self.overall_phase[qDrive] += phase_adjust
                return
            else:
                assert False, "Invalid gate"

            n_pulses = 1
            if not pihalf:
                n_pulses = 2

            if neg:
                phase_deg -= 180

            # print("phase_deg", phase_deg)
            self.safe_regwi(
                self.ch_page(self.swap_Q_chs[qDrive]),
                self.swap_rphase,
                self.deg2reg(phase_deg, gen_ch=self.swap_Q_chs[qDrive]),
            )
            if play:
                for i in range(n_pulses):
                    self.pulse(ch=self.swap_Q_chs[qDrive])

            if add_virtual_Z and play:
                virtual_Z = self.cfg.device.qubit.pulses.pi_EgGf_Q.phase[qDrive]
                if pihalf:
                    virtual_Z = self.cfg.device.qubit.pulses.pi_EgGf_Q.half_phase[qDrive]
                self.overall_phase[qDrive] += virtual_Z

            if sync_after:
                self.sync_all()

    def body(self):
        cfg = AttrDict(self.cfg)

        self.reset_and_sync()

        if "cool_qubits" in self.cfg.expt and self.cfg.expt.cool_qubits is not None:
            cool_idle = [self.cfg.device.qubit.pulses.pi_f0g1.idle[q] for q in self.cfg.expt.cool_qubits]
            if "cool_idle" in self.cfg.expt and self.cfg.expt.cool_idle is not None:
                cool_idle = self.cfg.expt.cool_idle
            self.active_cool(cool_qubits=self.cfg.expt.cool_qubits, cool_idle=cool_idle)

        if self.readout_cool:
            self.measure_readout_cool()

        # Get into the Eg-Gf subspace
        ZZ_qubit = None
        use_q3_init_switch = False
        if (self.qDrive == 3 and not self.wrong_init) or (self.qDrive == 2 and self.wrong_init):
            use_q3_init_switch = True

        if use_q3_init_switch:
            # print("WARNING: not initiating q0 in e for the q3/q1 swap")
            # print("initializing q0 in e")
            self.X_pulse(q=0, play=True)  # ZZ qubit for the q3/q1 swap
            ZZ_qubit = 0

        if not self.ground_state_init:
            self.X_pulse(
                q=self.qNotDrive,
                ZZ_qubit=ZZ_qubit,
                extra_phase=-self.overall_phase[self.qSort],
                pihalf=False,
                play=True,
            )

        # print("WARNING INITIATING IN GF")
        # self.X_pulse(
        #     q=self.qDrive, ZZ_qubit=ZZ_qubit, extra_phase=-self.overall_phase[self.qSort], pihalf=False, play=True
        # )
        # self.Xef_pulse(
        #     q=self.qDrive, ZZ_qubit=ZZ_qubit, extra_phase=-self.overall_phase[self.qSort], pihalf=False, play=True
        # )

        add_virtual_Z = False
        if "add_phase" in self.cfg.expt and self.cfg.expt.add_phase:
            add_virtual_Z = True

        # Set swap registers
        self.XEgGf_half_pulse(
            qDrive=self.qDrive,
            qNotDrive=self.qNotDrive,
            add_virtual_Z=add_virtual_Z,
            set_reg=True,
            play=False,
            reload=True,
            sync_after=True,
        )

        # Do all the gates given in the initialize except for the total gate
        for i in range(len(self.gate_list) - 1):
            self.cliffordEgGf(
                qDrive=self.qDrive,
                add_virtual_Z=add_virtual_Z,
                pulse_name=self.gate_list[i],
                play=True,
                sync_after=False,
            )
            # don't need to sync between gates because all are on the same channel
            # self.sync_all()
        # self.sync_all()

        # Do the inverse by applying the total gate with pi phase
        self.cliffordEgGf(
            qDrive=self.qDrive,
            add_virtual_Z=add_virtual_Z,
            pulse_name=self.gate_list[-1],
            inverted=True,
            play=True,
        )
        self.sync_all()

        # print("WARNING: adding an extra test pulse")
        # self.sync_all(self.us2cycles(1))
        # self.X_pulse(q=1, play=True)

        # Measure the population of just the e state when e/f are not distinguishable - check the g population
        setup_measure = None
        if "setup_measure" in self.cfg.expt:
            setup_measure = self.cfg.expt.setup_measure
        if setup_measure != None and "qDrive_ge" in setup_measure:
            self.X_pulse(q=self.qDrive, play=True)  # not sure whether needs some phase adjustment

        # Bring Gf to Ge, or stays in same state.
        if setup_measure != None and "qDrive_ef" in setup_measure:
            self.Xef_pulse(self.qDrive, play=True)  # not sure whether needs some phase adjustment

        self.measure(
            pulse_ch=self.measure_chs,
            adcs=self.adc_chs,
            adc_trig_offset=cfg.device.readout.trig_offset[0],
            wait=True,
            syncdelay=self.us2cycles(max([cfg.device.readout.relax_delay[q] for q in self.qubits])),
        )


# ===================================================================== #


class SimultaneousRBEgGfExperiment(Experiment):
    """
    Simultaneous Randomized Benchmarking Experiment
    Experimental Config:
    expt = dict(
        start: rb depth start - for interleaved RB, depth specifies the number of random gates
        step: step rb depth
        expts: number steps
        reps: number averages per unique sequence
        variations: number different sequences per depth
        gate_char: a single qubit clifford gate (str) to characterize. If not None, runs interleaved RB instead of regular RB.
        qubits: the qubits to perform simultaneous RB on. If using EgGf subspace, specify just qA (where qA, qB represents the Eg->Gf qubits)
        singleshot_reps: reps per state for singleshot calibration
        post_process: 'threshold' (uses single shot binning), 'scale' (scale by ge_avgs), or None
        measure_f: qubit: if not None, calibrates the single qubit f state measurement on this qubit and also runs the measurement twice to distinguish e and f states
        thresholds: (optional) don't rerun singleshot and instead use this
        ge_avgs: (optional) don't rerun singleshot and instead use this
        angles: (optional) don't rerun singleshot and instead use this

        test_leakage: True/False: if True, replaces all gates in gate list with pi*depth
        wrong_init: initializes in the wrong state, to test the transfer into gf when switch is in the wrong state
        ground_state_init: initializes input/output in gg instead of eg
    )
    """

    def __init__(self, soccfg=None, path="", prefix="SimultaneousRBEgGf", config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        qubits = self.cfg.expt.qubits

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

        assert len(self.cfg.expt.qubits) == 3, "this experiment is designed for qubits [0, 1, qDrive]"
        assert self.cfg.expt.qubits[0] == 0
        assert self.cfg.expt.qubits[1] == 1
        assert self.cfg.expt.qubits[2] in [2, 3]
        q0, qNotDrive, qDrive = self.cfg.expt.qubits

        self.qDrive = qDrive
        self.qNotDrive = 1

        data = dict()

        thresholds_q = ge_avgs_q = angles_q = fids_q = None
        if "post_process" not in self.cfg.expt.keys():  # threshold or scale
            self.cfg.expt.post_process = None

        self.measure_f_only = True
        if "measure_f_only" in self.cfg.expt:
            self.measure_f_only = self.cfg.expt.measure_f_only

        self.calib_order = []
        for i in ["g", "e"]:
            for j in ["g", "e"]:
                if not self.measure_f_only:
                    k_states = ["g", "e", "f"]
                else:
                    k_states = ["g", "f"]
                for k in k_states:
                    self.calib_order.append(i + j + k)
        print("calib_order", self.calib_order)

        data["xpts"] = []

        if "depths" not in self.cfg.expt or self.cfg.expt.depths is None:
            print("WARNING: depths not in expt config, calculating depths")
            depths = self.cfg.expt.start + self.cfg.expt.step * np.arange(self.cfg.expt.expts)
        else:
            print("depths", self.cfg.expt.depths)
            depths = self.cfg.expt.depths

        print("depths", depths)

        gate_list_variations = [None] * len(depths)

        data["counts_raw"] = np.zeros(
            (
                1 + len(self.cfg.expt.measure_f),
                self.cfg.expt.loops,
                len(depths),
                self.cfg.expt.variations,
                2 ** (len(self.cfg.expt.qubits)),
            )
        )

        data["thresholds_loops"] = []
        data["angles_loops"] = []
        data["ge_avgs_loops"] = []
        data["counts_calib_loops"] = []

        data["thresholds_f_loops"] = []
        data["angles_f_loops"] = []
        data["gf_avgs_loops"] = []
        data["counts_calib_f_loops"] = []

        full_mux_expt = False
        if "full_mux_expt" in self.cfg.expt:
            full_mux_expt = self.cfg.expt.full_mux_expt

        if "singleshots_reps_f" not in self.cfg.expt:
            self.cfg.expt.singleshot_reps_f = self.cfg.expt.singleshot_reps
        if "reps_f" not in self.cfg.expt:
            self.cfg.expt.reps_f = self.cfg.expt.reps

        if "loops" not in self.cfg.expt:
            self.cfg.expt.loops = 1
        print("running", self.cfg.expt.loops, "loops")

        self.test_leakage = False
        if "test_leakage" in self.cfg.expt:
            self.test_leakage = self.cfg.expt.test_leakage
        self.wrong_init = False
        if "wrong_init" in self.cfg.expt:
            self.wrong_init = self.cfg.expt.wrong_init

        # ================= #
        # Make sure the variations picked can run
        # ================= #
        if "validate_variations" in self.cfg.expt and self.cfg.expt.validate_variations:
            validate_variations = True
            print("Validating variations")
        else:
            validate_variations = False
            print("Skipping variations validation")
        for i_depth, depth in enumerate(tqdm(depths, disable=not progress or not validate_variations)):
            gate_list_variations[i_depth] = []
            for var in range(self.cfg.expt.variations):
                if "gate_char" in self.cfg.expt and self.cfg.expt.gate_char is not None:
                    gate_list, total_gate = interleaved_gate_sequence(depth, gate_char=self.cfg.expt.gate_char)
                else:
                    gate_list, total_gate = gate_sequence(depth)
                gate_list.append(total_gate)  # make sure to do the inverse gate

                # gate_list = ["I"]
                # gate_list = ["X", "I"]
                # gate_list = ["X", "X", "X", "X", "I"]
                # gate_list = ["X/2", "X/2", "-X/2", "-X/2", "I"]

                # gate_list = ["X/2", "X/2", "X/2", "X/2", "I"]
                # gate_list = ["-X/2,-Z/2", "X/2,Z/2", "X,Z/2", "-X/2"]
                # gate_list = ["X/2", "X", "Z", "X", "-Z", "-X/2", "I"]

                # gate_list.append("I")

                if self.test_leakage:
                    gate_list = ["X"] * depth + ["I"]

                # print("gate_list =", gate_list)

                gate_list_variations[i_depth].append(gate_list)

                if "validate_variations" in self.cfg.expt and self.cfg.expt.validate_variations:
                    if i_depth != len(depths) - 1:  # only validate for the last variation
                        continue
                    cfg_test = AttrDict(deepcopy(self.cfg))
                    cfg_test.reps = 10

                    randbench = RBEgGfProgram(
                        soccfg=self.soccfg,
                        cfg=cfg_test,
                        gate_list=gate_list,
                        qubits=[qNotDrive, qDrive],
                        qDrive=self.cfg.expt.qDrive,
                    )
                    popln, popln_err = randbench.acquire_rotated(
                        soc=self.im[self.cfg.aliases.soc],
                        progress=False,
                    )

        print("gate_list_variations=", gate_list_variations)
        for var in range(self.cfg.expt.variations):
            data["xpts"].append(depths)

        tomo_analysis = TomoAnalysis(nb_qubits=3)
        adcQA = self.cfg.hw.soc.adcs.readout.ch[q0]
        adcQB = self.cfg.hw.soc.adcs.readout.ch[qNotDrive]
        adcQC = self.cfg.hw.soc.adcs.readout.ch[qDrive]

        for loop in tqdm(range(self.cfg.expt.loops), disable=not progress or self.cfg.expt.loops == 1):
            print("Beginning loop", loop)

            if not self.measure_f_only:
                # ================= #
                # Get single shot calibration for all qubits
                # ================= #

                if (
                    "angles" in self.cfg.expt
                    and "thresholds" in self.cfg.expt
                    and "ge_avgs" in self.cfg.expt
                    and "counts_calib" in self.cfg.expt
                ):
                    angles_q = self.cfg.expt.angles
                    thresholds_q = self.cfg.expt.thresholds
                    ge_avgs_q = np.asarray(self.cfg.expt.ge_avgs)
                    counts_calib = self.cfg.expt.counts_calib
                    print("Re-using provided angles, thresholds, ge_avgs")
                else:
                    thresholds_q = [0] * 4
                    ge_avgs_q = [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]
                    angles_q = [0] * 4
                    fids_q = [0] * 4
                    counts_calib = []

                    # We really just need the single shot plots here, but convenient to use the ErrorMitigation tomo to do it
                    sscfg = AttrDict(deepcopy(self.cfg))
                    sscfg.expt.reps = sscfg.expt.singleshot_reps
                    sscfg.expt.tomo_qubits = self.cfg.expt.qubits

                    calib_prog_dict = dict()
                    for prep_state in tqdm(self.calib_order):
                        # print(prep_state)
                        sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state, apply_q1_pi2=False)
                        err_tomo = ErrorMitigationStateTomo3QProgram(soccfg=self.soccfg, cfg=sscfg)
                        err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=False)
                        calib_prog_dict.update({prep_state: err_tomo})

                    g_prog = calib_prog_dict["ggg"]
                    Ig, Qg = g_prog.get_shots(verbose=False)
                    shot_calib = dict(Ig=Ig, Qg=Qg, Ie=[], Qe=[])

                    # Get readout angle + threshold for qubits
                    for qi, q in enumerate(sscfg.expt.tomo_qubits):
                        calib_e_state = "ggg"
                        calib_e_state = calib_e_state[:qi] + "e" + calib_e_state[qi + 1 :]
                        e_prog = calib_prog_dict[calib_e_state]
                        Ie, Qe = e_prog.get_shots(verbose=False)
                        shot_calib["Ie"].append(Ie[q])
                        shot_calib["Qe"].append(Qe[q])
                        shot_data = dict(Ig=Ig[q], Qg=Qg[q], Ie=Ie[q], Qe=Qe[q])
                        print(f"Qubit ({q}) ge")
                        fid, threshold, angle = hist(
                            data=shot_data, plot=debug, verbose=False, amplitude_mode=full_mux_expt
                        )
                        thresholds_q[q] = threshold[0]
                        ge_avgs_q[q] = get_ge_avgs(
                            Igs=Ig[q], Qgs=Qg[q], Ies=Ie[q], Qes=Qe[q], amplitude_mode=full_mux_expt
                        )
                        angles_q[q] = angle
                        fids_q[q] = fid[0]
                        print(
                            f"ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[q]} \t threshold ge: {thresholds_q[q]}"
                        )

                    # Process the shots taken for the confusion matrix with the calibration angles
                    for prep_state in self.calib_order:
                        counts = calib_prog_dict[prep_state].collect_counts(
                            angle=angles_q, threshold=thresholds_q, amplitude_mode=full_mux_expt
                        )
                        counts_calib.append(counts)

                    print(f"thresholds={thresholds_q},")
                    print(f"angles={angles_q},")
                    print(f"ge_avgs={ge_avgs_q},")
                    print(f"counts_calib={np.array(counts_calib).tolist()}")

                    data["thresholds_loops"].append(thresholds_q)
                    data["angles_loops"].append(angles_q)
                    data["ge_avgs_loops"].append(ge_avgs_q)
                    data["counts_calib_loops"].append(np.array(counts_calib))

                # ================= #
                # Begin RB in g/e measurement
                # ================= #

                if "shot_avg" not in self.cfg.expt:
                    self.cfg.expt.shot_avg = 1

                # for i_depth, depth in enumerate(tqdm(depths, disable=not progress)):
                loop_order = range(len(depths))
                if loop % 2 == 1:
                    loop_order = range(len(depths) - 1, -1, -1)
                for i_depth in tqdm(loop_order, disable=not progress):
                    # print("depth", i_depth)
                    # print(f'depth {depth} gate list (last gate is the total gate)')
                    for var in range(self.cfg.expt.variations):
                        gate_list = gate_list_variations[i_depth][var]

                        randbench = RBEgGfProgram(
                            soccfg=self.soccfg,
                            cfg=self.cfg,
                            gate_list=gate_list,
                            qubits=[qNotDrive, qDrive],
                            qDrive=self.cfg.expt.qDrive,
                        )
                        # print(gate_list)
                        # print(randbench)
                        # # from qick.helpers import progs2json
                        # # print(progs2json([randbench.dump_prog()]))

                        assert self.cfg.expt.post_process is not None, "need post processing for RB to make sense!"
                        popln, popln_err = randbench.acquire_rotated(
                            soc=self.im[self.cfg.aliases.soc],
                            progress=False,
                            angle=angles_q,
                            threshold=thresholds_q,
                            ge_avgs=ge_avgs_q,
                            post_process=self.cfg.expt.post_process,
                            amplitude_mode=full_mux_expt,
                        )
                        assert (
                            self.cfg.expt.post_process == "threshold"
                        ), "Can only bin EgGf RB properly using threshold"

                        if self.cfg.expt.post_process == "threshold":
                            shots, _ = randbench.get_shots(
                                angle=angles_q, threshold=thresholds_q, amplitude_mode=full_mux_expt
                            )
                            # 000, 001, 010, 011, 100, 101, 110, 111
                            counts = np.array([tomo_analysis.sort_counts([shots[adcQA], shots[adcQB], shots[adcQC]])])
                            data["counts_raw"][0, loop, i_depth, var, :] = counts
                            # print('variation', var, 'gate list', gate_list, 'counts', counts)
                            # print("variation", var, "counts", counts)

                            # print("plotting shots")
                            # I_shot, Q_shot = randbench.get_shots(verbose=False)
                            # for qi, q in enumerate(sscfg.expt.tomo_qubits):
                            #     fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                            #     ax.scatter(
                            #         I_shot[q],
                            #         Q_shot[q],
                            #         marker=".",
                            #         edgecolor="None",
                            #         alpha=0.3,
                            #         color=default_colors[0],
                            #         label="depth %i, var %i" % (depths[i_depth], var),
                            #         zorder=1,
                            #     )
                            #     ax.scatter(
                            #         np.average(I_shot[q]),
                            #         np.average(Q_shot[q]),
                            #         marker="o",
                            #         facecolor=default_colors[0],
                            #         s=50,
                            #         edgecolor="black",
                            #         linewidth=1,
                            #         zorder=2,
                            #     )
                            #     ax.scatter(
                            #         shot_calib[f"Ig"][q],
                            #         shot_calib[f"Qg"][q],
                            #         marker=".",
                            #         edgecolor="None",
                            #         alpha=0.3,
                            #         color=default_colors[1],
                            #         label="g",
                            #         zorder=1,
                            #     )
                            #     ax.scatter(
                            #         shot_calib[f"Ie"][qi],
                            #         shot_calib[f"Qe"][qi],
                            #         marker=".",
                            #         edgecolor="None",
                            #         alpha=0.3,
                            #         color=default_colors[2],
                            #         label="e",
                            #         zorder=1,
                            #     )
                            #     ax.scatter(
                            #         np.average(shot_calib[f"Ig"][q]),
                            #         np.average(shot_calib[f"Qg"][q]),
                            #         marker="o",
                            #         facecolor=default_colors[1],
                            #         s=50,
                            #         edgecolor="black",
                            #         linewidth=1,
                            #         zorder=2,
                            #     )
                            #     ax.scatter(
                            #         np.average(shot_calib[f"Ie"][qi]),
                            #         np.average(shot_calib[f"Qe"][qi]),
                            #         marker="o",
                            #         facecolor=default_colors[2],
                            #         s=50,
                            #         edgecolor="black",
                            #         linewidth=1,
                            #         zorder=2,
                            #     )
                            #     ax.set_title(f"Qubit {q}")
                            #     ax.legend()
                            #     plt.show()

            # ================= #
            # Measure the same thing with g/f distinguishing
            # ================= #

            flip_threshold_all_q = [False] * 4
            # We are using regular mux4 for f readout!
            # flip_threshold_all_q[
            #     q_measure_f
            # ] = full_mux_expt  # if using amplitude mode and measuring at the f resonator frequency, f will have a lower amplitude than g/e
            counts_calib_f = []

            # ================= #
            # Get f state single shot calibration (this must be re-run if you just ran measurement with the standard readout)
            # ================= #

            thresholds_f_q = [0] * 4
            gf_avgs_q = [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]
            angles_f_q = [0] * 4
            fids_f_q = [0] * 4

            # We really just need the single shot plots here, but convenient to use the ErrorMitigation tomo to do it
            sscfg = AttrDict(deepcopy(self.cfg))
            sscfg.expt.reps = sscfg.expt.singleshot_reps_f
            sscfg.expt.tomo_qubits = (
                self.cfg.expt.qubits
            )  # the order of this was set earlier in code so 2nd qubit is the measure f qubit

            sscfg.expt.full_mux_expt = False
            sscfg.expt.resonator_reset = None
            sscfg.device.readout.frequency[qDrive] = sscfg.device.readout.frequency_ef[qDrive]
            sscfg.device.readout.gain = sscfg.device.readout.gain_ef
            # for q in range(4):
            #     if q not in self.cfg.expt.qubits:
            #         sscfg.device.readout.gain[q] = 1e-4
            sscfg.device.readout.readout_length = sscfg.device.readout.readout_length_ef
            local_amplitude_mode = sscfg.expt.full_mux_expt
            # local_amplitude_mode = True

            calib_prog_dict = dict()
            for prep_state in tqdm(self.calib_order):
                # print(prep_state)
                sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state, apply_q1_pi2=False)
                err_tomo = ErrorMitigationStateTomo3QProgram(soccfg=self.soccfg, cfg=sscfg)
                err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=False)
                calib_prog_dict.update({prep_state: err_tomo})

            g_prog = calib_prog_dict["ggg"]
            Ig, Qg = g_prog.get_shots(verbose=False)
            shot_calib = dict(Ig=Ig, Qg=Qg, If=[], Qf=[])

            # Get readout angle + threshold for qubits to distinguish g/f on one of the qubits
            for qi, q in enumerate(sscfg.expt.tomo_qubits):
                calib_f_state = "ggg"
                calib_f_state = calib_f_state[:qi] + f'{"f" if q == qDrive else "e"}' + calib_f_state[qi + 1 :]
                f_prog = calib_prog_dict[calib_f_state]
                If, Qf = f_prog.get_shots(verbose=False)
                shot_data = dict(Ig=Ig[q], Qg=Qg[q], Ie=If[q], Qe=Qf[q])
                shot_calib["If"].append(If[q])
                shot_calib["Qf"].append(Qf[q])
                print(f'Qubit ({q}){f" gf" if q == qDrive else " ge"}')
                fid, threshold, angle = hist(
                    data=shot_data, plot=debug, verbose=False, amplitude_mode=local_amplitude_mode
                )
                thresholds_f_q[q] = threshold[0]
                gf_avgs_q[q] = get_ge_avgs(
                    Igs=Ig[q], Qgs=Qg[q], Ies=If[q], Qes=Qf[q], amplitude_mode=local_amplitude_mode
                )
                angles_f_q[q] = angle
                fids_f_q[q] = fid[0]
                print(f'{"gf" if q == qDrive else "ge"} fidelity (%): {100*fid[0]}')

            # Process the shots taken for the confusion matrix with the calibration angles
            for prep_state in self.calib_order:
                counts = calib_prog_dict[prep_state].collect_counts(
                    angle=angles_f_q,
                    threshold=thresholds_f_q,
                    amplitude_mode=local_amplitude_mode,
                    flip_threshold_all_q=flip_threshold_all_q,
                )
                counts_calib_f.append(counts)

            print(f"thresholds_f={thresholds_f_q},")
            print(f"angles_f={angles_f_q},")
            print(f"gf_avgs={gf_avgs_q},")
            print(f"counts_calib_f={np.array(counts_calib_f).tolist()}")

            data["thresholds_f_loops"].append(thresholds_f_q)
            data["angles_f_loops"].append(angles_f_q)
            data["gf_avgs_loops"].append(gf_avgs_q)
            data["counts_calib_f_loops"].append(np.array(counts_calib_f))

            # ================= #
            # Begin RB for measure f, using same gate list as measure with g/e
            # ================= #

            # for i_depth, depth in enumerate(tqdm(depths, disable=not progress)):
            loop_order = range(len(depths))
            if loop % 2 == 1:
                loop_order = range(len(depths) - 1, -1, -1)
            for i_depth in tqdm(loop_order, disable=not progress):
                for var in range(self.cfg.expt.variations):
                    gate_list = gate_list_variations[i_depth][var]

                    rbcfg = deepcopy(self.cfg)
                    rbcfg.expt.full_mux_expt = False
                    rbcfg.expt.resonator_reset = None
                    rbcfg.device.readout.frequency[qDrive] = rbcfg.device.readout.frequency_ef[qDrive]
                    rbcfg.device.readout.gain = rbcfg.device.readout.gain_ef
                    # for q in range(4):
                    #     if q not in self.cfg.expt.qubits:
                    #         rbcfg.device.readout.gain[q] = 1e-4
                    rbcfg.device.readout.readout_length = rbcfg.device.readout.readout_length_ef
                    local_amplitude_mode = rbcfg.expt.full_mux_expt
                    # local_amplitude_mode = True

                    rbcfg.expt.reps = rbcfg.expt.reps_f

                    randbench = RBEgGfProgram(
                        soccfg=self.soccfg,
                        cfg=rbcfg,
                        gate_list=gate_list,
                        qubits=[qNotDrive, qDrive],
                        qDrive=self.cfg.expt.qDrive,
                    )
                    # print(randbench)
                    # # from qick.helpers import progs2json
                    # # print(progs2json([randbench.dump_prog()]))

                    assert self.cfg.expt.post_process is not None, "need post processing for RB to make sense!"
                    popln, popln_err = randbench.acquire_rotated(
                        soc=self.im[self.cfg.aliases.soc],
                        progress=False,
                        angle=angles_f_q,
                        threshold=thresholds_f_q,
                        ge_avgs=gf_avgs_q,
                        post_process=self.cfg.expt.post_process,
                        amplitude_mode=local_amplitude_mode,
                        flip_threshold_all_q=flip_threshold_all_q,
                    )
                    assert self.cfg.expt.post_process == "threshold", "Can only bin EgGf RB properly using threshold"

                    if self.cfg.expt.post_process == "threshold":
                        shots, _ = randbench.get_shots(
                            angle=angles_f_q,
                            threshold=thresholds_f_q,
                            amplitude_mode=local_amplitude_mode,
                            flip_threshold_all_q=flip_threshold_all_q,
                        )

                        # print("plotting shots")
                        # I_shot, Q_shot = randbench.get_shots(verbose=False)
                        # for qi, q in enumerate(sscfg.expt.tomo_qubits):
                        #     fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                        #     ax.scatter(
                        #         I_shot[q],
                        #         Q_shot[q],
                        #         marker=".",
                        #         edgecolor="None",
                        #         alpha=0.3,
                        #         color=default_colors[0],
                        #         label="depth %i, var %i" % (depths[i_depth], var),
                        #         zorder=1,
                        #     )
                        #     ax.scatter(
                        #         np.average(I_shot[q]),
                        #         np.average(Q_shot[q]),
                        #         marker="o",
                        #         facecolor=default_colors[0],
                        #         s=50,
                        #         edgecolor="black",
                        #         linewidth=1,
                        #         zorder=2,
                        #     )
                        #     ax.scatter(
                        #         shot_calib[f"Ig"][q],
                        #         shot_calib[f"Qg"][q],
                        #         marker=".",
                        #         edgecolor="None",
                        #         alpha=0.3,
                        #         color=default_colors[1],
                        #         label="g",
                        #         zorder=1,
                        #     )
                        #     ax.scatter(
                        #         np.average(shot_calib[f"Ig"][q]),
                        #         np.average(shot_calib[f"Qg"][q]),
                        #         marker="o",
                        #         facecolor=default_colors[1],
                        #         s=50,
                        #         edgecolor="black",
                        #         linewidth=1,
                        #         zorder=2,
                        #     )
                        #     ax.scatter(
                        #         shot_calib[f"If"][qi],
                        #         shot_calib[f"Qf"][qi],
                        #         marker=".",
                        #         edgecolor="None",
                        #         alpha=0.3,
                        #         color=default_colors[2],
                        #         label="f",
                        #         zorder=1,
                        #     )
                        #     ax.scatter(
                        #         np.average(shot_calib[f"If"][qi]),
                        #         np.average(shot_calib[f"Qf"][qi]),
                        #         marker="o",
                        #         facecolor=default_colors[2],
                        #         s=50,
                        #         edgecolor="black",
                        #         linewidth=1,
                        #         zorder=2,
                        #     )
                        #     ax.set_title(f"Qubit {q}")
                        #     ax.legend()
                        #     plt.show()

                        # 000, 002, 010, 012, 100, 102, 110, 112
                        counts = np.array([tomo_analysis.sort_counts([shots[adcQA], shots[adcQB], shots[adcQC]])])
                        data["counts_raw"][1, loop, i_depth, var, :] = counts
                        # print('variation', var, 'gate list', gate_list, 'counts', counts)
                        # print("variation", var, "counts", counts)

        data["xpts"] = np.reshape(data["xpts"], (self.cfg.expt.variations, len(self.cfg.expt.depths)))

        for k, a in data.items():
            # print(k)
            # print(a)
            # print(np.shape(a))
            data[k] = np.array(a)

        self.data = data
        return data

    def calib_index(self, str_stateBC, stateA=None):
        assert len(str_stateBC) == 2
        if stateA is None:
            if self.qDrive == 3:
                stateA = "e"
            elif self.qDrive == 2:
                stateA = "g"
            else:
                assert False
        assert f"{stateA}{str_stateBC}" in self.calib_order
        return self.calib_order.index(f"{stateA}{str_stateBC}")

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        q0, qNotDrive, qDrive = self.cfg.expt.qubits

        self.qDrive = qDrive
        self.qNotDrive = 1

        unique_depths = np.average(data["xpts"], axis=0)
        print(unique_depths)

        # assert self.measure_f
        self.measure_f_only = True
        if "measure_f_only" in self.cfg.expt:
            self.measure_f_only = self.cfg.expt.measure_f_only

        if not self.measure_f_only:
            data["counts_calib_total"] = np.concatenate(
                (data["counts_calib_loops"], data["counts_calib_f_loops"]), axis=2
            )
            data["counts_raw_total"] = np.concatenate(
                (data["counts_raw"][0] / self.cfg.expt.reps, data["counts_raw"][1] / self.cfg.expt.reps_f), axis=3
            )

            # ggg, gge, geg, gee, ggf, gef, egg, ege, eeg, eee, egf, eef
            data["poplns_2q_loops"] = np.zeros(
                shape=(self.cfg.expt.loops, len(unique_depths), self.cfg.expt.variations, 2 * 2 * 3)
            )

        elif self.measure_f_only:
            # ggg, ggf, geg, gef, egg, egf, eeg, eef
            data["poplns_2q_loops"] = np.zeros(
                shape=(self.cfg.expt.loops, len(unique_depths), self.cfg.expt.variations, 2 * 2 * 2)
            )
            data["counts_calib_total"] = data["counts_calib_f_loops"]
            data["counts_raw_total"] = data["counts_raw"][1] / self.cfg.expt.reps_f

        # print("counts calib total shape", np.shape(data["counts_calib_total"]))
        # print(data["counts_calib_total"])
        # print("counts raw shape", np.shape(data["counts_raw_total"]))
        # print(data['counts_raw'])

        for loop in range(self.cfg.expt.loops):
            for idepth, depth in enumerate(tqdm(unique_depths)):
                for ivar in range(self.cfg.expt.variations):
                    # after correcting readout error, counts corrected should correspond to counts in [ggg, gge, geg, gee, ggf, gef, egg, ege, eeg, eee, egf, eef] (the calib_order)
                    # instead of the raw counts:
                    # [gggA, ggeA, gegA, geeA, eggA, egeA, eegA, eeeA,
                    # gggB, ggfB, gegB, gefB, eggB, egfB, eegB, eefB]
                    # or if measure_f_only, [gg, gf, eg, ef] (the calib_order, which = raw counts order)
                    tomo_analysis = TomoAnalysis(nb_qubits=3)
                    counts_corrected = tomo_analysis.correct_readout_err(
                        [data["counts_raw_total"][loop, idepth, ivar]], data["counts_calib_total"][loop], verbose=False
                    )
                    # print("counts raw", data["counts_raw_total"][loop, idepth, ivar])
                    # print("counts_corrected", counts_corrected)
                    # print(data["counts_calib_total"][loop])
                    # counts_corrected = tomo_analysis.fix_neg_counts(counts_corrected)
                    data["poplns_2q_loops"][loop, idepth, ivar, :] = counts_corrected / np.sum(counts_corrected)

        data["poplns_2q"] = np.average(data["poplns_2q_loops"], axis=0)

        # print("poplns_2q_loops shape", np.shape(data["poplns_2q_loops"]))
        # print("poplns_2q shape", np.shape(data["poplns_2q"]))

        probs_eg = data["poplns_2q"][:, :, self.calib_index("eg")]
        probs_gf = data["poplns_2q"][:, :, self.calib_index("gf")]
        probs_gg = data["poplns_2q"][:, :, self.calib_index("gg")]
        probs_ef = data["poplns_2q"][:, :, self.calib_index("ef")]
        if not self.measure_f_only:
            # ggg, gge, geg, gee, ggf, gef, egg, ege, eeg, eee, egf, eef
            probs_ge = data["poplns_2q"][:, :, self.calib_index("ge")]
            probs_ee = data["poplns_2q"][:, :, self.calib_index("ee")]

        data["popln_eg_std"] = np.std(probs_eg, axis=1)
        data["popln_eg_avg"] = np.average(probs_eg, axis=1)
        data["popln_eg_err"] = np.std(probs_eg, axis=1) / np.sqrt(np.shape(probs_eg)[1])

        data["popln_gf_std"] = np.std(probs_gf, axis=1)
        data["popln_gf_err"] = np.std(probs_gf, axis=1) / np.sqrt(np.shape(probs_gf)[1])
        data["popln_gf_avg"] = np.average(probs_gf, axis=1)

        data["popln_gg_std"] = np.std(probs_gg, axis=1)
        data["popln_gg_avg"] = np.average(probs_gg, axis=1)
        data["popln_gg_err"] = np.std(probs_gg, axis=1) / np.sqrt(np.shape(probs_gg)[1])

        data["popln_ef_std"] = np.std(probs_ef, axis=1)
        data["popln_ef_avg"] = np.average(probs_ef, axis=1)
        data["popln_ef_err"] = np.std(probs_ef, axis=1) / np.sqrt(np.shape(probs_ef)[1])

        if not (self.measure_f_only):
            data["popln_ge_std"] = np.std(probs_ge, axis=1)
            data["popln_ge_avg"] = np.average(probs_ge, axis=1)
            data["popln_ge_err"] = np.std(probs_ge, axis=1) / np.sqrt(np.shape(probs_ge)[1])
            data["popln_ee_std"] = np.std(probs_ee, axis=1)
            data["popln_ee_avg"] = np.average(probs_ee, axis=1)
            data["popln_ee_err"] = np.std(probs_ee, axis=1) / np.sqrt(np.shape(probs_ee)[1])

        sum_prob_subspace = probs_eg + probs_gf
        data["popln_subspace"] = sum_prob_subspace
        data["popln_subspace_std"] = np.std(sum_prob_subspace, axis=1)
        data["popln_subspace_avg"] = np.average(sum_prob_subspace, axis=1)
        data["popln_subspace_err"] = np.std(sum_prob_subspace, axis=1) / np.sqrt(np.shape(sum_prob_subspace)[1])

        data["popln_eg_subspace"] = probs_eg / sum_prob_subspace
        data["popln_eg_subspace_std"] = np.std(probs_eg / sum_prob_subspace, axis=1)
        data["popln_eg_subspace_avg"] = np.average(probs_eg / sum_prob_subspace, axis=1)
        data["popln_eg_subspace_err"] = np.std(probs_eg / sum_prob_subspace, axis=1) / np.sqrt(
            np.shape(probs_eg / sum_prob_subspace)[1]
        )

        data["popln_gf_subspace"] = probs_gf / sum_prob_subspace
        data["popln_gf_subspace_std"] = np.std(probs_gf / sum_prob_subspace, axis=1)
        data["popln_gf_subspace_avg"] = np.average(probs_gf / sum_prob_subspace, axis=1)
        data["popln_gf_subspace_err"] = np.std(probs_gf / sum_prob_subspace, axis=1) / np.sqrt(
            np.shape(probs_gf / sum_prob_subspace)[1]
        )

        # print("shape sum prob_eg + prob_gf", np.shape(sum_prob_subspace))
        # print(
        #     "shape average sum over each depth",
        #     np.shape(data["popln_subspace_avg"]),
        #     "should equal",
        #     np.shape(unique_depths),
        # )

        if not fit:
            return data

        popt1, pcov1 = fitter.fitrb(unique_depths, data["popln_subspace_avg"])
        print("fit1 p1, a, offset", popt1)
        data["fit1"] = popt1
        data["fit1_err"] = pcov1
        p1, a, offset = popt1
        data["l1"] = fitter.leakage_err(p1, offset)
        data["l2"] = fitter.seepage_err(p1, offset)

        popt2, pcov2 = fitter.fitrb_l1_l2(unique_depths, data["popln_eg_avg"], p1=p1, offset=offset)
        print("fit2 a0, b0, c0, p2", popt2)
        data["fit2"] = popt2
        data["fit2_err"] = pcov2
        a0, b0, c0, p2 = popt2

        data["fidelity"], data["fidelity_err"] = fitter.rb_fidelity_l1_l2(
            d=2, p2=p2, l1=data["l1"], p2_err=pcov2[3][3], l1_err=pcov1[0][0]
        )

        popt3, pcov3 = fitter.fitrb(unique_depths, data["popln_eg_subspace_avg"])
        data["fit3"] = popt3
        data["fit3_err"] = pcov3

        return data

    def display(self, data=None, fit=True, show_all_vars=False):
        if data is None:
            data = self.data

        plt.figure(figsize=(8, 6))
        irb = "gate_char" in self.cfg.expt and self.cfg.expt.gate_char is not None
        title = f'{"Interleaved " + self.cfg.expt.gate_char + " Gate" if irb else ""} EgGf RB on Q{self.cfg.expt.qubits[0]}, Q{self.cfg.expt.qubits[1]}, Q{self.cfg.expt.qubits[2]}'

        plt.subplot(111, title=title, xlabel="Sequence Depth", ylabel="Population")
        depths = data["xpts"]
        unique_depths = np.average(depths, axis=0)
        flat_depths = np.concatenate(depths)
        print(flat_depths)
        flat_probs_eg = np.concatenate(data["poplns_2q"][:, :, self.calib_index("eg")])

        print("flat_probs_eg", flat_probs_eg)
        print("all poplns_2q\n", np.round(data["poplns_2q"], 3))
        flat_probs_subspace = np.concatenate(data["popln_subspace"])
        if show_all_vars:
            plt.plot(flat_depths, flat_probs_eg, "x", color="tab:grey")
            plt.plot(flat_depths, flat_probs_subspace, "v", color="tab:grey")

        probs_eg_avg = data["popln_eg_avg"]
        probs_eg_std = data["popln_eg_std"]
        probs_eg_err = data["popln_eg_err"]
        probs_subspace_avg = data["popln_subspace_avg"]
        probs_subspace_std = data["popln_subspace_std"]
        probs_subspace_err = data["popln_subspace_err"]
        probs_eg_subspace_avg = data["popln_eg_subspace_avg"]
        probs_eg_subspace_std = data["popln_eg_subspace_std"]
        probs_eg_subspace_err = data["popln_eg_subspace_err"]
        probs_gf_subspace_avg = data["popln_gf_subspace_avg"]
        probs_gf_subspace_std = data["popln_gf_subspace_std"]
        probs_gf_subspace_err = data["popln_gf_subspace_err"]

        probs_gg_avg = data["popln_gg_avg"]
        probs_gg_std = data["popln_gg_std"]
        probs_gg_err = data["popln_gg_err"]
        probs_gf_avg = data["popln_gf_avg"]
        probs_gf_std = data["popln_gf_std"]
        probs_gf_err = data["popln_gf_err"]
        probs_ef_avg = data["popln_ef_avg"]
        probs_ef_std = data["popln_ef_std"]
        probs_ef_err = data["popln_ef_err"]

        if not self.measure_f_only:
            probs_ge_avg = data["popln_ge_avg"]
            probs_ge_std = data["popln_ge_std"]
            probs_ge_err = data["popln_ge_err"]
            probs_ee_avg = data["popln_ee_avg"]
            probs_ee_std = data["popln_ee_std"]
            probs_ee_err = data["popln_ee_err"]
            print("probs_ge_avg", probs_ge_avg, "+/-", probs_ge_std)
            print("probs_ee_avg", probs_ee_avg, "+/-", probs_ee_std)

        print("prob_eg_avg", probs_eg_avg, "+/-", probs_eg_std)
        print("prob_subspace_avg", probs_subspace_avg, "+/-", probs_subspace_std)
        print("prob_eg_subspace_avg", probs_eg_subspace_avg, "+/-", probs_eg_subspace_std)
        # plt.errorbar(working_depths, avg_probs, fmt='o', yerr=2*std_dev_probs, color='k', elinewidth=0.75)
        plt.errorbar(
            unique_depths,
            probs_eg_avg,
            fmt="x",
            yerr=probs_eg_err,
            color=default_colors[0],
            elinewidth=0.75,
            label=f"{'(e)' if self.qDrive == 3 else '(g)'}eg probability",
        )
        plt.errorbar(
            unique_depths,
            probs_subspace_avg,
            fmt="v",
            yerr=probs_subspace_err,
            color=default_colors[1],
            elinewidth=0.75,
            label="subspace probability",
        )
        plt.errorbar(
            unique_depths,
            probs_eg_subspace_avg,
            fmt="o",
            yerr=probs_eg_subspace_err,
            color=default_colors[2],
            elinewidth=0.75,
            label=f"{'(e)' if self.qDrive == 3 else '(g)'}eg/subspace probability",
        )

        plt.errorbar(
            unique_depths,
            probs_gf_subspace_avg,
            fmt="o",
            yerr=probs_gf_subspace_err,
            color=default_colors[5 % len(default_colors)],
            elinewidth=0.75,
            label=f"{'(e)' if self.qDrive == 3 else '(g)'}gf/subspace probability",
        )

        plt.errorbar(
            unique_depths,
            probs_gg_avg,
            fmt="x",
            yerr=probs_gg_err,
            color=default_colors[3],
            elinewidth=0.75,
            label=f"{'(e)' if self.qDrive == 3 else '(g)'}gg probability",
        )

        plt.errorbar(
            unique_depths,
            probs_gf_avg,
            fmt="x",
            yerr=probs_gf_err,
            color=default_colors[4],
            elinewidth=0.75,
            label=f"{'(e)' if self.qDrive == 3 else '(g)'}gf probability",
        )

        plt.errorbar(
            unique_depths,
            probs_ef_avg,
            fmt="x",
            yerr=probs_ef_err,
            color=default_colors[0],
            elinewidth=0.75,
            label=f"{'(e)' if self.qDrive == 3 else '(g)'}ef probability",
        )

        if not self.measure_f_only:

            plt.errorbar(
                unique_depths,
                probs_ge_avg,
                fmt="x",
                yerr=probs_ge_err,
                color=default_colors[6 % len(default_colors)],
                elinewidth=0.75,
                label=f"{'(e)' if self.qDrive == 3 else '(g)'}ge probability",
            )

            plt.errorbar(
                unique_depths,
                probs_ee_avg,
                fmt="x",
                yerr=probs_ee_err,
                color=default_colors[7 % len(default_colors)],
                elinewidth=0.75,
                label=f"{'(e)' if self.qDrive == 3 else '(g)'}ee probability",
            )

        if fit:
            pcov1 = data["fit1_err"]
            # plt.plot(depths, avg_probs, 'o-', color='tab:blue')
            plt.plot(unique_depths, fitter.rb_func(unique_depths, *data["fit1"]), color=default_colors[1])
            print(
                f'Running {"interleaved " + self.cfg.expt.gate_char + " gate" if irb else "regular"} RB on EgGf subspace'
            )
            p1 = data["fit1"][0]
            print(f"Depolarizing parameter p1 from fit: {p1} +/- {np.sqrt(pcov1[0][0])}")
            # print(f'Average RB gate error: {data["error"]} +/- {np.sqrt(fitter.error_fit_err(pcov1, 2**(len(self.cfg.expt.qubits))))}')
            # print(f'\tFidelity=1-error: {1-data["error"]} +/- {np.sqrt(fitter.error_fit_err(pcov1, 2**(len(self.cfg.expt.qubits))))}')

            pcov2 = data["fit2_err"]
            # plt.plot(depths, avg_probs, 'o-', color='tab:blue')
            plt.plot(unique_depths, fitter.rb_decay_l1_l2(unique_depths, p1, *data["fit2"]), color=default_colors[0])
            print(
                f'Running {"interleaved " + self.cfg.expt.gate_char + " gate" if irb else "regular"} RB on EgGf subspace'
            )
            print(f'Depolarizing parameter p2 from fit: {data["fit2"][3]} +/- {np.sqrt(pcov2[3][3])}')
            print(f'Fidelity: {data["fidelity"]} +/- {data["fidelity_err"]}')
            print(f'Leakage L1: {data["l1"]}')
            print(f'Seepage L2: {data["l2"]}')

            pcov3 = data["fit3_err"][0][0]
            # plt.plot(depths, avg_probs, 'o-', color='tab:blue')
            plt.plot(unique_depths, fitter.rb_func(unique_depths, *data["fit3"]), color=default_colors[2])
            p = data["fit3"][0]
            print(f"Depolarizing parameter p from eg/subspace fit: {p} +/- {np.sqrt(pcov3)}")
            err = fitter.rb_error(p, d=2)
            print(f"Average RB gate error on eg/subspace: {err} +/- {np.sqrt(fitter.error_fit_err(pcov3, 2))}")
            print(f"\tFidelity of eg/subspace=1-error: {1-err} +/- {np.sqrt(fitter.error_fit_err(pcov3, 2))}")

        plt.grid(linewidth=0.3)
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        with self.datafile() as f:
            f.attrs["calib_order"] = json.dumps(self.calib_order, cls=NpEncoder)
        return self.fname


class EgGfLeakageExperiment(SimultaneousRBEgGfExperiment):
    def acquire(self, progress=False, debug=False):
        if "test_leakage" not in self.cfg.expt:
            self.cfg.expt.test_leakage = True
        assert self.cfg.expt.test_leakage

        if "ground_state_init" not in self.cfg.expt:
            self.cfg.expt.ground_state_init = False

        self.cfg.expt.variations = 1
        super().acquire(progress=progress, debug=debug)

    def calib_index(self, str_stateABC):
        assert len(str_stateABC) == 3 or len(str_stateABC) == 2
        if len(str_stateABC) == 2:  # pick the wrong switch state by default
            if self.qDrive == 2:
                stateA = "e"
            elif self.qDrive == 3:
                stateA = "g"
            else:
                assert False
            str_stateABC = f"{stateA}{str_stateABC}"
        assert f"{str_stateABC}" in self.calib_order
        return self.calib_order.index(f"{str_stateABC}")

    def analyze(self, data=None, fit=False, **kwargs):
        super().analyze(data=data, fit=fit, **kwargs)
        if data is None:
            data = self.data

        # G/B: good/bad switch state
        bad = "g" if self.qDrive == 3 else "e"
        good = "e" if self.qDrive == 3 else "g"

        self.prob_names_index_dict = dict(
            Beg=f"{bad}eg",
            Bgf=f"{bad}gf",
            Bge=f"{bad}ge",
            Bgg=f"{bad}gg",
            Geg=f"{good}eg",
            Ggf=f"{good}gf",
            Ggg=f"{good}gg",
        )
        self.probs_dict = dict()

        for probs_name in self.prob_names_index_dict.keys():
            # print(probs_name, self.prob_names_index_dict[probs_name])
            probs = data[f"poplns_2q"][:, :, self.calib_index(self.prob_names_index_dict[probs_name])]
            # print("hello???", probs)
            self.probs_dict[probs_name] = probs
        for probs_name in self.calib_order:
            # print(probs_name, self.prob_names_index_dict[probs_name])
            probs = data[f"poplns_2q"][:, :, self.calib_index(probs_name)]
            # print("hello???", probs)
            self.probs_dict[probs_name] = probs
        self.probs_dict["bad_subspace"] = self.probs_dict["Beg"] + self.probs_dict["Bgf"]
        self.probs_dict["good_subspace"] = self.probs_dict["Geg"] + self.probs_dict["Ggf"]
        self.probs_dict["Bgf_bad_subspace"] = self.probs_dict["Bgf"] / self.probs_dict["bad_subspace"]
        self.probs_dict["Beg_bad_subspace"] = self.probs_dict["Beg"] / self.probs_dict["bad_subspace"]
        self.probs_dict["Ggf_good_subspace"] = self.probs_dict["Ggf"] / self.probs_dict["good_subspace"]
        self.probs_dict["Geg_good_subspace"] = self.probs_dict["Geg"] / self.probs_dict["good_subspace"]

        for probs_name in self.probs_dict.keys():
            probs = self.probs_dict[probs_name]
            data[f"popln_{probs_name}_std"] = np.std(probs, axis=1)
            data[f"popln_{probs_name}_avg"] = np.average(probs, axis=1)
            data[f"popln_{probs_name}_err"] = np.std(probs, axis=1) / np.sqrt(np.shape(probs)[1])
            # print(probs_name, data[f"popln_{probs_name}_avg"])

        return data

    def display(self, data=None, fit=False, show_all_vars=False):
        if data is None:
            data = self.data

        plt.figure(figsize=(8, 6))
        irb = "gate_char" in self.cfg.expt and self.cfg.expt.gate_char is not None
        title = (
            f'{"Interleaved " + self.cfg.expt.gate_char + " Gate" if irb else ""} EgGf $\\times$ depth on Q{self.cfg.expt.qubits[0]}, Q{self.cfg.expt.qubits[1]}, Q{self.cfg.expt.qubits[2]} From {"Wrong" if self.cfg.expt.wrong_init else "Right"} Switch State'
            + (" gg" if self.cfg.expt.ground_state_init else "")
        )

        plt.subplot(111, title=title, xlabel="Sequence Depth", ylabel="Population")
        depths = data["xpts"]
        unique_depths = np.average(depths, axis=0)
        flat_depths = np.concatenate(depths)
        print(flat_depths)
        flat_probs_eg = np.concatenate(data["poplns_2q"][:, :, self.calib_index("eg")])

        # print("flat_probs_eg", flat_probs_eg)
        # print("all poplns_2q\n", np.round(data["poplns_2q"], 3))
        flat_probs_subspace = np.concatenate(data["popln_subspace"])
        if show_all_vars:
            plt.plot(flat_depths, flat_probs_eg, "x", color="tab:grey")
            plt.plot(flat_depths, flat_probs_subspace, "v", color="tab:grey")

        markers = ["x", "v", "o"]

        if self.cfg.expt.wrong_init:
            plot_names = ["Beg", "Bgf", "Bgg", "Geg", "Ggf", "bad_subspace", "Bgf_bad_subspace", "Beg_bad_subspace"]
            if self.cfg.expt.ground_state_init:
                plot_names = ["Bgf", "Bgg", "Bge", "ggg", "gge", "ggf"]
        else:
            plot_names = ["Geg", "Ggf", "ggg", "gge", "good_subspace", "Geg_good_subspace"]
            if self.cfg.expt.ground_state_init:
                plot_names.append("gge")
        # plot_names = ["Bgf"]
        # for i, probs_name in enumerate(self.probs_dict.keys()):
        for i, probs_name in enumerate(plot_names):
            plt.errorbar(
                unique_depths,
                data[f"popln_{probs_name}_avg"],
                fmt=markers[i % len(markers)] + "-",
                yerr=data[f"popln_{probs_name}_err"],
                color=default_colors[i % len(default_colors)],
                elinewidth=0.75,
                label=probs_name,
            )

        for plot_name in self.probs_dict.keys():
            print(plot_name, data[f"popln_{plot_name}_avg"][-1])

        plt.grid(linewidth=0.3)
        # plt.ylim(-0.05, 1.05)
        # plt.ylim(-0.01, 0.2)
        # plt.ylim(0.8, 1.01)
        plt.legend()
        plt.show()
