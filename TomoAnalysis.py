import logging

import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import yaml
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qick import *
from qick.helpers import gauss
from scipy.optimize import minimize

logger = logging.getLogger("qick.qick_asm")
logger.setLevel(logging.ERROR)

import os
import time

from tqdm import tqdm_notebook as tqdm

expt_path = os.getcwd() + "/data"
import itertools
import json
import sys

import qutip as qt
import qutip.visualization as qplt
import scipy as sp
from qutip_qip.operations import rz
from slab import AttrDict, get_next_filename
from slab.datamanagement import SlabFile
from slab.experiment import Experiment
from slab.instruments import InstrumentManager

sys.path.append(os.getcwd() + "/../../qutip_sims")
from PulseSequence import PulseSequence
from QSwitch import QSwitch
from scipy.optimize import minimize

style.use("S:\Connie\prx.mplstyle")

from experiments.clifford_averager_program import post_select_shots, ps_threshold_adjust

"""
Get effective drive rate in GHz
"""


def amp_eff(sigma_ns, sigma_n=4):
    """
    When this amp_eff is passed as a drive amplitude to the simulation,
    performs a full pi pulse when sigma_n*sigma_ns gaussian is played
    """
    return 1 / 2 / (sigma_ns * np.sqrt(2 * np.pi) * sp.special.erf(sigma_n / 2 / np.sqrt(2)))


def phase_to_other_drive(phase):
    """
    Return what phi should be when driving in the other quadrature, i.e. cos(wt+phi) <-> sin(wt-phi)
    """
    return phase


def ge_label_to_numeric_str(ge_label):
    """
    Convert a state str of g/e to 0/1 in string format
    """
    label_numeric = ""
    for char in ge_label:
        if char == "g":
            label_numeric += "0"
        elif char == "e":
            label_numeric += "1"
    return label_numeric


class TomoAnalysis:

    basis_list = ["Z", "X", "Y"]
    calib_list = ["g", "e"]
    psiZ = [qt.basis(2, 0), qt.basis(2, 1)]
    psiX = [1 / np.sqrt(2) * (psiZ[0] + psiZ[1]), 1 / np.sqrt(2) * (psiZ[0] - psiZ[1])]
    psiY = [1 / np.sqrt(2) * (psiZ[0] + 1j * psiZ[1]), 1 / np.sqrt(2) * (psiZ[0] - 1j * psiZ[1])]
    psi_dict = dict(Z=psiZ, X=psiX, Y=psiY)

    def __init__(
        self,
        nb_qubits=3,
        nb_qubits_tot=4,
        rfsoc_config=None,
        config_file="config_q3diamond_full688and638_reset.yml",
        meas_order=None,
        calib_order=None,
    ):

        self.nb_qubits = nb_qubits
        self.nb_qubits_tot = nb_qubits_tot
        if meas_order is None:
            # all possible permutations of the basis_list like ZZZ, ZZX, ZYX, etc.
            self.meas_order = np.array(["".join(x) for x in itertools.product(self.basis_list, repeat=nb_qubits)])
        else:
            self.meas_order = meas_order

        if calib_order is None:
            self.calib_order = np.array(["".join(x) for x in itertools.product(self.calib_list, repeat=nb_qubits)])
        else:
            self.calib_order = calib_order

        calib_order_numeric = np.zeros_like(self.calib_order)
        for i, label in enumerate(self.calib_order):
            calib_order_numeric[i] = ge_label_to_numeric_str(label)
        self.calib_order_numeric = calib_order_numeric

        # file a table with the first 2** number in binary
        state_range = 2**self.nb_qubits
        state_bin = [np.binary_repr(i, width=self.nb_qubits) for i in range(state_range)]
        psi_basis = dict()
        for S in self.meas_order:
            psi_basis.update({S: []})
            for state in state_bin:
                psi = qt.tensor([self.psi_dict[S][int(state[i])] for i, S in enumerate(S)]).unit()
                psi_basis[S].append(psi)

        psi_basis_flat = []
        for S in self.meas_order:
            for psi in psi_basis[S]:
                psi_basis_flat.append(psi)

        self.psi_basis = psi_basis
        self.psi_basis_flat = psi_basis_flat

        config_path = os.path.join("s:\\Connie\\experiments\\qram_tprocv1_expts\\configs\\", config_file)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.config_file = AttrDict(config)

        self.rfsoc_config = rfsoc_config

    def pauli(self, i):
        return [qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()][i]

    def s2i(self, s):
        """
        String to pauli matrix index
        """
        return int(np.char.find("IXYZ", s))

    def i2s(self, i):
        """
        Pauli matrix index to string
        """
        return "IXYZ"[i]

    def order(self, S):
        """
        Given a pauli matrix name, get the index at which it (or equivalent data) was measured
        """
        for i in range(len(S)):
            assert S[i] in "IXYZ"
        # For the purposes of getting counts, measuring in I basis and Z basis are interchangeable
        S = S.replace("I", "Z")
        # Return the index of the measurement basis
        return np.argwhere(self.meas_order == np.array([S]))[0][0]
        # print(order('X'))

    # ============ Counts from shots (readout post selection only) ============ #
    """
    shots_q should be shape (nb_qubits, reps)
    """

    def sort_counts(self, shots_q):
        counts = []
        shots_q = np.array(shots_q)
        assert shots_q.shape[0] == self.nb_qubits  # should be num tomo qubits, reps (0/1)
        assert len(shots_q.shape) == 2
        qcounts_e = shots_q
        qcounts_g = np.logical_not(qcounts_e)
        qcounts = np.array([qcounts_g, qcounts_e])  # shape: 2, num tomo qubits, reps (0/1)
        for q1_state in (0, 1):
            if self.nb_qubits == 1:
                counts.append(np.sum(qcounts[q1_state, 0]))
                continue

            assert self.nb_qubits > 1
            for q2_state in (0, 1):
                and_q1q2_counts = np.logical_and(qcounts[q1_state, 0], qcounts[q2_state, 1])
                if self.nb_qubits == 2:
                    counts.append(np.sum(and_q1q2_counts))
                    continue

                assert self.nb_qubits == 3, "sort counts only implemented up to 3 qubits"
                for q3_state in (0, 1):
                    and_q1q2q3_counts = np.logical_and(and_q1q2_counts, qcounts[q3_state, 2])
                    counts.append(np.sum(and_q1q2q3_counts))

        return np.array(counts)

    """
    Get the sorted raw counts for each prep state given raw iq data and apply post selection if
    requested.
    
    all_ishots_raw_q_preps: shape is (prep_states, adc_chs, n_init_readout+1, reps)
    nprep: if there are more rows in the array than needed for prep states, specify nprep
    thresholds is used to threshold the final iq shots, shape is nb_qubits_tot
    ps_adjust if specified should have shape nb_qubits_tot, only ps_qubits are adjusted/applied
    ge_avgs is used to adjust the ps_threshold, shape is (4, nb_qubits_tot) (Ig, Qg, Ie, Qe)
    If angles is specified, uses qshots to rotate prior to thresholding, shape is nb_qubits_tot
    """

    def counts_from_iqshots(
        self,
        tomo_qubits,
        all_ishots_raw_q_preps,
        thresholds,
        ge_avgs=None,
        nprep=None,
        ps_adjust=None,
        ps_qubits=None,
        n_init_readout=1,
        angles=None,
        all_qshots_raw_q_preps=None,
        verbose=False,
        amplitude_mode=False,
    ):
        assert len(all_ishots_raw_q_preps.shape) == 4
        assert all_ishots_raw_q_preps.shape[1] == self.nb_qubits_tot
        assert len(tomo_qubits) == self.nb_qubits
        if angles is not None:
            assert all_qshots_raw_q_preps is not None
            assert all_qshots_raw_q_preps.shape == all_ishots_raw_q_preps.shape
        if nprep is None:
            nprep = all_ishots_raw_q_preps.shape[0]

        counts_all = np.zeros((nprep, 2**self.nb_qubits))
        for iprep in range(nprep):
            ps_thresholds = thresholds
            if verbose:
                print("ps thresholds", ps_thresholds)
            if ps_adjust is not None:
                ps_thresholds = ps_threshold_adjust(
                    ps_thresholds_init=thresholds,
                    adjust=ps_adjust,
                    ge_avgs=ge_avgs,
                    angles=angles,
                    amplitude_mode=amplitude_mode,
                )
                if verbose:
                    print("new ps thresholds", ps_thresholds)

            shots_q = []
            for q in tomo_qubits:
                shots_ps = post_select_shots(
                    final_qubit=q,
                    all_ishots_raw_q=all_ishots_raw_q_preps[iprep],
                    all_qshots_raw_q=all_qshots_raw_q_preps[iprep] if angles is not None else None,
                    ps_thresholds=ps_thresholds,
                    ps_qubits=ps_qubits,
                    n_init_readout=n_init_readout,
                    angles=angles,
                    post_process="threshold",
                    thresholds=thresholds,
                    verbose=verbose,
                    amplitude_mode=amplitude_mode,
                )
                shots_q.append(shots_ps)
            shots_q = np.array(shots_q)
            counts_row = self.sort_counts(shots_q)
            # print('counts_row', counts_row)
            counts_all[iprep, :] = counts_row / np.sum(counts_row)

        return counts_all

    """
    Just a wrapper to get n_tomo and n_calib directly given a data set (organized as expected), cfg, and ps flags
    If any of ishots_raw, qshots_raw, calib_ishots_raw, calib_qshots_raw are not None, overrides that key in data
    """

    def n_tomo_calib_from_data(
        self,
        data,
        cfg,
        tomo_qubits=None,
        ps_adjust=None,
        ps_qubits=None,
        apply_ps=True,
        verbose=False,
        ishots_raw=None,
        qshots_raw=None,
        calib_ishots_raw=None,
        calib_qshots_raw=None,
    ):
        if apply_ps:
            assert cfg.expt.readout_cool, "Experiment was not run with post selection readouts!"
            assert "thresholds" in data
            assert "ge_avgs" in data
            assert "angles" in data
            assert "ishots_raw" in data
            assert "qshots_raw" in data
            assert "calib_ishots_raw" in data
            assert "calib_qshots_raw" in data

            amplitude_mode = False
            if "full_mux_expt" in cfg.expt:
                amplitude_mode = cfg.expt.full_mux_expt

            thresholds = np.array(data["thresholds"])
            ge_avgs = np.array(data["ge_avgs"])
            angles = np.array(data["angles"])

            if ishots_raw is None:
                ishots_raw = data["ishots_raw"]
            if qshots_raw is None:
                qshots_raw = data["qshots_raw"]
            if calib_ishots_raw is None:
                calib_ishots_raw = data["calib_ishots_raw"]
            if calib_qshots_raw is None:
                calib_qshots_raw = data["calib_qshots_raw"]
            if tomo_qubits is None:
                tomo_qubits = cfg.expt.tomo_qubits

            n_tomo_raw = self.counts_from_iqshots(
                tomo_qubits=tomo_qubits,
                all_ishots_raw_q_preps=np.array(ishots_raw),
                all_qshots_raw_q_preps=np.array(qshots_raw),
                thresholds=thresholds,
                ge_avgs=ge_avgs,
                ps_adjust=ps_adjust,
                ps_qubits=ps_qubits,
                n_init_readout=1,
                angles=angles,
                verbose=verbose,
                amplitude_mode=amplitude_mode,
            )

            # print()
            # print('n calib')
            n_calib = self.counts_from_iqshots(
                tomo_qubits=tomo_qubits,
                all_ishots_raw_q_preps=np.array(calib_ishots_raw),
                all_qshots_raw_q_preps=np.array(calib_qshots_raw),
                thresholds=thresholds,
                ge_avgs=ge_avgs,
                nprep=len(self.calib_order),
                ps_adjust=ps_adjust,
                ps_qubits=ps_qubits,
                n_init_readout=1,
                angles=angles,
                verbose=verbose,
                amplitude_mode=amplitude_mode,
            )
        else:
            assert "counts_tomo" in data
            assert "counts_calib" in data
            n_tomo_raw = np.array(data["counts_tomo"])
            n_calib = np.array(data["counts_calib"])

        return n_tomo_raw, n_calib

    # =========================== Rho from counts ====================================== #
    """
    Convert single shot measurements into counts for each of the 9 measurement axes: 4x9=36 elements in the count array, then into the initial rho from experiment which will likely be unphysical.
    """

    def Tij(self, n, S):
        """
        n should be length 2^q array containing single shot counts of measuring 000, 001, 010, 011, ... for measurement along axes i, j, k
        Converts n to Tij for use in the T^\dag T = rho matrix where measurement axes are S1, S2, S3: I, X, Y, Z
        """
        # convert S in an array of int
        S = [int(S[i]) for i in range(self.nb_qubits)]

        for i in range(len(S)):
            assert 0 <= S[i] <= 3  # S[i] represent pauli matrix indices
        signs = [1] * (2**self.nb_qubits)
        for icalib, calib in enumerate(self.calib_order):
            for q in range(self.nb_qubits):
                if S[q] > 0 and calib[q] == "e":
                    signs[icalib] *= -1
        return np.sum(np.multiply(signs, n)) / np.sum(n)

    def rho_from_counts(self, n):
        """
        Construct rho by adding together Tij for each of the 4^3 = 64 combinations of tensor product of 3 Pauli matrices
        Total number of matrices to add up should be d^2, d=2^n_qubits
        """
        qb_list = [qt.qeye(2)] * self.nb_qubits
        rho = 0 * qt.tensor(*qb_list)

        # loop over all possible 3 qubit pauli matrices
        pauli_range = 4**self.nb_qubits
        state_base4 = [np.base_repr(i, base=4) for i in range(pauli_range)]
        state_base4 = [state.zfill(self.nb_qubits) for state in state_base4]

        for state in state_base4:
            idx = ""
            for qb in range(self.nb_qubits):
                idx += self.i2s(int(state[qb]))
            o = self.order(idx)
            rho += (
                self.Tij(n[o : o + 1], state)
                * qt.tensor(*[self.pauli(int(state[qb])) for qb in range(self.nb_qubits)])
                / 2**self.nb_qubits
            )

        rho = rho.tidyup(1e-10)  # remove small elements
        rho = rho.full()
        assert np.around(np.trace(rho), 1), f"Trace of rho from counts is {np.trace(rho)}"
        return rho

    # =========================== Generate test data ====================================== #
    def generate_counts(self, rho_id, n_tot, psi_basis_flat=None, evol_mats=None, noise=1.0):
        # print(evol_mats)
        assert psi_basis_flat is not None or evol_mats is not None
        if psi_basis_flat is not None:
            print("Generating with psi basis")
            n = []
            for psi in psi_basis_flat:
                # measure in psi basis
                n.append(n_tot * (psi.dag() * rho_id * psi).tr())
            n = np.reshape(np.array(n), (len(self.psi_basis.keys()), 2**self.nb_qubits))

        elif evol_mats is not None:
            print("Generating with evol mats")
            evals, evecs = np.linalg.eig(rho_id)
            evals = np.real(evals)
            n_evol = []
            for basis in self.meas_order:
                evol_mat = evol_mats[basis]  # evol_mat sends each psi_ij to the evolved ket
                n_basis_th = np.zeros(2**self.nb_qubits)
                for i in range(2**self.nb_qubits):
                    evec_evol = evol_mat @ evecs.T[i]
                    if evals[i] > 1e-12:
                        n_basis_th += evals[i] * n_tot * abs(evec_evol) ** 2  # add counts weighted by eval
                n_evol.append(n_basis_th)
            n_evol = np.array(n_evol)
            return n_evol

        # introduce gaussian noise
        if noise is not None:
            for n_psi in n:
                n_meas = sum(n_psi)
                new_nlast = -1
                while new_nlast < 0:
                    new_n_excludelast = np.random.normal(loc=n_psi[:-1], scale=noise * np.sqrt(n_psi[:-1]))
                    # preserve original total count per measurement
                    new_nlast = n_meas - sum(new_n_excludelast)
                n_psi[:-1] = np.round(new_n_excludelast)
                n_psi[-1] = np.round(new_nlast)
        return n

    # =========================== Error Mitigation ====================================== #
    """
    See qiskit measurement error mitigation procedure: [https://qiskit.org/textbook/ch-quantum-hardware/measurement-error-mitigation.html](https://qiskit.org/textbook/ch-quantum-hardware/measurement-error-mitigation.html)
    """

    def correct_readout_err_legacy(self, n, n_conf):
        n = np.array(n, dtype=float)
        conf_mat = np.array(n_conf, dtype=float)
        assert len(n.shape) == 2  # 2d array
        assert len(conf_mat.shape) == 2  # 2d array
        old_sum = sum(n[0])
        n_out_states = np.shape(conf_mat)[0]  # number of possible states that we are correcting our counts into
        for r, row in enumerate(conf_mat):
            conf_mat[r] /= sum(row)  # normalize so counts for each state prep sum to 1
        conf_mat = np.transpose(conf_mat)  # want counts for each state prep on columns
        # Check the determinant to make sure we are not running into machine precision
        # det = np.linalg.det(conf_mat)
        # print('DETERMINANT', det)
        if np.shape(conf_mat)[0] == np.shape(conf_mat)[1]:  # square matrix
            conf_mat_inv = np.linalg.inv(conf_mat)
        else:
            conf_mat_inv = np.linalg.pinv(conf_mat)
        # print('conf mat transpose', conf_mat)
        # print('inv conf mat transpose', conf_mat_inv)
        # C_id = invM . C_noisy
        n = np.array(n, dtype=float)
        out_n = np.zeros(shape=(np.shape(n)[0], n_out_states))
        for r in range(np.shape(n)[0]):  # correct each set of measurements (rows of n)
            # print('n[r]', r, n[r])
            out_n[r] = (conf_mat_inv @ n[r].T).T
            out_n[r] *= old_sum / sum(
                n[r]
            )  # scale so total counts in each row of out_n is same as total counts in each row of n
        return np.around(out_n, decimals=5)

    """
    From https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/ignis/measurement_error_mitigation.ipynb
    """
    # def correct_readout_err(self, n, n_conf, verbose=False, method='SLSQP'):

    #     n= np.array(n ,dtype=float)
    #     conf_mat = np.array(n_conf, dtype=float)

    #     # normalize the conf_mat
    #     for r, row in enumerate(conf_mat):
    #         conf_mat[r] /= sum(row)

    #     # normalize the counts
    #     if len(n.shape) == 1: n /= sum(n)
    #     else:
    #         for r, row in enumerate(n):
    #             n[r] /= sum(row)

    #     # transpose the shots so that each measurement is a column
    #     n = np.transpose(n)

    #     # define the objective function
    #     def objective(x_flat, n, conf_mat, shape):
    #         x = np.reshape(x_flat, shape)
    #         return np.linalg.norm(conf_mat @ x - n)
    #     # define the constraints
    #     def constraint_norm(x_flat, shape):
    #         x = np.reshape(x_flat, shape)
    #         return np.sum(x, axis=0) - 1 # constrain to ensure that each column sums to 1

    #     def constraint_norm_lower(x_flat, shape, tol):
    #         x = np.reshape(x_flat, shape)
    #         return np.sum(x, axis=0) - (1 - tol) # constrain to ensure that each column sums to 1

    #     def constraint_norm_upper(x_flat, shape, tol):
    #         x = np.reshape(x_flat, shape)
    #         return (1 + tol) - np.sum(x, axis=0) # constrain to ensure that each column sums to 1

    #     # constrain to ensure that each element is in [0,1]
    #     def constraint_zero(x_flat):
    #         return x_flat

    #     # define the initial guess
    #     x0_flat = n.flatten()
    #     shape = np.shape(n)

    #     if method == 'SLSQP':
    #         cons = [{'type': 'eq', 'fun': constraint_norm, 'args': (shape,)},
    #                 {'type': 'ineq', 'fun': constraint_zero}]
    #     elif method == 'COBYLA':
    #         cons = [{'type': 'ineq', 'fun': constraint_norm_lower, 'args': (shape, 1e-10)}]
    #         cons += [{'type': 'ineq', 'fun': constraint_norm_upper, 'args': (shape, 1e-10)}]
    #         cons += [{'type': 'ineq', 'fun': constraint_zero}]

    #     # minimize the objective function

    #     result = minimize(objective, x0_flat, args=(n, conf_mat, shape),
    #                       constraints=cons, method=method, options={'ftol': 1e-20, 'disp': False}, tol=1e-20)
    #     n_corrected = result.x.reshape(shape)

    #     # if result.fun > 1e-1:
    #     #     print('Optimization did not converge')
    #     #     print('result', result)

    #     if result.success is False:
    #         print('Optimization did not converge')

    #     if verbose:
    #         print('result', result)

    #     return n_corrected.T

    def correct_readout_err(self, n, n_conf, verbose=False, method="SLSQP"):

        n = np.array(n, dtype=float)
        conf_mat = np.array(n_conf, dtype=float)

        # normalize the conf_mat
        for r, row in enumerate(conf_mat):
            conf_mat[r] /= sum(row)

        # normalize the counts
        if len(n.shape) == 1:
            n /= sum(n)
        else:
            for r, row in enumerate(n):
                n[r] /= sum(row)

        # define the objective function
        def objective(x_flat, n, conf_mat):
            x_flat = x_flat.flatten()
            x_flat = np.reshape(x_flat, (len(x_flat), 1))
            n = n.flatten()
            n = np.reshape(n, (len(n), 1))
            # print('conf_mat', conf_mat.T.shape)
            # print('x_flat', x_flat.shape)
            # print('n', n.shape)

            return np.linalg.norm(conf_mat.T @ x_flat - n)

        # define the constraints
        def constraint_norm(x_flat):
            return np.sum(x_flat, axis=0) - 1  # constrain to ensure that each column sums to 1

        def constraint_norm_lower(x_flat, tol):
            return np.sum(x_flat, axis=0) - (1 - tol)  # constrain to ensure that each column sums to 1

        def constraint_norm_upper(x_flat, tol):
            return (1 + tol) - np.sum(x_flat, axis=0)  # constrain to ensure that each column sums to 1

        # constrain to ensure that each element is in [0,1]
        def constraint_zero(x_flat):
            return x_flat

        if len(n.shape) == 1:
            n = np.reshape(n, (1, len(n)))
        conf_mat_shape = np.shape(conf_mat)
        shape_n = np.shape(n)
        out_n = np.zeros((shape_n[0], conf_mat_shape[0]))
        for r, n_row in enumerate(n):
            # transpose the shots so that each measurement is a column
            n_row = np.transpose(n_row)

            # define the initial guess
            # x0_flat = n_row.flatten()
            x0_flat = np.zeros_like(conf_mat[:, 0])

            if method == "SLSQP":
                cons = [{"type": "eq", "fun": constraint_norm}, {"type": "ineq", "fun": constraint_zero}]
            elif method == "COBYLA":
                cons = [{"type": "ineq", "fun": constraint_norm_lower, "args": (1e-10)}]
                cons += [{"type": "ineq", "fun": constraint_norm_upper, "args": (1e-10)}]
                cons += [{"type": "ineq", "fun": constraint_zero}]

            # minimize the objective function
            result = minimize(
                objective,
                x0_flat,
                args=(n_row, conf_mat),
                constraints=cons,
                method=method,
                options={"ftol": 1e-20, "disp": False},
                tol=1e-20,
            )
            n_corrected = result.x.flatten()
            out_n[r] = n_corrected

        if result.success is False:
            print("Optimization did not converge")

        if verbose:
            print("result", result)

        return out_n
        # return n_corrected.T

    def fix_neg_counts_legacy(self, counts):
        counts = np.array(counts)
        assert len(counts.shape) == 2  # 2d array

        for i_n, n in enumerate(counts):
            orig_sum = sum(n)
            while len(n[n < 0]) > 0:  # repeat while still has neg counts
                # print(i_n, n)
                assert orig_sum > 0, "Negative sum of counts"
                most_neg_ind = np.argmin(n)
                n += abs(n[most_neg_ind]) / (len(n) - 1)
                n[most_neg_ind] = 0
            n *= orig_sum / sum(n)
        return counts

    # =========================== Cholesky-esque decomposition ====================================== #
    """
    Use MLE to map unphysical initial rho to a physical density matrix 
    """

    def diag_indices_k(self, a, k):
        rows, cols = np.diag_indices_from(a)
        if k < 0:
            return rows[-k:], cols[:k]
        elif k > 0:
            return rows[:-k], cols[k:]
        else:
            return rows, cols

    def T_flat2mat(self, t):
        d = int(np.sqrt(len(t)))
        t_complex = np.zeros(int(d + (len(t) - d) / 2), dtype=complex)
        t_complex[:d] = t[:d]
        for i in range(d, len(t_complex)):
            t_complex[i] = t[d + (i - d) * 2] + 1j * t[d + (i - d) * 2 + 1]
        T_mat = np.zeros(shape=(d, d), dtype=complex)
        start_i = 0
        for k in range(d):
            T_mat[self.diag_indices_k(T_mat, -k)] = t_complex[start_i : start_i + (d - k)]
            start_i += d - k
        return np.array(T_mat)

    def t_from_rho(self, rho):
        """
        Cholesky-Banachiewicz algorithm - valid for positive definite matrices, so need to ensure no divide by 0 errors. I think this actually works better than the James et al method eq. 4.6 (https://journals.aps.org/pra/pdf/10.1103/PhysRevA.64.052312)?
        """
        t = []
        T = np.zeros(shape=np.shape(rho))
        d = np.shape(rho)[0]
        for i in range(d):
            for j in range(i + 1):
                sum = 0
                for k in range(j):
                    sum += T[i, k] * T[j, k]
                if i == j:
                    T[i, i] = np.sqrt(rho[i, i] - sum)
                else:
                    Tjj = T[j, j]
                    if Tjj == 0:
                        T[i, j] = 0
                    else:
                        T[i, j] = (rho[i, j] - sum) / Tjj

        t = np.diagonal(T)
        for k in range(1, d):
            t_complex = np.diag(T, k=-k)
            for t_i in t_complex:
                t = np.append(t, [np.real(t_i), np.imag(t_i)])
        return np.real(t)

    def rho_from_t(self, t):
        T = self.T_flat2mat(t)
        rho = T @ T.conj().T
        return rho / np.trace(rho)

    # =========================== ZZ correction functions ====================================== #

    def state_name_to_state_ind(self, state_name_ge, cutoffs):
        """
        Given state name in terms of g/e, return the state index in the full Hamiltonian basis
        """
        assert len(state_name_ge) == len(cutoffs)
        state_ind = 0
        state_numeric = ge_label_to_numeric_str(state_name_ge)
        nq_hamiltonian = len(cutoffs)
        for i_q in range(nq_hamiltonian):
            state_numeric_q = int(state_numeric[i_q])
            if i_q + 1 == nq_hamiltonian:
                state_ind += state_numeric_q
            else:
                state_ind += state_numeric_q * np.prod(cutoffs[i_q + 1 :])
        return state_ind

    def get_evol_mats(
        self,
        tomo_qubits,
        qfreqs,
        alphas,
        ZZs,
        pulse_dict,
        ZZ_qubit=None,
        soccfg=None,
        dt=0.01,
        debug=False,
        cutoffs=None,
    ):
        """
        If ZZ_qubit is not None, does all the simulations adding another qubit to the Hamiltonian, but not doing tomo on that qubit; uses last index of qfreqs, alphas, ZZs to refer to the ZZ qubit

        """

        assert len(pulse_dict.items()) == len(self.meas_order)
        evol_mats = dict()
        if cutoffs is None:
            cutoffs = [2] * (self.nb_qubits + (1 if ZZ_qubit is not None else 0))
        if ZZ_qubit is not None:
            assert len(tomo_qubits) == self.nb_qubits
            assert ZZ_qubit not in tomo_qubits
            assert len(qfreqs) == self.nb_qubits + 1
            assert len(alphas) == self.nb_qubits + 1
            assert ZZs.shape == (self.nb_qubits + 1, self.nb_qubits + 1)
            assert len(cutoffs) == self.nb_qubits + 1
        device = QSwitch(qubit_freqs=qfreqs, alphas=alphas, ZZs=ZZs, cutoffs=cutoffs, useZZs=True)

        # Get the pi/2 pulse lengths for all tomo qubits
        pulse_cfgs_XXX = pulse_dict["X" * self.nb_qubits]
        pi2_lens = [0] * self.nb_qubits
        pi2_types = [None] * self.nb_qubits
        if self.rfsoc_config is None:
            assert soccfg is not None
        for pulse_name, pulse_cfg in pulse_cfgs_XXX.items():
            if pulse_cfg["flag"] != "ZZcorrection":
                continue
            drive_qubit = int(pulse_cfg["name"][-1])  # get the qubit number
            assert drive_qubit in tomo_qubits

            # convert drive qubit out of n to drive qubit in an nQ Hamiltonian
            drive_qubit = tomo_qubits.index(drive_qubit)

            pi2_types[drive_qubit] = pulse_cfg["type"]
            pulse_type = pi2_types[drive_qubit]
            if pulse_type == "gauss":
                pulse_len = pulse_cfg["sigma"]
            elif pulse_type == "const":
                pulse_len = pulse_cfg["length"]
            elif pulse_type == "IQpulse":
                pass

            if soccfg is not None:
                self.rfsoc_config = soccfg

            if pulse_type != "IQpulse":
                pi2_lens[drive_qubit] = self.rfsoc_config.cycles2us(pulse_len, gen_ch=pulse_cfg["ch"]) * 1e3
            else:
                pi2_lens[drive_qubit] = pulse_cfg["times_us"][-1]
        assert 0 not in pi2_lens

        for basis, pulse_cfgs in tqdm(pulse_dict.items()):
            if debug:
                print(basis)
            seq = PulseSequence()
            for drive_qubit in range(self.nb_qubits):  # drive qubit is the index out of [tomo_qubits]
                # For Z basis measurement, do not play a pulse
                if basis[drive_qubit] == "Z":
                    continue

                # Find the appropriate pulse cfg for the pulse within the experiment on this basis
                for pulse_name, pulse_cfg in pulse_cfgs.items():
                    # print('flag', pulse_cfg['flag'])
                    if pulse_cfg["flag"] != "ZZcorrection":
                        continue
                    print(pulse_name)
                    pulse_qubit = int(pulse_cfg["name"][-1])  # get the qubit number
                    assert pulse_qubit in tomo_qubits
                    # convert drive qubit out of 4 to drive qubit in an nQ Hamiltonian, proceed if this is the drive qubit we want
                    if tomo_qubits.index(pulse_qubit) != drive_qubit:
                        continue

                    ch = pulse_cfg["ch"]
                    freq = pulse_cfg["freq_MHz"]
                    # freq += 4000 if freq < 1000 else 0 # qubit drive has mixer freq
                    wd = 2 * np.pi * freq * 1e-3
                    # phase = pulse_cfg['phase_deg'] * np.pi/180
                    phase = phase_to_other_drive(
                        pulse_cfg["phase_deg"] * np.pi / 180
                    )  # convert from the sin wavefunction used in definition of rfsoc drive to the cos wavefunction used for simulations
                    pulse_type = pulse_cfg["type"]

                    start_state = "g" * self.nb_qubits
                    if ZZ_qubit is not None:
                        start_state += "e"
                    new_state = start_state[:drive_qubit] + "e" + start_state[drive_qubit + 1 :]

                    if pulse_cfg["type"] != "IQpulse":
                        pi2_sigma = pi2_lens[drive_qubit]  # ns
                        assert pi2_sigma > 1  # 1 ns

                        # Account for qick having a different definition of sigma in the gaussian
                        pi2_sigma /= np.sqrt(2)
                        sigma_n = 4 * np.sqrt(2)

                        # Figure out whether we are dividing the length or not for the pi/2 pulse and set the flags appropriately
                        # Default: assume pi/2 length is pi pulse length/2

                        amp = amp_eff(pi2_sigma * 2, sigma_n=sigma_n)  # This is the amplitude for a pi pulse
                        pihalf = True  # Determines whether the simulation will play the gaussian for the full Tpi (False) or half Tpi (True)

                        if "X" in pulse_cfg["name"] or "Y" in pulse_cfg["name"]:
                            if "half" not in pulse_cfg["name"]:
                                # Dividing gain instead of length in the experiment pi/2 pulse
                                amp /= 2
                                pihalf = False

                        # print('WARNING OVERRIDING THE PI/2 DETERMINATION')
                        # amp /= 2
                        # pihalf = False
                        # pi2_sigma *= 2

                        if debug:
                            print(
                                "add pi/2 pulse on qubit",
                                tomo_qubits[drive_qubit],
                                "with freq",
                                freq,
                                "length",
                                pi2_sigma,
                                "amp",
                                amp,
                            )

                        device.add_precise_pi_pulse(
                            seq,
                            start_state,
                            new_state,
                            amp=amp,
                            pihalf=pihalf,
                            drive_qubit=drive_qubit,
                            wd=wd,
                            phase=phase,
                            type=pulse_type,
                            t_pulse=pi2_sigma,
                            sigma_n=sigma_n,
                        )
                    else:
                        I_ghz_vs_ns = np.array(pulse_cfg["I_mhz_vs_us"]) * 1e-3
                        Q_ghz_vs_ns = np.array(pulse_cfg["Q_mhz_vs_us"]) * 1e-3
                        times_ns = np.array(pulse_cfg["times_us"]) * 1e3

                        if debug:
                            print(
                                "add (robust) pi/2 pulse on qubit",
                                tomo_qubits[drive_qubit],
                                "with freq",
                                freq,
                            )

                        if debug:
                            plt.figure()
                            plt.title(f"Q{tomo_qubits[drive_qubit]}")
                            plt.plot(times_ns, I_ghz_vs_ns, label="I")
                            plt.plot(times_ns, Q_ghz_vs_ns, label="Q")
                            plt.ylabel("Drive Amplitude (GHz)")
                            plt.xlabel("Time (ns)")
                            plt.legend()
                            plt.show()

                        seq.pulse_IQ(
                            wd=wd,
                            amp=1.0,
                            pulse_levels=[start_state, new_state],
                            I_values=I_ghz_vs_ns,
                            Q_values=Q_ghz_vs_ns,
                            times=times_ns,
                            drive_qubit=drive_qubit,
                            phase=phase,
                        )

                    # # print('all params')
                    # print('fd', wd/2/np.pi)
                    # print('pulse length', pi2_sigma)
                    # print('amp', amp)
                    # print('phase rad', phase)
                    # print('drive q', drive_qubit)

            total_length = seq.get_end_time()
            if total_length != 0:
                nsteps = int(total_length // dt + 1)
                times = np.linspace(0, total_length, num=nsteps)

                # ====== PLOT PULSE SEQUENCE ====== #
                if debug:
                    envelope_seq = seq.get_envelope_seq()
                    pulse_amps = seq.get_pulse_amps()
                    pulse_freqs = seq.get_pulse_freqs()
                    pulse_lens = seq.get_pulse_lengths()
                    drive_funcs = seq.get_pulse_seq()
                    envelope_v_times = []
                    for i in range(len(envelope_seq)):
                        envelope_v_time = [drive_funcs[i](t) * 1e3 for t in times]
                        envelope_v_times.append(envelope_v_time)
                        plt.plot(times, envelope_v_time)
                    plt.xlabel("Time (ns)")
                    plt.ylabel("Drive Amplitude (MHz)")
                    plt.title("Pulse Sequence")
                    plt.show()

            evol_mats.update({basis: []})

            # EVOLVE THE INITIAL STATES
            for init_state in self.calib_order:
                # psi0 = qt.ket(str(state0) + str(state1) + str(state2))
                if ZZ_qubit is not None:
                    init_state += "e"
                if debug:
                    print("init state", init_state)
                psi0 = device.state(init_state)
                if total_length == 0:
                    evol_ket = psi0
                else:
                    evol_ket_all_times = device.evolve(
                        psi0, seq, times, nsteps=100000, use_str_solve=False, progress=False
                    )
                    evol_ket = device.evolve_unrotate(
                        times=[times[-1]], result=[evol_ket_all_times[-1]], progress=False
                    )[-1]

                # Extract state indices for 2 level system
                states_inds = []
                for psi_name in self.calib_order:
                    if ZZ_qubit is not None:
                        psi_name += "e"
                    states_inds.append(self.state_name_to_state_ind(psi_name, cutoffs))
                if debug:
                    print("truncating to states inds", states_inds)

                _evol_ket = evol_ket.full()
                _evol_ket = _evol_ket[states_inds, :]
                idnq = qt.tensor(*[qt.basis(2, 0)] * self.nb_qubits)
                evol_ket = qt.Qobj(_evol_ket, dims=idnq.dims).unit()
                evol_mats[basis].append(evol_ket.unit().full())

                if not debug:
                    continue
                print("evolution ket result")
                print(evol_ket.full())

                if total_length != 0:
                    # ====== PLOT STATE EVOLUTION ====== #
                    # print('from', str(state0) + str(state1) + str(state2))
                    print("from", init_state)
                    ref_states = self.calib_order
                    fig, ax = plt.subplots()
                    for ref_state in ref_states:
                        if ZZ_qubit is not None:
                            ref_state += "e"
                        state = device.state(ref_state)
                        probs = [np.abs(state.overlap(evol_ket_all_times[t])) ** 2 for t in range(len(times))]
                        print(ref_state, "probabilty", probs[-1])
                        ax.plot(times, probs, label=f"$|{ref_state}\\rangle_D$")

                    ax.legend(ncol=2)
                    ax.set_ylim(-0.05, 1.05)
                    ax.set_xlabel("Time (ns)")
                    ax.set_ylabel("Probability")
                    # ax.grid(linewidth=0.3)
                    # plt.title(f'{basis} start from {str(state0)+str(state1)+str(state2)}')
                    ax.set_title(f"{basis} start from {init_state}")
                    # plt.show()

            evol_mats[basis] = np.hstack(evol_mats[basis])

            if debug:
                print()

        return evol_mats

    def get_evol_mats_from_yaml(
        self,
        tomo_qubits,
        yaml_cfg,
        pulse_dict,
        ZZ_qubit=None,
        cutoffs=None,
        soccfg=None,
        dt=0.01,
        debug=False,
        evol_mats_path=None,
        evol_mats_filename=None,
    ):
        f_ge = np.array(
            [f + (4000 if f < 1000 else 0) for f in np.reshape(yaml_cfg.device.qubit.f_ge, (4, 4)).diagonal()]
        )  # MHz
        f_ef = np.array(
            [f + (4000 if f < 1000 else 0) for f in np.reshape(yaml_cfg.device.qubit.f_ef, (4, 4)).diagonal()]
        )  # Mhz

        hamiltonian_qubits = np.copy(tomo_qubits)
        if ZZ_qubit is not None:
            hamiltonian_qubits = np.append(tomo_qubits, ZZ_qubit)
        print("TOMO QUBITS", tomo_qubits)
        print("HAMILTONIAN QUBITS", hamiltonian_qubits)

        ZZs_4q = np.zeros((4, 4))
        reshaped_f_ge = np.reshape(yaml_cfg.device.qubit.f_ge, (4, 4))
        for row in range(4):
            ZZs_4q[row, :] = reshaped_f_ge[row, :] - reshaped_f_ge[row, row]

        f_ge = np.array([f_ge[q] for q in hamiltonian_qubits])
        f_ef = np.array([f_ef[q] for q in hamiltonian_qubits])
        alphas = f_ef - f_ge  # MHz
        ZZs = np.zeros(shape=(len(hamiltonian_qubits), len(hamiltonian_qubits)))
        ZZs = ZZs_4q[hamiltonian_qubits][:, hamiltonian_qubits]

        print("qubit freqs", f_ge)
        print("alphas", alphas)
        print("ZZs (MHz)", ZZs)
        # print(pulse_dict)

        if evol_mats_path is None:
            evol_mats_path = "S:\\QRAM\\qram_4QR2\\evol_mats"
            # evol_mats_path = os.path.join(os.getcwd(), "evol_mats")

        if evol_mats_filename is None:
            evol_mats_filename = f"evol_mats_"
            for q in tomo_qubits:
                evol_mats_filename += f"{q}"
            if ZZ_qubit is not None:
                evol_mats_filename += f"_ZZ{ZZ_qubit}"
            evol_mats_filename += ".npz"
        print("Will save to filename", evol_mats_filename)

        evol_mats = self.get_evol_mats(
            tomo_qubits=tomo_qubits,
            qfreqs=f_ge * 1e-3,
            alphas=alphas * 1e-3,
            ZZs=ZZs * 1e-3,
            ZZ_qubit=ZZ_qubit,
            cutoffs=cutoffs,
            dt=dt,
            pulse_dict=pulse_dict,
            soccfg=soccfg,
            debug=debug,
        )

        evol_mats_file_path = os.path.join(evol_mats_path, evol_mats_filename)
        np.savez(evol_mats_file_path, **evol_mats)
        print(f"Saved evol mats to file {evol_mats_file_path}")

        return evol_mats

    def rho_from_counts_ZZcorrection(self, n_tomo, evol_mats):
        evol_paulis = []
        evol_pauli_measurements = []

        # loop over all possible 3 qubit pauli matrices
        pauli_range = 4**self.nb_qubits
        state_base4 = [np.base_repr(i, base=4) for i in range(pauli_range)]
        state_base4 = [state.zfill(self.nb_qubits) for state in state_base4]

        for state in state_base4:
            idx = ""
            s_vec = []
            basis = ""
            for qb in range(self.nb_qubits):
                _s = self.i2s(int(state[qb]))
                idx += _s
                if _s == "I":
                    _s = "Z"
                basis += _s

                # convert everything but I to Z here, because we are rotating from the Z axis
                if int(state[qb]) != self.s2i("I"):
                    s_vec.append(self.s2i("Z"))
                else:
                    s_vec.append(int(state[qb]))
            o = self.order(idx)

            evol_mat = evol_mats[basis]  # evol_mat sends each psi_ij to the evolved ket
            evol_paulis.append(evol_mat.conj().T @ qt.tensor([self.pauli(s) for s in s_vec]).full() @ evol_mat)
            evol_pauli_measurements.append(self.Tij(n_tomo[o : o + 1], state))

        norm_mats, norm_measurements = self.orthonormalization(evol_paulis, evol_pauli_measurements)

        qb_list = [qt.qeye(2)] * self.nb_qubits
        rho = 0 * qt.tensor(*qb_list).full()

        for o in range(len(norm_measurements)):
            rho += norm_measurements[o] * norm_mats[o] / len(norm_mats)
        return rho

    # =========================== MLE functions =========================== #

    # =========================== MLE with optimization =========================== #
    def run_MLE_optimizer(self, n_tomo, rho_guess=None, rho_id=None, method="L-BFGS-B", decimals=10, maxiter=10000000):
        # methods = 'Nelder-Mead' 'Powell' 'CG' 'BFGS' 'Newton-CG' 'L-BFGS-B' 'TNC' 'COBYLA' 'SLSQP' 'dogleg' 'trust-ncg'
        if rho_guess is None:
            rho_guess = self.rho_from_counts(n_tomo)
        # psi_basis = get_psi_basis(qubits)
        def likelihood(t_arr, n_tomo):
            n_tot = np.sum(n_tomo[0, :])
            rho = self.rho_from_t(t_arr)
            rho = rho / np.trace(rho)
            val = 0
            for psi, n_val in zip(self.psi_basis_flat, n_tomo.flatten()):
                psi = psi.full()
                proj = (psi.conj().T @ rho @ psi)[0][0]
                if proj != 0:
                    val += np.real(abs(n_tot * proj - n_val) ** 2 / (n_tot * proj))
            # return np.log(val)
            return val

        optvals = sp.optimize.minimize(
            likelihood,
            self.t_from_rho(rho_guess),
            args=(n_tomo),
            method=method,
            options={"maxiter": maxiter, "maxfun": maxiter},
        )
        print(f"Convergence: {optvals.success}")
        return np.around(self.rho_from_t(optvals.x), decimals=decimals)

    def run_MLE_ZZ_optimizer(
        self, n_tomo, evol_mats, rho_guess=None, rho_id=None, method="L-BFGS-B", decimals=10, maxiter=1000000
    ):
        # methods = 'Nelder-Mead' 'Powell' 'CG' 'BFGS' 'Newton-CG' 'L-BFGS-B' 'TNC' 'COBYLA' 'SLSQP' 'dogleg' 'trust-ncg'
        if rho_guess is None:
            rho_guess = self.rho_from_counts(n_tomo)
        n_tot = np.sum(n_tomo[0, :])
        # psi_basis = get_psi_basis(qubits)
        def likelihood(t_arr, n_tomo):
            rho = self.rho_from_t(t_arr)
            rho = rho / np.trace(rho)
            val = 0
            evals, evecs = np.linalg.eig(rho)
            evals = np.real(evals)
            n_evol = []
            for basis in self.meas_order:
                evol_mat = evol_mats[basis]  # evol_mat sends each psi_ij to the evolved ket
                n_basis_th = np.zeros(2**self.nb_qubits)
                for i in range(2**self.nb_qubits):
                    if evals[i] < 1e-12:
                        continue
                    evec_evol = evol_mat @ evecs.T[i]
                    n_basis_th += evals[i] * n_tot * abs(evec_evol) ** 2  # add array of probabilities weighted by eval
                n_evol.append(n_basis_th)
            n_evol = np.array(n_evol)
            for n_th, n_expt in zip(n_evol.flatten(), n_tomo.flatten()):
                if n_th != 0:
                    val += abs((n_th - n_expt) ** 2 / n_th)
            return val

        optvals = sp.optimize.minimize(
            likelihood,
            self.t_from_rho(rho_guess),
            args=(n_tomo),
            method=method,
            options={"maxiter": maxiter, "maxfun": maxiter},
        )
        print(f"Convergence: {optvals.success}")
        return np.around(self.rho_from_t(optvals.x), decimals=decimals)

    # =========================== Analytical MLE with ZZ =========================== #
    """
    How to construct the expectation values in the evolved bases?

    If we have a perfect rotation (no ZZ): let's say we want to measure sigmax, the evals are +/-1.
    During the measurement, we apply a rotation R to the state, which takes us to the sigmaz basis.
    Next, we perform a projective measurement onto the Z axis.
    This gives us full information about the state as if we had performed the measurement in the X axis,
    but this is only true because the act of measuring projects us onto the same basis that we are getting information in.

    In the ZZ correction case: let's say we are again doing the "$\sigma_x$" measurement. But now, we have a bad rotation $R'$, which takes $|X_+\rangle$ to $R'|X_+\rangle=|Z_+ + \phi\rangle $. If we now project this measurement against the $Z$ axis, what is the basis we have now measured? One might guess that it is in $X'=X+\phi$. However, unlike in the perfect rotation case, because the rotation $R'$ (let's say this took us through total angle $\theta=90+\phi$) does not take us directly from $|X_+\rangle$ to $|Z_+\rangle$, doing the projective measurement along the $Z$ axis (aka getting the g/e probabilities in the $Z$ axis) does NOT translate to having that same expectation value in the $X'$ axis. This can be seen in the extreme case where the ZZ is so bad that we do not perform any rotation at all (i.e. $\phi=-90$): if we start in the $|0\rangle$ state and tried to measure in the $X$ basis, we should expect 50/50 g/e counts. However, we now perform the $R'$ rotation (which does nothing) to try measure along the $X$ axis and project onto the $Z$ axis and we get probabilities of 100/0 for g/e. But if we then try to extrapolate these counts to actually be in the basis of $X'$, $X'=X-90=-Z$. So we would have said, this means we measured 100/0 g/e in the $-Z$ basis, which is flipped from what we actually did! 

    Instead, what the measurement along the $Z+\phi$ axis tells us is NOT the expectation value of the operator $\tilde{\sigma}_x=R' \sigma_x (R')^{-1}$ (the component of $\rho$ in the $X+\phi$ basis)  as might be most intuitive, but rather the expectation value of the operator $(R')^{-1} \sigma_z R'$ (the component of $\rho$ in the $Z-\theta$ basis, which is NOT the same as $X+\phi$). In other words, the projection of $|\psi + \theta\rangle$ onto $|Z_+\rangle$ is the same as the projection of $|\psi\rangle$ onto $|Z_+ - \theta\rangle$. So we should take the counts that we measure in each of the tilde bases and convert it to an expectation value using +/- 1 for measuring g/e (long story short, same Tij from before still works for the expectation value), then use the set of "measured operators" as $(R')^{-1} \sigma_z R'$ to reconstruct rho_guess. 

    You can check that this setup works for the example above: start in $\psi=|0\rangle$, do a $R'$ that does nothing, so we end up measuring 100/0 for g/e along the $Z$ axis. We interpret this as the counts in the $Z-\theta=Z-(90+\phi)=Z-90-(-90)=Z$ axis, which is correct!  
    This method maintains the assumptions of the paper:
    - Basis remains orthonormal since that's the point of the Gram Schmidt
    - Basis remains Hermitian since we are only doing linear transformations
    - Noise remains Gaussian since we are only doing linear transformations
    """

    def orthonormalization(self, evol_paulis, evol_pauli_measurements):
        """
        modified gram-schmidt https://www.math.uci.edu/~ttrogdon/105A/html/Lecture23.html
        replace inner product (projection) with tr(sigma_i sigma_j)/dim
        """

        def inner_prod_mats(mat1, mat2):
            return np.trace(mat1.conj().T @ mat2)

        evol_paulis = np.array(evol_paulis)
        process_mats = np.copy(evol_paulis)
        norm_mats = np.zeros_like(evol_paulis)
        process_measurements = np.copy(evol_pauli_measurements)
        norm_measurements = np.zeros_like(evol_pauli_measurements)
        dim = len(evol_paulis)  # dimensions (number of basis vectors) to orthonormalize over
        for j in range(dim):
            norm_factor = np.sqrt(inner_prod_mats(process_mats[j], process_mats[j]))
            # print(j, norm_factor)
            norm_mats[j] = process_mats[j] / norm_factor
            # print(j, norm_mats[j])
            norm_measurements[j] = process_measurements[j] / norm_factor
            # subtract off the part that is parallel to the good matrix
            for k in range(j + 1, dim):
                proj_factor = inner_prod_mats(process_mats[k], norm_mats[j])
                process_mats[k] -= proj_factor * norm_mats[j]
                process_measurements[k] -= proj_factor * norm_measurements[j]
        return norm_mats * np.sqrt(dim), norm_measurements * np.sqrt(dim)

    def run_MLE_analytical(self, n_tomo, ZZ_correction=False, evol_mats=None):
        """
        Change of basis method 10.1103/PhysRevLett.108.070502
        """
        # step 0: get the mu "u" matrix
        rho_guess = self.rho_from_counts(n_tomo)
        # print('rho guess', rho_guess)
        if ZZ_correction:
            assert evol_mats is not None
            rho_guess = self.rho_from_counts_ZZcorrection(n_tomo, evol_mats)
        # print('trace', np.trace(rho_guess))

        # step 1
        u_evals, u_evecs = np.linalg.eig(
            rho_guess
        )  # evecs are normalized, evecs[:,i] is evec corresponding to evals[i]
        # sort from largest to smallest
        idx = u_evals.argsort()[::-1]
        u_evals = u_evals[idx]
        u_evecs = u_evecs[:, idx]
        # print('evals', u_evals)

        # step 2, 3
        dim = len(rho_guess[0])
        rho_evals = [0] * dim  # MLE evals
        a = 0
        stop_i = dim  # 1 indexed
        check = u_evals[stop_i - 1] + a / stop_i
        while check < 0:
            rho_evals[stop_i - 1] = 0
            a += u_evals[stop_i - 1]
            stop_i -= 1
            check = u_evals[stop_i - 1] + a / stop_i

        # step 4
        for j in range(stop_i):
            rho_evals[j] = u_evals[j] + a / stop_i
        # print('rho evals', rho_evals)

        # step 5
        rho_opt = np.zeros_like(rho_guess)
        for j in range(dim):
            u_evec = np.array([u_evecs[:, j]]).T
            # print(u_evec)
            rho_opt += rho_evals[j] * (u_evec @ u_evec.conj().T)

        return rho_opt

    # ====================================================== #
    """
    Most general run_MLE
    """

    def run_MLE(
        self,
        n_tomo,
        analytical=True,
        ZZ_correction=False,
        evol_mats=None,
        rho_guess=None,
        rho_id=None,
        method="L-BFGS-B",
        maxiter=1000000,
    ):
        if analytical:
            return self.run_MLE_analytical(n_tomo, ZZ_correction=ZZ_correction, evol_mats=evol_mats)
        else:
            if ZZ_correction:
                return self.run_MLE_ZZ_optimizer(
                    n_tomo, evol_mats, rho_guess=rho_guess, rho_id=rho_id, method=method, maxiter=maxiter
                )
            else:
                return self.run_MLE_optimizer(
                    n_tomo, rho_guess=rho_guess, rho_id=rho_id, method=method, maxiter=maxiter
                )

    # =========================== PLOTTING FUNCTIONS =========================== #
    def show_mat_2d(
        self, mat, ax, title, labels, cmax=1, show_cbar=True, show_xticks=True, show_yticks=True, show=True
    ):
        """
        Plot an arbitrary 2D matrix with labels
        """
        plt.sca(ax)
        plt.title(title, fontsize=18)
        plt.imshow(np.real(mat), cmap="RdBu")
        # hinton(np.real(mat), xlabels=labels, ylabels=labels)
        if show_xticks:
            plt.xticks(np.arange(len(mat)), labels, fontsize=14, rotation=45)
        else:
            plt.xticks([])
        if show_yticks:
            plt.yticks(np.arange(len(mat)), labels, fontsize=14)
        else:
            plt.yticks([])
        # Loop over data dimensions and create text annotations.
        for ii in range(len(mat)):
            for jj in range(len(mat)):
                plt.text(
                    ii,
                    jj,
                    round(mat[jj, ii], 3),
                    ha="center",
                    va="center",
                    color="w",
                    size=9 + 6 / self.nb_qubits,
                    rotation=45,
                )
        if show_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(cax=cax, ticks=[-cmax, 0, cmax])
            cbar.ax.tick_params(labelsize=14)
        plt.clim(vmin=-cmax, vmax=cmax)
        if show:
            plt.tight_layout()
            plt.show()

    def show_plot_rho_2d(self, rho_test, rho_id=None, title=None, cmax=None, savetitle=None, size=None):
        """
        Plot real and imag parts of rho, optionally also with a comparison ideal rho
        """
        # if savetitle is None:
        #     plt.style.use("default")
        # else:
        #     plt.style.use("dark_background")

        labels = self.calib_order_numeric

        if size is None:
            if rho_id is None:
                size = (9.5, 5)
            else:
                size = (9.5, 10)

        if rho_id is None:
            fig = plt.figure(figsize=size)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
        else:
            fig = plt.figure(figsize=size)
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)
        plt.suptitle(title, fontsize=18)
        if cmax is None:
            cmax = np.max(np.abs(np.array([np.real(rho_test), np.imag(rho_test), np.real(rho_id), np.imag(rho_id)])))
        self.show_mat_2d(
            np.real(rho_test),
            ax=ax1,
            title="Re[$\\rho_{MLE}$]",
            labels=labels,
            cmax=cmax,
            show_cbar=False,
            show_yticks=True,
            show=False,
        )
        self.show_mat_2d(
            np.imag(rho_test),
            ax=ax2,
            title="Im[$\\rho_{MLE}$]",
            labels=labels,
            cmax=cmax,
            show_cbar=True,
            show_yticks=False,
            show=False,
        )
        if rho_id is not None:
            self.show_mat_2d(
                np.real(rho_id),
                ax=ax3,
                title="Re[$\\rho_{Ideal}$]",
                labels=labels,
                cmax=cmax,
                show_cbar=False,
                show_yticks=True,
                show=False,
            )
            self.show_mat_2d(
                np.imag(rho_id),
                ax=ax4,
                title="Im[$\\rho_{Ideal}$]",
                labels=labels,
                cmax=cmax,
                show_cbar=True,
                show_yticks=False,
                show=False,
            )
        plt.tight_layout()

        if savetitle is not None:
            plt.savefig(savetitle, format="svg", bbox_inches="tight", transparent=True)
        plt.show()

    def show_plot_rho_3d(
        self, rho_test, rho_id=None, title="", zmin=None, zmax=None, width=0.75, elev=30, azim=-20, savetitle=None
    ):
        if savetitle is None:
            plt.style.use("default")
        else:
            plt.style.use("dark_background")
        fig = plt.figure(figsize=(15, 7))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")

        labels = self.calib_order_numeric

        _xx, _yy = np.meshgrid(np.arange(4), np.arange(4))
        x, y = _xx.ravel(), _yy.ravel()
        if zmax is None:
            zmax = np.max(np.array([np.real(rho_test), np.imag(rho_test), np.real(rho_id), np.imag(rho_id)]))
        if zmin is None:
            zmin = np.min(
                (0, np.min(np.array([np.real(rho_test), np.imag(rho_test), np.real(rho_id), np.imag(rho_id)])))
            )

        ax1.view_init(elev=elev, azim=azim)
        ax1.set_xticks(np.arange(4), minor=False)
        ax1.set_xticklabels(labels, fontdict=None, minor=False, fontsize=16)
        ax1.set_yticks(np.arange(1, 5, 1), minor=False)
        ax1.set_yticklabels(labels, fontdict=None, minor=False, fontsize=16)
        for t in ax1.zaxis.get_major_ticks():
            t.label.set_fontsize(16)
        ax1.bar3d(x, y, z=np.zeros((16)), dx=width, dy=width, dz=np.real(rho_id).flatten(), edgecolor="k", alpha=0)
        ax1.bar3d(
            x,
            y,
            z=np.zeros((16)),
            dx=0.95 * width,
            dy=width,
            dz=np.real(rho_test).flatten(),
            color="cornflowerblue",
            edgecolor="mediumblue",
            alpha=1.0,
        )
        ax1.set_zlim(zmin, zmax)
        ax1.set_title("Re[$\\rho$]", fontsize=20)

        ax2.view_init(elev=elev, azim=azim)
        ax2.set_xticks(np.arange(4), minor=False)
        ax2.set_xticklabels(labels, fontdict=None, minor=False, fontsize=16)
        ax2.set_yticks(np.arange(1, 5, 1), minor=False)
        ax2.set_yticklabels(labels, fontdict=None, minor=False, fontsize=16)
        for t in ax2.zaxis.get_major_ticks():
            t.label.set_fontsize(16)
        ax2.bar3d(x, y, z=np.zeros((16)), dx=width, dy=width, dz=np.imag(rho_id).flatten(), edgecolor="k", alpha=0)
        ax2.bar3d(
            x,
            y,
            z=np.zeros((16)),
            dx=0.95 * width,
            dy=width,
            dz=np.imag(rho_test).flatten(),
            color="cornflowerblue",
            edgecolor="mediumblue",
            alpha=1.0,
        )
        ax2.set_zlim(zmin, zmax)
        ax2.set_title("Im[$\\rho$]", fontsize=20)

        plt.suptitle(title, fontsize=22)
        plt.tight_layout()

        if savetitle is not None:
            plt.savefig(savetitle, format="svg", bbox_inches="tight", transparent=True)
        plt.show()

    # =========================== VIRTUAL ROTATIONS =========================== #
    def z_gate_nq(self, phis):  # expects phis in deg
        gates = []
        for phi in phis:
            gates.append(rz(np.pi / 180 * phi))
        return qt.tensor(*gates)

    def opt_virtualZ_MLE(self, rho_MLE, rho_id, phis=None, progress=True, verbose=True):  # phis in deg
        """
        The experimental density matrix from MLE may be offset from the simulated/ideal density matrix by a Z gate - due to different pulse times, ac stark shifts, etc.
        """
        phis_vec = []
        if phis is None:
            for i in range(self.nb_qubits):
                _phi_vec = np.linspace(0, 360, 100)
                phis_vec.append(_phi_vec)
        else:
            phis_vec = phis

        best_fid = 0
        best_phis = [0] * self.nb_qubits
        best_rho_MLE = rho_MLE

        # define an n_qubit dimensional zeros array
        fids_grid = np.zeros([len(phis_vec[i]) for i in range(self.nb_qubits)])
        phi_grid = np.array(np.meshgrid(*phis_vec, indexing="ij"))  # shape: (nqubits, nphi0, nphi1, ...)
        phi_grid = np.moveaxis(phi_grid, 0, -1)
        phi_grid = np.reshape(phi_grid, (np.prod(phi_grid.shape[:-1]), phi_grid.shape[-1]))
        # print(phi_grid.shape)

        for i, phi in enumerate(tqdm(phi_grid, disable=not progress)):
            idx = np.unravel_index(i, fids_grid.shape)
            # print(i, phi, idx)
            z_phi = self.z_gate_nq(phi)
            rho_MLE_rot = (z_phi * rho_MLE * z_phi.dag()).unit()
            fid = qt.fidelity(rho_MLE_rot, rho_id) ** 2
            fids_grid[idx] = fid
            if fid > best_fid:
                best_fid = fid
                best_phis = phi
                best_rho_MLE = rho_MLE_rot
        if verbose:
            print(f"Improved fidelity by (%) {(best_fid - qt.fidelity(rho_MLE, rho_id)**2)*100}")
        return best_rho_MLE, best_phis, best_fid, fids_grid

    # =========================== The Function =========================== #
    def get_rho_from_counts(
        self,
        n_tomo_raw,
        n_calib,
        method="analytical",
        correct_readout=True,
        correct_neg_counts=True,
        ZZ_correction=False,
        pulse_dict=None,
        evol_mats=None,
        tomo_qubits=None,
        rho_guess=None,
        rho_id=None,
        verbose=False,
    ):
        if ZZ_correction:
            assert evol_mats is not None

        if method == "MLE":
            analytical = False
            method = "L-BFGS-B"
            maxiter = 1000000
        elif method == "analytical":
            analytical = True
            rho_guess = None
            method = None
            maxiter = None
            rho_id = None
        else:
            assert False, "method must be either MLE or analytical"

        n_tomo_corrected = n_tomo_raw
        if correct_readout:
            n_tomo_corrected = self.correct_readout_err(n_tomo_raw, n_calib)
        # if correct_neg_counts: n_tomo_corrected = self.fix_neg_counts_legacy(n_tomo_corrected)

        if verbose:
            print("n_tomo_corrected")
            print(n_tomo_corrected)

        return self.run_MLE(
            n_tomo_corrected,
            analytical=analytical,
            ZZ_correction=ZZ_correction,
            evol_mats=evol_mats,
            rho_guess=rho_guess,
            rho_id=rho_id,
            method=method,
            maxiter=maxiter,
        )
