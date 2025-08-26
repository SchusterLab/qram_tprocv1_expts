# import logging

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize

import os
import time

from tqdm import tqdm_notebook as tqdm

expt_path = os.getcwd() + "/data"
import itertools

import qutip as qt
import scipy as sp
from qutip_qip.operations import rz
from scipy.optimize import minimize


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

# Calculating colors for best contrast in tomo plotting
# Thanks chatgpt
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))

def luminance(r, g, b):
    def linearize(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r_lin, g_lin, b_lin = map(linearize, (r, g, b))
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

def contrast_ratio(l1, l2):
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)

def colormap_hex(value, clim, cmap_name="RdBu"):
    vmin, vmax = clim
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm(value))
    return mcolors.to_hex(rgba)


def best_greyscale_for_contrast(value, clim, cmap_name="RdBu", reduce_crowded=True):
    """
    Given a matplotlib colormap and a value, return the 
    hex value for a gresyscale color with best contrast against the value on this colormap.
    If reduce_crowded, values that are close to 0 are scaled to lighter colors to avoid overcrowding.
    """
    bg_hex = colormap_hex(value, clim, cmap_name=cmap_name)
    bg_rgb = hex_to_rgb(bg_hex)
    bg_lum = luminance(*bg_rgb)

    best_contrast = 0
    best_gray = None

    for i in range(0, 256):
        gray = i / 255
        lum = luminance(gray, gray, gray)
        contrast = contrast_ratio(bg_lum, lum)
        hex_gray = "#{:02x}{:02x}{:02x}".format(i, i, i)
        if contrast > best_contrast:
            best_contrast = contrast
            best_gray = i

    if reduce_crowded:
        if abs(value) < 0.1 * (clim[1] - clim[0]):
            best_gray += (255 - best_gray) / 2
        best_gray = int(best_gray)

    hex_gray = "#{:02x}{:02x}{:02x}".format(best_gray, best_gray, best_gray)
    return hex_gray


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

    # =========================== Error Mitigation ====================================== #
    """
    From https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/ignis/measurement_error_mitigation.ipynb
    """

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
            # print("result", result)
            for r, n_row in enumerate(n):
                # transpose the shots so that each measurement is a column
                n_row = np.transpose(n_row)
                print("minimization error", objective(n_corrected, n_row, conf_mat))
        return out_n
        # return n_corrected.T

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

    # =========================== Analytical MLE with ZZ =========================== #
    def orthonormalization(self, evol_paulis, evol_pauli_measurements):
        """
        modified gram-schmidt
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
            # rho_evals[stop_i - 1] = 0
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

    # =========================== PLOTTING FUNCTIONS =========================== #
    def show_mat_2d(
        self,
        mat,
        ax,
        title,
        labels,
        cmax=1,
        show_cbar=True,
        show_xticks=True,
        show_yticks=True,
        show=True,
        tick_rotation=45,
    ):
        """
        Plot an arbitrary 2D matrix with labels
        """
        plt.sca(ax)
        plt.title(title, fontsize=18)
        plt.imshow(np.real(mat), cmap="RdBu")
        ax.tick_params(axis="both", which="minor", left=False, bottom=False, right=False, top=False)
        if show_xticks:
            plt.xticks(np.arange(len(mat)), labels, fontsize=17, rotation=tick_rotation)
        else:
            plt.xticks([])
        if show_yticks:
            plt.yticks(np.arange(len(mat)), labels, fontsize=17)
        else:
            ax.set_yticks(np.arange(len(mat)), labels=[""] * len(mat))
            ax.tick_params(axis="y", which="minor", labelleft=False, left=False)
        
        clim = (-cmax, cmax)
        # Loop over data dimensions and create text annotations.
        for ii in range(len(mat)):
            for jj in range(len(mat)):
                value = mat[jj, ii]
                text_color = best_greyscale_for_contrast(value, clim, cmap_name="RdBu")
                plt.text(
                    ii,
                    jj,
                    round(value, 2),
                    ha="center",
                    va="center",
                    color=text_color,
                    size=13 + 6 / self.nb_qubits - (1 if abs(value) < 0.1 else 0),
                    rotation=45,
                )
        if show_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(cax=cax, ticks=[-cmax, 0, cmax])
            cbar.ax.tick_params(labelsize=17)
        plt.clim(vmin=clim[0], vmax=clim[1])
        if show:
            plt.tight_layout()
            plt.show()

    def show_plot_rho_2d(
        self,
        rho_test,
        rho_id=None,
        title=None,
        cmax=None,
        state_num=True,
        savetitle=None,
        size=None,
        ideal_name="ideal",
    ):
        """
        Plot real and imag parts of rho, optionally also with a comparison ideal rho
        """

        if not state_num:
            labels = [f"$|{state}\\rangle$" for state in self.calib_order]
            tick_rotation = 90
        else:
            labels = [f"$|{state}\\rangle$" for state in self.calib_order_numeric]
            tick_rotation = 60

        if size is None:
            if rho_id is None:
                width = 10
                height = 5
            else:
                width = 10
                height = 10
            size = (width, height)

        if rho_id is None:
            fig, axes = plt.subplots(1, 2, figsize=size)
            ax1 = axes[0]
            ax2 = axes[1]
        else:
            fig, axes = plt.subplots(2, 2, gridspec_kw={"height_ratios": [1, 1]}, figsize=size)
            ax1 = axes[0, 0]
            ax2 = axes[0, 1]
            ax3 = axes[1, 0]
            ax4 = axes[1, 1]
        plt.suptitle(title, fontsize=18)
        if cmax is None:
            cmax = np.max(np.abs(np.array([np.real(rho_test), np.imag(rho_test), np.real(rho_id), np.imag(rho_id)])))
        self.show_mat_2d(
            np.real(rho_test),
            ax=ax1,
            title="Re[$\\rho_{\\mathrm{MLE}}$]",
            labels=labels,
            cmax=cmax,
            show_cbar=False,
            show_yticks=True,
            show=False,
            tick_rotation=tick_rotation,
        )
        self.show_mat_2d(
            np.imag(rho_test),
            ax=ax2,
            title="Im[$\\rho_{\\mathrm{MLE}}$]",
            labels=labels,
            cmax=cmax,
            show_cbar=True,
            show_yticks=False,
            show=False,
            tick_rotation=tick_rotation,
        )
        if rho_id is not None:
            self.show_mat_2d(
                np.real(rho_id),
                ax=ax3,
                title="Re[$\\rho_{\\mathrm{" + ideal_name + "}}$]",
                labels=labels,
                cmax=cmax,
                show_cbar=False,
                show_yticks=True,
                show=False,
                tick_rotation=tick_rotation,
            )
            self.show_mat_2d(
                np.imag(rho_id),
                ax=ax4,
                title="Im[$\\rho_{\\mathrm{" + ideal_name + "}}$]",
                labels=labels,
                cmax=cmax,
                show_cbar=True,
                show_yticks=False,
                show=False,
                tick_rotation=tick_rotation,
            )
        plt.tight_layout()

        if savetitle is not None:
            plt.savefig(savetitle, bbox_inches="tight", transparent=True)
            print("Saved", savetitle)
        plt.show()

        return fig

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

        for i, phi in enumerate(tqdm(phi_grid, disable=not progress)):
            idx = np.unravel_index(i, fids_grid.shape)
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

    # ============= High level function to get density matrix from counts ============= #
    def get_rho_from_counts(
        self,
        n_tomo_raw,
        n_calib,
        correct_readout=True,
        ZZ_correction=False,
        evol_mats=None,
        verbose=False,
    ):
        if ZZ_correction:
            assert evol_mats is not None

        n_tomo_corrected = n_tomo_raw
        if correct_readout:
            n_tomo_corrected = self.correct_readout_err(n_tomo_raw, n_calib)

        if verbose:
            print("n_tomo_corrected")
            print(n_tomo_corrected)

        return self.run_MLE_analytical(
            n_tomo_corrected,
            ZZ_correction=ZZ_correction,
            evol_mats=evol_mats,
        )
