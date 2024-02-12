# Author: Connie 2022/02/17

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from copy import deepcopy

from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

from experiments.single_qubit.single_shot_edit import hist
from experiments.clifford_averager_program import CliffordAveragerProgram, CliffordEgGfAveragerProgram
from experiments.two_qubit.length_rabi_EgGf import LengthRabiEgGfProgram
from experiments.two_qubit.twoQ_state_tomography import ErrorMitigationStateTomo2QProgram

import experiments.fitting as fitter

"""
Define matrices representing (all) Clifford gates for single
qubit in the basis of Z, X, Y, -Z, -X, -Y, indicating
where on the 6 cardinal points of the Bloch sphere the
+Z, +X, +Y axes go after each gate. Each Clifford gate
can be uniquely identified just by checking where +X and +Y
go.
"""
clifford_1q = dict()
clifford_1q['Z'] = np.matrix([[1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 1, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0]])
clifford_1q['X'] = np.matrix([[0, 0, 0, 1, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1],
                             [1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 1, 0, 0, 0]])
clifford_1q['Y'] = np.matrix([[0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 1, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1]])
clifford_1q['Z/2'] = np.matrix([[1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0]])
clifford_1q['X/2'] = np.matrix([[0, 0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 1, 0],
                             [1, 0, 0, 0, 0, 0]])
clifford_1q['Y/2'] = np.matrix([[0, 0, 0, 0, 1, 0],
                             [1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 1]])
clifford_1q['-Z/2'] = np.matrix([[1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0]])
clifford_1q['-X/2'] = np.matrix([[0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0]])
clifford_1q['-Y/2'] = np.matrix([[0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1]])
clifford_1q['I'] = np.diag([1]*6)

# Read pulse as a matrix product acting on state (meaning apply pulses in reverse order of the tuple)
two_step_pulses= [
    ('X','Z/2'), ('X/2','Z/2'), ('-X/2','Z/2'),
    ('Y','Z/2'), ('Y/2','Z/2'), ('-Y/2','Z/2'),
    ('X','Z'), ('X/2','Z'), ('-X/2','Z'),
    ('Y','Z'), ('Y/2','Z'), ('-Y/2','Z'),
    ('X','-Z/2'), ('X/2','-Z/2'), ('-X/2','-Z/2'),
    ('Y','-Z/2'), ('Y/2','-Z/2'), ('-Y/2','-Z/2'),
]
# Get rid of repeats
for pulse in two_step_pulses:
    new_mat = clifford_1q[pulse[0]] @ clifford_1q[pulse[1]]
    repeat = False
    for existing_pulse_name, existing_pulse in clifford_1q.items():
        if np.array_equal(new_mat, existing_pulse):
            # print(pulse, existing_pulse_name)
            repeat = True
    if not repeat: clifford_1q[pulse[0]+','+pulse[1]] = new_mat
clifford_1q_names = list(clifford_1q.keys())

for name, matrix in clifford_1q.items():
    z_new = np.argmax(matrix[:,0]) # +Z goes to row where col 0 is 1
    x_new = np.argmax(matrix[:,1]) # +X goes to row where col 1 is 1
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
        pulse_n_seq = (len(clifford_1q_names)*np.random.rand(rb_depth)).astype(int)
    if debug: print('pulse seq', pulse_n_seq)
    pulse_name_seq = [clifford_1q_names[n] for n in pulse_n_seq]
    psi_nz = np.matrix([[1, 0, 0, 0, 0, 0]]).transpose()
    psi_nx = np.matrix([[0, 1, 0, 0, 0, 0]]).transpose()
    for n in pulse_n_seq: # n is index in clifford_1q_names
        gates = clifford_1q_names[n].split(',')
        for gate in reversed(gates): # Apply matrices from right to left of gates
            psi_nz = clifford_1q[gate][0] @ psi_nz
            psi_nx = clifford_1q[gate][0] @ psi_nx
    psi_nz = psi_nz.flatten()
    psi_nx = psi_nx.flatten()
    if debug: print('+Z axis after seq:', psi_nz, '+X axis after seq:', psi_nx)
    for clifford in clifford_1q_names: # Get the clifford equivalent to the total seq
        if clifford_1q[clifford][1] == (np.argmax(psi_nz), np.argmax(psi_nx)):
            total_clifford = clifford
            break
    if debug: print('Total gate matrix:\n', clifford_1q[total_clifford][0])
    return pulse_name_seq, total_clifford

def interleaved_gate_sequence(rb_depth, gate_char:str, debug=False):
    """
    Generate RB gate sequence with rb_depth random gates interleaved with gate_char
    Returns the total gate list (including the interleaved gates) and the total
    Clifford gate equivalent to the total pulse sequence.
    """
    pulse_n_seq_rand = (len(clifford_1q_names)*np.random.rand(rb_depth)).astype(int)
    pulse_n_seq = []
    assert gate_char in clifford_1q_names
    n_gate_char = clifford_1q_names.index(gate_char)
    if debug: print('n gate char:', n_gate_char, clifford_1q_names[n_gate_char])
    for n_rand in pulse_n_seq_rand:
        pulse_n_seq.append(n_rand)
        pulse_n_seq.append(n_gate_char)
    return gate_sequence(len(pulse_n_seq), pulse_n_seq=pulse_n_seq, debug=debug)    

if __name__ == '__main__':
    print('Clifford gates:', clifford_1q_names)
    print('Total number Clifford gates:', len(clifford_1q_names))
    pulse_name_seq, total_clifford = gate_sequence(2, debug=True)
    print('Pulse sequence:', pulse_name_seq)
    print('Total clifford of seq:', total_clifford)
    gate_char = 'X/2'
    print()
    print('Interleaved RB with gate', gate_char)
    pulse_name_seq, total_clifford = interleaved_gate_sequence(2, gate_char=gate_char, debug=True)
    print('Pulse sequence:', pulse_name_seq)
    print('Total clifford of seq:', total_clifford)

# ===================================================================== #

class SimultaneousRBProgram(CliffordAveragerProgram):
    """
    RB program for single qubit gates
    """

    def clifford(self, qubit, pulse_name:str, extra_phase=0, inverted=False, play=False):
        """
        Convert a clifford pulse name into the function that performs the pulse.
        If inverted, play the inverse of this gate (the extra phase is added on top of the inversion)
        """
        pulse_name = pulse_name.upper()
        assert pulse_name in clifford_1q_names
        gates = pulse_name.split(',')

        # Normally gates are applied right to left, but if inverted apply them left to right
        gate_order = reversed(gates)
        if inverted:
            gate_order = gates
        for gate in gate_order:
            pulse_func = None
            if gate == 'I': continue
            if 'X' in gate: pulse_func = self.X_pulse
            elif 'Y' in gate: pulse_func = self.Y_pulse
            elif 'Z' in gate: pulse_func = self.Z_pulse
            else: assert False, 'Invalid gate'

            neg = '-' in gate
            if inverted: neg = not neg
            pulse_func(qubit, pihalf='/2' in gate, neg=neg, extra_phase=extra_phase, play=play, reload=False) # very important to not reload unless necessary to save memory on the gen
            # print(self.overall_phase[qubit])

    def __init__(self, soccfg, cfg, gate_list, qubit_list):
        # gate_list should include the total gate!
        # qubit_list should specify the qubit on which each random gate will be applied
        self.gate_list = gate_list
        self.qubit_list = qubit_list
        super().__init__(soccfg, cfg)

    def body(self):
        # Phase reset all channels
        for ch in self.gen_chs.keys():
            if self.gen_chs[ch]['mux_freqs'] is None: # doesn't work for the mux channels
                # print('resetting', ch)
                self.setup_and_pulse(ch=ch, style='const', freq=100, phase=0, gain=100, length=10, phrst=1)
            self.sync_all()
        self.sync_all(10)

        # Do all the gates given in the initialize except for the total gate, measure
        cfg=AttrDict(self.cfg)
        for i in range(len(self.gate_list) - 1):
            self.clifford(qubit=self.qubit_list[i], pulse_name=self.gate_list[i], play=True)
            self.sync_all()

        # Do the inverse by applying the total gate with pi phase
        # This is actually wrong!!! need to apply an inverse total gate for each qubit!!
        self.clifford(qubit=self.qubit_list[-1], pulse_name=self.gate_list[-1], inverted=True, play=True)
        self.sync_all(self.us2cycles(0.01)) # align channels and wait 10ns

        measure_chs = self.res_chs
        if self.res_ch_types[0] == 'mux4': measure_chs = self.res_chs[0]
        self.measure(
            pulse_ch=measure_chs, 
            adcs=self.adc_chs,
            adc_trig_offset=cfg.device.readout.trig_offset[0],
            wait=True,
            syncdelay=self.us2cycles(max([cfg.device.readout.relax_delay[q] for q in self.qubits])))

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
        use_EgGf_subspace: specifies whether to run RB treating EgGf as the TLS subspace
        qubits: the qubits to perform simultaneous RB on. If using EgGf subspace, specify just qA (where qA, qB represents the Eg->Gf qubits)
        singleshot_reps: reps per state for singleshot calibration
        post_process: 'threshold' (uses single shot binning), 'scale' (scale by ge_avgs), or None
        thresholds: (optional) don't rerun singleshot and instead use this
        ge_avgs: (optional) don't rerun singleshot and instead use this
        angles: (optional) don't rerun singleshot and instead use this
    )
    """

    def __init__(self, soccfg=None, path='', prefix='SimultaneousRB', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        qubits = self.cfg.expt.qubits

        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update({key3: [value3]*num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})

        adc_chs = self.cfg.hw.soc.adcs.readout.ch
        
        # ================= #
        # Get single shot calibration for all qubits
        # ================= #

        post_process = self.cfg.expt.post_process
        thresholds_q = None
        ge_avgs_q = None
        angles_q = None
        fids_q = None
        if 'angles' in self.cfg.expt and 'thresholds' in self.cfg.expt and 'ge_avgs' in self.cfg.expt:
            angles_q = self.cfg.expt.angles
            thresholds_q = self.cfg.expt.thresholds
            ge_avgs_q = np.asarray(self.cfg.expt.ge_avgs)
            print('Re-using provided angles, thresholds, ge_avgs')
        else:
                thresholds_q = [0]*4
                ge_avgs_q = [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]
                angles_q = [0]*4
                fids_q = [0]*4

                # We really just need the single shot plots here, but convenient to use the ErrorMitigation tomo to do it
                sscfg = AttrDict(deepcopy(self.cfg))
                sscfg.expt.reps = sscfg.expt.singleshot_reps
                sscfg.expt.tomo_qubits = self.cfg.expt.qubits
                if not self.cfg.expt.use_EgGf_subspace: # single qubit RB
                    sscfg.expt.tomo_qubits = [sscfg.expt.tomo_qubits[0], (sscfg.expt.tomo_qubits[0] + 1) % 4] # super hacky way to just add another qubit so can use the error tomo program to do single shot calib
                else: sscfg.expt.tomo_qubits = [sscfg.expt.tomo_qubits[0], 1]

                calib_prog_dict = dict()
                calib_order = ['gg', 'ge', 'eg', 'ee']
                for prep_state in tqdm(calib_order):
                    # print(prep_state)
                    sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state, apply_q1_pi2=False)
                    err_tomo = ErrorMitigationStateTomo2QProgram(soccfg=self.soccfg, cfg=sscfg)
                    err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=False, debug=debug)
                    calib_prog_dict.update({prep_state:err_tomo})

                g_prog = calib_prog_dict['gg']
                Ig, Qg = g_prog.get_shots(verbose=False)

                # Get readout angle + threshold for qubits
                for qi, q in enumerate(sscfg.expt.tomo_qubits):
                    calib_e_state = 'gg'
                    calib_e_state = calib_e_state[:qi] + 'e' + calib_e_state[qi+1:]
                    e_prog = calib_prog_dict[calib_e_state]
                    Ie, Qe = e_prog.get_shots(verbose=False)
                    shot_data = dict(Ig=Ig[q], Qg=Qg[q], Ie=Ie[q], Qe=Qe[q])
                    print(f'Qubit  ({q})')
                    fid, threshold, angle = hist(data=shot_data, plot=True, verbose=False)
                    thresholds_q[q] = threshold[0]
                    ge_avgs_q[q] = [np.average(Ig[q]), np.average(Qg[q]), np.average(Ie[q]), np.average(Qe[q])]
                    angles_q[q] = angle
                    fids_q[q] = fid[0]
                    print(f'ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[q]} \t threshold ge: {thresholds_q[q]}')

                if debug:
                    print(f'thresholds={thresholds_q}')
                    print(f'angles={angles_q}')
                    print(f'ge_avgs={ge_avgs_q}')

                thresholds_q = np.asarray(thresholds_q)
                angles_q = np.asarray(angles_q)
                ge_avgs_q = np.asarray(ge_avgs_q)

        # ================= #
        # Begin RB
        # ================= #

        if 'shot_avg' not in self.cfg.expt: self.cfg.expt.shot_avg=1
        a = [[] for _ in range(len(qubits))]
        data={"xpts":[], "avgi":deepcopy(a), "avgq":deepcopy(a), "amps":deepcopy(a), "phases":deepcopy(a), "avgi_err":deepcopy(a), "avgq_err":deepcopy(a)}

        depths = self.cfg.expt.start + self.cfg.expt.step * np.arange(self.cfg.expt.expts)
        for depth in tqdm(depths):
            # print(f'depth {depth} gate list (last gate is the total gate)')
            data['xpts'].append([])
            for iq, q in enumerate(qubits):
                data["avgi"][iq].append([])
                data["avgi_err"][iq].append([])
            for var in range(self.cfg.expt.variations):
                if 'gate_char' in self.cfg.expt and self.cfg.expt.gate_char is not None:
                    gate_list, total_gate = interleaved_gate_sequence(depth, gate_char=self.cfg.expt.gate_char)
                else: gate_list, total_gate = gate_sequence(depth)
                gate_list.append(total_gate) # make sure to do the inverse gate
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

                # print('variation', var)
                qubit_list = np.random.choice(self.cfg.expt.qubits, size=len(gate_list)-1)

                if self.cfg.expt.use_EgGf_subspace: randbench = RBEgGfProgram(soccfg=self.soccfg, cfg=self.cfg, gate_list=gate_list, qubits=self.cfg.expt.qubits, qDrive=self.cfg.expt.qDrive)
                else: randbench = SimultaneousRBProgram(soccfg=self.soccfg, cfg=self.cfg, gate_list=gate_list, qubit_list=qubit_list)
                # print(randbench)
                # from qick.helpers import progs2json
                # print(progs2json([randbench.dump_prog()]))

                # print('angles', angles_q)
                # print('ge avgs', ge_avgs_q)
                # angles_q = thresholds_q = ge_avgs_q = None

                # print(gate_list)
                avgi, avgi_err = randbench.acquire_rotated(soc=self.im[self.cfg.aliases.soc], progress=False, angle=angles_q, threshold=thresholds_q, ge_avgs=ge_avgs_q, post_process=post_process)
                for iq, q in enumerate(qubits):
                    data["avgi"][iq][-1].append(avgi[adc_chs[q]])
                    # print(depth, var, iq, avgi)
                    data["avgi_err"][iq][-1].append(avgi_err[adc_chs[q]])
                data['xpts'][-1].append(depth)
                # try:
                #     avgi, avgi_err = randbench.acquire_rotated(soc=self.im[self.cfg.aliases.soc], progress=False, angle=angles_q, threshold=thresholds_q, ge_avgs=ge_avgs_q, post_process=post_process)
                #     for iq, q in enumerate(qubits):
                #         avgi = avgi[adc_chs[q]]
                #         data["avgi"][iq][-1].append(avgi)
                #         # print(depth, var, iq, avgi)
                #         data["avgi_err"][iq][-1].append(avgi_err[adc_chs[q]])
                #     data['xpts'][-1].append(depth)
                # except Exception as e:
                #     print(e)
                #     print('Varation', var, 'failed in depth', depth)
                #     continue


                # print(1-data['avgi'][0][-1], gate_list)
            # data['xpts'].append(depth)

        # for k, arr in data.items():
        #     if isinstance(arr, tuple):
        #         data[k]=(np.array(a) for a in arr)
        #     else: data[k] = np.array(arr)
        # print(data['avgi'])
        for k, a in data.items():
            data[k] = np.array(a)
        # print(np.shape(data['avgi'][iq]))

        self.data=data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data

        qubits = self.cfg.expt.qubits
        data['probs'] = [None] * len(qubits)
        data['fit'] = [None] * len(qubits)
        data['fit_err'] = [None] * len(qubits)
        data['error'] = [100.] * len(qubits)
        data['std_dev_probs'] = [None] * len(qubits)
        data['med_probs'] = [None] * len(qubits)
        data['avg_probs'] = [None] * len(qubits)
        for iq, q in enumerate(qubits):
            probs = np.zeros_like(data['avgi'][iq])
            for depth in range(len(data['avgi'][iq])):
                if self.cfg.expt.post_process is not None: probs[depth] = 1 - np.asarray(data['avgi'][iq][depth])
                else: probs[depth] = np.asarray(data['avgi'][iq][depth]) # when not doing post processing, avgi is 'avgi', avgq is 'avgi_err'
                # data['xpts'][depth] = np.asarray(data['xpts'][depth])
            # for ival, val in enumerate(data['avgi'][iq]):
            #     if val is None: probs[ival] = np.nan
            #     else: probs[ival] = 1 - val # want fidelity = 1 when end in |g>
            # probs = np.reshape(probs, (self.cfg.expt.expts, self.cfg.expt.variations))
            # probs = np.average(probs, axis=1) # average over variations
            probs = np.asarray(probs)
            data['xpts'] = np.asarray(data['xpts'])
            data['probs'][iq] = probs
            # probs = np.reshape(probs, (self.cfg.expt.expts, self.cfg.expt.variations))
            std_dev_probs = []
            med_probs = []
            avg_probs = []
            working_depths = []
            depths = data['xpts']
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
            flat_probs = np.concatenate(data['probs'][iq])
            # depths = self.cfg.expt.start + self.cfg.expt.step * np.arange(self.cfg.expt.expts)
            # popt, pcov = fitter.fitrb(depths[:-4], med_probs[:-4])
            # popt, pcov = fitter.fitrb(depths, med_probs)
            # print(working_depths, avg_probs)
            # popt, pcov = fitter.fitrb(working_depths, avg_probs)
            data['std_dev_probs'][iq] = std_dev_probs
            data['med_probs'][iq] = med_probs
            data['avg_probs'][iq] = avg_probs
            data['working_depths'] = working_depths
            if fit:
                popt, pcov = fitter.fitrb(flat_depths, flat_probs)
                data['fit'][iq] = popt
                data['fit_err'][iq] = pcov
                data['error'][iq] = fitter.rb_error(popt[0], d=2**(len(self.cfg.expt.qubits)))
        return data

    def display(self, qubit, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 
        
        qubits = self.cfg.expt.qubits
        iq = np.argwhere(np.array(qubits) == qubit)[0][0]

        plt.figure(figsize=(8,6))
        irb = 'gate_char' in self.cfg.expt and self.cfg.expt.gate_char is not None
        use_EgGf = self.cfg.expt.use_EgGf_subspace
        title = f'{"Interleaved " + self.cfg.expt.gate_char + " Gate" if irb else ""} {"EgGf" if use_EgGf else ""} RB on {(str(self.cfg.expt.qubits[0]) + ", " + str(self.cfg.expt.qubits[1])) if use_EgGf else ("Q" + str(qubit))}'

        plt.subplot(111, title=title, xlabel="Sequence Depth", ylabel="Population in g")
        depths = data['xpts']
        flat_depths = np.concatenate(depths)
        flat_probs = np.concatenate(data['probs'][iq])
        plt.plot(flat_depths, flat_probs, 'x')

        probs_vs_depth = data['probs'][iq]
        # probs_vs_depth = np.reshape(data['probs'][iq], (self.cfg.expt.expts, self.cfg.expt.variations))
        std_dev_probs = data['std_dev_probs'][iq]
        med_probs = data['med_probs'][iq]
        avg_probs = data['avg_probs'][iq]
        working_depths = data['working_depths']
        plt.errorbar(working_depths, avg_probs, fmt='o-', yerr=2*std_dev_probs, color='tab:blue', elinewidth=0.75)

        if fit:
            cov_p = data['fit_err'][iq][0][0]
            fit_plt_xpts = range(working_depths[-1]+1)
            # plt.plot(depths, avg_probs, 'o-', color='tab:blue')
            plt.plot(fit_plt_xpts, fitter.rb_func(fit_plt_xpts, *data["fit"][iq]))
            print(f'Running {"interleaved " + self.cfg.expt.gate_char + " gate" if irb else "regular"} RB {"on EgGf subspace" if use_EgGf else ""}')
            print(f'Depolarizing parameter p from fit: {data["fit"][iq][0]} +/- {np.sqrt(cov_p)}')
            print(f'Average RB gate error: {data["error"][iq]} +/- {np.sqrt(fitter.error_fit_err(cov_p, 2**(len(self.cfg.expt.qubits))))}')
            print(f'\tFidelity=1-error: {1-data["error"][iq]} +/- {np.sqrt(fitter.error_fit_err(cov_p, 2**(len(self.cfg.expt.qubits))))}')

        plt.grid(linewidth=0.3)
        if self.cfg.expt.post_process is not None: plt.ylim(-0.1, 1.1)
        plt.show()
    
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)

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
        if qA == 1: qSort = qB
        qDrive = 1
        if 'qDrive' in self.cfg.expt and self.cfg.expt.qDrive is not None:
            qDrive = self.cfg.expt.qDrive
        qNotDrive = -1
        if qA == qDrive: qNotDrive = qB
        else: qNotDrive = qA
        self.qDrive = qDrive
        self.qNotDrive = qNotDrive
        self.qSort = qSort
        super().__init__(soccfg, cfg)

    def cliffordEgGf(self, qDrive, qNotDrive, pulse_name:str, extra_phase=0, inverted=False, play=False):
        """
        Convert a clifford pulse name (in the Eg-Gf subspace) into the function that performs the pulse.
        If inverted, play the inverse of this gate (the extra phase is added on top of the inversion)
        """
        pulse_name = pulse_name.upper()
        assert pulse_name in clifford_1q_names
        gates = pulse_name.split(',')

        # Normally gates are applied right to left, but if inverted apply them left to right
        gate_order = reversed(gates)
        if inverted:
            gate_order = gates
        for gate in gate_order:
            pulse_func = None
            if gate == 'I': continue
            if 'X' in gate: pulse_func = self.XEgGf_pulse
            elif 'Y' in gate: pulse_func = self.YEgGf_pulse
            elif 'Z' in gate: pulse_func = self.ZEgGf_pulse
            else: assert False, 'Invalid gate'

            neg = '-' in gate
            if inverted: neg = not neg
            pulse_func(qDrive=qDrive, qNotDrive=qNotDrive, pihalf='/2' in gate, neg=neg, extra_phase=extra_phase, play=play, reload=False)
            # print(self.overall_phase[qubit])

    def body(self):
        cfg=AttrDict(self.cfg)

        # Phase reset all channels
        for ch in self.gen_chs.keys():
            if self.gen_chs[ch]['mux_freqs'] is None: # doesn't work for the mux channels
                # print('resetting', ch)
                self.setup_and_pulse(ch=ch, style='const', freq=100, phase=0, gain=100, length=10, phrst=1)
            self.sync_all()
        self.sync_all(10)

        # Get into the Eg-Gf subspace
        self.X_pulse(self.qNotDrive, extra_phase=-self.overall_phase[self.qSort], pihalf=False, play=True) # this is the g->e pulse from CliffordAveragerProgram, always have the "overall phase" of the normal qubit subspace be 0 because it is just a state prep pulse
        self.sync_all(5)

        # self.setup_and_pulse(ch=self.qubit_chs[1], style='arb', freq=self.f_Q1_ZZ_regs[self.qA], phase=self.deg2reg(-90, gen_ch=self.qA), gain=self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[self.qA] // 2, waveform=f'qubit1_ZZ{self.qA}')
        # self.sync_all(10)

        # Do all the gates given in the initialize except for the total gate
        for i in range(len(self.gate_list) - 1):
            self.cliffordEgGf(qDrive=self.qDrive, qNotDrive=self.qNotDrive, pulse_name=self.gate_list[i], play=True)
            self.sync_all()

        # self.Xef_pulse(q=1, play=True)
        # qB = 1
        # self.setup_and_pulse(ch=self.qubit_chs[qB], style="arb", freq=self.f_ef_regs[qB], phase=0, gain=cfg.device.qubit.pulses.pi_ef.gain[qB], waveform=f"pi_ef_qubit{qB}") #, phrst=1)

        # Do the inverse by applying the total gate with pi phase
        self.cliffordEgGf(qDrive=self.qDrive, qNotDrive=self.qNotDrive, pulse_name=self.gate_list[-1], inverted=True, play=True)
        self.sync_all(5)

        # Go back to measurement subspace
        self.X_pulse(self.qNotDrive, extra_phase=-self.overall_phase[self.qSort], play=True)

        measure_chs = self.res_chs
        if self.res_ch_types[0] == 'mux4': measure_chs = self.res_chs[0]
        self.measure(
            pulse_ch=measure_chs, 
            adcs=self.adc_chs,
            adc_trig_offset=cfg.device.readout.trig_offset[0],
            wait=True,
            syncdelay=self.us2cycles(max([cfg.device.readout.relax_delay[q] for q in self.qubits])))