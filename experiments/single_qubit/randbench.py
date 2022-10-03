# Author: Connie 2022/02/17

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from copy import deepcopy

from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

from experiments.single_qubit.single_shot import hist
from experiments.clifford_averager_program import CliffordAveragerProgram
from experiments.two_qubit.twoQ_state_tomography import ErrorMitigationStateTomo2QProgram

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

def gate_sequence(rb_depth, debug=False):
    """
    Generate RB forward gate sequence of length rb_depth as a list of pulse names;
    also return the Clifford gate that is equivalent to the total pulse sequence.
    The effective inverse is pi phase + the total Clifford.
    """
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

def rb_func(depth, alpha, a, b):
    return a*alpha**depth + b

# !! something seems wrong about this function
def rb_error(alpha, dim): # dim = number of qubits
    return (1-alpha) * (dim - 1)/dim

if __name__ == '__main__':
    print('Clifford gates:', clifford_1q_names)
    print('Total number Clifford gates:', len(clifford_1q_names))
    pulse_name_seq, total_clifford = gate_sequence(2, debug=True)
    print('Pulse sequence:', pulse_name_seq)
    print('Total clifford of seq:', total_clifford)

# ===================================================================== #

class SimultaneousRBProgram(CliffordAveragerProgram):
    """
    RB program for single qubit gates
    """
    def clifford(self, qubit, pulse_name:str, extra_phase=0, play=False):
        """
        Convert a clifford pulse name into the function that performs the pulse.
        """
        pulse_name = pulse_name.upper()
        assert pulse_name in clifford_1q_names
        gates = pulse_name.split(',')
        for gate in reversed(gates):
            print(gate)
            if gate == 'I': continue
            if 'X' in gate:
                print('/2' in gate)
                self.X_pulse(qubit, pihalf='/2' in gate, neg='-' in gate, extra_phase=extra_phase, play=play)
            elif 'Y' in gate:
                self.Y_pulse(qubit, pihalf='/2' in gate, neg='-' in gate, extra_phase=extra_phase, play=play)
            else:
                self.Z_pulse(qubit, pihalf='/2' in gate, neg='-' in gate, extra_phase=extra_phase, play=play)
    
    def __init__(self, soccfg, cfg, gate_list, qubit_list):
        # gate_list should include the inverse!
        # qubit_list should specify the qubit on which each random gate will be applied
        self.gate_list = gate_list
        self.qubit_list = qubit_list
        super().__init__(soccfg, cfg)
    
    def body(self):
        # Do all the gates given in the initialize except for the total gate, measure
        cfg=AttrDict(self.cfg)
        for i in range(len(self.gate_list) - 1):
            self.clifford(qubit=self.qubit_list[i], pulse_name=self.gate_list[i], play=True)
            self.sync_all()

        # Do the inverse by applying the total gate with pi phase
        # This is actually wrong!!! need to apply an inverse total gate for each qubit!!
        self.clifford(qubit=self.qubit_list[-1], pulse_name=self.gate_list[-1], extra_phase=180, play=True)
        self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns

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
        start: rb depth start
        step: step rb depth
        expts: number steps
        reps: number averages per unique sequence
        variations: number different sequences per depth
        qubits: the qubits to perform simultaneous RB on
        singleshot: if true, uses threshold
        singleshot_reps: reps per state for singleshot calibration
        shot_avg: number shots to average over when classifying via threshold
        thresholds: (optional) don't rerun singleshot and instead use this
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

        thresholds_q = None
        angles_q = None
        fids_q = None
        if self.cfg.expt.singleshot: 
            if 'thresholds' in self.cfg.expt and 'angles' in self.cfg.expt:
                thresholds_q = self.cfg.expt.thresholds
                angles_q = self.cfg.expt.angles
            else:
                thresholds_q = [0]*4
                angles_q = [0]*4
                fids_q = [0]*4
                sscfg = AttrDict(deepcopy(self.cfg))
                sscfg.expt.reps = sscfg.expt.singleshot_reps

                # g states for q0, q1
                if 0 in qubits or 1 in qubits:
                    sscfg.expt.qubits = [0, 1]
                    sscfg.expt.state_prep_kwargs = dict(prep_state='gg')
                    err_tomo = ErrorMitigationStateTomo2QProgram(soccfg=self.soccfg, cfg=sscfg)
                    err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=True, debug=debug)
                    Ig, Qg = err_tomo.get_shots(verbose=False)

                # e states for q0, q1
                for q, prep_state in enumerate(['eg', 'ge']):
                    if q not in qubits: continue
                    sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state)
                    err_tomo = ErrorMitigationStateTomo2QProgram(soccfg=self.soccfg, cfg=sscfg)
                    err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=True, debug=debug)
                    Ie, Qe = err_tomo.get_shots(verbose=False)
                    shot_data = dict(Ig=Ig[q], Qg=Qg[q], Ie=Ie[q], Qe=Qe[q])
                    print(f'Qubit  ({q})')
                    fid, threshold, angle = hist(data=shot_data, plot=True, verbose=False)
                    thresholds_q[q] = threshold[0]
                    angles_q[q] = angle
                    fids_q[q] = fid[0]
                    print(f'ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[q]} \t threshold ge: {thresholds_q[q]}')

                # g states for q2, q3
                if 2 in qubits or 3 in qubits:
                    sscfg.expt.qubits = [2, 3]
                    sscfg.expt.state_prep_kwargs = dict(prep_state='gg')
                    err_tomo = ErrorMitigationStateTomo2QProgram(soccfg=self.soccfg, cfg=sscfg)
                    err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=True, debug=debug)
                    Ig, Qg = err_tomo.get_shots(verbose=False)

                # e states for q2, q3
                for q, prep_state in enumerate(['eg', 'ge'], start=2):
                    if q not in qubits: continue
                    sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state)
                    err_tomo = ErrorMitigationStateTomo2QProgram(soccfg=self.soccfg, cfg=sscfg)
                    err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=True, debug=debug)
                    Ie, Qe = err_tomo.get_shots(verbose=False)
                    shot_data = dict(Ig=Ig[q], Qg=Qg[q], Ie=Ie[q], Qe=Qe[q])
                    print(f'Qubit  ({q})')
                    fid, threshold, angle = hist(data=shot_data, plot=True, verbose=False)
                    thresholds_q[q] = threshold[0]
                    angles_q[q] = angle
                    fids_q[q] = fid[0]
                    print(f'ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[q]} \t threshold ge: {thresholds_q[q]}')

            print(f'thresholds={thresholds_q}')
            print(f'angles={angles_q}')

        # ================= #
        # Begin protocol stepping
        # ================= #

        if 'shot_avg' not in self.cfg.expt: self.cfg.expt.shot_avg=1
        a = [[] for _ in range(len(qubits))]
        data={"xpts":[], "avgi":deepcopy(a), "avgq":deepcopy(a), "amps":deepcopy(a), "phases":deepcopy(a), "avgi_err":deepcopy(a), "avgq_err":deepcopy(a)}

        depths = self.cfg.expt.start + self.cfg.expt.step * np.arange(self.cfg.expt.expts)
        for depth in tqdm(depths):
            # print(f'depth {depth} gate list (last gate is the total gate)')
            for var in range(self.cfg.expt.variations):
                gate_list, total_gate = gate_sequence(depth)
                gate_list.append(total_gate) # make sure to do the inverse gate
                # gate_list = ['Z', '-X/2,Z/2', '-X/2,-Z/2']
                # gate_list = ['X']*4
                gate_list = ['X/2', '-X/2', 'X']
                print(gate_list)
                qubit_list = np.random.choice(self.cfg.expt.qubits, size=len(gate_list)-1)
                # qubit_list = np.random.choice(self.cfg.expt.qubits, size=depth)
                randbench = SimultaneousRBProgram(soccfg=self.soccfg, cfg=self.cfg, gate_list=gate_list, qubit_list=qubit_list)
                avgi, avgq, avgi_err, avgq_err = randbench.acquire_threshold(soc=self.im[self.cfg.aliases.soc], progress=False, angle=angles_q, threshold=thresholds_q, shot_avg=self.cfg.expt.shot_avg)

                for iq, q in enumerate(qubits):
                    avgi = avgi[adc_chs[q]]
                    avgq = avgq[adc_chs[q]]
                    amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
                    phase = np.angle(avgi+1j*avgq) # Calculating the phase
                    data["avgi"][iq].append(avgi)
                    # print(data['avgi'][iq])
                    data["avgq"][iq].append(avgq)
                    data["avgi_err"][iq].append(avgi_err[adc_chs[q]])
                    data["avgq_err"][iq].append(avgq_err[adc_chs[q]])
                    data["amps"][iq].append(amp)
                    data["phases"][iq].append(phase)
                data['xpts'].append(depth)
            # data['xpts'].append(depth)

        # for k, arr in data.items():
        #     if isinstance(arr, tuple):
        #         data[k]=(np.array(a) for a in arr)
        #     else: data[k] = np.array(arr)
        for k, a in data.items():
            data[k] = np.array(a)
        print(np.shape(data['avgi'][iq]))

        self.data=data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data

        qubits = self.cfg.expt.qubits
        data['probs'] = [None] * len(qubits)
        data['fit'] = [None] * len(qubits)
        data['error'] = [100.] * len(qubits)
        print(np.shape(data['avgi'][0]))
        for iq, q in enumerate(qubits):
            if not self.cfg.expt.singleshot: probs = data['amps'][iq]
            else: probs = 1 - data['avgi'][iq] # want fidelity = 1 when end in |g>
            # probs = np.reshape(probs, (self.cfg.expt.expts, self.cfg.expt.variations))
            # probs = np.average(probs, axis=1) # average over variations
            print(np.shape(probs))
            data['probs'][iq] = probs
            if fit:
                popt, pcov = curve_fit(rb_func, xdata=data['xpts'], ydata=probs)
                data['fit'][iq] = popt
                data['error'][iq] = rb_error(popt[0], dim=len(self.cfg.expt.qubits))
        return data

    def display(self, qubit, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 
        
        qubits = self.cfg.expt.qubits
        iq = np.argwhere(np.array(qubits) == qubit)[0][0]

        plt.figure(figsize=(10,6))
        plt.subplot(111, title=f"Simultaneous RB with qubits {qubits}: Qubit {qubit}",
                    xlabel="Sequence Depth", ylabel="Fidelity")
        plt.plot(data["xpts"], data["probs"][iq],'o')
        if "fit" in data:
            fit_plt_xpts = range(data["xpts"][-1])
            plt.plot(fit_plt_xpts, rb_func(fit_plt_xpts, *data["fit"][iq]))
            print(f'Alpha: {data["fit"][iq][0]}')
            print(f'Error: {data["error"][iq]}')

        plt.grid(linewidth=0.3)
        if self.cfg.expt.singleshot:
            plt.ylim(-0.02, 1.02)
        plt.show()

    
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)