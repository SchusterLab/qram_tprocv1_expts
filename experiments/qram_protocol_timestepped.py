import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from copy import deepcopy
import json

from qick import *
from qick.helpers import gauss
from slab import Experiment, NpEncoder, AttrDict

import experiments.fitting as fitter

from experiments.single_qubit.single_shot import hist
from experiments.clifford_averager_program import CliffordAveragerProgram
from experiments.two_qubit.twoQ_state_tomography import AbstractStateTomo2QProgram, ErrorMitigationStateTomo2QProgram, sort_counts

class QramProtocolProgram(AbstractStateTomo2QProgram):
    def initialize(self):
        super().initialize()
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        self.cfg.expt.state_prep_kwargs = None
        self.all_qubits = self.cfg.all_qubits

        self.swap_chs = self.cfg.hw.soc.dacs.swap.ch
        self.swap_ch_types = self.cfg.hw.soc.dacs.swap.type

        self.f_EgGf_regs = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(self.cfg.device.qubit.f_EgGf, self.swap_chs)]

        # declare swap dac indexed by qA (since the the drive is always applied to qB)
        for qA in self.all_qubits:
            if qA == 1: continue
            mixer_freq = 0
            if self.swap_ch_types[qA] == 'int4':
                mixer_freq = self.cfg.hw.soc.dacs.swap.mixer_freq[qA]
            if self.swap_chs[qA] not in self.prog_gen_chs: 
                self.declare_gen(ch=self.swap_chs[qA], nqz=self.cfg.hw.soc.dacs.swap.nyquist[qA], mixer_freq=mixer_freq)
            self.prog_gen_chs.append(self.swap_chs[qA])

        # get aliases for the sigmas we need in clock cycles
        self.pi_EgGf_types = self.cfg.device.qubit.pulses.pi_EgGf.type
        assert all(type == 'flat_top' for type in self.pi_EgGf_types)
        self.pi_EgGf_sigmas_us = self.cfg.device.qubit.pulses.pi_EgGf.sigma

        # update timestep in outer loop over averager program
        self.timestep_us = cfg.expt.timestep

        # add qubit pulses to respective channels
        for q in self.all_qubits:
            if q != 1:
                pi_EgGf_sigma_cycles = self.us2cycles(self.pi_EgGf_sigmas_us[q], gen_ch=self.swap_chs[1])
                if self.pi_EgGf_types[q] == 'gauss':
                    self.add_gauss(ch=self.swap_chs[q], name=f"pi_EgGf_swap{q}", sigma=pi_EgGf_sigma_cycles, length=pi_EgGf_sigma_cycles*4)
                elif self.pi_EgGf_types[q] == 'flat_top':
                    sigma_ramp_cycles = 3
                    self.add_gauss(ch=self.swap_chs[q], name=f"pi_EgGf_swap{q}_ramp", sigma=sigma_ramp_cycles, length=sigma_ramp_cycles*4)

        self.sync_all(200)

    def handle_next_pulse(self, count_us, ch, freq_reg, type, phase, gain, sigma_us, waveform):
        if type == 'gauss':
            new_count_us = count_us + 4 * sigma_us
        else: # const or flat_top
            new_count_us = count_us + sigma_us

        if new_count_us <= self.timestep_us: # fit entire pulse
            # print('full pulse')
            if type == 'gauss':
                self.setup_and_pulse(ch=ch, style='arb', freq=freq_reg, phase=phase, gain=gain, waveform=waveform)
            elif type == 'flat_top':
                sigma_ramp_cycles = 3
                flat_length_cycles = self.us2cycles(sigma_us, gen_ch=ch) - sigma_ramp_cycles*4
                self.setup_and_pulse(ch=ch, style='flat_top', freq=freq_reg, phase=phase, gain=gain, length=flat_length_cycles, waveform=f"{waveform}_ramp")
            elif type == 'const':
                self.setup_and_pulse(ch=ch, style='const', freq=freq_reg, phase=phase, gain=gain, length=self.us2cycles(sigma_us, gen_ch=ch))

        elif count_us < self.timestep_us: # fit part of pulse
            cut_length_us = self.timestep_us - count_us
            # print('cut length', cut_length_us)
            if type == 'gauss' :
                sigma_cycles = self.us2cycles(cut_length_us / 4, gen_ch=ch)
                if sigma_cycles > 0:
                    # print(1e3*self.timestep_us, 'sigma cycles', sigma_cycles)
                    self.add_gauss(ch=ch, name=f"{waveform}_cut", sigma=sigma_cycles, length=4*sigma_cycles)
                    self.setup_and_pulse(ch=ch, style='arb', freq=freq_reg, phase=phase, gain=gain, waveform=f"{waveform}_cut")
            elif type == 'const':
                cut_length_cycles = self.us2cycles(cut_length_us, gen_ch=ch)
                if cut_length_cycles > 3: # pulse length needs to be at least 2 cycles, add another cycle for buffer
                    self.setup_and_pulse(ch=ch, style='const', freq=freq_reg, phase=phase, gain=gain, length=cut_length_cycles)
            elif type == 'flat_top' :
                sigma_ramp_cycles = 3
                flat_length_cycles = self.us2cycles(cut_length_us, gen_ch=ch) - sigma_ramp_cycles*4
                if flat_length_cycles >= 0:
                    self.setup_and_pulse(ch=ch, style='flat_top', freq=freq_reg, phase=phase, gain=gain, length=flat_length_cycles, waveform=f"{waveform}_ramp")

        # else: already done with protocol for this timestep
        return new_count_us

    def collect_counts_post_select(self, angle=None, threshold=None, postselect=True, postselect_q=1):
        if not postselect: return self.collect_counts(angle, threshold)

        avgi, avgq = self.get_shots(angle=angle)
        # collect shots for all adcs, then sorts into e, g based on >/< threshold and angle rotation
        shots = np.array([np.heaviside(avgi[i] - threshold[i], 0) for i in range(len(self.adc_chs))])

        qA, qB = self.cfg.expt.tomo_qubits
        shots_psq0 = [[], []] # postselect qubit measured as 0
        shots_psq1 = [[], []] # postselect qubit measured as 1
        for i_shot, shot_psq in enumerate(shots[postselect_q]):
            if shot_psq:
                shots_psq1[0].append(shots[qA][i_shot])
                shots_psq1[1].append(shots[qB][i_shot])
            else:
                shots_psq0[0].append(shots[qA][i_shot])
                shots_psq0[1].append(shots[qB][i_shot])
        
        counts_psq0 = sort_counts(shotsA=shots_psq0[0], shotsB=shots_psq0[1])
        counts_psq1 = sort_counts(shotsA=shots_psq1[0], shotsB=shots_psq1[1])
        return (counts_psq0, counts_psq1)


    def state_prep_pulse(self, qubits=None, **kwargs):
        cfg=AttrDict(self.cfg)

        # ================= #
        # Initial states
        # ================= #

        init_state = self.cfg.expt.init_state

        if init_state == '|0>|0+1>':
            self.Y_pulse(q=1, play=True, pihalf=True)
            self.sync_all()

        elif init_state == '|1>|0>':
            self.Y_pulse(q=0, play=True, pihalf=False)
            self.sync_all()

        elif init_state == '|1>|0+1>':
            self.Y_pulse(q=0, play=True)
            self.sync_all(0)

            # pi2_Q1_ZZ_sigma_cycles = self.us2cycles(self.pi_Q1_ZZ_sigmas_us[0], gen_ch=self.qubit_chs[1])
            phase = self.deg2reg(-90, gen_ch=self.qubit_chs[1]) # +Y/2 -> 0+1
            # self.add_gauss(ch=self.qubit_chs[1], name='qubit1_ZZ0_half', sigma=pi2_Q1_ZZ_sigma_cycles, length=4*pi2_Q1_ZZ_sigma_cycles)
            # self.setup_and_pulse(ch=self.qubit_chs[1], style='arb', freq=self.f_Q1_ZZ_regs[0], phase=phase, gain=self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[0] // 2, waveform='qubit1_ZZ0_half')
            self.setup_and_pulse(ch=self.qubit_chs[1], style='arb', freq=self.f_Q1_ZZ_regs[0], phase=phase, gain=self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[0]//2, waveform='qubit1_ZZ0')
            self.sync_all()

        elif init_state == '|1>|1>':
            self.Y_pulse(q=0, play=True)
            self.sync_all(0)

            self.setup_and_pulse(ch=self.qubit_chs[1], style='arb', freq=self.f_Q1_ZZ_regs[0], phase=0, gain=self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[0], waveform='qubit1_ZZ0')
            self.sync_all()

        elif init_state == '|0+1>|0+1>':
            self.Y_pulse(q=0, play=True, pihalf=True) # -> 0+1
            self.sync_all()

            freq_reg = int(np.average([self.f_Q1_ZZ_regs[0], self.f_ge_regs[1]]))
            gain = int(np.average([self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[0], self.cfg.device.qubit.pulses.pi_ge.gain[1]]))
            sigma_us = np.average([self.pi_Q1_ZZ_sigmas_us[0]/2, self.pi_sigmas_us[1]/2])
            pi2_sigma_cycles = self.us2cycles(sigma_us, gen_ch=self.qubit_chs[1])
            phase = self.deg2reg(-90, gen_ch=self.qubit_chs[1]) # +Y/2 -> 0+1
            self.add_gauss(ch=self.qubit_chs[1], name='qubit1_semiZZ0_half', sigma=pi2_sigma_cycles, length=4*pi2_sigma_cycles)
            self.setup_and_pulse(ch=self.qubit_chs[1], style='arb', freq=freq_reg, phase=phase, gain=gain, waveform='qubit1_semiZZ0_half')
            self.sync_all()

        elif init_state == '|0+1>|0>':
            self.Y_pulse(q=0, play=True, pihalf=True) # -> 0+1
            self.sync_all()

        elif init_state == '|0+1>|1>':
            self.Y_pulse(q=1, play=True, pihalf=False) # -> 1
            self.sync_all()

            ZZs = np.reshape(self.cfg.device.qubit.ZZs, (4,4))
            waveform = f'qubit0_ZZ1'
            freq = self.freq2reg(self.cfg.device.qubit.f_ge[0] + ZZs[0, 1], gen_ch=self.qubit_chs[0])
            gain = self.cfg.device.qubit.pulses.pi_ge.gain[0] //2
            sigma_cycles = self.us2cycles(self.pi_sigmas_us[0], gen_ch=self.qubit_chs[0])
            self.add_gauss(ch=self.qubit_chs[0], name=waveform, sigma=sigma_cycles, length=4*sigma_cycles)
            self.setup_and_pulse(ch=self.qubit_chs[0], style='arb', freq=freq, phase=self.deg2reg(-90, gen_ch=self.qubit_chs[0]), gain=gain, waveform=waveform)
            self.sync_all()

        elif init_state == '|0+i1>|0+1>':
            self.X_pulse(q=0, play=True, pihalf=True, neg=True) # -> 0+i1
            self.sync_all()

            freq_reg = int(np.average([self.f_Q1_ZZ_regs[0], self.f_ge_regs[1]]))
            gain = int(np.average([self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[0], self.cfg.device.qubit.pulses.pi_ge.gain[1]]))
            sigma_us = np.average([self.pi_Q1_ZZ_sigmas_us[0]/2, self.pi_sigmas_us[1]/2])
            pi2_sigma_cycles = self.us2cycles(sigma_us, gen_ch=self.qubit_chs[1])
            phase = self.deg2reg(-90, gen_ch=self.qubit_chs[1]) # +Y/2 -> 0+1
            self.add_gauss(ch=self.qubit_chs[1], name='qubit1_semiZZ0_half', sigma=pi2_sigma_cycles, length=4*pi2_sigma_cycles)
            self.setup_and_pulse(ch=self.qubit_chs[1], style='arb', freq=freq_reg, phase=phase, gain=gain, waveform='qubit1_semiZZ0_half')
            self.sync_all()
        
        else:
            assert False, 'Init state not valid'

        # ================= #
        # Begin protocol
        # ================= #

        count_us = 0
        self.end_times_us = []

        # apply Eg-Gf with qA=0: 1. eggg -> gfgg [path 1]
        count_us = self.handle_next_pulse(count_us=count_us, ch=self.swap_chs[0], freq_reg=self.f_EgGf_regs[0], type=self.pi_EgGf_types[0], phase=0, gain=cfg.device.qubit.pulses.pi_EgGf.gain[0], sigma_us=self.pi_EgGf_sigmas_us[0], waveform='pi_EgGf_swap0')
        if count_us < self.timestep_us: self.end_times_us.append(count_us)
        self.sync_all()

        # apply Eg-Gf with qA=2: 2. gfgg -> ggeg [path 1]
        count_us = self.handle_next_pulse(count_us=count_us, ch=self.swap_chs[2], freq_reg=self.f_EgGf_regs[2], type=self.pi_EgGf_types[2], phase=self.deg2reg(-90, gen_ch=self.swap_chs[2]), gain=cfg.device.qubit.pulses.pi_EgGf.gain[2], sigma_us=self.pi_EgGf_sigmas_us[2], waveform='pi_EgGf_swap2')
        if count_us < self.timestep_us: self.end_times_us.append(count_us)
        self.sync_all()

        # 3. apply pi pulse on Q1 - need to average pi pulses corresponding to eegg -> eggg (pi_Q1_ZZ with qB=0), ggeg -> geeg (pi_Q1_ZZ with qB=2), gegg -> gggg (pi on Q1) [divisional pi pulse between two paths of protocol]
        freq_reg = self.f_Q1_ZZ_regs[0]
        gain = cfg.device.qubit.pulses.pi_Q1_ZZ.gain[0]
        sigma_us = self.pi_Q1_ZZ_sigmas_us[0]
        # freq_reg = int(np.average([self.f_Q1_ZZ_regs[0], self.f_Q1_ZZ_regs[2], self.f_ge_regs[1]]))
        # gain = int(np.average([cfg.device.qubit.pulses.pi_Q1_ZZ.gain[0], cfg.device.qubit.pulses.pi_Q1_ZZ.gain[2], self.cfg.device.qubit.pulses.pi_ge.gain[1]]))
        # sigma_us = np.average([self.pi_Q1_ZZ_sigmas_us[0], self.pi_Q1_ZZ_sigmas_us[2], self.pi_sigmas_us[1]])
        count_us = self.handle_next_pulse(count_us=count_us, ch=self.qubit_chs[1], freq_reg=freq_reg, type=self.pi_Q1_ZZ_types[0], phase=0, gain=gain, sigma_us=sigma_us, waveform='qubit1_ZZ0')
        if count_us < self.timestep_us: self.end_times_us.append(count_us)
        self.sync_all()

        # apply Eg-Gf with qA=0: 4. eggg -> gfgg [path 2]
        count_us = self.handle_next_pulse(count_us=count_us, ch=self.swap_chs[0], freq_reg=self.f_EgGf_regs[0], type=self.pi_EgGf_types[0], phase=0, gain=cfg.device.qubit.pulses.pi_EgGf.gain[0], sigma_us=self.pi_EgGf_sigmas_us[0], waveform='pi_EgGf_swap0')
        if count_us < self.timestep_us: self.end_times_us.append(count_us)
        self.sync_all()

        # apply Eg-Gf with qA=3: 5. gfgg -> ggge [path 2]
        count_us = self.handle_next_pulse(count_us=count_us, ch=self.swap_chs[3], freq_reg=self.f_EgGf_regs[3], type=self.pi_EgGf_types[3], phase=self.deg2reg(-90, gen_ch=self.swap_chs[3]), gain=cfg.device.qubit.pulses.pi_EgGf.gain[3], sigma_us=self.pi_EgGf_sigmas_us[3], waveform='pi_EgGf_swap3')
        if count_us < self.timestep_us: self.end_times_us.append(count_us)
        self.sync_all()

        # 6. apply pi pulse on Q1 - need to average pi pulses corresponding to ggge -> gege (pi_Q1_ZZ with qB=3), geeg -> ggeg (pi_Q1_ZZ with qB=2), gegg -> gggg (pi on Q1) [path 2, which should also affect path 1: geeg -> ggeg]
        freq_reg = self.f_Q1_ZZ_regs[3]
        gain = cfg.device.qubit.pulses.pi_Q1_ZZ.gain[3]
        sigma_us = self.pi_Q1_ZZ_sigmas_us[3]
        # freq_reg = int(np.average([self.f_Q1_ZZ_regs[3], self.f_Q1_ZZ_regs[2], self.f_ge_regs[1]]))
        # gain = int(np.average([cfg.device.qubit.pulses.pi_Q1_ZZ.gain[3], cfg.device.qubit.pulses.pi_Q1_ZZ.gain[2], self.cfg.device.qubit.pulses.pi_ge.gain[1]]))
        # sigma_us = np.average([self.pi_Q1_ZZ_sigmas_us[3], self.pi_Q1_ZZ_sigmas_us[2], self.pi_sigmas_us[1]])
        count_us = self.handle_next_pulse(count_us=count_us, ch=self.qubit_chs[1], freq_reg=freq_reg, type=self.pi_Q1_ZZ_types[3], phase=0, gain=gain, sigma_us=sigma_us, waveform='qubit1_ZZ3')
        if count_us < self.timestep_us: self.end_times_us.append(count_us)
        self.sync_all()
        print(f'Total protocol time (us): {count_us}')

        # if self.cfg.expt.post_select:
        #     self.setup_measure(qubit=1, basis='X', play=True, flag=None)

        # # ================= #
        # # Reverse protocol
        # # ================= #

        # # 6. apply pi pulse on Q1 - need to average pi pulses corresponding to ggge -> gege (pi_Q1_ZZ with qB=3), geeg -> ggeg (pi_Q1_ZZ with qB=2), gegg -> gggg (pi on Q1) [path 2, which should also affect path 1: geeg -> ggeg]
        # freq_reg = int(np.average([self.f_Q1_ZZ_regs[3], self.f_Q1_ZZ_regs[2], self.f_ge_regs[1]]))
        # gain = int(np.average([cfg.device.qubit.pulses.pi_Q1_ZZ.gain[3], cfg.device.qubit.pulses.pi_Q1_ZZ.gain[2], self.cfg.device.qubit.pulses.pi_ge.gain[1]]))
        # sigma_us = np.average([self.pi_Q1_ZZ_sigmas_us[3], self.pi_Q1_ZZ_sigmas_us[2], self.pi_sigmas_us[1]])
        # count_us = self.handle_next_pulse(count_us=count_us, ch=self.qubit_chs[1], freq_reg=freq_reg, type=self.pi_Q1_ZZ_types[3], phase=0, gain=gain, sigma_us=sigma_us, waveform='qubit1_ZZ3')
        # if count_us < self.timestep_us: self.end_times_us.append(count_us)
        # self.sync_all()

        # # apply Eg-Gf with qA=3: 5. gfgg -> ggge [path 2]
        # count_us = self.handle_next_pulse(count_us=count_us, ch=self.swap_chs[3], freq_reg=self.f_EgGf_regs[3], type=self.pi_EgGf_types[3], phase=0, gain=cfg.device.qubit.pulses.pi_EgGf.gain[3], sigma_us=self.pi_EgGf_sigmas_us[3], waveform='pi_EgGf_swap3')
        # if count_us < self.timestep_us: self.end_times_us.append(count_us)
        # self.sync_all()

        # # apply Eg-Gf with qA=0: 4. eggg -> gfgg [path 2]
        # count_us = self.handle_next_pulse(count_us=count_us, ch=self.swap_chs[0], freq_reg=self.f_EgGf_regs[0], type=self.pi_EgGf_types[0], phase=0, gain=cfg.device.qubit.pulses.pi_EgGf.gain[0], sigma_us=self.pi_EgGf_sigmas_us[0], waveform='pi_EgGf_swap0')
        # if count_us < self.timestep_us: self.end_times_us.append(count_us)
        # self.sync_all()

        # # 3. apply pi pulse on Q1 - need to average pi pulses corresponding to eegg -> eggg (pi_Q1_ZZ with qB=0), ggeg -> geeg (pi_Q1_ZZ with qB=2), gegg -> gggg (pi on Q1) [divisional pi pulse between two paths of protocol]
        # freq_reg = int(np.average([self.f_Q1_ZZ_regs[0], self.f_Q1_ZZ_regs[2], self.f_ge_regs[1]]))
        # gain = int(np.average([cfg.device.qubit.pulses.pi_Q1_ZZ.gain[0], cfg.device.qubit.pulses.pi_Q1_ZZ.gain[2], self.cfg.device.qubit.pulses.pi_ge.gain[1]]))
        # sigma_us = np.average([self.pi_Q1_ZZ_sigmas_us[0], self.pi_Q1_ZZ_sigmas_us[2], self.pi_sigmas_us[1]])
        # count_us = self.handle_next_pulse(count_us=count_us, ch=self.qubit_chs[1], freq_reg=freq_reg, type=self.pi_Q1_ZZ_types[0], phase=0, gain=gain, sigma_us=sigma_us, waveform='qubit1_ZZ0')
        # if count_us < self.timestep_us: self.end_times_us.append(count_us)
        # self.sync_all()

        # # apply Eg-Gf with qA=2: 2. gfgg -> ggeg [path 1]
        # count_us = self.handle_next_pulse(count_us=count_us, ch=self.swap_chs[2], freq_reg=self.f_EgGf_regs[2], type=self.pi_EgGf_types[2], phase=0, gain=cfg.device.qubit.pulses.pi_EgGf.gain[2], sigma_us=self.pi_EgGf_sigmas_us[2], waveform='pi_EgGf_swap2')
        # if count_us < self.timestep_us: self.end_times_us.append(count_us)
        # self.sync_all()

        # # apply Eg-Gf with qA=0: 1. eggg -> gfgg [path 1]
        # count_us = self.handle_next_pulse(count_us=count_us, ch=self.swap_chs[0], freq_reg=self.f_EgGf_regs[0], type=self.pi_EgGf_types[0], phase=0, gain=cfg.device.qubit.pulses.pi_EgGf.gain[0], sigma_us=self.pi_EgGf_sigmas_us[0], waveform='pi_EgGf_swap0')
        # if count_us < self.timestep_us: self.end_times_us.append(count_us)
        # self.sync_all()


        # # wait any remaining time
        # # print('us left', self.timestep_us-count_us)
        # if count_us < self.timestep_us:
        #     self.sync_all(self.us2cycles(self.timestep_us - count_us))
        # self.sync_all()

class QramProtocolExperiment(Experiment):
    """
    Qram protocol over time sweep
    Experimental Config
    expt = dict(
        start: start protocol time [us],
        step: time step, 
        expts: number of different time experiments, 
        reps: number of reps per time step,
        tomo_2q: True/False whether to do 2q state tomography on state at last time step
        tomo_qubits: the qubits on which to do the 2q state tomo
        singleshot_reps: reps per state for singleshot calibration
        post_process: 'threshold' (uses single shot binning), 'scale' (scale by ge_avgs), or None
        calib_apply_q1_pi2: initialize Q1 in 0+1 for all calibrations
        thresholds: (optional) don't rerun singleshot and instead use this
        ge_avgs: (optional) don't rerun singleshot and instead use this
        angles: (optional) don't rerun singleshot and instead use this
    )
    """

    def __init__(self, soccfg=None, path='', prefix='qram_protocol', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
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

        timesteps = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        print('timesteps', timesteps)
        
        data={"xpts":[], "avgi":[[],[],[],[]], "avgq":[[],[],[],[]], "avgi_err":[[],[],[],[]], "avgq_err":[[],[],[],[]], "amps":[[],[],[],[]], "phases":[[],[],[],[]]}

        self.meas_order = ['ZZ', 'ZX', 'ZY', 'XZ', 'XX', 'XY', 'YZ', 'YX', 'YY']
        # self.meas_order = ['ZZ']
        self.calib_order = ['gg', 'ge', 'eg', 'ee'] # should match with order of counts for each tomography measurement 
        self.tomo_qubits = self.cfg.expt.tomo_qubits
        if self.cfg.expt.post_select: data.update({'counts_tomo_ps0':[], 'counts_tomo_ps1':[],'counts_calib':[]})
        else: data.update({'counts_tomo':[], 'counts_calib':[]})

        # ================= #
        # Get single shot calibration for qubits
        # ================= #

        post_process = self.cfg.expt.post_process
        thresholds_q = ge_avgs_q = angles_q = fids_q = None
        if 'angles' in self.cfg.expt and 'thresholds' in self.cfg.expt and 'ge_avgs' in self.cfg.expt and not self.cfg.expt.tomo_2q:
            angles_q = self.cfg.expt.angles
            thresholds_q = self.cfg.expt.thresholds
            ge_avgs_q = np.asarray(self.cfg.expt.ge_avgs)
            print('Re-using provided angles, thresholds, ge_avgs')
        else:
            thresholds_q = [0]*4
            ge_avgs_q = [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]
            angles_q = [0]*4
            fids_q = [0]*4

            sscfg = AttrDict(deepcopy(self.cfg))
            sscfg.expt.reps = sscfg.expt.singleshot_reps

            # Error mitigation measurements: prep in gg, ge, eg, ee to recalibrate measurement angle and measure confusion matrix
            calib_prog_dict = dict()
            for prep_state in tqdm(self.calib_order):
                # print(prep_state)
                sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state, apply_q1_pi2=sscfg.expt.calib_apply_q1_pi2)
                err_tomo = ErrorMitigationStateTomo2QProgram(soccfg=self.soccfg, cfg=sscfg)
                err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=False, debug=debug)
                calib_prog_dict.update({prep_state:err_tomo})

            g_prog = calib_prog_dict['gg']
            Ig, Qg = g_prog.get_shots(verbose=False)
            threshold = [0]*num_qubits_sample
            angle = [0]*num_qubits_sample

            # Get readout angle + threshold for qubits
            for qi, q in enumerate(sscfg.expt.tomo_qubits):
                calib_e_state = 'gg'
                calib_e_state = calib_e_state[:qi] + 'e' + calib_e_state[qi+1:]
                e_prog = calib_prog_dict[calib_e_state]
                Ie, Qe = e_prog.get_shots(verbose=False)
                shot_data = dict(Ig=Ig[q], Qg=Qg[q], Ie=Ie[q], Qe=Qe[q])
                print(f'Qubit  ({q})')
                fid, threshold, angle = hist(data=shot_data, plot=progress, verbose=False)
                thresholds_q[q] = threshold[0]
                ge_avgs_q[q] = [np.average(Ig[q]), np.average(Qg[q]), np.average(Ie[q]), np.average(Qe[q])]
                angles_q[q] = angle
                fids_q[q] = fid[0]
                print(f'ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[q]} \t threshold ge: {thresholds_q[q]}')
            
            # Process the shots taken for the confusion matrix with the calibration angles (for tomography)
            for prep_state in self.calib_order:
                counts = calib_prog_dict[prep_state].collect_counts(angle=angles_q, threshold=thresholds_q)
                data['counts_calib'].append(counts)


            if self.cfg.expt.expts > 1 or (self.cfg.expt.post_select and 1 not in self.cfg.expt.tomo_qubits): # Do single shot for non-tomo qubits also
                
                sscfg.expt.tomo_qubits = []
                for q in range(4):
                    if q not in self.cfg.expt.tomo_qubits: sscfg.expt.tomo_qubits.append(q)
                assert len(sscfg.expt.tomo_qubits) == 2

                # We really just need the single shot plots here, but convenient to use the ErrorMitigation tomo to do it
                calib_prog_dict = dict()
                for prep_state in tqdm(self.calib_order):
                    # print(prep_state)
                    sscfg.expt.state_prep_kwargs = dict(prep_state=prep_state)
                    err_tomo = ErrorMitigationStateTomo2QProgram(soccfg=self.soccfg, cfg=sscfg)
                    err_tomo.acquire(self.im[sscfg.aliases.soc], load_pulses=True, progress=False, debug=debug)
                    calib_prog_dict.update({prep_state:err_tomo})

                g_prog = calib_prog_dict['gg']
                Ig, Qg = g_prog.get_shots(verbose=False)
                threshold = [0]*num_qubits_sample
                angle = [0]*num_qubits_sample

                # Get readout angle + threshold for qubits
                for qi, q in enumerate(sscfg.expt.tomo_qubits):
                    calib_e_state = 'gg'
                    calib_e_state = calib_e_state[:qi] + 'e' + calib_e_state[qi+1:]
                    e_prog = calib_prog_dict[calib_e_state]
                    Ie, Qe = e_prog.get_shots(verbose=False)
                    shot_data = dict(Ig=Ig[q], Qg=Qg[q], Ie=Ie[q], Qe=Qe[q])
                    print(f'Qubit  ({q})')
                    fid, threshold, angle = hist(data=shot_data, plot=progress, verbose=False)
                    thresholds_q[q] = threshold[0]
                    ge_avgs_q[q] = [np.average(Ig[q]), np.average(Qg[q]), np.average(Ie[q]), np.average(Qe[q])]
                    angles_q[q] = angle
                    fids_q[q] = fid[0]
                    print(f'ge fidelity (%): {100*fid[0]} \t angle (deg): {angles_q[q]} \t threshold ge: {thresholds_q[q]}')

        print(f'thresholds={thresholds_q}')
        print(f'angles={angles_q}')
        print(f'ge_avgs={ge_avgs_q}')

        # ================= #
        # Begin protocol stepping
        # ================= #

        adc_chs = self.cfg.hw.soc.adcs.readout.ch
        self.pulse_dict = dict()
        if 'post_select' not in self.cfg.expt: cfg.expt.post_select = False

        for time_i, timestep in enumerate(tqdm(timesteps, disable=not progress)):
            self.cfg.expt.timestep = float(timestep)
            self.cfg.all_qubits = range(4)

            # Perform 2q state tomo only on last timestep
            if self.cfg.expt.tomo_2q and time_i == len(timesteps) - 1:
                for basis in tqdm(self.meas_order):
                    # print(basis)
                    cfg = AttrDict(deepcopy(self.cfg))
                    cfg.expt.basis = basis
                    tomo_prog = QramProtocolProgram(soccfg=self.soccfg, cfg=cfg)
                    # from qick.helpers import progs2json
                    # print('basis', basis)
                    # print(progs2json([tomo_prog.dump_prog()]))
                    # print()
                    avgi, avgq = tomo_prog.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False, debug=debug)
                    counts = tomo_prog.collect_counts_post_select(angle=angles_q, threshold=thresholds_q, postselect=self.cfg.expt.post_select, postselect_q=1)
                    if cfg.expt.post_select:
                        data['counts_tomo_ps0'].append(counts[0])
                        data['counts_tomo_ps1'].append(counts[1])
                    else: data['counts_tomo'].append(counts)
                    self.pulse_dict.update({basis:tomo_prog.pulse_dict})

            else:
                self.cfg.expt.basis = 'ZZ'
                protocol_prog = QramProtocolProgram(soccfg=self.soccfg, cfg=self.cfg)
                avgi, avgq = protocol_prog.acquire_rotated(soc=self.im[self.cfg.aliases.soc], progress=False, angle=angles_q, threshold=thresholds_q, ge_avgs=ge_avgs_q, post_process=post_process)

                for q in range(4):
                    data['avgi'][q].append(avgi[adc_chs[q]])
                    data['avgq'][q].append(avgq[adc_chs[q]])
                    # data['avgi_err'][q].append(avgi_err[adc_chs[q]])
                    # data['avgq_err'][q].append(avgq_err[adc_chs[q]])
                    data['amps'][q].append(np.abs(avgi[adc_chs[q]]+1j*avgi[adc_chs[q]]))
                    data['phases'][q].append(np.angle(avgi[adc_chs[q]]+1j*avgi[adc_chs[q]]))

                data['xpts'].append(float(timestep))

        if self.cfg.expt.expts > 1:
            data['end_times'] = protocol_prog.end_times_us
            print('end times', protocol_prog.end_times_us)

        for k, a in data.items():
            data[k] = np.array(a)
        
        self.data = data

        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data
        return data

    def display(self, data=None, err=True, saveplot=False, **kwargs):
        if data is None:
            data=self.data 

        xpts_ns = data['xpts']*1e3

        if self.cfg.expt.post_process == 'threshold' or self.cfg.expt.post_process == 'scale':
            plt.figure(figsize=(14,8))
            if saveplot: plt.style.use('dark_background')
            plt_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            # plt.title(f"Qram Protocol", fontsize=20)

            if err:
                plt.errorbar(xpts_ns, data["avgi"][0], fmt='o-', yerr=data["avgi_err"][0], label='Q0')
                plt.errorbar(xpts_ns, data["avgi"][1], fmt='o-', yerr=data["avgi_err"][1], label='Q1')
                plt.errorbar(xpts_ns, data["avgi"][2], fmt='o-', yerr=data["avgi_err"][2], label='Q2')
                plt.errorbar(xpts_ns, data["avgi"][3], fmt='o-', yerr=data["avgi_err"][3], label='Q3')

            else:
                plt.plot(xpts_ns, data["avgi"][0],'-', marker='o', markersize=8, label='Q0')
                plt.plot(xpts_ns, data["avgi"][1],'-', marker='*', markersize=12, label='Q1')
                plt.plot(xpts_ns, data["avgi"][2],'-', marker='s', markersize=7, label='Q2')
                plt.plot(xpts_ns, data["avgi"][3],'-', marker='^', markersize=8, label='Q3')

            # plt.fill_between(xpts_ns, data["avgi"][0] - data["avgi_err"][0], data["avgi"][0] + data["avgi_err"][0], color=plt_colors[0], alpha=0.4, linestyle='-', edgecolor=plt_colors[0])
            # plt.fill_between(xpts_ns, data["avgi"][1] - data["avgi_err"][1], data["avgi"][1] + data["avgi_err"][1], color=plt_colors[1], alpha=0.4, linestyle='-', edgecolor=plt_colors[1])
            # plt.fill_between(xpts_ns, data["avgi"][2] - data["avgi_err"][2], data["avgi"][2] + data["avgi_err"][2], color=plt_colors[2], alpha=0.4, linestyle='-', edgecolor=plt_colors[2])
            # plt.fill_between(xpts_ns, data["avgi"][3] - data["avgi_err"][3], data["avgi"][3] + data["avgi_err"][3], color=plt_colors[3], alpha=0.4, linestyle='-', edgecolor=plt_colors[3])

            end_times = data['end_times']
            for end_time in end_times:
                plt.axvline(1e3*end_time, color='0.4', linestyle='--')

            if self.cfg.expt.post_process == 'threshold':
                plt.ylim(-0.02, 1.02)
            plt.legend(fontsize=26)
            plt.xlabel('Time [ns]', fontsize=26)
            plt.ylabel("Population", fontsize=26)
            plt.tick_params(labelsize=24)
            # plt.grid(linewidth=0.3)

        else:
            plt.figure(figsize=(14,20))
            plt.subplot(421, title=f'Qubit 0', ylabel="I [adc level]")
            plt.plot(xpts_ns, data["avgi"][0],'o-')
            plt.subplot(422, title=f'Qubit 0', ylabel="Q [adc level]")
            plt.plot(xpts_ns, data["avgq"][0],'o-')

            plt.subplot(423, title=f'Qubit 1', ylabel="I [adc level]")
            plt.plot(xpts_ns, data["avgi"][1],'o-')
            plt.subplot(424, title=f'Qubit 1', ylabel="Q [adc level]")
            plt.plot(xpts_ns, data["avgq"][1],'o-')

            plt.subplot(425, title=f'Qubit 2', ylabel="I [adc level]")
            plt.plot(xpts_ns, data["avgi"][2],'o-')
            plt.subplot(426, title=f'Qubit 2', ylabel="Q [adc level]")
            plt.plot(xpts_ns, data["avgq"][2],'o-')

            plt.subplot(427, title=f'Qubit 3', xlabel='Time [ns]', ylabel="I [adc level]")
            plt.plot(xpts_ns, data["avgi"][3],'o-')
            plt.subplot(428, title=f'Qubit 3', xlabel='Time [ns]', ylabel="Q [adc level]")
            plt.plot(xpts_ns, data["avgq"][3],'o-')

        plt.tight_layout()

        if saveplot:
            plot_filename = 'qram_protocol.png'
            plt.savefig(plot_filename, format='png', bbox_inches='tight', transparent = True)
            print('Saved', plot_filename)

        plt.show()


    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        self.cfg.all_qubits = [1,2,3,4]
        super().save_data(data=data)
        # print(self.pulse_dict)
        with self.datafile() as f:
            f.attrs['pulse_dict'] = json.dumps(self.pulse_dict, cls=NpEncoder)
            f.attrs['meas_order'] = json.dumps(self.meas_order, cls=NpEncoder)
            f.attrs['calib_order'] = json.dumps(self.calib_order, cls=NpEncoder)
        return self.fname