# # Author: Ziqian 11/08/2023

# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.optimize import curve_fit
# from copy import deepcopy
# import random

# from qick import *
# from qick.helpers import gauss

# from slab import Experiment, AttrDict
# from tqdm import tqdm_notebook as tqdm

# # from experiments.single_qubit.single_shot import hist, HistogramProgram

# from experiments.single_qubit.single_shot_ziqian import hist, HistogramProgram


# import experiments.fitting as fitter

# """
# Single qubit RB sequence generator
# Gate set = {I, +-X/2, +-Y/2, +-Z/2, X, Y, Z}
# """
# ## generate sequences of random pulses
# ## 1:Z,   2:X, 3:Y
# ## 4:Z/2, 5:X/2, 6:Y/2
# ## 7:-Z/2, 8:-X/2, 9:-Y/2
# ## 0:I
# ## Calculate inverse rotation
# matrix_ref = {}
# # Z, X, Y, -Z, -X, -Y
# matrix_ref['0'] = np.matrix([[1, 0, 0, 0, 0, 0],  
#                                 [0, 1, 0, 0, 0, 0],
#                                 [0, 0, 1, 0, 0, 0],
#                                 [0, 0, 0, 1, 0, 0],
#                                 [0, 0, 0, 0, 1, 0],
#                                 [0, 0, 0, 0, 0, 1]])  # identity 
# matrix_ref['1'] = np.matrix([[1, 0, 0, 0, 0, 0],
#                                 [0, 0, 0, 0, 1, 0],
#                                 [0, 0, 0, 0, 0, 1],
#                                 [0, 0, 0, 1, 0, 0],
#                                 [0, 1, 0, 0, 0, 0],
#                                 [0, 0, 1, 0, 0, 0]]) 
# matrix_ref['2'] = np.matrix([[0, 0, 0, 1, 0, 0],
#                                 [0, 1, 0, 0, 0, 0],
#                                 [0, 0, 0, 0, 0, 1],
#                                 [1, 0, 0, 0, 0, 0],
#                                 [0, 0, 0, 0, 1, 0],
#                                 [0, 0, 1, 0, 0, 0]])
# matrix_ref['3'] = np.matrix([[0, 0, 0, 1, 0, 0],
#                                 [0, 0, 0, 0, 1, 0],
#                                 [0, 0, 1, 0, 0, 0],
#                                 [1, 0, 0, 0, 0, 0],
#                                 [0, 1, 0, 0, 0, 0],
#                                 [0, 0, 0, 0, 0, 1]])
# matrix_ref['4'] = np.matrix([[1, 0, 0, 0, 0, 0],
#                                 [0, 0, 0, 0, 0, 1],
#                                 [0, 1, 0, 0, 0, 0],
#                                 [0, 0, 0, 1, 0, 0],
#                                 [0, 0, 1, 0, 0, 0],
#                                 [0, 0, 0, 0, 1, 0]])
# matrix_ref['5'] = np.matrix([[0, 0, 1, 0, 0, 0],
#                                 [0, 1, 0, 0, 0, 0],
#                                 [0, 0, 0, 1, 0, 0],
#                                 [0, 0, 0, 0, 0, 1],
#                                 [0, 0, 0, 0, 1, 0],
#                                 [1, 0, 0, 0, 0, 0]])
# matrix_ref['6'] = np.matrix([[0, 0, 0, 0, 1, 0],
#                                 [1, 0, 0, 0, 0, 0],
#                                 [0, 0, 1, 0, 0, 0],
#                                 [0, 1, 0, 0, 0, 0],
#                                 [0, 0, 0, 1, 0, 0],
#                                 [0, 0, 0, 0, 0, 1]])
# matrix_ref['7'] = np.matrix([[1, 0, 0, 0, 0, 0],
#                                 [0, 0, 1, 0, 0, 0],
#                                 [0, 0, 0, 0, 1, 0],
#                                 [0, 0, 0, 1, 0, 0],
#                                 [0, 0, 0, 0, 0, 1],
#                                 [0, 1, 0, 0, 0, 0]])
# matrix_ref['8'] = np.matrix([[0, 0, 0, 0, 0, 1],
#                                 [0, 1, 0, 0, 0, 0],
#                                 [1, 0, 0, 0, 0, 0],
#                                 [0, 0, 1, 0, 0, 0],
#                                 [0, 0, 0, 0, 1, 0],
#                                 [0, 0, 0, 1, 0, 0]])
# matrix_ref['9'] = np.matrix([[0, 1, 0, 0, 0, 0],
#                                 [0, 0, 0, 1, 0, 0],
#                                 [0, 0, 1, 0, 0, 0],
#                                 [0, 0, 0, 0, 1, 0],
#                                 [1, 0, 0, 0, 0, 0],
#                                 [0, 0, 0, 0, 0, 1]])

# def no2gate(no):
#     g = 'I'
#     if no==1:
#         g = 'Z'
#     elif no==2:
#         g = 'X'
#     elif no==3:
#         g = 'Y'
#     elif no==4:
#         g = 'Z/2'
#     elif no==5:
#         g = 'X/2'
#     elif no==6:
#         g = 'Y/2'
#     elif no==7:
#         g = '-Z/2'
#     elif no==8:
#         g = '-X/2'
#     elif no==9:
#         g = '-Y/2'   

#     return g

# def gate2no(g):
#     no = 0
#     if g=='Z':
#         no = 1
#     elif g=='X':
#         no = 2
#     elif g=='Y':
#         no = 3
#     elif g=='Z/2':
#         no = 4
#     elif g=='X/2':
#         no = 5
#     elif g=='Y/2':
#         no = 6
#     elif g=='-Z/2':
#         no = 7
#     elif g=='-X/2':
#         no = 8
#     elif g=='-Y/2':
#         no = 9  

#     return no

# def generate_sequence(rb_depth, iRB_gate_no=-1, debug=False, matrix_ref=matrix_ref):
#     gate_list = []
#     for ii in range(rb_depth):
#         gate_list.append(random.randint(0, 9))
#         if iRB_gate_no > -1:   # performing iRB
#             gate_list.append(iRB_gate_no)

#     a0 = np.matrix([[1], [0], [0], [0], [0], [0]])
#     anow = a0
#     for i in gate_list:
#         anow = np.dot(matrix_ref[str(i)], anow)
#     anow1 = np.matrix.tolist(anow.T)[0]
#     max_index = anow1.index(max(anow1))
#     # inverse of the rotation
#     inverse_gate_symbol = ['-Y/2', 'X/2', 'X', 'Y/2', '-X/2']
#     if max_index == 0:
#         pass
#     else:
#         gate_list.append(gate2no(inverse_gate_symbol[max_index-1]))
#     if debug:
#         print(gate_list)
#         print(max_index)
#     return gate_list

# class SingleRBrun(AveragerProgram):
#     """
#     RB program for single qubit gates
#     """

#     def __init__(self, soccfg, cfg):
#         # gate_list should include the total gate!
#         self.gate_list =  cfg.expt.running_list
#         self.cfg = AttrDict(cfg)
#         self.cfg.update(self.cfg.expt)

#         # copy over parameters for the acquire method
#         self.cfg.reps = cfg.expt.reps
#         super().__init__(soccfg, cfg)

#     def initialize(self):
#         cfg = AttrDict(self.cfg)
#         self.cfg.update(cfg.expt)
#         qTest = 0

#         self.adc_chs = cfg.hw.soc.adcs.readout.ch
#         self.res_chs = cfg.hw.soc.dacs.readout.ch
#         self.res_ch_types = cfg.hw.soc.dacs.readout.type
#         self.qubit_chs = cfg.hw.soc.dacs.qubit.ch
#         self.qubit_ch_types = cfg.hw.soc.dacs.qubit.type

#         gen_chs = []

#         self.f_ge = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_chs)
#         self.f_ef = self.freq2reg(cfg.device.qubit.f_ef, gen_ch=self.qubit_chs)

#         self.q_rps = self.ch_page(self.qubit_chs) # get register page for qubit_chs
#         self.f_ge_reg = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_chs)

#         self.f_res_reg = self.freq2reg(cfg.device.readout.frequency, gen_ch=self.res_chs, ro_ch=self.adc_chs)
#         self.readout_lengths_dac = self.us2cycles(self.cfg.device.readout.readout_length, gen_ch=self.res_chs) 
#         self.readout_lengths_adc = 1+self.us2cycles(self.cfg.device.readout.readout_length, ro_ch=self.adc_chs) 

#         self.declare_readout(ch=self.adc_chs, length=self.readout_lengths_adc, freq=cfg.device.readout.frequency, gen_ch=self.res_chs)
#         self.declare_gen(ch=self.qubit_chs, nqz=cfg.hw.soc.dacs.qubit.nyquist)
#         gen_chs.append(self.qubit_chs)


#         self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_chs)
#         self.pi_gain = cfg.device.qubit.pulses.pi_ge.gain
#         self.hpi_sigma = self.us2cycles(cfg.device.qubit.pulses.hpi_ge.sigma, gen_ch=self.qubit_chs)
#         self.hpi_gain = cfg.device.qubit.pulses.hpi_ge.gain


#         # define all 2 different pulses
#         self.add_gauss(ch=self.qubit_chs, name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)
#         self.add_gauss(ch=self.qubit_chs, name="hpi_qubit", sigma=self.hpi_sigma, length=self.hpi_sigma*4)


#         self.set_pulse_registers(ch=self.res_chs, style="const", freq=self.f_res_reg, phase=self.deg2reg(
#             cfg.device.readout.phase), gain=cfg.device.readout.gain, length=self.readout_lengths_dac)

#         self.sync_all(self.us2cycles(0.2))

#     def body(self):
#         cfg = AttrDict(self.cfg)

#         self.vz = 0   # virtual Z phase in degree
#         qTest = 0
#         for ii in self.cfg.expt.running_list:
#             # add gate
#             if ii == 0:
#                 pass
#             if ii == 1:  #'Z'
#                 self.vz += 180 
#             if ii == 2:  #'X'
#                 self.setup_and_pulse(ch=self.qubit_chs, style="arb", freq=self.f_ge_reg,
#                                  phase=self.deg2reg(0+self.vz), gain=self.pi_gain, waveform="pi_qubit")
#                 self.sync_all()
#             if ii == 3:  #'Y'
#                 self.setup_and_pulse(ch=self.qubit_chs, style="arb", freq=self.f_ge_reg,
#                                  phase=self.deg2reg(-90+self.vz), gain=self.pi_gain, waveform="pi_qubit")
#                 self.sync_all()
#             if ii == 4:  #'Z/2'
#                 self.vz += 90
#             if ii == 5:  #'X/2'
#                 self.setup_and_pulse(ch=self.qubit_chs, style="arb", freq=self.f_ge_reg,
#                                  phase=self.deg2reg(0+self.vz), gain=self.hpi_gain, waveform="hpi_qubit")
#                 self.sync_all()
#             if ii == 6:  #'Y/2'
#                 self.setup_and_pulse(ch=self.qubit_chs, style="arb", freq=self.f_ge_reg,
#                                  phase=self.deg2reg(-90+self.vz), gain=self.hpi_gain, waveform="hpi_qubit")
#                 self.sync_all()
#             if ii == 7:  #'-Z/2'
#                 self.vz -= 90
#             if ii == 8:  #'-X/2'
#                 self.setup_and_pulse(ch=self.qubit_chs, style="arb", freq=self.f_ge_reg,
#                                  phase=self.deg2reg(-180+self.vz), gain=self.hpi_gain, waveform="hpi_qubit")
#                 self.sync_all()
#             if ii == 9:  #'-Y/2'
#                 self.setup_and_pulse(ch=self.qubit_chs, style="arb", freq=self.f_ge_reg,
#                                  phase=self.deg2reg(90+self.vz), gain=self.hpi_gain, waveform="hpi_qubit")
#                 self.sync_all()
                
#         # align channels and wait 50ns and measure
#         self.sync_all(self.us2cycles(0.05))
#         self.measure(
#             pulse_ch=self.res_chs,
#             adcs=[self.adc_chs],
#             adc_trig_offset=cfg.device.readout.trig_offset,
#             wait=True,
#             syncdelay=self.us2cycles(cfg.device.readout.relax_delay)
#         )

#     def collect_shots(self):
#         # collect shots for the relevant adc and I and Q channels
#         cfg=AttrDict(self.cfg)
#         # print(np.average(self.di_buf[0]))
#         shots_i0 = self.di_buf[0] / self.readout_lengths_adc
#         shots_q0 = self.dq_buf[0] / self.readout_lengths_adc
#         return shots_i0, shots_q0

# # ===================================================================== #
# # play the pulse
# class SingleRB(Experiment):
#     def __init__(self, soccfg=None, path='', prefix='SingleRB', config_file=None, progress=None):
#             super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)
    
#     def acquire(self, progress=False, debug=False):
#         qubits = self.cfg.expt.qubit

#         # expand entries in config that are length 1 to fill all qubits
#         q_ind = self.cfg.expt.qubit
#         for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
#             for key, value in subcfg.items() :
#                 if isinstance(value, list):
#                     subcfg.update({key: value[q_ind]})
#                 elif isinstance(value, dict):
#                     for key2, value2 in value.items():
#                         for key3, value3 in value2.items():
#                             if isinstance(value3, list):
#                                 value2.update({key3: value3[q_ind]}) 

#         adc_chs = self.cfg.hw.soc.adcs.readout.ch
        
#         # ================= #
#         # Get single shot calibration for all qubits
#         # ================= #

#         # g states for q0
#         data=dict()
#         sscfg = AttrDict(deepcopy(self.cfg))
#         sscfg.expt.reps = sscfg.expt.singleshot_reps

#         # Ground state shots
#         # cfg.expt.reps = 10000
#         sscfg.expt.qubit = 0 # this looks like its hardcoding the qubit to 0 change??? 
#         print(sscfg.expt.qubit)
#         sscfg.expt.rounds = 1 # this also looks hardcoded
#         sscfg.expt.pulse_e = False
#         sscfg.expt.pulse_f = False
#         # print(sscfg)

#         data['Ig'] = []
#         data['Qg'] = []
#         data['Ie'] = []
#         data['Qe'] = []
#         histpro = HistogramProgram(soccfg=self.soccfg, cfg=sscfg)
#         avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug)
#         data['Ig'], data['Qg'] = histpro.collect_shots()

#         # Excited state shots
#         sscfg.expt.pulse_e = True 
#         sscfg.expt.pulse_f = False
#         histpro = HistogramProgram(soccfg=self.soccfg, cfg=sscfg)
#         avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, debug=debug)
#         data['Ie'], data['Qe'] = histpro.collect_shots()
#         # print(data)

#         fids, thresholds, angle = hist(data=data, plot=True, verbose=True, span=self.cfg.expt.span)
#         data['fids'] = fids
#         data['angle'] = angle
#         data['thresholds'] = thresholds


#         print(f'ge fidelity (%): {100*fids[0]}')
#         print(f'rotation angle (deg): {angle}')
#         print(f'threshold ge: {thresholds[0]}')

#         data['Idata'] = []
#         data['Qdata'] = []
#         for var in tqdm(range(self.cfg.expt.variations)):   # repeat each depth by variations
#             # generate random gate list
#             self.cfg.expt.running_list = generate_sequence(self.cfg.expt.rb_depth, iRB_gate_no=self.cfg.expt.IRB_gate_no)
#             print(self.cfg.expt.running_list)

        
#             rb_shot = SingleRBrun(soccfg=self.soccfg, cfg=self.cfg)
#             self.prog = rb_shot
#             avgi, avgq = rb_shot.acquire(
#                 self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False, debug=debug)
#             II, QQ = rb_shot.collect_shots()
#             data['Idata'].append(II)
#             data['Qdata'].append(QQ)
            
#         self.data = data

#         return data
    
#     def save_data(self, data=None):
#         print(f'Saving {self.fname}')
#         super().save_data(data=data)
#         return self.fname
# # ===================================================================== #
