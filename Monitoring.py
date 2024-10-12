import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys 
import os 

cmap = plt.get_cmap('Dark2')
plt.rcParams['figure.figsize'] = [10,6]
plt.rcParams.update({'font.size': 12})
plt.rcParams['animation.html'] = 'jshtml'
expt_path=os.getcwd()+'/data'
sys.path.append(os.getcwd()+'/../../qutip_sims')
sys.path.append('../qick/')
sys.path.append('../slab/')
sys.path.append('/home/xilinx/jupyter_notebooks/')


from qick import *
from qick.helpers import gauss
from tqdm import tqdm_notebook as tqdm

import time
expt_path=os.getcwd()+'/data'
import scipy as sp
import json
import yaml
import experiments as meas
import Pyro4.util
from copy import deepcopy
import scipy.constants as const

import qutip as qt
import qutip.visualization as qplt

from slab.instruments import *
from slab.experiment import Experiment
from slab.datamanagement import SlabFile
from slab import get_next_filename, AttrDict

from QSwitch import QSwitch
from PulseSequence import PulseSequence
from TomoAnalysis import TomoAnalysis
import time 
import experiments.fitting as fitter

class Monitoring(): 
    
    # define dictionary of parameters for each experiment
    
    param_dict_default = {
        't1': {'reps': 100,
               'span': 300,
               'npts': 25,
               'gain_pulse': None,
               'freq_qb': None,
               'rounds': 10,
               'value': None,
               'value_err': None,
               'stored': [], 
               'stored_err': [],
               'time': []
               },
        't2': {'reps_spectro':100,
               'span_spectro':10,
               'npts_spectro': 100,
                'rounds_spectro':10,
                'probe_length':5, 
                'probe_gain':50,
                'reps_ramsey': 200,
                'step_ramsey': 75,
                'npts_ramsey': 100,
                'freq_ramsey': 0.25,
                'ramsey_round': 4,
                'gain_pulse': None,
                'freq_qb': None,
                'freq_qb_err': None, 
                'value': None, 
                'value_err': None,
                'stored': [],
                'stored_err': [],
                'freq_stored': [],
                'freq_err_stored': [],
                'time': []
                },
        'pi': {'reps': 100,
                     'npts': 50,
                     'freq_qb': None,
                     'rounds': 10, 
                     'value': None,
                     'value_half': None,
                     'stored': [],
                     'time': [],
                     'err_amp_start': 0, 
                     'err_amp_step': 1, 
                     'err_amp_expts':5, 
                     'err_amp_reps': 1000,
                     'err_amp_loops': 20,
                     },
        
        
        'temp': {'reps': 500,
                'rounds': 200,
                'npts' : 2,
                'freq_qb': None,
                    'value': None,
                    'value_err': None,
                    'cool_qb': [0,1,2,3],
                    'stored': [],
                    'time': []
                    },
        

        'zz': {'spectro': {'reps_spectro': 100,
                        'span_spectro': 40,
                        'npts_spectro': 100,
                        'rounds_spectro': 10,
                        'probe_length': 1,
                        'probe_gain': 150,
                        'freq_qb': [],
                        'gain_pulse': [],
                        'freq_qb_err': [],
                        },
               'ramsey': {'reps_ramsey': 100,
                        'step_ramsey': 5,
                        'npts_ramsey': 150,
                        'freq_ramsey': 1,
                        'ramsey_round': 10,
                        'gain_pulse': [],
                        'freq_qb': [], 
                        'freq_qb_err': [],
},
               'pi': {'reps': 100,
                            'npts': 50,
                            'rounds': 10, 
                            'err_amp_start': 0,
                            'err_amp_step': 1,
                            'err_amp_expts': 10,
                            'err_amp_reps': 1000,
                            'err_amp_loops': 20,
                            'freq_qb': [],
                            'gain_pulse': [],
                            'gain_pulse_half': [],
                            }
               }
    }
               
    def __init__(self,
                 param_dict,
                 qubits = [0,1,2,3],
                 ip_address='10.108.30.56',
                 config_file='config_q3diamond_full688and638_reset.yml', 
                 save_path='data_241007',
                 debug=False,
                 using_LO=False, 
                 live_plot=True):
        
       
        self.debug = debug
        self.save_path = save_path
        # load experiment config and rfsoc config
        self.im = InstrumentManager(ns_address=ip_address)
        config_path = os.path.join(os.getcwd(), config_file)
        self.config_path = config_path
        with open(config_path, 'r') as cfg_file:
            yaml_cfg = yaml.safe_load(cfg_file)
        self.config_file = AttrDict(yaml_cfg)
        if self.debug:
            print('config loaded')
            print(self.config_file)
        self.rfsoc_config = QickConfig(self.im[yaml_cfg['aliases']['soc']].get_cfg())
        if self.debug:
            print('rfsoc config loaded')
            print(self.rfsoc_config)
            
        self.expt_path = os.path.join(os.getcwd(), 'data', 'data_240617')
        if debug:
            print('Data will be stored in:')
            print(self.expt_path)
            
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
            
        self.qubits = qubits
        self.param_dict = param_dict
        self.qubit_watched = list(self.param_dict.keys())
        
        # update the length of the zz param 
        
        self.param_dict_default['zz']['spectro']['freq_qb'] = [None]*len(self.qubit_watched)
        self.param_dict_default['zz']['spectro']['freq_qb_err'] = [None]*len(self.qubit_watched)
        self.param_dict_default['zz']['spectro']['gain_pulse'] = [None]*len(self.qubit_watched)
        self.param_dict_default['zz']['ramsey']['freq_qb'] = [None]*len(self.qubit_watched)
        self.param_dict_default['zz']['ramsey']['freq_qb_err'] = [None]*len(self.qubit_watched)
        self.param_dict_default['zz']['ramsey']['gain_pulse'] = [None]*len(self.qubit_watched)
        self.param_dict_default['zz']['pi']['gain_pulse'] = [None]*len(self.qubit_watched)
        self.param_dict_default['zz']['pi']['gain_pulse_half'] = [None]*len(self.qubit_watched)
        self.param_dict_default['zz']['pi']['freq_qb'] = [None]*len(self.qubit_watched)
        
        
        self.ZZ_mat = {'ge': {'value': np.zeros((len(self.qubits), len(self.qubits))),
                              'value_err': np.zeros((len(self.qubits), len(self.qubits)))},
                       'ef': {'value': np.zeros((len(self.qubits), len(self.qubits))),
                              'value_err': np.zeros((len(self.qubits), len(self.qubits)))}
                        }
        
        self.ZZ_mat = AttrDict(self.ZZ_mat)
        
        # self.ZZ_eg_conf = np.reshape(self.config_file.device.qubit.ZZs, (len(self.qubits), len(self.qubits)))
        # # temporary fix for the ZZ matrix
        # self.ZZ_ef_conf = self.ZZ_eg_conf


        # for each experiment check if the parameter is empty 
        t1_keys = self.param_dict_default['t1'].keys()
        t2_keys = self.param_dict_default['t2'].keys()
        pi_pulse_keys = self.param_dict_default['pi'].keys()
        
        for idxq, qubit_i in enumerate(self.qubit_watched):
            self.param_dict[qubit_i] = AttrDict(self.param_dict[qubit_i])
            
            if 't1' in self.param_dict[qubit_i].keys():
                if 'ge' in self.param_dict[qubit_i].t1.keys():
                    for key in t1_keys:
                        if key not in self.param_dict[qubit_i].t1.ge.keys():
                            self.param_dict[qubit_i].t1.ge[key] = self.param_dict_default['t1'][key]
                            
                if 'ef' in self.param_dict[qubit_i].t1.keys():
                    for key in t1_keys:
                        if key not in self.param_dict[qubit_i].t1.ef.keys():
                            self.param_dict[qubit_i].t1.ef[key] = self.param_dict_default['t1'][key]
            
            if 't2' in self.param_dict[qubit_i].keys():
                if 'ge' in self.param_dict[qubit_i].t2.keys():
                    for key in t2_keys:
                        if key not in self.param_dict[qubit_i].t2.ge.keys():
                            self.param_dict[qubit_i].t2.ge[key] = self.param_dict_default['t2'][key]
                if 'ef' in self.param_dict[qubit_i].t2.keys():
                    for key in t2_keys:
                        if key not in self.param_dict[qubit_i].t2.ef.keys():
                            self.param_dict[qubit_i].t2.ef[key] = self.param_dict_default['t2'][key]
                            
            if 'pi' in self.param_dict[qubit_i].keys():
                if 'ge' in self.param_dict[qubit_i].pi.keys():
                    for key in pi_pulse_keys:
                        if key not in self.param_dict[qubit_i].pi.ge.keys():
                            self.param_dict[qubit_i].pi.ge[key] = self.param_dict_default['pi'][key]
                if 'ef' in self.param_dict[qubit_i].pi.keys():
                    for key in pi_pulse_keys:
                        if key not in self.param_dict[qubit_i].pi.ef.keys():
                            self.param_dict[qubit_i].pi.ef[key] = self.param_dict_default['pi'][key]           
                            
            if 'temp' in self.param_dict[qubit_i].keys():
                for key in self.param_dict_default['temp'].keys():
                    if key not in self.param_dict[qubit_i].temp.keys():
                        self.param_dict[qubit_i].temp[key] = self.param_dict_default['temp'][key]
                        
                            
            if 'zz' in self.param_dict[qubit_i].keys():
                if 'spectro' in self.param_dict[qubit_i].zz.keys():
                    if 'ge' in self.param_dict[qubit_i].zz.spectro.keys():
                        for key in self.param_dict_default['zz']['spectro'].keys():
                            if key not in self.param_dict[qubit_i].zz.spectro.ge.keys():
                                self.param_dict[qubit_i].zz.spectro.ge[key] = deepcopy(self.param_dict_default['zz']['spectro'][key])
                    if 'ef' in self.param_dict[qubit_i].zz.spectro.keys():
                        for key in self.param_dict_default['zz']['spectro'].keys():
                            if key not in self.param_dict[qubit_i].zz.spectro.ef.keys():
                                self.param_dict[qubit_i].zz.spectro.ef[key] = deepcopy(self.param_dict_default['zz']['spectro'][key])
                if 'ramsey' in self.param_dict[qubit_i].zz.keys():
                    if 'ge' in self.param_dict[qubit_i].zz.ramsey.keys():
                        for key in self.param_dict_default['zz']['ramsey'].keys():
                            if key not in self.param_dict[qubit_i].zz.ramsey.ge.keys():
                                self.param_dict[qubit_i].zz.ramsey.ge[key] = deepcopy(self.param_dict_default['zz']['ramsey'][key])
                    if 'ef' in self.param_dict[qubit_i].zz.ramsey.keys():
                        for key in self.param_dict_default['zz']['ramsey'].keys():
                            if key not in self.param_dict[qubit_i].zz.ramsey.ef.keys():
                                self.param_dict[qubit_i].zz.ramsey.ef[key] = deepcopy(self.param_dict_default['zz']['ramsey'][key])
                if 'pi' in self.param_dict[qubit_i].zz.keys():
                    if 'ge' in self.param_dict[qubit_i].zz.pi.keys():
                        for key in self.param_dict_default['zz']['pi'].keys():
                            if key not in self.param_dict[qubit_i].zz.pi.ge.keys():
                                self.param_dict[qubit_i].zz.pi.ge[key] = deepcopy(self.param_dict_default['zz']['pi'][key])
                    if 'ef' in self.param_dict[qubit_i].zz.pi.keys():
                        for key in self.param_dict_default['zz']['pi'].keys():
                            if key not in self.param_dict[qubit_i].zz.pi.ef.keys():
                                self.param_dict[qubit_i].zz.pi.ef[key] = deepcopy(self.param_dict_default['zz']['pi'][key])
   
                # if 'ZZ_matrix' in self.param_dict[qubit_i].zz.keys():
                #     if 'ge' in self.param_dict[qubit_i].zz.spectro.keys():
                #         self.param_dict[qubit_i].zz.ZZ_matrix.ge = self.param_dict_default['zz']['ZZ_matrix']
                #     if 'ef' in self.param_dict[qubit_i].zz.spectro.keys():
                #         self.param_dict[qubit_i].zz.ZZ_matrix.ef = self.param_dict_default['zz']['ZZ_matrix']
                        
                # if 'ZZ_matrix_err' in self.param_dict[qubit_i].zz.keys():
                #     if 'ge' in self.param_dict[qubit_i].zz.spectro.keys():
                #         self.param_dict[qubit_i].zz.ZZ_matrix_err.ge = self.param_dict_default['zz']['ZZ_matrix']
                #     if 'ef' in self.param_dict[qubit_i].zz.spectro.keys():
                #         self.param_dict[qubit_i].zz.ZZ_matrix_err.ef = self.param_dict_default['zz']['ZZ_matrix']
         
         
         # define the color list 
         
        self.colors = cmap(np.linspace(0, 1, len(self.qubits)))

      
        # define the experiments 
    
        
        self.expt_t1_ge = []
        self.expt_t1_ef = []
        self.expt_t2_ge = []
        self.expt_t2_ef = []
        self.expt_pi_ge = []
        self.expt_pi_ef = []
        self.expt_temp = []
        
        for idxq, qubit_i in enumerate(self.qubits):
            
            if qubit_i in self.qubit_watched:
                
                _param_exp= self.param_dict[qubit_i]
                
                if 't1' in _param_exp.keys():
                    if 'ge' in _param_exp.t1.keys():
                        _expt_t1_ge = Experiment(
                        path=self.expt_path,
                        prefix=f"t1_time_sweep_qubit{qubit_i}",
                        config_file=config_path)
                        _expt_t1_ge.data = dict(xpts=[], avgi=[], avgq=[], amps=[], t1_fit=[], t1_fit_err=[], times=[]) 
                    else:
                        _expt_t1_ge = dict()
                        
                    self.expt_t1_ge.append(_expt_t1_ge)
                        
                    if 'ef' in _param_exp.t1.keys():
                        _expt_t1_ef = Experiment(
                        path=self.expt_path,
                        prefix=f"t1EF_time_sweep_qubit{qubit_i}",
                        config_file=config_path)
                        _expt_t1_ef.data = dict(xpts=[], avgi=[], avgq=[], amps=[], t1_fit=[], t1_fit_err=[], times=[])
                    else:
                        _expt_t1_ef = dict()
                        
                    self.expt_t1_ef.append(_expt_t1_ef)
                            
                if 't2' in _param_exp.keys():
                    if 'ge' in _param_exp.t2.keys():
                        _expt_t2_ge = Experiment(
                        path=self.expt_path,
                        prefix=f"t2r_time_sweep_qubit{qubit_i}",
                        config_file=config_path)
                        _expt_t2_ge.data = dict(xpts=[], avgi=[], avgq=[], amps=[], t2_fit=[], t2_fit_err=[], times=[], freq_qb=[], freq_qb_err=[])
                    else:
                        _expt_t2_ge = dict()
                        
                    self.expt_t2_ge.append(_expt_t2_ge)
                        
                    if 'ef' in _param_exp.t2.keys():
                        _expt_t2_ef = Experiment(
                        path=expt_path,
                        prefix=f"t2rEF_time_sweep_qubit{qubit_i}",
                        config_file=config_path)
                        _expt_t2_ef.data = dict(xpts=[], avgi=[], avgq=[], amps=[], t2_fit=[], t2_fit_err=[], times=[], freq_qb=[], freq_qb_err=[])
                    else:
                        _expt_t2_ef = dict()
                            
                    self.expt_t2_ef.append(_expt_t2_ef)
                            
                if 'pi' in _param_exp.keys():
                    if 'ge' in _param_exp.pi.keys():
                        _expt_pi_ge = Experiment(
                        path=expt_path,
                        prefix=f"pi_qubit{qubit_i}",
                        config_file=config_path)
                        _expt_pi_ge.data = dict(xpts=[], avgi=[], avgq=[], amps=[], times=[], pi_gain=[])
                    else:
                        _expt_pi_ge = dict()
                    
                    self.expt_pi_ge.append(_expt_pi_ge)
                        
                    if 'ef' in _param_exp.pi.keys():
                        _expt_pi_ef = Experiment(
                        path=expt_path,
                        prefix=f"pi_EF_qubit{qubit_i}",
                        config_file=config_path)
                        _expt_pi_ef.data = dict(xpts=[], avgi=[], avgq=[], amps=[], times=[], pi_gain=[])
                    else:
                        _expt_pi_ef = dict()
                    
                    self.expt_pi_ef.append(_expt_pi_ef)
                
                if 'temp' in _param_exp.keys():
                    _expt_temp = Experiment(
                    path=expt_path,
                    prefix=f"temp_qubit{qubit_i}",
                    config_file=config_path)
                    _expt_temp.data = dict(xpts=[], avgi=[], avgq=[], amps=[], times=[], temp=[], pop=[])
                    self.expt_temp.append(_expt_temp)    
                    
                else:
                    self.expt_temp.append(dict())
     

                if 'zz' in _param_exp.keys():
                    if 'ge' in _param_exp.zz.keys():
                        for qubit_ZZ in self.qubit_watched:
                            _expt_zz_ge = Experiment(
                            path=expt_path,
                            prefix=f"zz_qubit{qubit_i}{qubit_ZZ}",
                            config_file=config_path)
                            _expt_zz_ge.data = dict(xpts=[], avgi=[], avgq=[], amps=[], times=[], freq_qb=[], freq_qb_err=[])
                            
                        # self.expt_zz_ge.append(_expt_zz_ge)
                        
            else:
                self.expt_t1_ge.append(dict())
                self.expt_t1_ef.append(dict())
                self.expt_t2_ge.append(dict())
                self.expt_t2_ef.append(dict())
                self.expt_pi_ge.append(dict())
                self.expt_pi_ef.append(dict())
                self.expt_temp.append(dict())
                # self.expt_zz_ge.append(dict())
                
                
                
        # define the list of experiments to be watched
        
        # creat a dictionary of all experiment where the value is stored 
        expt_keys = []
        subspace_list = ['ge', 'ef']
        # check all the experiments of each qubits and fill the list with it 
        
        for idxq, qubit_i in enumerate(self.qubit_watched):
            _exps = self.param_dict[qubit_i].keys()
            for exp in _exps:
                _subspace = self.param_dict[qubit_i][exp].keys()
                if np.any([_sub in _subspace for _sub in subspace_list]):
                    for subexp in _subspace:
                        expt_keys.append(f'{exp}_{subexp}')
                        if exp == 't2':
                            expt_keys.append(f'freq_{subexp}')
                else:
                    expt_keys.append(exp)
                    
        # delete the duplicates
        expt_keys = set(expt_keys)
        self.expt_list = expt_keys
        
        
        
        # create the dictionary of the experiments that can be monitored versus time 
        
        # first remove the zz experiments
        
        expt_keys = [exp for exp in expt_keys if 'zz' not in exp]


        # initialize the live plot if needed

        self.live_plot = live_plot
        
        if live_plot: 
            
            # first count the number of experiments that countain 'ge' and 'ef'
            n_ge = np.sum(['ge' in _exp for _exp in expt_keys])
            n_ef = np.sum(['ef' in _exp for _exp in expt_keys])
            
            # if temp is in the list, add 1 to the number of ge experiments
            if 'temp' in expt_keys:
                n_ge += 1
                
            self.n_ge = n_ge
            self.n_ef = n_ef
                
            # create the figure and the axis
            self.fig, self.ax = plt.subplots(n_ge+n_ef, 1, figsize=(5, 2*(n_ge+n_ef)), sharex=True)
     
    def update_plot(self):

        # each time it is called, update the plot with the new values
        
        expt_keys = self.expt_list
        
        # delete the zz 
        expt_keys = [exp for exp in expt_keys if 'zz' not in exp]
        
        # check if the temp is in the list
        
        if 'temp' in expt_keys:
            expt_keys.remove('temp')
            temp = True
        else:
            temp = False
        
        # sort the experiment order such that the ge and ef are separated
        

        expt_subspace = sorted(expt_keys, key=lambda x: x.split('_')[1])
        
        if temp:
            expt_subspace.append('temp')
        
        
        
        for idx_plot in range(self.n_ge + self.n_ef): 
            self.ax[idx_plot].cla()
            
            if expt_subspace[idx_plot] == 'temp':
                self.ax[idx_plot].set_ylabel('Temperature [K]')
                
                for idxq, qubit_i in enumerate(self.qubit_watched):
                    if 'temp' in self.param_dict[qubit_i].keys():
                        x = np.array(self.param_dict[qubit_i]['temp']['time'])
                        y = np.array(self.param_dict[qubit_i]['temp']['stored'])
                        self.ax[idx_plot].plot(x/60/60, y, 'o-', label=f'Q{qubit_i}', color=self.colors[idxq])

                self.ax[idx_plot].legend()
            else:
                expt, subspace = expt_subspace[idx_plot].split('_')

                
                # replace all the characters by its capital letter
                _expt = expt
                _subspace = '|'+subspace+'>'
                ylabel = f'{_expt} {_subspace}'
                self.ax[idx_plot].set_ylabel(ylabel)
                
                for idxq, qubit_i in enumerate(self.qubit_watched):
                    
                    if expt == 'freq':
                        if subspace in self.param_dict[qubit_i]['t2'].keys():
                            x = np.array(self.param_dict[qubit_i]['t2'][subspace]['time'])
                            y = np.array(self.param_dict[qubit_i]['t2'][subspace]['freq_stored'])
                            y_err = np.array(self.param_dict[qubit_i]['t2'][subspace]['freq_err_stored'])
                            if len(y) > 1:
                                y = y - np.mean(y)
                                self.ax[idx_plot].errorbar(x/60/60, y, yerr=y_err, label=f'Q{qubit_i}', fmt='o-', elinewidth=0.75, color=self.colors[idxq])
                    else:
                        if expt in self.param_dict[qubit_i].keys():
                            if subspace in self.param_dict[qubit_i][expt].keys():
                                x = np.array(self.param_dict[qubit_i][expt][subspace]['time'])
                                y = np.array(self.param_dict[qubit_i][expt][subspace]['stored'])
                                if 'stored_err' in self.param_dict[qubit_i][expt][subspace].keys():
                                    y_err = np.array(self.param_dict[qubit_i][expt][subspace]['stored_err'])
                                else:
                                    y_err = None

                                self.ax[idx_plot].errorbar(x/60/60, y, yerr=y_err, label=f'Q{qubit_i}', fmt='o-', elinewidth=0.75, color=self.colors[idxq])
                            
                self.ax[idx_plot].legend()
                
            self.ax[-1].set_xlabel('Time [h]')
                
        self.fig.canvas.draw()
   
    def measure_t1(self, start_time=None, expt=None, param_exp=None, qubit_test=None, EF=False, debug=False, save=True): 
        
        # load parameters
        
        if param_exp is None:
            if EF:
                param_exp = self.param_dict[qubit_test].t1.ef
            else:
                param_exp = self.param_dict[qubit_test].t1.ge

        reps = param_exp.reps
        span = param_exp.span
        npts = param_exp.npts
        gain_pulse = param_exp.gain_pulse
        freq_qb = param_exp.freq_qb
        rounds = param_exp.rounds
        
        # load exps 
        
        if expt is None:
            if EF:
                expt = self.expt_t1_ef[qubit_test]
            else:
                expt = self.expt_t1_ge[qubit_test]
        
        if EF: 
            t1 = meas.T1Experiment(
            soccfg=self.rfsoc_config,
            path=self.expt_path,
            prefix="t1EF"+f"_qubit{qubit_test}",
            config_file=self.config_path)
        else:
            t1 = meas.T1Experiment(
            soccfg=self.rfsoc_config,
            path=self.expt_path,
            prefix="t1"+f"_qubit{qubit_test}",
            config_file=self.config_path)
            
        t1.cfg = AttrDict(deepcopy(self.config_file))  
        
        if EF:
            if freq_qb is not None:
                idx_qb = (len(self.qubits)+1)*qubit_test
                t1.cfg.device.qubit.f_ef[idx_qb] = freq_qb
                
        else:
            if freq_qb is not None:
                idx_qb = (len(self.qubits)+1)*qubit_test
                t1.cfg.device.qubit.f_ge[idx_qb] = freq_qb
            

        if gain_pulse is not None:
            idx_qb = (len(self.qubits)+1)*qubit_test
            if EF: 
                t1.cfg.device.qubit.pulses.pi_ef.gain[idx_qb] = int(gain_pulse)
            else:
                t1.cfg.device.qubit.pulses.pi_ge.gain[idx_qb] = int(gain_pulse)
                
                
        t1.cfg.expt = dict(start=0, # wait time [us]
            step=span/npts,
            expts=npts,
            reps=reps,
            rounds=rounds,
            qTest=qubit_test,
            checkEF=EF,
        )

        t1.go(analyze=True, display=False, progress=debug, save=False)
        t1_fit, t1_fit_cov= meas.fitting.get_best_fit(t1.data) #, fitter.expfunc)
        t1_fit_err = np.sqrt(t1_fit_cov[3][3])
        if debug:
            print('T1: %i +/- %i' % (t1_fit[3], t1_fit_err))
    
        if save:
            expt.cfg = t1.cfg
            expt.data['xpts'].append(t1.data['xpts'])
            expt.data['avgi'].append(t1.data['avgi'])
            expt.data['avgq'].append(t1.data['avgq'])
            expt.data['amps'].append(t1.data['amps'])
            expt.data['t1_fit'].append(t1_fit[3])
            expt.data['t1_fit_err'].append(np.sqrt(t1_fit_err[3][3]))
            if start_time is not None:
                expt.data['times'].append(time.time()-start_time)
            # save data
            expt.save_data()
                
        if debug:
            fig, ax = plt.subplots(figsize=(5, 2))
            x_data = t1.data['xpts']
            y_data = t1.data['amps']
            p_fit = t1.data['fit_amps']
            x = np.linspace(x_data[0], x_data[-1], 100)
            y = fitter.expfunc(x, *p_fit)
        
            ax.scatter(x_data, y_data, color=self.colors[qubit_test], label=f'Q{qubit_test}')
            ax.plot(x, y, label='fit', color='black')
            ax.set_xlabel('Time [us]')
            ax.set_ylabel('Amplitude')
            if EF:
                ax.set_title('T1 EF', fontsize=12)
            else:
                ax.set_title('T1', fontsize=12)
            # print the t1 time
            ax.text(0.6, 0.8, 'T1: %i +/- %i' % (t1_fit[3], t1_fit_err), transform=ax.transAxes, fontsize=12)
            ax.legend()
            
        
        return t1_fit[3], t1_fit_err
    
    def measure_ramsey(self, start_time=None, expt=None, param_exp=None, qubit_test=None, qubit_ZZ=None,  EF=False, debug=False, save=True):
        
        if param_exp is None: 
            if qubit_ZZ is None:
                if EF:
                    param_exp = self.param_dict[qubit_test].t2.ef
                else:
                    param_exp = self.param_dict[qubit_test].t2.ge
            else:
                if EF:
                    param_exp = self.param_dict[qubit_test].zz.ramsey.ef    
                else:
                    param_exp = self.param_dict[qubit_test].zz.ramsey.ge
                if 't2' in self.param_dict[qubit_ZZ].keys():
                    if 'ge' in self.param_dict[qubit_ZZ].t2.keys():
                        param_exp_zz = self.param_dict[qubit_ZZ].t2.ge
                else:
                    param_exp_zz = None
                          
        reps = param_exp.reps_ramsey
        step = param_exp.step_ramsey
        npts = param_exp.npts_ramsey
        freq = param_exp.freq_ramsey
        rounds = param_exp.ramsey_round
        gain_pulse = param_exp.gain_pulse
        freq_qb = param_exp.freq_qb
        

        if qubit_ZZ is not None:
            freq_qb = freq_qb[qubit_ZZ]
            gain_pulse = gain_pulse[qubit_ZZ]
            
            if param_exp_zz is not None:
                    freq_qb_zz = param_exp_zz.freq_qb
                    gain_pulse_zz = param_exp_zz.gain_pulse
            else:
                freq_qb_zz = None
                gain_pulse_zz = None


        if expt is None:
            if qubit_ZZ is None:
                if EF:
                    expt = self.expt_t2_ef[qubit_test]
                else:
                    expt = self.expt_t2_ge[qubit_test]

        
        if qubit_ZZ is None:      
            if EF:
                ramsey = meas.RamseyExperiment(
                    soccfg=self.rfsoc_config,
                    path=self.expt_path,
                    prefix=f"ramseyEF_qubit{qubit_test}",
                    config_file=self.config_path)
            else:
                ramsey = meas.RamseyExperiment(
                    soccfg=self.rfsoc_config,
                    path=self.expt_path,
                    prefix=f"ramsey_qubit{qubit_test}",
                    config_file=self.config_path)
                
        else:
            if EF:
                ramsey = meas.RamseyExperiment(
                    soccfg=self.rfsoc_config,
                    path=self.expt_path,
                    prefix=f"ramsey_zzEF_qubit{qubit_test}{qubit_ZZ}",
                    config_file=self.config_path)
            else:
                ramsey = meas.RamseyExperiment(
                    soccfg=self.rfsoc_config,
                    path=self.expt_path,
                    prefix=f"ramsey_zz_qubit{qubit_test}{qubit_ZZ}",
                    config_file=self.config_path)
                
        ramsey.cfg = AttrDict(deepcopy(self.config_file))
        
        if qubit_ZZ is not None:
            idx_zz = (len(self.qubits)+1)*qubit_ZZ
            if freq_qb_zz is not None:
                ramsey.cfg.device.qubit.f_ge[idx_zz] = freq_qb_zz
            if gain_pulse_zz is not None:
                ramsey.cfg.device.qubit.pulses.pi_ge.gain[idx_zz] = gain_pulse_zz
  
        if EF:
            idx_qb = (len(self.qubits)+1)*qubit_test
            if qubit_ZZ is not None:
                idx_qb += (qubit_ZZ-qubit_test)
            
            if freq_qb is not None:
                ramsey.cfg.device.qubit.f_ef[qubit_test] = freq_qb
            if gain_pulse is not None:
                ramsey.cfg.device.qubit.pulses.pi_ef.gain[qubit_test] = gain_pulse
        else:
            if freq_qb is not None:
                ramsey.cfg.device.qubit.f_ge[qubit_test] = freq_qb
            if gain_pulse is not None:
                ramsey.cfg.device.qubit.pulses.pi_ge.gain[qubit_test] = gain_pulse
            
                           
        ramsey.cfg.expt = dict(
            start=0,
            expts=npts,
            step=self.rfsoc_config.cycles2us(step),
            ramsey_freq=freq,
            reps=reps,
            rounds=rounds,
            checkEF=EF,
            qTest=qubit_test,
            qZZ=qubit_ZZ,
        )
        
        
        ramsey.go(analyze=True, display=False, progress=debug, save=False)
        t2r_fit, t2r_fit_err, t2r_adjust = meas.fitting.get_best_fit(ramsey.data, get_best_data_params=['f_adjust_ramsey'])      
        delta_freq = np.min(np.abs(t2r_adjust))
        
        p_cov = ramsey.data['fit_err_amps']
        delta_freq_err = np.sqrt(np.diag(p_cov))[1]

        
        
        if debug:
            print('T2R: %i +/- %i' % (t2r_fit[3], np.sqrt(t2r_fit_err[3][3])))
            fig, ax = plt.subplots(figsize=(5, 3))
            x_data = ramsey.data['xpts']
            y_data = ramsey.data['amps']
            x = np.linspace(x_data[0], x_data[-1], 100)
            p_fit = ramsey.data['fit_amps']
            y = fitter.decaysin(x, *p_fit)
            ax.plot(x_data, y_data, color=self.colors[qubit_test], label=f'Q{qubit_test}')
            ax.plot(x, y, label='fit', color='black')
            ax.set_xlabel('Time [us]')
            ax.set_ylabel('Amplitude')
            if EF:
                ax.set_title('Ramsey EF', fontsize=12)
            else:
                ax.set_title('Ramsey', fontsize=12)
            ax.text(0.6, 0.9, 'T2: %i +/- %i' % (t2r_fit[3], np.sqrt(t2r_fit_err[3][3])), transform=ax.transAxes, fontsize=12)
            ax.legend()
        
        if save: 
            if qubit_ZZ is None:
                expt.cfg = ramsey.cfg
                expt.data['xpts'].append(ramsey.data['xpts'])
                expt.data['avgi'].append(ramsey.data['avgi'])
                expt.data['avgq'].append(ramsey.data['avgq'])
                expt.data['amps'].append(ramsey.data['amps'])
                expt.data['t2r_fit'].append(t2r_fit[3])
                expt.data['t2r_fit_err'].append(np.sqrt(t2r_fit_err[3][3]))
                if start_time is not None:
                    expt.data['times'].append(time.time()-start_time)
                expt.data['freq_qb'].append(freq_qb + delta_freq)
                expt.data['freq_qb_err'].append(delta_freq_err)
                # save data
                expt.save_data()
                
        return freq_qb + delta_freq, delta_freq_err, t2r_fit[3], np.sqrt(t2r_fit_err[3][3])
  
    def measure_pi_pulse(self, start_time=None, expt=None, param_exp=None, qubit_test=None, EF=False, debug=False, save=True, qubit_ZZ=None, temp=False, pulse_ge=True):

        if temp:
            EF=True # override EF to True for temperature measurement
            fit = False
        else:
            fit = True        
        
        if param_exp is None:
            if qubit_ZZ is None:
                if EF:
                    param_exp = self.param_dict[qubit_test].pi.ef
                else:
                    param_exp = self.param_dict[qubit_test].pi.ge
            else:
                if EF:
                    param_exp = self.param_dict[qubit_test].zz.pi.ef
                else:
                    param_exp = self.param_dict[qubit_test].zz.pi.ge
                if 't2' in self.param_dict[qubit_test].keys():
                    if 'ge' in self.param_dict[qubit_test].t2.keys():
                        param_exp_zz = self.param_dict[qubit_ZZ].t2.ge
                else:
                    param_exp_zz = None
                    
                                    
        reps = param_exp.reps
        npts = param_exp.npts
        freq_qb = param_exp.freq_qb
        rounds = param_exp.rounds
        
        if qubit_ZZ is not None:
            freq_qb = freq_qb[qubit_ZZ]
            
            if param_exp_zz is not None:
                freq_qb_zz = param_exp_zz.freq_qb
                gain_pulse_zz = param_exp_zz.gain_pulse
            else:
                freq_qb_zz = None
                gain_pulse_zz = None
                
                    
        if qubit_ZZ is None:
            if expt is None:
                if EF:
                    expt = self.expt_pi_ef[qubit_test]
                else:
                    expt = self.expt_pi_ge[qubit_test]
                
        if qubit_ZZ is None:
            if EF:
                amprabi = meas.AmplitudeRabiExperiment(
                    soccfg=self.rfsoc_config,
                    path=self.expt_path,
                    prefix="amp_rabi_EF"+f"_qubit{qubit_test}",
                    config_file=self.config_path) 
            else:
                amprabi = meas.AmplitudeRabiExperiment(
                    soccfg=self.rfsoc_config,
                    path=self.expt_path,
                    prefix="amp_rabi"+f"_qubit{qubit_test}",
                    config_file=self.config_path)
                
        else:
            if EF:
                amprabi = meas.AmplitudeRabiExperiment(
                    soccfg=self.rfsoc_config,
                    path=self.expt_path,
                    prefix="amp_rabi_zzEF"+f"_qubit{qubit_test}{qubit_ZZ}",
                    config_file=self.config_path)
            else:
                amprabi = meas.AmplitudeRabiExperiment(
                    soccfg=self.rfsoc_config,
                    path=self.expt_path,
                    prefix="amp_rabi_zz"+f"_qubit{qubit_test}{qubit_ZZ}",
                    config_file=self.config_path)
                
                
            
        amprabi.cfg = AttrDict(deepcopy(self.config_file))
        
        
        if EF:
            idx_qb = (len(self.qubits)+1)*qubit_test
            if qubit_ZZ is not None:
                idx_qb += (qubit_ZZ-qubit_test)
            if freq_qb is not None:
                amprabi.cfg.device.qubit.f_ef[qubit_test] = freq_qb
                

            if 'pi' in self.param_dict[qubit_test].keys():
                if 'ef' in self.param_dict[qubit_test].pi.keys():
                    gain_max = self.param_dict[qubit_test].pi.ef.value 
                else: 
                    gain_max = None
            
            else:
                gain_max = None
           
            if gain_max is None:
                gain_max = self.config_file.device.qubit.pulses.pi_ef.gain[idx_qb]
                
            
            pi_len = self.config_file.device.qubit.pulses.pi_ef.sigma[idx_qb]
        
        else:
            idx_qb = (len(self.qubits)+1)*qubit_test
            if qubit_ZZ is not None:
                idx_qb += (qubit_ZZ-qubit_test)
            if freq_qb is not None:
                amprabi.cfg.device.qubit.f_ge[qubit_test] = freq_qb
                
                
            if 'pi' in self.param_dict[qubit_test].keys():
                if 'ge' in self.param_dict[qubit_test].pi.keys():
                    gain_max = self.param_dict[qubit_test].pi.ge.value
                else:
                    gain_max = None
            else:
                gain_max = None
         
            if gain_max is None:
                gain_max = self.config_file.device.qubit.pulses.pi_ge.gain[idx_qb]
                
            pi_len = self.config_file.device.qubit.pulses.pi_ge.sigma[idx_qb]
            
        if not temp:
            gain_max = 1.5*gain_max
            

        span = min(32000, gain_max)
        
        # qubit_list = [qubit_test]
        # if qubit_ZZ is not None:
        #     qubit_list = [qubit_ZZ, qubit_test] # for the amp_rabi the qubits are inverted
        

            
        if qubit_ZZ is not None:
            idx_zz = (len(self.qubits)+1)*qubit_ZZ
            if freq_qb_zz is not None:
                amprabi.cfg.device.qubit.f_ge[idx_zz] = freq_qb_zz
            if gain_pulse_zz is not None: 
                amprabi.cfg.device.qubit.pulses.pi_ge.gain[idx_zz] = gain_pulse_zz
            
            

        amprabi.cfg.expt = dict(
            start=0,
            step=int(span/npts),
            expts=npts,
            reps=reps,
            rounds=rounds,
            qTest=qubit_test,
            qZZ=qubit_ZZ,
            checkEF=EF,
            sigma_test=pi_len,
            pulse_type='gauss',
            pulse_ge=pulse_ge,
        )
        
        amprabi.go(analyze=False, display=False, progress=debug, save=False)
        if fit:
            amprabi.analyze(fit=fit)
            amprabi_fit, amprabi_fit_err = meas.fitting.get_best_fit(amprabi.data)
            p = amprabi_fit
            if p[2] > 180: p[2] = p[2] - 360
            elif p[2] < -180: p[2] = p[2] + 360
            if p[2] < 0:
                pi_gain = int(np.round(((1/2 - p[2]/180)/2/p[1]), 0))
            else:
                pi_gain = int(np.round(((3/2 - p[2]/180)/2/p[1]), 0))
        
                
            if pi_gain > 32000:
                print(f'WARNING: Fit pi pulse gain is long ({pi_gain}), please double check!')
                pi_gain = 30000
            
            if debug:
                print('pi pulse gain: %i' % pi_gain)
                
                fig, ax = plt.subplots(figsize=(5, 3))
                x_data = amprabi.data['xpts']
                y_data = amprabi.data['amps']
                x = np.linspace(x_data[0], x_data[-1], 100)
                p_fit = amprabi.data['fit_amps']
                y = fitter.sinfunc(x, *p_fit)
                
                ax.scatter(x_data, y_data, color=self.colors[qubit_test], label=f'Q{qubit_test}')
                ax.plot(x, y, label='fit', color='black')
                ax.set_xlabel('Gain')
                ax.set_ylabel('Amplitude')
                if EF:
                    ax.set_title(r'$\pi$ pulse EF', fontsize=12)
                else:
                    ax.set_title(r'$\pi$ pulse EG', fontsize=12)
                ax.text(0.6, 0.2, 'Gain: %i' % pi_gain, transform=ax.transAxes, fontsize=12)
                
                ax.legend()
                
        else:
            pi_gain = None
                
        if save:
            expt.cfg = amprabi.cfg
            expt.data['xpts'].append(amprabi.data['xpts'])
            expt.data['avgi'].append(amprabi.data['avgi'])
            expt.data['avgq'].append(amprabi.data['avgq'])
            expt.data['amps'].append(amprabi.data['amps'])
            if start_time is not None:
                expt.data['times'].append(time.time()-start_time)
            expt.data['pi_gain'].append(pi_gain)
            
        contrast = (max(amprabi.data['amps']) - min(amprabi.data['amps']))/2
            
        return pi_gain, contrast
    
    
    def measure_spectro(self, qubit_test, qubit_zz=None, debug=False, save=False, EF=False, param_exp=None, expt=None): 
        
        if qubit_zz is not None: 
            assert qubit_test != qubit_zz, 'Qubit test and qubit zz must be different'   
            
        
        if param_exp is None:
            if qubit_zz is not None:
                if EF:
                    param_exp = self.param_dict[qubit_test].zz.spectro.ef
                else:
                    param_exp = self.param_dict[qubit_test].zz.spectro.ge
                    
                if 't2' in self.param_dict[qubit_zz].keys():
                    if 'ge' in self.param_dict[qubit_zz].t2.keys():
                        param_exp_qb_zz = self.param_dict[qubit_zz].t2.ge
                else: 
                    param_exp_qb_zz = None
            else:
                if EF:
                    param_exp = self.param_dict[qubit_test].t2.ef
                else:
                    param_exp = self.param_dict[qubit_test].t2.ge 
                    
                    
        reps = param_exp.reps_spectro
        span = param_exp.span_spectro
        npts = param_exp.npts_spectro
        rounds = param_exp.rounds_spectro
        probe_length = param_exp.probe_length
        probe_gain = param_exp.probe_gain
        freq_qb_test = param_exp.freq_qb

        if qubit_zz is not None: 
            freq_qb_test = freq_qb_test[qubit_zz]
            if param_exp_qb_zz is not None:
                freq_qb_zz = param_exp_qb_zz.freq_qb
                gain_pulse = param_exp_qb_zz.gain_pulse
            else:  
                freq_qb_zz = None
                gain_pulse = None
                
            # if freq_qb_zz is None:
            #         freq_qb_zz = self.config_file.device.qubit.f_ge[qubit_zz]
            # if gain_pulse is None:
            #         gain_pulse = self.config_file.device.qubit.pulses.pi_ge.gain[qubit_zz]
                        
        # if freq_qb_test is None:
        #     if EF:
        #         freq_qb_test = self.config_file.device.qubit.f_ef[qubit_test]
        #     else:
        #         freq_qb_test = self.config_file.device.qubit.f_ge[qubit_test]
        
        
        # load experiment
        
        if qubit_zz is not None:
            if EF:
                prefix = f"qubit_EF_ZZ_spectroscopy_qubit{qubit_test}{qubit_zz}"
            else:
                prefix = f"qubit_ZZ_spectroscopy_qubit{qubit_test}{qubit_zz}"
        else: 
            if EF:
                prefix = f"qubit_EF_spectroscopy_qubit{qubit_test}"
            else:
                prefix = f"qubit_spectroscopy_qubit{qubit_test}"

            
            
        qspec = meas.PulseProbeSpectroscopyExperiment(
                    soccfg=self.rfsoc_config,
                    path=self.expt_path,
                    prefix=prefix,
                    config_file=self.config_path)
    
        # qspec.cfg = AttrDict(deepcopy(self.config_file))
        if EF:
            idx_qb = (len(self.qubits)+1)*qubit_test
            if qubit_zz is not None:
                idx_qb += (qubit_zz-qubit_test)
            if freq_qb_test is not None:
                qspec.cfg.device.qubit.f_ef[idx_qb] = freq_qb_test      
        else:
            idx_qb = (len(self.qubits)+1)*qubit_test
            if qubit_zz is not None:
                idx_qb += (qubit_zz-qubit_test)
            if freq_qb_test is not None:
                qspec.cfg.device.qubit.f_ge[idx_qb] = freq_qb_test
        
        if qubit_zz is not None:
            idx_zz = (len(self.qubits)+1)*qubit_zz
            if freq_qb_zz is not None:
                qspec.cfg.device.qubit.f_ge[idx_zz] = freq_qb_zz
            if gain_pulse is not None:
                qspec.cfg.device.qubit.pulses.pi_ge.gain[idx_zz] = gain_pulse
                
        if freq_qb_test is None:
            idx_qb = (len(self.qubits)+1)*qubit_test
            if qubit_zz is not None:
                idx_qb += (qubit_zz-qubit_test)
            if EF:
                if self.config_file.device.qubit.f_ef[idx_qb] == 0:
                    idx_qb = (len(self.qubits)+1)*qubit_test
                freq_qb_test = self.config_file.device.qubit.f_ef[idx_qb]
            else:
                if self.config_file.device.qubit.f_ge[idx_qb] == 0:
                    idx_qb = (len(self.qubits)+1)*qubit_test
                freq_qb_test = self.config_file.device.qubit.f_ge[idx_qb]

        print(freq_qb_test, span)
                 
        qspec.cfg.expt = dict(
        start=freq_qb_test-span/2,
        step=span/npts,
        expts=npts, # Number of experiments stepping from start
        reps=reps, # Number of averages per point
        rounds=rounds, # Number of start to finish sweeps to average over
        length=probe_length, # qubit 0 probe constant pulse length [us]
        gain=int(probe_gain), # pulse gain for qubit we are measuring
        pulse_type='const',
        qTest=qubit_test,
        qZZ=qubit_zz,
        checkEF=EF,
        )
        
              
        qspec.go(analyze=False, display=False, progress=debug, save=False)
                    
                    
        # ============ POST PROCESSING ============ #
        best_signs = [1, 0, 0]
        best_fit_err_i = np.inf
        best_fit_err_q = np.inf
        for sign in [1, -1]:
            try: 
                qspec.analyze(fit=True, signs=[1, sign, sign])
                fit_err_amps = qspec.data['fit_err_amps'][2][2]
                fit_err_i = qspec.data['fit_err_avgi'][2][2]
                fit_err_q = qspec.data['fit_err_avgq'][2][2]
                if fit_err_i < best_fit_err_i:
                    best_signs[1] = sign
                    best_fit_err_i = fit_err_i
                if fit_err_q < best_fit_err_q:
                    best_signs[2] = sign
                    best_fit_err_q = fit_err_q
            except: 
                pass
        if fit_err_amps == np.inf and best_fit_err_i == np.inf and best_fit_err_q == np.inf:
            qspec.display(fit=False)
            print(f'WARNING: All fits failed for Q{qubit_test} due to Q{qubit_zz} in e, please manually fix!')

        qspec.analyze(fit=True, signs=best_signs)
        qZZspec_fit, qZZspec_fit_err = meas.fitting.get_best_fit(qspec.data)
        
        if qubit_zz is not None:
            print('qubit test', qubit_test, 'qubit zz', qubit_zz)
            if EF:
                
                idx_qb = (len(self.qubits)+1)*qubit_test
                freq_bare = self.config_file.device.qubit.f_ef[idx_qb]
                freq_bare_err = 0
                
                print('EF')
                if 't2' in self.param_dict[qubit_test].keys():
                    if 'ef' in self.param_dict[qubit_test].t2.keys():
                        if self.param_dict[qubit_test].t2.ef.freq_qb is not None:
                            freq_bare = self.param_dict[qubit_test].t2.ef.freq_qb
                            freq_bare_err = self.param_dict[qubit_test].t2.ef.freq_qb_err

            else:
                print('GE')
                
                idx_qb = (len(self.qubits)+1)*qubit_test
                freq_bare = self.config_file.device.qubit.f_ge[idx_qb]
                freq_bare_err = 0
                
                if 't2' in self.param_dict[qubit_test].keys():
                    print('t2')
                    if 'ge' in self.param_dict[qubit_test].t2.keys():
                        print('ge')
                        if self.param_dict[qubit_test].t2.ge.freq_qb is not None:
                            freq_bare = self.param_dict[qubit_test].t2.ge.freq_qb
                            freq_bare_err = self.param_dict[qubit_test].t2.ge.freq_qb_err

                    
            ZZ = qZZspec_fit[2] - freq_bare
            ZZ_err = np.sqrt(qZZspec_fit_err[2][2] + freq_bare_err**2)
        
        if debug: 
            if qubit_zz is not None:
                print(f'Qubit test {qubit_test}, Qubit ZZ {qubit_zz}')
                print(f'ZZ {ZZ} +/- {ZZ_err}')
            else:
                print(f'Qubit test {qubit_test}')
                print(f'freq_qb {qZZspec_fit[2]} +/- {qZZspec_fit_err[2][2]}')
                
            
            fig, ax = plt.subplots(figsize=(5, 3))
            
            x_data = qspec.data['xpts']
            y_data = qspec.data['amps']
            x = np.linspace(x_data[0], x_data[-1], 100)
            p_fit = qspec.data['fit_amps']
            y = fitter.lorfunc(x, *p_fit)
            ax.plot(x_data, y_data, color=self.colors[qubit_test], label=f'Q{qubit_test}')
            ax.plot(x, y, label='fit', color='black')
            
            ax.set_xlabel('Frequency [GHz]')
            ax.set_ylabel('Amplitude')
            if qubit_zz is not None:
                if EF:
                    ax.set_title('ZZ spectroscopy EF', fontsize=12)
                else:
                    ax.set_title('ZZ spectroscopy', fontsize=12)
                ax.text(0.6, 0.8, 'ZZ: %.3f +/- %.3f' % (ZZ, ZZ_err), transform=ax.transAxes, fontsize=12)
            else:
                if EF:
                    ax.set_title('Spectroscopy EF', fontsize=12)
                else:
                    ax.set_title('Spectroscopy', fontsize=12)
                ax.text(0.6, 0.8, 'Freq: %.3f +/- %.3f' % (qZZspec_fit[2], qZZspec_fit_err[2][2]), transform=ax.transAxes, fontsize=12)
            ax.legend()
                
        if qubit_zz is not None:
            return ZZ, ZZ_err, qZZspec_fit[2], qZZspec_fit_err[2][2]
        
        else:
            return qZZspec_fit[2], qZZspec_fit_err[2][2]

    def measure_t2(self, start_time=None, expt=None, param_exp=None, qubit_i=None, EF=False, debug=False, save=True, accurate_mode=True):
        
        if param_exp is None:
                if EF:
                    param_exp = self.param_dict[qubit_i].t2.ef
                else:
                    param_exp = self.param_dict[qubit_i].t2.ge
                    
                        
        reps_spectro = param_exp.reps_spectro
        span_spectro = param_exp.span_spectro
        npts_spectro = param_exp.npts_spectro
        rounds_spectro = param_exp.rounds_spectro
        probe_length = param_exp.probe_length
        probe_gain = param_exp.probe_gain
        reps_ramsey = param_exp.reps_ramsey
        step_ramsey = param_exp.step_ramsey
        npts_ramsey = param_exp.npts_ramsey
        freq_ramsey = param_exp.freq_ramsey
        ramsey_round = param_exp.ramsey_round
        freq_qb = param_exp.freq_qb
        gain_pulse = param_exp.gain_pulse
  
        if freq_qb is None:
            idx_qb = (len(self.qubits)+1)*qubit_i
            if EF:
                freq_qb = self.config_file.device.qubit.f_ef[idx_qb]
            else:
                freq_qb = self.config_file.device.qubit.f_ge[idx_qb]
        
        if expt is None:
            if EF:
                expt = self.expt_t2_ef[qubit_i]
            else:
                expt = self.expt_t2_ge[qubit_i]
                
        
        # load spectroscopy
        
        # fill the dictionary with the values of the spectroscopy
        
        dict_spectro = {'reps_spectro': reps_spectro,
                        'span_spectro': span_spectro,
                        'npts_spectro': npts_spectro,
                        'rounds_spectro': rounds_spectro,
                        'probe_length': probe_length,
                        'probe_gain': probe_gain,
                        'reps_ramsey': reps_ramsey,
                        'step_ramsey': step_ramsey,
                        'npts_ramsey': npts_ramsey,
                        'freq_ramsey': freq_ramsey,
                        'ramsey_round': ramsey_round,
                        'gain_pulse': gain_pulse,
                        'freq_qb': freq_qb}
        
        dict_spectro = AttrDict(dict_spectro)
        
        freq_qb, freq_qb_err = self.measure_spectro(qubit_test=qubit_i, debug=debug, save=False, EF=EF, param_exp=dict_spectro, expt=expt)
        
        
        # fill the dictionary with the values of the ramsey
        
        if not accurate_mode:
        
            ramsey_dict = {'reps_ramsey': reps_ramsey,
                        'step_ramsey': step_ramsey,
                        'npts_ramsey': npts_ramsey,
                        'freq_ramsey': freq_ramsey,
                        'ramsey_round': ramsey_round,
                        'gain_pulse': gain_pulse,
                        'freq_qb': freq_qb}
            
            ramsey_dict = AttrDict(ramsey_dict)
            
            freq_qb, freq_qb_err, t2r_fit, t2r_fit_err = self.measure_ramsey(start_time=start_time, expt=expt, param_exp=ramsey_dict, qubit_test=qubit_i, EF=EF, debug=debug, save=save)
            
        else:
            
            ramsey_first = {'reps_ramsey': reps_ramsey,
                        'step_ramsey': int(step_ramsey/10),
                        'npts_ramsey': int(npts_ramsey),
                        'freq_ramsey': freq_ramsey*8,
                        'ramsey_round': ramsey_round,
                        'gain_pulse': gain_pulse,
                        'freq_qb': freq_qb}
            
            ramsey_first = AttrDict(ramsey_first)
            
            freq_qb, freq_qb_err, t2r_fit, t2r_fit_err = self.measure_ramsey(start_time=start_time, expt=expt, param_exp=ramsey_first, qubit_test=qubit_i, EF=EF, debug=debug, save=save)
            
            ramsey_second = {'reps_ramsey': reps_ramsey,
                        'step_ramsey': int(step_ramsey*2),
                        'npts_ramsey': int(3*npts_ramsey/2),
                        'freq_ramsey': freq_ramsey/3,
                        'ramsey_round': ramsey_round,
                        'gain_pulse': gain_pulse,
                        'freq_qb': freq_qb}
            
            if EF:
                ramsey_second['step_ramsey'] = int(step_ramsey)
            
            ramsey_second = AttrDict(ramsey_second)
            
            _freq_qb, _freq_qb_err, t2r_fit, t2r_fit_err = self.measure_ramsey(start_time=start_time, expt=expt, param_exp=ramsey_second, qubit_test=qubit_i, EF=EF, debug=debug, save=save)
            
        
        return freq_qb, freq_qb_err, t2r_fit, t2r_fit_err
  
    def measure_temp(self, qubit_i, debug=False, save=True, start_time=None, expt=None, param_exp=None):
        
        if expt is None:
            expt = self.expt_temp[qubit_i]
        
        
        if param_exp is None:
            param_exp = self.param_dict[qubit_i].temp
        
        reps = param_exp.reps
        npts = param_exp.npts
        freq_qb = param_exp.freq_qb
        rounds = param_exp.rounds
        
        if freq_qb is None:
            if self.param_dict[qubit_i].t2.ge.freq_qb is not None:
                freq_qb = self.param_dict[qubit_i].t2.ge.freq_qb*1e6
            else:
                idx_qb = (len(self.qubits)+1)*qubit_i
                freq_qb = self.config_file.device.qubit.f_ge[idx_qb]*1e6
            
        if expt is None:
            expt = self.expt_temp[qubit_i]
                        
        # no_eg pulse 
        gain_no_eg, contrast_no_eg = self.measure_pi_pulse(start_time=start_time, expt=expt, param_exp=param_exp,
                                                           qubit_test=qubit_i, EF=True, debug=debug, save=save, temp=True, pulse_ge=False)
        
        
        
        # copy the dictionary and reps and rounds
        _param_exp = param_exp.copy()
        _param_exp['reps'] = 100
        _param_exp['rounds'] = 10
        _param_exp = AttrDict(_param_exp)
                        
        
        # eg pulse
        gain_eg, contrast_eg = self.measure_pi_pulse(start_time=start_time, expt=expt, param_exp=_param_exp,
                                                     qubit_test=qubit_i, EF=True, debug=debug, save=save, temp=True, pulse_ge=True)
        
        
        
        T = -1e3*const.h*freq_qb/(const.k*np.log(contrast_no_eg/(contrast_eg + contrast_no_eg)))
        n_th = 1/(np.exp(const.h*freq_qb/(const.k*T*1e-3))-1)
        
        
        
        
        if debug:
            print(f'Qubit {qubit_i}')
            print(f'Gain no eg: {gain_no_eg}')
            print(f'Gain eg: {gain_eg}')
            print(f'Temperature: {T} mK')
            print(f'n_th: {n_th}')
            
        if save:
            expt.data['temp'].append(T)
            expt.data['pop'].append(n_th)
            
        return T
   
    def measure_error_amplification(self, qubit_test, qubit_ZZ=None, debug=False, save=True, start_time=None, expt=None, param_exp=None, pi_half=False, EF=False, divide_length=False):
        

            
        if param_exp is None:
            if qubit_ZZ is None:
                if EF: 
                    param_exp = self.param_dict[qubit_test].pi.ef
                else:
                    param_exp = self.param_dict[qubit_test].pi.ge
            else:
                if EF:
                    param_exp = self.param_dict[qubit_test].zz.pi.ef
                else:
                    param_exp = self.param_dict[qubit_test].zz.pi.ge
                if 'pi' in self.param_dict[qubit_test].keys():
                    if 'ge' in self.param_dict[qubit_test].t2.keys():
                        param_exp_zz = self.param_dict[qubit_ZZ].pi.ge
                else:
                    param_exp_zz = None

                            
        start = param_exp.err_amp_start
        step = param_exp.err_amp_step
        expts = param_exp.err_amp_expts
        reps = param_exp.err_amp_reps
        loops = param_exp.err_amp_loops
        freq_qb = param_exp.freq_qb

        if pi_half:
            test_pi_half = True
            gain = param_exp.value_half
        else:
            test_pi_half = False
            gain = param_exp.value
  
        if qubit_ZZ is not None:
            freq_qb = freq_qb[qubit_ZZ]
            gain_pulse = gain_pulse[qubit_ZZ]
            
            if param_exp_zz is not None:
                gain_pulse_zz = param_exp_zz.value
                freq_qb_zz = param_exp_zz.freq_qb
            else:
                gain_pulse_zz = None
                freq_qb_zz = None
            
            # if freq_qb_zz is None:
            #     if EF:
            #         freq_qb_zz = self.config_file.device.qubit.f_ef[qubit_ZZ]
            #     else:
            #         freq_qb_zz = self.config_file.device.qubit.f_ge[qubit_ZZ]
            
            # if gain_pulse_zz is None:
            #     if EF:
            #         gain_pulse_zz = self.config_file.device.qubit.pulses.pi_ef.gain[qubit_ZZ]
            #     else:
            #         gain_pulse_zz = self.config_file.device.qubit.pulses.pi_ge.gain[qubit_ZZ]

            
        # if freq_qb is None:
        #     if qubit_ZZ is None:
        #         if EF:
        #             freq_qb = self.config_file.device.qubit.f_ef[qubit_test]
        #         else:
        #             freq_qb = self.config_file.device.qubit.f_ge[qubit_test]
        #     else:
        #         if EF:
        #             freq_qb = self.config_file.device.qubit.f_ef[qubit_test] + self.config_file.device.qubit.ZZsEF[qubit_test, qubit_ZZ]
        #         else:
        #             freq_qb = self.config_file.device.qubit.f_ge[qubit_test] + self.config_file.device.qubit.ZZs[qubit_test, qubit_ZZ]
                    

        
        # if gain is None:
        #     print('Gain is None')
        #     if pi_half:
        #         print('pi_half')
        #         if EF:
        #             print('EF')
        #             gain = self.config_file.device.qubit.pulses.pi_ef.half_gain[qubit_test]
        #             print(gain)
        #             if gain is None:
        #                 print('gain is None')
        #                 gain = self.config_file.device.qubit.pulses.pi_ef.gain[qubit_test]//2
        #                 print(gain)
        #         else:
        #             print('GE')
        #             gain = self.config_file.device.qubit.pulses.pi_ge.half_gain[qubit_test]
        #             print(gain)
        #             if gain is None:
        #                 print('gain is None')
        #                 gain = self.config_file.device.qubit.pulses.pi_ge.gain[qubittesti]//2
        #     else:
        #         print('pi')
        #         if EF:
        #             print('EF')
        #             gain = self.config_file.device.qubit.pulses.pi_ef.gain[qubit_test]
        #             print(gain)
        #         else:
        #             print('GE')
        #             gain = self.config_file.device.qubit.pulses.pi_ge.gain[qubit_test]
                       
            
        if qubit_ZZ is None:
            if EF: 
                npulsecalib = meas.NPulseExperiment(
                soccfg=self.rfsoc_config,
                path=self.expt_path,
                prefix=f"NPulse_EF_ExptQ{qubit_test}",
                config_file=self.config_path)
            else:
                npulsecalib = meas.NPulseExperiment(
                soccfg=self.rfsoc_config,
                path=self.expt_path,
                prefix=f"NPulse_ExptQ{qubit_test}",
                config_file=self.config_path)
                
        else:
            if EF:
                npulsecalib = meas.NPulseExperiment(
                soccfg=self.rfsoc_config,
                path=self.expt_path,
                prefix=f"NPulse_ZZEF_ExptQ{qubit_test}{qubit_ZZ}",
                config_file=self.config_path)
            else:
                npulsecalib = meas.NPulseExperiment(
                soccfg=self.rfsoc_config,
                path=self.expt_path,
                prefix=f"NPulse_ZZ_ExptQ{qubit_test}{qubit_ZZ}",
                config_file=self.config_path)
                
        npulsecalib.cfg = AttrDict(deepcopy(self.config_file))
        
        
        idx_qb = (len(self.qubits)+1)*qubit_test
        if qubit_ZZ is not None:
            idx_qb += (qubit_ZZ-qubit_test)
        
 
        if EF:
            if freq_qb is not None:
                npulsecalib.cfg.device.qubit.f_ef[idx_qb] = freq_qb
            if pi_half:
                if gain is not None:
                    npulsecalib.cfg.device.qubit.pulses.pi_ef.half_gain[idx_qb] = gain
            else:
                if gain is not None:
                    npulsecalib.cfg.device.qubit.pulses.pi_ef.gain[idx_qb] = gain
                    
        else:                
            if freq_qb is not None:
                npulsecalib.cfg.device.qubit.f_ge[idx_qb] = freq_qb
            if pi_half:
                if gain is not None:
                    npulsecalib.cfg.device.qubit.pulses.pi_ge.half_gain[idx_qb] = gain
            else:
                if gain is not None:
                    npulsecalib.cfg.device.qubit.pulses.pi_ge.gain[idx_qb] = gain
            
                                
        npulsecalib.cfg.expt = dict(
            start=start,
            step=step,
            expts=expts,
            reps=reps,
            loops=loops,
            pulse_type='gauss',
            checkEF=EF,
            qTest=qubit_test,
            qZZ=qubit_ZZ,
            test_pi_half=test_pi_half,
            post_process='scale',
            singleshot_reps=15000,
            error_amp=True,
            )
    
        npulsecalib.go(analyze=False, display=False, progress=debug, save=False)
        
        
        data = npulsecalib.data
        npulsecalib.analyze(fit=True)
        npulsecalib_fit = data['fit_amps']
        npulsecalib_fit_err = data['fit_err_amps']

        
        # npulsecalib_fit, npulsecalib_fit_err = meas.fitting.get_best_fit(data)
        
        
        
        angle_err = npulsecalib_fit[1]
        angle_err_err = npulsecalib_fit_err[1][1]
        amp_ratio = (180 - angle_err)/180
        new_amp = int(np.round(gain/amp_ratio))
        
        if debug:
            
            print(f'Qubit {qubit_i}')
            print(f'Gain: {gain}')
            print(f'Angle error: {angle_err} +/- {angle_err_err}')
            print(f'New gain: {new_amp}')
            
            if test_pi_half: fit_func = fitter.probg_Xhalf
            else: fit_func = fitter.probg_X

            fig, ax = plt.subplots(figsize=(5, 3))
            x_data = data['xpts']
            y_data = data['amps']
            x = x_data
            p_fit = data['fit_amps']
            y = fit_func(x, *p_fit)
            ax.plot(x_data, y_data, color=self.colors[qubit_test], label=f'Q{qubit_test}')
            ax.plot(x, y, label='fit', color='black')
            print('x_data', x_data)
            print('y_data', y_data)
            print('p_fit', p_fit)
            print('y', y)
            if test_pi_half:
                ax.set_xlabel(r'X_{\pi/2}.X_{\pi/2}^{2n}')
            else:
                ax.set_xlabel(r'X_{\pi/2}.X_{\pi}^{n}')
            ax.set_xlabel('Gain')
            ax.set_ylabel('Amplitude')
            if EF:
                ax.set_title('Error amplification EF', fontsize=12)
            else:
                ax.set_title('Error amplification', fontsize=12)
            ax.text(0.6, 0.2, f'Gain: {new_amp}', transform=ax.transAxes, fontsize=12)
            ax.legend()
            
        return new_amp, angle_err, angle_err_err
        
    def report_value(self):
        
        expt_keys = self.expt_list
        
        # delete the zz 
        expt_keys = [key for key in expt_keys if 'zz' not in key]
        
        
        # fill a dictionary with the values of each experiment
        expt_values = {}
        
        for key in expt_keys:
            list = [None]*len(self.qubits)
            list_err = [None]*len(self.qubits)
            expt_values[key] = [list, list_err]
            
            
        # only those experiments can be performed on different subspaces
        _exp_list = ['t1', 't2', 'pi', 'freq']
      
        for key in expt_keys:
            for idxq, qubit_i in enumerate(self.qubits):
                if qubit_i in self.param_dict.keys():
                    if np.any([_exp in key for _exp in _exp_list]):
                        exp, subexp = key.split('_')
                        
                        if exp == 'freq':
                            if subexp in self.param_dict[qubit_i].t2.keys():
                                expt_values[key][0][idxq] = self.param_dict[qubit_i].t2[subexp].freq_qb
                                expt_values[key][1][idxq] = self.param_dict[qubit_i].t2[subexp].freq_qb_err
                        else:
                            if exp in self.param_dict[qubit_i].keys():
                                if subexp in self.param_dict[qubit_i][exp].keys():
                                    expt_values[key][0][idxq] = self.param_dict[qubit_i][exp][subexp].value
                                    if 'pi' not in key:
                                        expt_values[key][1][idxq] = self.param_dict[qubit_i][exp][subexp].value_err
                                    
              
                    elif 'temp' in key: 
                        for idxq, qubit_i in enumerate(self.qubits):
                            if qubit_i in self.param_dict.keys():
                                if 'temp' in self.param_dict[qubit_i].keys():
                                    expt_values[key][0][idxq] = self.param_dict[qubit_i].temp.value
 

        return expt_values
               
    def measure_all(self, live_plotting=False, debug=False, save=True, report=True, start_time=None):
    
    
        # for idxq, qubit_i in enumerate(self.qubit_watched):
            
        #     if debug: print(f'Qubit {qubit_i}')
            
        #     if 't2' in self.param_dict[qubit_i].keys():
        #         if 'ge' in self.param_dict[qubit_i].t2.keys():
        #             try: 
        #                 time_saved = False
                    
        #                 freq_qb,freq_qb_err, t2_ge, t2_ge_err = self.measure_t2(start_time, qubit_i=qubit_i, EF=False, debug=debug, save=save)
                        
                        
        #                 if freq_qb_err/freq_qb > 0.1:
        #                     print(f'Warning: GE frequency is too large for qubit {qubit_i}')
        #                     continue
        #                 else:
        #                     self.param_dict[qubit_i].t2.ge.freq_qb = freq_qb
        #                     self.param_dict[qubit_i].t2.ge.freq_qb_err = freq_qb_err 
        #                     if 't1' in self.param_dict[qubit_i].keys():
        #                         if 'ge' in self.param_dict[qubit_i].t1.keys():
        #                             self.param_dict[qubit_i].t1.ge.freq_qb = freq_qb
        #                     if 'pi' in self.param_dict[qubit_i].keys():
        #                         if 'ge' in self.param_dict[qubit_i].pi.keys():
        #                             self.param_dict[qubit_i].pi.ge.freq_qb = freq_qb
                                    
        #                     if live_plotting:
        #                         self.param_dict[qubit_i].t2.ge.time.append(time.time()-start_time)

        #                         time_saved = True
        #                         self.param_dict[qubit_i].t2.ge.freq_stored.append(freq_qb)
        #                         self.param_dict[qubit_i].t2.ge.freq_err_stored.append(freq_qb_err)
        #                         self.param_dict[qubit_i].t2.ge.stored.append(0)
        #                         self.param_dict[qubit_i].t2.ge.stored_err.append(0)
                        
        #                 if t2_ge_err/t2_ge > 0.5:
        #                     print(f'Warning: GE t2 error is too large for qubit {qubit_i}')
        #                     continue
        #                 else:       
        #                     self.param_dict[qubit_i].t2.ge.value = t2_ge
        #                     self.param_dict[qubit_i].t2.ge.value_err = t2_ge_err
                                                        
                            
                            
        #                     if live_plotting:

        #                         if not time_saved:
        #                             self.param_dict[qubit_i].t2.ge.time.append(time.time()-start_time)
        #                             self.param_dict[qubit_i].t2.ge.freq_stored.append(0)
        #                             self.param_dict[qubit_i].t2.ge.freq_err_stored.append(0)
        #                             self.param_dict[qubit_i].t2.ge.stored.append(t2_ge)
        #                             self.param_dict[qubit_i].t2.ge.stored_err.append(t2_ge_err)
        #                         else:
        #                             self.param_dict[qubit_i].t2.ge.stored[-1] = t2_ge
        #                             self.param_dict[qubit_i].t2.ge.stored_err[-1] = t2_ge_err

                    

        #             except Exception as e:
        #                 print(f'Error in T2 GE: {e}')
                    
        #         if 'ef' in self.param_dict[qubit_i].t2.keys():
        #             try:
        #                 time_saved = False
        #                 freq_qb, freq_qb_err, t2_ef, t2_ef_err = self.measure_t2(start_time, qubit_i=qubit_i, EF=True, debug=debug, save=save)
                        
                        
        #                 if freq_qb_err/freq_qb > 0.1:
        #                     print(f'Warning: EF frequency error is too large for qubit {qubit_i}')
        #                     continue
        #                 else:
        #                     self.param_dict[qubit_i].t2.ef.freq_qb = freq_qb
        #                     self.param_dict[qubit_i].t2.ef.freq_qb_err = freq_qb_err
                            
        #                     if 't1' in self.param_dict[qubit_i].keys():
        #                         if 'ef' in self.param_dict[qubit_i].t1.keys():
        #                             self.param_dict[qubit_i].t1.ef.freq_qb = freq_qb
        #                     if 'pi' in self.param_dict[qubit_i].keys():
        #                         if 'ef' in self.param_dict[qubit_i].pi.keys():
        #                             self.param_dict[qubit_i].pi.ef.freq_qb = freq_qb
                                    
        #                     if live_plotting:
        #                         self.param_dict[qubit_i].t2.ef.time.append(time.time()-start_time)
        #                         time_saved = True
        #                         self.param_dict[qubit_i].t2.ef.freq_stored.append(freq_qb)
        #                         self.param_dict[qubit_i].t2.ef.freq_err_stored.append(freq_qb_err)
        #                         self.param_dict[qubit_i].t2.ef.stored.append(0)
        #                         self.param_dict[qubit_i].t2.ef.stored_err.append(0)
                    
                        
        #                 if t2_ef_err/t2_ef > 0.5:
        #                     print(f'Warning: EF t2 error is too large for qubit {qubit_i}')
        #                     continue
        #                 else:  
        #                     self.param_dict[qubit_i].t2.ef.value = t2_ef
        #                     self.param_dict[qubit_i].t2.ef.value_err = t2_ef_err
                            
        #                     if live_plotting:
        #                         if not time_saved:
        #                             self.param_dict[qubit_i].t2.ef.time.append(time.time()-start_time)
        #                             self.param_dict[qubit_i].t2.ef.freq_stored.append(0)
        #                             self.param_dict[qubit_i].t2.ef.freq_err_stored.append(0)
        #                             self.param_dict[qubit_i].t2.ef.stored.append(t2_ef)
        #                             self.param_dict[qubit_i].t2.ef.stored_err.append(t2_ef_err)
        #                         else:
        #                             self.param_dict[qubit_i].t2.ef.stored[-1] = t2_ef
        #                             self.param_dict[qubit_i].t2.ef.stored_err[-1] = t2_ef_err
     
        #             except Exception as e:
        #                 print(f'Error in T2 EF: {e}')
                
        #     if 'pi' in self.param_dict[qubit_i].keys():
        #         if 'ge' in self.param_dict[qubit_i].pi.keys():
        #             try:
        #                 pi_gain, constrast = self.measure_pi_pulse(start_time, qubit_test=qubit_i, EF=False, debug=debug, save=save)
                        
        #                 self.param_dict[qubit_i].pi.ge.value = pi_gain
        #                 if 't2' in self.param_dict[qubit_i].keys():
        #                     if 'ge' in self.param_dict[qubit_i].t2.keys():
        #                         self.param_dict[qubit_i].t2.ge.pi_gain = pi_gain
        #                 if 't1' in self.param_dict[qubit_i].keys():
        #                     if 'ge' in self.param_dict[qubit_i].t1.keys():
        #                         self.param_dict[qubit_i].t1.ge.pi_gain = pi_gain
                        
        #                 if live_plotting:
        #                     self.param_dict[qubit_i].pi.ge.stored.append(pi_gain)
        #                     self.param_dict[qubit_i].pi.ge.time.append(time.time()-start_time)
                            
                            
        #             except Exception as e:
        #                 print(f'Error in pi pulse GE: {e}')
                    
        #         if 'ef' in self.param_dict[qubit_i].pi.keys():
        #             try:
        #                 pi_gain, constrast = self.measure_pi_pulse(start_time, qubit_test=qubit_i, EF=True, debug=debug, save=save)
                        
        #                 self.param_dict[qubit_i].pi.ef.value = pi_gain
        #                 if 't2' in self.param_dict[qubit_i].keys():
        #                     if 'ef' in self.param_dict[qubit_i].t2.keys():
        #                         self.param_dict[qubit_i].t2.ef.pi_gain = pi_gain
        #                 if 't1' in self.param_dict[qubit_i].keys():
        #                     if 'ef' in self.param_dict[qubit_i].t1.keys():
        #                         self.param_dict[qubit_i].t1.ef.pi_gain = pi_gain
                        
        #                 if live_plotting:
        #                     self.param_dict[qubit_i].pi.ef.stored.append(pi_gain)
        #                     self.param_dict[qubit_i].pi.ef.time.append(time.time()-start_time)
                            
                        
        #             except Exception as e:
        #                 print(f'Error in pi pulse EF: {e}')
                    
        #     if 't1' in self.param_dict[qubit_i].keys():
        #         if 'ge' in self.param_dict[qubit_i].t1.keys():
        #             try:
        #                 t1_ge, t1_ge_err = self.measure_t1(start_time, qubit_test=qubit_i, EF=False, debug=debug, save=save)
                        
        #                 if t1_ge_err/t1_ge > 0.5:
        #                     print(f'Warning: T1GE error is too large for qubit {qubit_i}')
        #                 else:
        #                     self.param_dict[qubit_i].t1.ge.value = t1_ge
        #                     self.param_dict[qubit_i].t1.ge.value_err = t1_ge_err
                            
        #                     if live_plotting:
        #                         self.param_dict[qubit_i].t1.ge.stored.append(t1_ge)
        #                         self.param_dict[qubit_i].t1.ge.stored_err.append(t1_ge_err)
        #                         self.param_dict[qubit_i].t1.ge.time.append(time.time()-start_time)


        #             except Exception as e:
        #                 print(f'Error in T1 GE: {e}')
                    
        #         if 'ef' in self.param_dict[qubit_i].t1.keys():
        #             try:
        #                 t1_ef, t1_ef_err = self.measure_t1(start_time, qubit_test=qubit_i, EF=True, debug=debug, save=save)
                        
        #                 if t1_ef_err/t1_ef > 0.5:
        #                     print(f'Warning: T1EF error is too large for qubit {qubit_i}')
        #                 else:
        #                     self.param_dict[qubit_i].t1.ef.value = t1_ef
        #                     self.param_dict[qubit_i].t1.ef.value_err = t1_ef_err
                            
        #                     if live_plotting:
        #                         self.param_dict[qubit_i].t1.ef.stored.append(t1_ef)
        #                         self.param_dict[qubit_i].t1.ef.stored_err.append(t1_ef_err)
        #                         self.param_dict[qubit_i].t1.ef.time.append(time.time()-start_time)

        #             except Exception as e: 
        #                 print(f'Error in T1 EF: {e}')
                        
                        
        #     if 'temp' in self.param_dict[qubit_i].keys():
        #         try:
        #             T = self.measure_temp(qubit_i, debug=debug, save=save, start_time=start_time)
                    
        #             self.param_dict[qubit_i].temp.value = T
                    
                    
        #             if live_plotting:
        #                 self.param_dict[qubit_i].temp.stored.append(self.param_dict[qubit_i].temp.value)
        #                 self.param_dict[qubit_i].temp.time.append(time.time()-start_time)
                        
        #         except Exception as e:
        #             print(f'Error in temperature: {e}')
                        
                        
                
                        
        
        if report:     
            report = self.report_value()                        
            print('-------------- measured parameters -----------------:')
        
            for key in report.keys():
                print(f'Qubit {key}: {report[key]}')
   
            print('--------------- yaml config parameters -------------:') 
            
            f_ge = np.array(self.config_file.device.qubit.f_ge)
            f_ef = np.array(self.config_file.device.qubit.f_ef)
            pi_ge = np.array(self.config_file.device.qubit.pulses.pi_ge.gain)
            pi_ef = np.array(self.config_file.device.qubit.pulses.pi_ef.gain)

            idx_print = (len(self.qubits)+1)*np.array(self.qubits)

            
            
            
            
            
            print('Qubit ge frequencies', f_ge[idx_print])
            print('Qubit ef frequencies', f_ef[idx_print])
            print('Qubit pi eg gains', pi_ge[idx_print])
            print('Qubit pi ef gains', pi_ef[idx_print])

    def measure_all_ZZ(self, debug=False, save=True, qubits_test=None, qubits_zz=None):
        
        if qubits_test is None:
            qubits_test = self.qubit_watched
        if qubits_zz is None:
            qubits_zz = self.qubit_watched


        
        for idxq, qubit_test in enumerate(self.qubit_watched):
            # check if the qubit is in the list of qubits to be tested
            if qubit_test not in qubits_test:
                continue
            for idxq2, qubit_zz in enumerate(self.qubit_watched):
                # check if the qubit is in the list of qubits to be tested
                if qubit_zz not in qubits_zz:
                    continue
                
                if qubit_zz == qubit_test:
                    continue
                
                if 'spectro' in self.param_dict[qubit_test].zz.keys():
                    if 'ge' in self.param_dict[qubit_test].zz.spectro.keys():
                        
                        print(f'Qubit test {qubit_test}, Qubit ZZ {qubit_zz}')
                        print('spectro GE')
 
                        # try:
                        ZZ, ZZ_err, freq_qb, freq_qb_err = self.measure_spectro(qubit_test=qubit_test, qubit_zz=qubit_zz, debug=debug, save=save, EF=False)
                        # append the values to the parameters

                        self.ZZ_mat.ge.value[qubit_test, qubit_zz] = ZZ
                        self.ZZ_mat.ge.value_err[qubit_test, qubit_zz] = ZZ_err
                        print('qubit test', qubit_test)
                        print('qubit zz', qubit_zz)
                        self.param_dict[qubit_test].zz.spectro.ge.freq_qb[qubit_zz] = freq_qb
                        self.param_dict[qubit_test].zz.spectro.ge.freq_qb_err[qubit_zz] = freq_qb_err
                        
                        if 'ramsey' in self.param_dict[qubit_test].zz.keys():
                            if 'ge' in self.param_dict[qubit_test].zz.ramsey.keys():
                                self.param_dict[qubit_test].zz.ramsey.ge.freq_qb[qubit_zz] = freq_qb
                                

                                
                                
                                
                        if 'pi' in self.param_dict[qubit_test].zz.keys():
                            if 'ge' in self.param_dict[qubit_test].zz.pi.keys():
                                self.param_dict[qubit_test].zz.pi.ge.freq_qb[qubit_zz] = freq_qb
                                

                        if debug:
                            print(f'Qubit test {qubit_test}, Qubit ZZ {qubit_zz}')
                            print('spectro GE')
                            print(f'freq_qb: {freq_qb} +/- {freq_qb_err}')
                            print(f'ZZ: {ZZ} +/- {ZZ_err}')
                                                

                        # except Exception as e:
                        #     print(f'Error in ZZ GE: {e}')

                    if 'ef' in self.param_dict[qubit_test].zz.spectro.keys():
                        # try:
                        ZZ, ZZ_err, freq_qb, freq_qb_err = self.measure_spectro(qubit_test=qubit_test, qubit_zz=qubit_zz, debug=debug, save=save, EF=True)
                        
                        self.ZZ_mat.ef.value[qubit_test, qubit_zz] = ZZ
                        self.ZZ_mat.ef.value_err[qubit_test, qubit_zz] = ZZ_err
                        self.param_dict[qubit_test].zz.spectro.ef.freq_qb[qubit_zz] = freq_qb
                        self.param_dict[qubit_test].zz.spectro.ef.freq_qb_err[qubit_zz] = freq_qb_err

                        
                        
                        if 'ramsey' in self.param_dict[qubit_test].zz.keys():
                            if 'ef' in self.param_dict[qubit_test].zz.ramsey.keys():
                                self.param_dict[qubit_test].zz.ramsey.ef.freq_qb[qubit_zz] = freq_qb

                                
                        if 'pi' in self.param_dict[qubit_test].zz.keys():
                            if 'ef' in self.param_dict[qubit_test].zz.pi.keys():
                                self.param_dict[qubit_test].zz.pi.ef.freq_qb[qubit_zz] = freq_qb
    
                                
                        if debug:
                            print(f'Qubit test {qubit_test}, Qubit ZZ {qubit_zz}')
                            print('spectro EF')
                            print(f'freq_qb: {freq_qb} +/- {freq_qb_err}')
                            print(f'ZZ: {ZZ} +/- {ZZ_err}')

                        # except Exception as e:
                        #     print(f'Error in ZZ EF: {e}')          
                            
                if 'ramsey' in self.param_dict[qubit_test].zz.keys():
                    if 'ge' in self.param_dict[qubit_test].zz.ramsey.keys():
                        # try: 
                        freq_qb, freq_err, t2, t2_err = self.measure_ramsey(qubit_test=qubit_test, qubit_ZZ=qubit_zz, EF=False, debug=debug, save=save)
                        
                        idx_qb = (len(self.qubits)+1)*qubit_test
                        freq_bare = self.config_file.device.qubit.f_ge[idx_qb]
                        freq_bare_err = 0
                    
                        if 't2' in self.param_dict[qubit_test].keys():
                            if self.param_dict[qubit_test].t2.ge.freq_qb is not None:
                                freq_bare = self.param_dict[qubit_test].t2.ge.freq_qb
                                freq_bare_err = self.param_dict[qubit_test].t2.ge.freq_qb_err

                        ZZ = freq_qb - freq_bare
                        ZZ_err = np.sqrt(freq_err**2 + freq_bare_err**2)
                        
                        self.ZZ_mat.ge.value[qubit_test, qubit_zz] = ZZ
                        self.ZZ_mat.ge.value_err[qubit_test, qubit_zz] = ZZ_err
                        self.param_dict[qubit_test].zz.ramsey.ge.freq_qb[qubit_zz] = freq_qb
                        self.param_dict[qubit_test].zz.ramsey.ge.freq_qb_err[qubit_zz] = freq_qb_err
                        
                        
                        if 'pi' in self.param_dict[qubit_test].zz.keys():
                            if 'ge' in self.param_dict[qubit_test].zz.pi.keys():
                                self.param_dict[qubit_test].zz.pi.ge.freq_qb[qubit_zz] = freq_qb
                                
                        
                        if debug:
                            print(f'Qubit test {qubit_test}, Qubit ZZ {qubit_zz}')
                            print('ramsey GE')
                            print(f'freq_qb: {freq_qb} +/- {freq_err}')
                            print(f'freq_bare: {freq_bare} +/- {freq_bare_err}')
                            print(f'ZZ: {ZZ} +/- {ZZ_err}')
                            
                #         # except Exception as e:
                #         #     print(f'Error in ZZ GE ramsey: {e}')
                            
                    if 'ef' in self.param_dict[qubit_test].zz.ramsey.keys():
                        # try:
                        freq_qb, freq_err, t2, t2_err = self.measure_ramsey(qubit_test=qubit_test, qubit_ZZ=qubit_zz, EF=True, debug=debug, save=save)
                        
                        idx_qb = (len(self.qubits)+1)*qubit_test
                        freq_bare = self.config_file.device.qubit.f_ef[idx_qb]
                        freq_bare_err = 0
                        
                        
                        if 't2' in self.param_dict[qubit_test].keys():
                            if self.param_dict[qubit_test].t2.ef.freq_qb is not None:
                                freq_bare = self.param_dict[qubit_test].t2.ef.freq_qb
                                freq_bare_err = self.param_dict[qubit_test].t2.ef.freq_qb_err

                        ZZ = freq_qb - freq_bare
                        ZZ_err = np.sqrt(freq_err**2 + freq_bare_err**2)
                        
                        if debug:
                            print(f'Qubit test {qubit_test}, Qubit ZZ {qubit_zz}')
                            print('ramsey EF')
                            print(f'freq_qb: {freq_qb} +/- {freq_err}')
                            print(f'freq_bare: {freq_bare} +/- {freq_bare_err}')
                            print(f'ZZ: {ZZ} +/- {ZZ_err}')
                        
                        
                        self.ZZ_mat.ef.value[qubit_test, qubit_zz] = ZZ
                        self.ZZ_mat.ef.value_err[qubit_test, qubit_zz] = ZZ_err

                        self.param_dict[qubit_test].zz.ramsey.ef.freq_qb[qubit_zz] = freq_qb
                        self.param_dict[qubit_test].zz.ramsey.ef.freq_qb_err[qubit_zz] = freq_qb_err
                        

                        if 'pi' in self.param_dict[qubit_test].zz.keys():
                            if 'ef' in self.param_dict[qubit_test].zz.pi.keys():
                                self.param_dict[qubit_test].zz.pi.ef.freq_qb[qubit_zz] = freq_qb
                        
                        
                        
                            
                        # except Exception as e:
                        #     print(f'Error in ZZ EF ramsey: {e}')
                            
                if 'pi': 
                    if 'ge' in self.param_dict[qubit_test].zz.pi.keys():
                        try:
                            pi_gain, constrast = self.measure_pi_pulse(start_time=None, expt=None, param_exp=None, qubit_test=qubit_test, EF=False, debug=debug, save=save, qubit_ZZ=qubit_zz)
                            self.param_dict[qubit_test].zz.ramsey.ge.gain_pulse[qubit_zz] = pi_gain
                            
                            if 'spectro' in self.param_dict[qubit_test].zz.keys():
                                if 'ge' in self.param_dict[qubit_test].zz.spectro.keys():
                                    self.param_dict[qubit_test].zz.spectro.ge.gain_pulse[qubit_zz] = pi_gain
                            if 'ramsey' in self.param_dict[qubit_test].zz.keys():
                                if 'ge' in self.param_dict[qubit_test].zz.ramsey.keys():
                                    self.param_dict[qubit_test].zz.ramsey.ge.gain_pulse[qubit_zz] = pi_gain

                            
                        except Exception as e:
                            print(f'Error in pi pulse GE: {e}')
                        
                        
                        
                    if 'ef' in self.param_dict[qubit_test].zz.pi.keys():
                        try:
                            pi_gain, constrast = self.measure_pi_pulse(start_time=None, expt=None, param_exp=None, qubit_test=qubit_test, EF=True, debug=debug, save=save, qubit_ZZ=qubit_zz)
                            self.param_dict[qubit_test].zz.ramsey.ef.gain_pulse[qubit_zz] = pi_gain
                            # update the value of the pi pulse in the pi pulse dictionary
                            if 'spectro' in self.param_dict[qubit_test].zz.keys():
                                if 'ef' in self.param_dict[qubit_test].zz.spectro.keys():
                                    self.param_dict[qubit_test].zz.spectro.ef.gain_pulse[qubit_zz] = pi_gain
                            if 'ramsey' in self.param_dict[qubit_test].zz.keys():
                                if 'ef' in self.param_dict[qubit_test].zz.ramsey.keys():
                                    self.param_dict[qubit_test].zz.ramsey.ef.gain_pulse[qubit_zz] = pi_gain

                        except Exception as e:
                            print(f'Error in pi pulse EF: {e}')
                
                  
        # print the values of the ZZ matrix
        print('ZZ matrix GE')
        print(self.ZZ_mat.ge.value)
        print('ZZ matrix GE error')
        print(self.ZZ_mat.ge.value_err)
        print('ZZ matrix EF')
        print(self.ZZ_mat.ef.value)
        print('ZZ matrix EF error')
        print(self.ZZ_mat.ef.value_err)
        if 'pi' in self.param_dict[qubit_test].zz.keys():
            if 'ge' in self.param_dict[qubit_test].zz.pi.keys():
                print('pi pulse gain GE')
                print(self.param_dict[qubit_test].zz.pi.ge)
        if 'pi' in self.param_dict[qubit_test].zz.keys():
            if 'ef' in self.param_dict[qubit_test].zz.pi.keys():
                print('pi pulse gain EF')
                print(self.param_dict[qubit_test].zz.pi.ef)
             
    def reset_values(self):
        
        # reset the stored value and time for each experiment
        
        expt_list = self.expt_list
        # remove the ZZ experiments
        expt_keys = [key for key in expt_list if 'zz' not in key]
            
        # only those experiments can be performed on different subspaces
        _exp_list = ['t1', 't2', 'pi']
      
        for key in expt_keys:
            for idxq, qubit_i in enumerate(self.qubits):
                if qubit_i in self.param_dict.keys():
                    if np.any([_exp in key for _exp in _exp_list]):
                        exp, subexp = key.split('_')
                        if exp in self.param_dict[qubit_i].keys():
                            if subexp in self.param_dict[qubit_i][exp].keys():
                                self.param_dict[qubit_i][exp][subexp].stored = []
                                self.param_dict[qubit_i][exp][subexp].time = []
                                if 'pi' not in key:
                                    self.param_dict[qubit_i][exp][subexp].stored_err = []
                                if 't2' in key:
                                    self.param_dict[qubit_i][exp][subexp].freq_stored = []
                                    self.param_dict[qubit_i][exp][subexp].freq_err_stored = []
                            
                    elif 'temp' in key: 
                        for idxq, qubit_i in enumerate(self.qubits):
                            if qubit_i in self.param_dict.keys():
                                if 'temp' in self.param_dict[qubit_i].keys():
                                    self.param_dict[qubit_i].temp.stored = []
                                    self.param_dict[qubit_i].temp.time = []

    def monitor_all(self, debug=False, save=False, reset=True):
        
        # first reset the values of the parameters
        
        if reset:
            self.reset_values()
        start_time = time.time()

        try: 
            while True:
                self.measure_all(live_plotting=True, debug=debug, save=False, report=False, start_time=start_time)
                self.update_plot()
                
                
        except KeyboardInterrupt:
            print('Monitoring stopped')
            report = self.report_value()

            print('-------------- measured parameters -----------------:')
        
            for key in report.keys():
                print(f'Qubit {key}: {report[key]}')
   
            print('--------------- yaml config parameters -------------:')  
            print('Qubit ge frequencies', self.config_file.device.qubit.f_ge)
            print('Qubit ef frequencies', self.config_file.device.qubit.f_ef)
            print('Qubit pi eg gains', self.config_file.device.qubit.pulses.pi_ge.gain)
            print('Qubit pi ef gains', self.config_file.device.qubit.pulses.pi_ef.gain)
            
            # save the values of the parameters in a npz file
            if save:
                for qubit_i in self.qubits:
                    data_i = {}
                    if 't2' in self.param_dict[qubit_i].keys():
                        if 'ge' in self.param_dict[qubit_i].t2.keys():
                            data_i['t2_ge'] = self.param_dict[qubit_i].t2.ge.stored
                            data_i['t2_ge_err'] = self.param_dict[qubit_i].t2.ge.stored_err
                            data_i['freq_ge'] = self.param_dict[qubit_i].t2.ge.freq_stored
                            data_i['freq_ge_err'] = self.param_dict[qubit_i].t2.ge.freq_err_stored
                            data_i['time_t2_ge'] = self.param_dict[qubit_i].t2.ge.time
                        if 'ef' in self.param_dict[qubit_i].t2.keys():
                            data_i['t2_ef'] = self.param_dict[qubit_i].t2.ef.stored
                            data_i['t2_ef_err'] = self.param_dict[qubit_i].t2.ef.stored_err
                            data_i['freq_ef'] = self.param_dict[qubit_i].t2.ef.freq_stored
                            data_i['freq_ef_err'] = self.param_dict[qubit_i].t2.ef.freq_err_stored
                            data_i['time_t2_ef'] = self.param_dict[qubit_i].t2.ef.time
                    if 't1' in self.param_dict[qubit_i].keys():
                        if 'ge' in self.param_dict[qubit_i].t1.keys():
                            data_i['t1_ge'] = self.param_dict[qubit_i].t1.ge.stored
                            data_i['t1_ge_err'] = self.param_dict[qubit_i].t1.ge.stored_err
                            data_i['time_t1_ge'] = self.param_dict[qubit_i].t1.ge.time
                        if 'ef' in self.param_dict[qubit_i].t1.keys():
                            data_i['t1_ef'] = self.param_dict[qubit_i].t1.ef.stored
                            data_i['t1_ef_err'] = self.param_dict[qubit_i].t1.ef.stored_err
                            data_i['time_t1_ef'] = self.param_dict[qubit_i].t1.ef.time
                    if 'pi' in self.param_dict[qubit_i].keys():
                        if 'ge' in self.param_dict[qubit_i].pi.keys():
                            data_i['pi_ge'] = self.param_dict[qubit_i].pi.ge.stored
                            data_i['time_pi_ge'] = self.param_dict[qubit_i].pi.ge.time
                        if 'ef' in self.param_dict[qubit_i].pi.keys():
                            data_i['pi_ef'] = self.param_dict[qubit_i].pi.ef.stored
                            data_i['time_pi_ef'] = self.param_dict[qubit_i].pi.ef.time
                    if 'temp' in self.param_dict[qubit_i].keys():
                        data_i['temp'] = self.param_dict[qubit_i].temp.stored
                        data_i['time_temp'] = self.param_dict[qubit_i].temp.time
                        
                    # save the data in a npz file in the save folder
                    data_folder = 'data/' + self.save_path
                    np.savez(data_folder + f'/qubit_{qubit_i}_data', **data_i)
                        
    # def pulse_cablibration(self, qubits_test, pi_half, debug=False, save=True):    
        
           
                    


            

            



            
            
            

   
                        
        

        
        
    

        
        

