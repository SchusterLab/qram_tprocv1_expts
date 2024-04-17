import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm
from datetime import datetime
import experiments.fitting as fitter


class T1Program(RAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
    
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)

        self.qubits = self.cfg.expt.qubits
        qA, qB = self.qubits

        # Declare resonatr ADCs
        self.adc_chA = cfg.hw.soc.adcs.readout.ch[qA]
        self.adc_chB = cfg.hw.soc.adcs.readout.ch[qB]

        # Declare qubit/ resonator DACs
        self.res_chA = cfg.hw.soc.dacs.readout.ch[qA]
        self.res_chB = cfg.hw.soc.dacs.readout.ch[qB]

        self.res_ch_type = cfg.hw.soc.dacs.readout.type
        self.qubit_chA = cfg.hw.soc.dacs.qubit.ch[qA]
        self.qubit_chB = cfg.hw.soc.dacs.qubit.ch[qA]

        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type

        self.q_rpA = self.ch_page(self.qubit_chA) # get register page for qubit_ch
        self.q_rpB = self.ch_page(self.qubit_chB) # get register page for qubit_ch
        self.r_wait = 3
        self.safe_regwi(self.q_rpA, self.r_wait, self.us2cycles(cfg.expt.startA))
        self.safe_regwi(self.q_rpB, self.r_wait, self.us2cycles(cfg.expt.startB))
        
        self.f_geA = self.freq2reg(cfg.device.qubit.f_ge[qA], gen_ch=self.qubit_chA)
        self.f_geB = self.freq2reg(cfg.device.qubit.f_ge[qB], gen_ch=self.qubit_chB)

        self.f_res_regA = self.freq2reg(cfg.device.readout.frequency[qA], gen_ch=self.res_chA, ro_ch=self.adc_chA)
        self.f_res_regB = self.freq2reg(cfg.device.readout.frequency[qB], gen_ch=self.res_chB, ro_ch=self.adc_chB)

        self.readout_length_dacA = self.us2cycles(cfg.device.readout.readout_length[qA], gen_ch=self.res_chA)
        self.readout_length_dacB = self.us2cycles(cfg.device.readout.readout_length[qB], gen_ch=self.res_chB)


        self.readout_length_adcA = self.us2cycles(cfg.device.readout.readout_length[qA], ro_ch=self.adc_chA)
        self.readout_length_adcB = self.us2cycles(cfg.device.readout.readout_length[qB], ro_ch=self.adc_chB)


        self.readout_length_adcA += 1 # ensure the rounding of the clock ticks calculation doesn't mess up the buffer
        self.readout_length_adcB += 1 # ensure the rounding of the clock ticks calculation doesn't mess up the buffer

        # declare res dacs
        mask = None
        mixer_freq = 0 # MHz
        mux_freqs = None # MHz
        mux_gains = None
        ro_chA = self.adc_chA
        ro_chB = self.adc_chB

        if self.res_ch_type == 'int4':
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
        elif self.res_ch_type == 'mux4':
            assert self.res_ch == 6
            mask = [0, 1, 2, 3] # indices of mux_freqs, mux_gains list to play
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
            mux_freqs = [0]*4
            mux_freqs[cfg.expt.qubit] = cfg.device.readout.frequency
            mux_gains = [0]*4
            mux_gains[cfg.expt.qubit] = cfg.device.readout.gain
        
        self.declare_gen(ch=self.res_chA, nqz=cfg.hw.soc.dacs.readout.nyquist[qA], mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_chA)
        self.declare_gen(ch=self.res_chB, nqz=cfg.hw.soc.dacs.readout.nyquist[qB], mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_chB)

        # declare qubit dacs
        mixer_freq = 0
        if self.qubit_ch_type == 'int4':
            mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq
        self.declare_gen(ch=self.qubit_chA, nqz=cfg.hw.soc.dacs.qubit.nyquist[qA], mixer_freq=mixer_freq)
        self.declare_gen(ch=self.qubit_chB, nqz=cfg.hw.soc.dacs.qubit.nyquist[qB], mixer_freq=mixer_freq)

        # declare adcs
        self.declare_readout(ch=self.adc_chA, length=self.readout_length_adcA, freq=cfg.device.readout.frequency[qA], gen_ch=self.res_chA)
        self.declare_readout(ch=self.adc_chB, length=self.readout_length_adcB, freq=cfg.device.readout.frequency[qB], gen_ch=self.res_chB)

        self.pi_sigmaA = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qA], gen_ch=self.qubit_chA)
        self.pi_sigmaB = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qB], gen_ch=self.qubit_chB)

        # add qubit and readout pulses to respective channels
        if self.cfg.device.qubit.pulses.pi_ge.type[qA] == 'gauss':
            self.add_gauss(ch=self.qubit_chA, name="pi_qubitA", sigma=self.pi_sigmaA, length=self.pi_sigmaA*4)
            self.set_pulse_registers(ch=self.qubit_chA, style="arb", freq=self.f_geA, phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain[qA], waveform="pi_qubitA")


            self.add_gauss(ch=self.qubit_chB, name="pi_qubitB", sigma=self.pi_sigmaB, length=self.pi_sigmaB*4)
            self.set_pulse_registers(ch=self.qubit_chB, style="arb", freq=self.f_geB, phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain[qB], waveform="pi_qubitB")
        else:
            self.set_pulse_registers(ch=self.qubit_chA, style="const", freq=self.f_geA, phase=0, gain=cfg.expt.startA, length=self.pi_sigmaA)
            self.set_pulse_registers(ch=self.qubit_chB, style="const", freq=self.f_geB, phase=0, gain=cfg.expt.startB, length=self.pi_sigmaB)


        
        self.set_pulse_registers(ch=self.res_chA, style="const", freq=self.f_res_regA, phase=0, gain=cfg.device.readout.gain[qA], length=self.readout_length_dacA)
        self.set_pulse_registers(ch=self.res_chB, style="const", freq=self.f_res_regB, phase=0, gain=cfg.device.readout.gain[qB], length=self.readout_length_dacB)

        self.sync_all(200)

    def body(self):
        cfg=AttrDict(self.cfg)
        self.pulse(ch=self.qubit_chA)
        self.pulse(ch=self.qubit_chB)

        self.qubits = self.cfg.expt.qubits
        qA, qB = self.qubits

        if self.cfg.expt.startA < self.cfg.expt.startB:
            self.sync(self.q_rpA, self.r_wait) #self.us2cycles(self.cfg.expt.startA)
            self.pulse(ch=self.res_chA, t=0 )
            self.sync(self.q_rpB, self.us2cycles(self.cfg.expt.startB - self.cfg.expt.startA))
            self.pulse(ch=self.res_chB, t=0 )

        else: 
            self.sync(self.q_rpB, self.us2cylces(self.cfg.expt.startB))
            self.pulse(ch=self.res_chB, t=0 )
            self.sync(self.q_rpA, self.us2cycles(self.cfg.expt.startA - self.cfg.expt.startB))
            self.pulse(ch=self.res_chA, t=0 )


        self.trigger([self.adc_chA], pins=None, adc_trig_offset= self.us2cycles(cfg.device.readout.trig_offset[qA]))
        self.trigger([self.adc_chB], pins=None, adc_trig_offset=self.us2cycles(cfg.device.readout.trig_offset[qB]))

       
        self.wait_all()
        sync_delay = self.us2cycles(max([cfg.device.readout.relax_delay[qA],cfg.device.readout.relax_delay[qB]]))
        self.sync_all(self, sync_delay)

        # self.sync(self.q_rp, self.r_wait) # wait for the time stored in the wait variable register


        #self.measure(pulse_ch=[self.res_chA, self.res_chB], 
        #      adcs=[self.adc_chA, self.adc_chB],
        #      adc_trig_offset=cfg.device.readout.trig_offset[qA],
        #      wait=True,
        #      syncdelay=self.us2cycles(max([cfg.device.readout.relax_delay[qA],cfg.device.readout.relax_delay[qB]])))
        
    def collect_shots(self):
        # collect shots for the relevant adc and I and Q channels
        cfg=AttrDict(self.cfg)
        shots_i0A = self.di_buf[0] / self.readout_length_adcA #[self.cfg.expt.qubit]
        shots_q0A = self.dq_buf[0] / self.readout_length_adcA #[self.cfg.expt.qubit]
        shots_i0B = self.di_buf[1] / self.readout_length_adcB #[self.cfg.expt.qubit]
        shots_q0B = self.dq_buf[1] / self.readout_length_adcB #[self.cfg.expt.qubit]
        return shots_i0A, shots_q0A, shots_i0B, shots_q0B
    
    def update(self):
        self.mathi(self.q_rpA, self.r_wait, self.r_wait, '+', self.us2cycles(self.cfg.expt.step)) # update wait time
        self.mathi(self.q_rpB, self.r_wait, self.r_wait, '+', self.us2cycles(self.cfg.expt.step)) # update wait time

# ====================================================== #
class T1_2qbExperiment(Experiment):
    """
    T1 Experiment
    Experimental Config:
    expt = dict(
        start: wait time sweep start [us]
        step: wait time sweep step
        expts: number steps in sweep
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
    )
    """

    def __init__(self, soccfg=None, path='', prefix='T1', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        # q_ind = self.cfg.expt.qubit
        # for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
        #     for key, value in subcfg.items() :
        #         if isinstance(value, list):
        #             subcfg.update({key: value[q_ind]})
        #         elif isinstance(value, dict):
        #             for key2, value2 in value.items():
        #                 for key3, value3 in value2.items():
        #                     if isinstance(value3, list):
        #                         value2.update({key3: value3[q_ind]})                                

        t1 = T1Program(soccfg=self.soccfg, cfg=self.cfg)
        x_ptsA, avgiA, avgqA = t1.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug)        

        shots_iA, shots_qA, shots_iB, shots_qB = t1.collect_shots()

       

        avgiA = avgiA[0][0]
        avgqA = avgqA[0][0]
        ampsA = np.abs(avgiA+1j*avgqA) # Calculating the magnitude
        phasesA = np.angle(avgiA+1j*avgqA) # Calculating the phase      

        avgiB = avgiB[0][1]
        avgqB = avgqB[0][1]
        ampsB = np.abs(avgiB+1j*avgqB) # Calculating the magnitude
        phasesB = np.angle(avgiB+1j*avgqB) # Calculating the phase      
        data={'xptsA': x_ptsA, 'xptsB': x_ptsB, 'avgiA':avgiA, 'avgiB':avgiB,'avgqA':avgqA, 'avgqB':avgqB, 'ampsA':ampsA, 'ampsB':ampsB,'phasesA':phasesA, 'phasesB':phasesB, 'raw_iA': shots_iA,'raw_iB': shots_iB, 'raw_A': shots_qA, 'raw_B': shots_qB}  

        self.data=data
        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data
            
        # fitparams=[y-offset, amp, x-offset, decay rate]
        # Remove the last point from fit in case weird edge measurements
        data['fit_ampsA'], data['fit_err_ampsA'] = fitter.fitexp(data['xptsA'][:-1], data['ampsA'][:-1], fitparams=None)
        data['fit_ampsB'], data['fit_err_ampsB'] = fitter.fitexp(data['xptsB'][:-1], data['ampsB'][:-1], fitparams=None)


        data['fit_avgiA'], data['fit_err_avgiA'] = fitter.fitexp(data['xptsA'][:-1], data['avgiA'][:-1], fitparams=None)
        data['fit_avgiB'], data['fit_err_avgiB'] = fitter.fitexp(data['xptsB'][:-1], data['avgiB'][:-1], fitparams=None)
        pA = data['fit_avgiA']
        pB = data['fit_avgiB']
        data['T1_iA'] = pA[3]
        data['T1_err_iA'] = np.sqrt(data['fit_err_avgiA'][3][3])
        data['T1_iB'] = pB[3]   
        data['T1_err_iB'] = np.sqrt(data['fit_err_avgiB'][3][3])

        data['fit_avgqA'], data['fit_err_avgqA'] = fitter.fitexp(data['xptsA'][:-1], data['avgqA'][:-1], fitparams=None)
        data['fit_avgqB'], data['fit_err_avgqB'] = fitter.fitexp(data['xptsB'][:-1], data['avgqB'][:-1], fitparams=None)

        return data

    # def display(self, data=None, fit=True, **kwargs):
    #     if data is None:
    #         data=self.data 

    #     plt.figure(figsize=(10,10))
    #     plt.subplot(211, title="$T_1$", ylabel="I [ADC units]")
    #     plt.plot(data["xpts"][:-1], data["avgi"][:-1],'o-')
    #     if fit:
    #         p = data['fit_avgi']
    #         pCov = data['fit_err_avgi']
    #         captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
    #         plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgi"]), label=captionStr)
    #         plt.legend()
    #         print(f'Fit T1 avgi [us]: {data["fit_avgi"][3]}')
    #         data["err_ratio_i"] = np.sqrt(data['fit_err_avgi'][3][3])/data['fit_avgi'][3]
    #     plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC units]")
    #     plt.plot(data["xpts"][:-1], data["avgq"][:-1],'o-')
    #     if fit:
    #         p = data['fit_avgq']
    #         pCov = data['fit_err_avgq']
    #         captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
    #         plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgq"]), label=captionStr)
    #         plt.legend()
    #         print(f'Fit T1 avgq [us]: {data["fit_avgq"][3]}')
    #         data["err_ratio_q"] = np.sqrt(data['fit_err_avgq'][3][3])/data['fit_avgq'][3]


    #     plt.show()
    
    # def save_T1_Values(self, fit = True, data = None): 
    #     if data is None:
    #         data=self.data  
        
    #     length_scan = len(data["xpts"])
    #     t1_points = np.linspace(0, length_scan, self.cfg.expt.num_saved_points)
    #     data['t1_save_i'] = np.zeros(len(t1_points)-1)
    #     data['t1_save_q'] = np.zeros(len(t1_points)-1)

    #     if fit:
    #         for i in range(len(t1_points)-1):
    #             data['t1_save_i'][i] = data["avgi"][int(t1_points[i])]
    #         for i in range(len(t1_points)-1):
    #             data['t1_save_q'][i] = data["avgq"][int(t1_points[i])]
        
    # def save_data(self, data=None):
    #     print(f'Saving {self.fname}')
    #     super().save_data(data=data)
    #     return self.fname

# ====================================================== #

class T1_2qbContinuous(Experiment):
    """
    T1 Continuous
    Experimental Config:
    expt = dict(
        startA: wait time sweep start for qA [us]
        startB: wait time sweep start for qA [us]
        step: wait time sweep step
        expts: number steps in sweep
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
    )
    """
    def __init__(self, soccfg=None, path='', prefix='T1Continuous', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)
    
    def acquire(self, progress=False, debug=False):
        # q_ind = self.cfg.expt.qubit
        # for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
        #     for key, value in subcfg.items() :
        #         if isinstance(value, list):
        #             subcfg.update({key: value[q_ind]})
        #         elif isinstance(value, dict):
        #             for key2, value2 in value.items():
        #                 for key3, value3 in value2.items():
        #                     if isinstance(value3, list):
        #                         value2.update({key3: value3[q_ind]})                                
        t1A = T1Program(soccfg=self.soccfg, cfg=self.cfg)
        x_ptsA, avgiA, avgqA = t1A.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug)        

        shots_iA, shots_qA, shots_iB, shots_qB = t1A.collect_shots()
       


        avgiA = avgiA[0][0]
        avgqA = avgqA[0][0]
        ampsA = np.abs(avgiA+1j*avgqA) # Calculating the magnitude
        phasesA = np.angle(avgiA+1j*avgqA) # Calculating the phase      

        avgiB = avgiB[0][0]
        avgqB = avgqB[0][0]
        ampsB = np.abs(avgiB+1j*avgqB) # Calculating the magnitude
        phasesB = np.angle(avgiB+1j*avgqB) # Calculating the phase      

        now = datetime.now()
        current_time = np.array([now.strftime("%H:%M:%S")])
        data={'xptsA': x_ptsA, 'xptsB': x_ptsB, 'avgiA':avgiA, 'avgiB':avgiB,'avgqA':avgqA, 'avgqB':avgqB, 'ampsA':ampsA, 'ampsB':ampsB,'phasesA':phasesA, 'phasesB':phasesB, 'time':current_time, 'raw_iA': shots_iA,'raw_iB': shots_iB, 'raw_A': shots_qA, 'raw_B': shots_qB}   
        
        self.data=data
        return data




'''
    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data
            
        # fitparams=[y-offset, amp, x-offset, decay rate]
        # Remove the last point from fit in case weird edge measurements
        data['fit_amps'], data['fit_err_amps'] = fitter.fitexp(data['xpts'][:-1], data['amps'][:-1], fitparams=None)
        data['fit_avgi'], data['fit_err_avgi'] = fitter.fitexp(data['xpts'][:-1], data['avgi'][:-1], fitparams=None)
        data['fit_avgq'], data['fit_err_avgq'] = fitter.fitexp(data['xpts'][:-1], data['avgq'][:-1], fitparams=None)
        return data

        
    def display(self, data=None, fit=True, show = False, **kwargs):
        if data is None:
            data=self.data 
    
        plt.figure(figsize=(10,10))
        plt.subplot(211, title="$T_1$", ylabel="I [ADC units]")
        plt.plot(data["xpts"], data["avgi"],'o-', label = 'Current Data')
        plt.plot(self.cfg.expt.prev_data_x, self.cfg.expt.prev_data_i,'o-', label = 'Previous Data')
        if fit:
            p = data['fit_avgi']
            pCov = data['fit_err_avgi']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgi"]), label=captionStr)
            plt.legend()
            print(f'Fit T1 avgi [us]: {data["fit_avgi"][3]}')
            data["err_ratio_i"] = np.sqrt(data['fit_err_avgi'][3][3])/data['fit_avgi'][3]
        plt.legend()
        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC units]")
        plt.plot(data["xpts"], data["avgq"],'o-', label = 'Current Data')
        plt.plot(self.cfg.expt.prev_data_x, self.cfg.expt.prev_data_q,'o-', label = 'Previous Data')
        if fit:
            p = data['fit_avgq']
            pCov = data['fit_err_avgq']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgq"]), label=captionStr)
            plt.legend()
            print(f'Fit T1 avgq [us]: {data["fit_avgq"][3]}')
            data["err_ratio_q"] = np.sqrt(data['fit_err_avgq'][3][3])/data['fit_avgq'][3]

        plt.legend()
        if show:
            plt.show() 
'''
    


        
