import experiments as meas
import config

def make_tof(soc, expt_path, results_config_path, qubit_i):

    tof = meas.ToFCalibrationExperiment(soccfg=soc,
    path=expt_path,
    prefix=f"adc_trig_offset_calibration_qubit{qubit_i}",
    config_file=results_config_path)

    tof.cfg.expt = dict(pulse_length=0.5, # [us]
    readout_length=1.0, # [us]
    trig_offset=0, # [clock ticks]
    gain=30000, # blast the power just for the RFSoC calibration
    frequency=tof.cfg.device.readout.frequency[qubit_i], # [MHz]
    reps=1000, # Number of averages per point
    qubit=qubit_i) 

    tof.cfg.device.readout.relax_delay[qubit_i]=0.1 # wait time between experiments [us]
    return tof

def make_rspec_coarse(soc, expt_path, results_config_path, qubit_i, start=7000, span=250, reps=800, npts=5000):
    rspec = meas.ResonatorSpectroscopyExperiment(
    soccfg=soc,
    path=expt_path,
    prefix=f"resonator_spectroscopy_coarse",
    config_file=results_config_path,   
    )

    rspec.cfg.expt = dict(
        start = start, #Lowest resontaor frequency
        step=span/npts, # min step ~1 Hz
        expts=npts, # Number experiments stepping from start
        reps= reps, # Number averages per point 
        pulse_e=False, # add ge pi pulse prior to measurement
        pulse_f=False, # add ef pi pulse prior to measurement
        qubit=qubit_i,
    )

    rspec.cfg.device.readout.relax_delay = 5 # Wait time between experiments [us]
    return rspec

def make_rspec_fine(soc, expt_path, results_config_path, qubit_i, j, center=7000, span=5, reps=500, smart=True):
    
    rspec = meas.ResonatorSpectroscopyExperiment(
    soccfg=soc,
    path=expt_path,
    prefix=f"resonator_spectroscopy_res{j}",
    config_file=results_config_path,   
    )
    npts = 1000 

    rspec.cfg.expt = dict(
        start = center-span/2, #Lowest resontaor frequency
        step=span/npts, # min step ~1 Hz
        smart=smart,
        kappa=0.35,
        expts=npts, # Number experiments stepping from start
        reps= reps, # Number averages per point 
        pulse_e=False, # add ge pi pulse prior to measurement
        pulse_f=False, # add ef pi pulse prior to measurement
        qubit=qubit_i,
    )

    rspec.cfg.device.readout.relax_delay = 5 # Wait time between experiments [us]
    return rspec

def make_rpowspec(soc, expt_path, cfg_file, qubit_i, res_freq, span_f=5, npts_f=250, span_gain=27000, start_gain=5000, npts_gain=10, reps=500, smart=False):

    rpowspec = meas.ResonatorPowerSweepSpectroscopyExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"ResonatorPowerSweepSpectroscopyExperiment_qubit{qubit_i}",
        config_file=cfg_file,
    )

    rpowspec.cfg.expt = dict(
        start_f = res_freq-span_f/2, # resonator frequency to be mixed up [MHz]
        step_f = span_f/npts_f, # min step ~1 Hz, 
        smart = smart, 
        expts_f=npts_f, # Number experiments stepping freq from start
        start_gain=start_gain,
        step_gain=int(span_gain/npts_gain), # Gain step size
        expts_gain=npts_gain+1, # Number experiments stepping gain from start
        reps= reps, # Number averages per point
        pulse_e=False, # add ge pi pulse before measurement
        pulse_f=False, # add ef pi pulse before measurement
        qubit=qubit_i,  
    ) 

    rpowspec.cfg.device.readout.relax_delay = 5 # Wait time between experiments [us]    
    rpowspec.cfg.device.readout.readout_length = 5
    return rpowspec

def make_qspec(soc, expt_path, cfg_path, qubit_i, span=None, npts=1500, reps=50, rounds=20, gain=None, coarse=True):

    if coarse is True and span is None:
        span=700 
        prefix = f"qubit_spectroscopy_coarse_qubit{qubit_i}"
    elif span is None:
        span=3
        prefix = f"qubit_spectroscopy_fine_qubit{qubit_i}"
    else:
        prefix = f"qubit_spectroscopy_qubit{qubit_i}"

    if coarse is True and gain is None: 
        gain=1000
    elif gain is None:
        gain=100

    qspec = meas.PulseProbeSpectroscopyExperiment(
    soccfg=soc,
    path = expt_path, 
    prefix = f"qubit_spectroscopy_coarse_qubit{qubit_i}",
    config_file=cfg_path,
    )

    qspec.cfg.expt = dict(
        start= qspec.cfg.device.qubit.f_ge[qubit_i]-span/2, # qubit frequency to be mixed up [MHz]
        step = span/npts, # min step ~1 Hz
        expts = npts, # Number experiments stepping from start
        reps = reps, # Number averages per point
        rounds = rounds, #Number of start to finish sweeps to average over 
        length = 50, # qubit probe constant pulse length [us]
        gain = gain, #qubit pulse gain  
        pulse_type = 'const', 
        #pulse_type = 'gauss',  
        qubit = qubit_i,
    ) 

    qspec.cfg.device.readout.relax_delay = 10 # Wait time between experiments [us]
    return qspec

def make_lengthrabi(soc, expt_path, cfg_path, qubit_i, npts = 200, reps = 1000, gain = 2000, num_pulses = 1):
    lengthrabi = meas.LengthRabiExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"length_rabi_qubit{qubit_i}",
        config_file=cfg_path,
    )

    lengthrabi.cfg.expt = dict(
        start =  0.0025, 
        step=  soc.cycles2us(1), # [us] this is the samllest possible step size (size of clock cycle)
        expts= npts, 
        reps= reps,
        gain =  gain, #qubit gain [DAC units]
        #gain=lengthrabi.cfg.device.qubit.pulses.pi_ge.gain[qubit_i],
        pulse_type='gauss',
        # pulse_type='const',
        checkZZ=False,
        checkEF=False, 
        qubits=[qubit_i],
        num_pulses = 1, #number of pulses to play, must be an odd number 
    )

    return lengthrabi

def make_amprabi(soc, expt_path, cfg_path, qubit_i, npts = 100, reps = 1000, rounds=1, gain=10000):
    #auto_cfg.device.qubit.pulses.pi_ge.gain[qubit_i]
    amprabi = meas.AmplitudeRabiExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"amp_rabi_qubit{qubit_i}",
        config_file=cfg_path,
        )
    auto_cfg = config.load(cfg_path)
    span = 2*gain
    amprabi.cfg.expt = dict(     
        start=0,
        step=int(span/npts), # [dac level]
        expts=npts,
        reps=reps,
        rounds=rounds,
        sigma_test= auto_cfg.device.qubit.pulses.pi_ge.sigma[qubit_i], # gaussian sigma for pulse length - overrides config [us]
        checkZZ=False,
        checkEF=False, 
        qubits=[qubit_i],
        pulse_type='gauss',
        # pulse_type='const',
        num_pulses = 1, #number of pulses to play, must be an odd number in order to achieve a pi rotation at pi length/ num_pulses 
    )
    return amprabi

def make_t2r(soc, expt_path, cfg_path, qubit_i, npts = 350, reps = 400, rounds=2, step=0.1, ramsey_freq=0.1):
    t2r = meas.RamseyExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"ramsey_qubit{qubit_i}",
        config_file=cfg_path,
    )

    t2r.cfg.expt = dict(
        start=0, # wait time tau [us]
        #step=soc.cycles2us(10), # [us] make sure nyquist freq = 0.5 * (1/step) > ramsey (signal) freq!
        step= step, # [us]
        expts=npts,
        ramsey_freq=ramsey_freq, # [MHz]
        reps=reps,
        rounds=rounds, # set this = 1 (computer asking for 20 rounds --> faster if I don't have to communicate between computer and board)
        qubits=[qubit_i],
        checkZZ=False,
        checkEF=False,
    )

    return t2r

def make_t1(soc, expt_path, cfg_path, qubit_i,span=600, npts=200, reps=1000, rounds=1):

    t1 = meas.T1Experiment(
      soccfg=soc,
      path=expt_path,
      prefix=f"t1_qubit{qubit_i}",
      config_file= cfg_path,
    )

    span = span 
    npts = npts

    t1.cfg.expt = dict(
        start=0, # wait time [us]
        step=int(span/npts), 
        expts=npts,
        reps=reps, # number of times we repeat a time point 
        rounds=rounds, # number of start to finish sweeps to average over
        qubit=qubit_i,
        length_scan = span, # length of the scan in us
        num_saved_points = 10, # number of points to save for the T1 continuous scan 
    )

    return t1

def make_t1_cont(soc, expt_path, cfg_path, qubit_i, reps=1000, rounds=1):
    t1_cont = meas.T1Continuous(
            soccfg=soc,
            path=expt_path,
            prefix=f"t1_continuous_qubit{qubit_i}",
            config_file= cfg_path,
        )

    span = t1.cfg.expt.length_scan 
    npts = t1.cfg.expt.num_saved_points #eventually need to change this to t1.cfg.expt.num_saved_points

    t1_cont.cfg.expt = dict(
        start=0, # wait time [us]
        step=int(span/npts), 
        expts=npts,
        reps=reps, # number of times we repeat a time point 
        rounds=rounds, # number of start to finish sweeps to average over
        qubit=qubit_i
    )

    return t1_cont


def make_singleshot(soc, expt_path, cfg_path, qubit_i, reps=10000):

    shot = meas.HistogramExperiment(
    soccfg=soc,
    path=expt_path,
    prefix=f"single_shot_qubit{qubit_i}",
    config_file= cfg_path,
    )

    shot.cfg.expt = dict(
        reps=reps,
        check_e = True, 
        check_f=False,
        qubit=qubit_i,
    )

    return shot

def make_singleshot_opt(soc, expt_path, cfg_path, qubit_i, reps=10000, span_f =1, npts_f =5, start_gain =2000, span_gain=14000, npts_gain=5, start_len=3.0, span_len=25.0, npts_len=5):

    shotopt = meas.SingleShotOptExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"single_shot_opt_qubit{qubit_i}",
        config_file=cfg_path,
    )
    
    shotopt.cfg.expt = dict(
        reps=100000,
        qubit=qubit_i,

        start_f=shotopt.cfg.device.readout.frequency[qubit_i] - 0.5*span_f,
        step_f=span_f/npts_f,
        expts_f=npts_f,

        start_gain=start_gain,
        step_gain=span_gain/npts_gain,
        expts_gain=npts_gain,

        start_len=start_len,
        step_len=span_len/npts_len,
        expts_len=npts_len,
    )

    return shotopt


def make_amprabiEF(soc, expt_path, config_path, qubit_i, span=20000, npts=101, reps=100, rounds=40):
    
    amprabiEF = meas.AmplitudeRabiExperiment(
        soccfg=soc,
        path=expt_path,
        prefix="amp_rabi_EF"+f"_qubit{qubit_i}",
        config_file=config_path,
    )

    amprabiEF.cfg.expt = dict(
        start=0, # qubit gain [dac level]
        step=int(span/npts), # [dac level]
        expts=npts,
        reps=reps,
        rounds=rounds,
        pulse_type='gauss',
        # sigma_test=0.013, # gaussian sigma for pulse length - default from cfg [us]
        checkZZ=False,
        # checkEF=True, 
        # pulse_ge=True,
        # cool_qubits=[1],
        # cool_idle=9.1, # us
        # check heating from swap
        checkEF=True, 
        pulse_ge=False,
        apply_EgGf=True,
        qubits_EgGf=[2, 1],
        qDrive=2)

    return amprabiEF

def make_acstark(soc, expt_path, config_path, qubit_i, span_f=100, npts_f=300, span_gain=10000, npts_gain=25):
    acspec = meas.ACStarkShiftPulseProbeExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"ac_stark_shift_qubit{qubit_i}",
        config_file=config_path,
    )

    pump_params=dict(
    ch=1,
    type='full',
    nyquist=2,
    )

    acspec.cfg.expt = dict(        
        start_f=acspec.cfg.device.qubit.f_ge[qubit_i]-0.25*span_f, # Pulse frequency [MHz]
        step_f=span_f/npts_f,
        expts_f=npts_f,
        start_gain=0, 
        step_gain=int(span_gain/npts_gain),
        expts_gain=npts_gain+1,
        pump_params=pump_params,
    
        pump_freq=acspec.cfg.device.qubit.f_ge[qubit_i]-20,
        # pump_freq=acspec.cfg.device.qubit.f_EgGf[2],
        pump_length=10, # [us]
        qubit_length=1, # [us]
        qubit_gain=2814,
        pulse_type='const',
        reps=100,
        rounds=10, # Number averages per point
        qubit=qubit_i,
    )
    acspec.cfg.device.readout.relax_delay = 25
    return acspec
