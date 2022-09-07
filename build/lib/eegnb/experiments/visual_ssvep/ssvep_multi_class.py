from concurrent.futures.process import _MAX_WINDOWS_WORKERS
import os
from this import d
from time import time
from glob import glob
from random import choice

import numpy as np
from pandas import DataFrame
from psychopy import visual, core, event
from random import shuffle
from eegnb import generate_save_fn
import pdb
import copy
__title__ = "Visual SSVEP"


def present(n_stimuli=2, n_rounds=6, inter_trial_time=0.5, trial_duration=10, eeg=None, save_fn=None, mode='2 stimuli'):
    iti = inter_trial_time# inter-trial interval
    soa = trial_duration # seconds
    #jitter = 0.2 # jitter the SOA
    duration = n_rounds * (n_stimuli *(soa + iti))#+0.5*jitter))
    
    n_trials = n_rounds * (n_stimuli+1)

    record_duration = np.float32(duration)
    if mode == '2 stimuli':
        markernames = [0, 1, 2]
    elif mode == '7 stimuli':
        markernames = [0, 1, 2, 3, 4, 5, 6, 7]
        
    elif mode == '8 stimuli':
        markernames = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    elif mode == '9 stimuli':
        markernames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  
    # Setup trial 
    indices_to_pause_at = [len(markernames)*i-1 for i in range(1, n_rounds)]
   
    #print(trials_to_pause_at)
    #pdb.set_trace()
    #trials_to_pause_at = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209
    sample_stim_freq = list(range(0, len(markernames)))
    
    stim_freq = []
    for round_i in range(n_rounds):
            shuffle(sample_stim_freq)
            stim_freq.append(copy.deepcopy(sample_stim_freq))
    stim_freq = np.array(stim_freq).flatten('C')
    #stim_freq = np.array(np.random.uniform(0, len(markernames)-1, n_trials), dtype=np.int)

    #stim_freq = np.random.binomial(markernames, 0.5, n_trials)
    print(f"Stim freq: {stim_freq}")

    trials = DataFrame(dict(stim_freq=stim_freq, timestamp=np.zeros(n_trials)))
  
    print(f"Trials: {trials}")
    # Set up graphics
    mywin = visual.Window([1600, 900], monitor="testMonitor", units="deg", fullscr=True)
    grating = visual.GratingStim(win=mywin, mask="circle", size=80, sf=0.2)
    grating_neg = visual.GratingStim(
        win=mywin, mask="circle", size=80, sf=0.2, phase=0.5
    )
    fixation = visual.GratingStim(
        win=mywin, size=0.2, pos=[0, 0], sf=0.2, color=[1, 0, 0], autoDraw=True
    )

    # Generate the possible ssvep frequencies based on monitor refresh rate
    def get_possible_ssvep_freqs(frame_rate, stim_type="single"):
        """Get possible SSVEP stimulation frequencies.
        Utility function that returns the possible SSVEP stimulation
        frequencies and on/off pattern based on screen refresh rate.
        Args:
            frame_rate (float): screen frame rate, in Hz
        Keyword Args:
            stim_type (str): type of stimulation
                'single'-> single graphic stimulus (the displayed object
                    appears and disappears in the background.)
                'reversal' -> pattern reversal stimulus (the displayed object
                    appears and is replaced by its opposite.)
        Returns:
            (dict): keys are stimulation frequencies (in Hz), and values are
                lists of tuples, where each tuple is the number of (on, off)
                periods of one stimulation cycle
        For more info on stimulation patterns, see Section 2 of:
            Danhua Zhu, Jordi Bieger, Gary Garcia Molina, and Ronald M. Aarts,
            "A Survey of Stimulation Methods Used in SSVEP-Based BCIs,"
            Computational Intelligence and Neuroscience, vol. 2010, 12 pages,
            2010.
        """

        max_period_nb = int(frame_rate / 6)
        periods = np.arange(max_period_nb) + 1

        if stim_type == "single":
            freqs = dict()
            for p1 in periods:
                for p2 in periods:
                    f = frame_rate / (p1 + p2)
                    try:
                        freqs[f].append((p1, p2))
                    except:
                        freqs[f] = [(p1, p2)]
        elif stim_type == "reversal":
            freqs = {frame_rate / p: [(p, p)] for p in periods[::-1]}
        print(f"freqs: {freqs}")
        return freqs


    def init_flicker_stim(frame_rate, cycle, soa): 
        """Initialize flickering stimulus.
        Get parameters for a flickering stimulus, based on the screen refresh
        rate and the desired stimulation cycle.
        Args:
            frame_rate (float): screen frame rate, in Hz
            cycle (tuple or int): if tuple (on, off), represents the number of
                'on' periods and 'off' periods in one flickering cycle. This
                supposes a "single graphic" stimulus, where the displayed object
                appears and disappears in the background.
                If int, represents the number of total periods in one cycle.
                This supposes a "pattern reversal" stimulus, where the
                displayed object appears and is replaced by its opposite.
            soa (float): stimulus duration, in s
        Returns:
            (dict): dictionary with keys
                'cycle' -> tuple of (on, off) periods in a cycle
                'freq' -> stimulus frequency
                'n_cycles' -> number of cycles in one stimulus trial
        """
        if isinstance(cycle, tuple):
            stim_freq = frame_rate / sum(cycle)
            n_cycles = int(soa * stim_freq) # number of cycles in one stimulus trial
        else:
            stim_freq = frame_rate / cycle
            cycle = (cycle, cycle) 
            n_cycles = int(soa * stim_freq) / 2 # Number of cycles in one stimulus trial

        return {"cycle": cycle, "freq": stim_freq, "n_cycles": n_cycles}
    #-------------------------------------------------------------------------
    # Generate the possible ssvep frequencies based on monitor refresh rate. The standard monitor refresh rate is 60Hz
    # Set up stimuli
    print("-----------------------------------------------------")
    print("mode: ", mode)
    print("-----------------------------------------------------")
    frame_rate = np.round(mywin.getActualFrameRate())  # Frame rate, in Hz
    freqs = get_possible_ssvep_freqs(frame_rate, stim_type="reversal")
    if mode == '2 stimuli':
        stim_patterns = [
            init_flicker_stim(frame_rate, 2, soa),    # call init_flicker_stim with 2 cycles. Meaning 30 hz
            init_flicker_stim(frame_rate, 3, soa),    # call init_flicker_stim with 3 cycles. Meaning 20 hz
        ]
    elif mode == '7 stimuli':
        stim_patterns = [

            init_flicker_stim(frame_rate, freqs[6.0][0], soa),# 6Hz    
            init_flicker_stim(frame_rate, freqs[7.5][0], soa), # 7.5Hz
            init_flicker_stim(frame_rate, freqs[10.0][0], soa), # 10Hz
            init_flicker_stim(frame_rate, freqs[12.0][0], soa), # 12.5Hz
            init_flicker_stim(frame_rate, freqs[15.0][0], soa), # 15Hz
            init_flicker_stim(frame_rate, 2, soa),    # call init_flicker_stim with 2 cycles. Meaning 30 hz
            init_flicker_stim(frame_rate, 3, soa),    # call init_flicker_stim with 3 cycles. Meaning 20 hz
        ]
    elif mode == '8 stimuli':
        stim_patterns = [

            init_flicker_stim(frame_rate, freqs[6.0], soa),# 6Hz    
            init_flicker_stim(frame_rate, freqs[7.5], soa), # 7.5Hz
            init_flicker_stim(frame_rate, freqs[10.0], soa), # 10Hz
            init_flicker_stim(frame_rate, freqs[12.0][0], soa), # 12.5Hz
            init_flicker_stim(frame_rate, freqs[15.0][0], soa), # 15Hz
            init_flicker_stim(frame_rate, freqs[20.0][0], soa), # 20Hz
            init_flicker_stim(frame_rate, freqs[30.0][0], soa), # 30Hz
            init_flicker_stim(frame_rate, freqs[60.0][0], soa), # 60Hz
        ]
    elif mode == '9 stimuli':
        stim_patterns = [
            init_flicker_stim(frame_rate, freqs[6.0][0], soa),# 6Hz
            init_flicker_stim(frame_rate, freqs[6.666666666666667][0], soa), #6.7Hz
            init_flicker_stim(frame_rate, freqs[7.5][0], soa), # 7.5Hz
            init_flicker_stim(frame_rate, freqs[8.571428571428571][0], soa), # 18.6Hz
            init_flicker_stim(frame_rate, freqs[10.0][0], soa), # 10Hz
            init_flicker_stim(frame_rate, freqs[12.0][0], soa), # 12.5Hz
            init_flicker_stim(frame_rate, freqs[15.0][0], soa), # 15Hz
            init_flicker_stim(frame_rate, freqs[20.0][0], soa), # 20Hz
            init_flicker_stim(frame_rate, freqs[30.0][0], soa), # 30Hz

        ]
 
   
    # print(
    #     (
    #         "These Flickering frequencies (Hz): {}\n".format(
    #             [stim_patterns[0]["freq"], stim_patterns[1]["freq"]]
    #         )
    #     )
    # )

    # Show the instructions screen
    show_instructions(duration)

    # start the EEG stream, will delay 5 seconds to let signal settle
    if eeg:
        if save_fn is None:  # If no save_fn passed, generate a new unnamed save file
            save_fn = generate_save_fn(eeg.device_name, "visual_ssvep", "unnamed")
            print(
                f"No path for a save file was passed to the experiment. Saving data to {save_fn}"
            )
        eeg.start(save_fn, duration=record_duration) # Start the EEG stream with the desired duration (in seconds)

    # Iterate through trials
    start = time()
    for ii, trial in trials.iterrows():
        # Intertrial interval
        core.wait(iti) #+ np.random.rand() * jitter) # Wait for the intertrial interval (iti) plus a random jitter

        # Select stimulus frequency
        ind = trials["stim_freq"].iloc[ii] 
    
        # Push sample
        if eeg:
            timestamp = time()
            if eeg.backend == "muselsl":
                marker = [markernames[ind]]
            else:
                marker = markernames[ind]
            eeg.push_sample(marker=marker, timestamp=timestamp) # Push timestamp marker to notate start of trial
        if marker == 0:
            core.wait(trial_duration)
        else:
            try:
                # Present flickering stim
                corrected_ind = ind - 1
                for _ in range(int(stim_patterns[corrected_ind]["n_cycles"])): # Iterate through stimulation frequencies
                    grating.setAutoDraw(True) # Draw the grating
                    for _ in range(int(stim_patterns[corrected_ind]["cycle"][0])): # Iterate through on periods
                        mywin.flip() # Flip the screen
                    grating.setAutoDraw(False)  # Turn off grating
                    grating_neg.setAutoDraw(True) # Draw the negative grdating
                    for _ in range(stim_patterns[corrected_ind]["cycle"][1]): # Iterate through off periods
                        mywin.flip() # Flip the screen
                    grating_neg.setAutoDraw(False)

                    # offset
                    mywin.flip()
            except:
                pdb.set_trace()
        # if len(event.getKeys()) > 0 or (time() - start) > record_duration:
        #     break
        if ii in indices_to_pause_at:
            show_instructions_block_pause()
        event.clearEvents()

    # Cleanup
    if eeg:
        eeg.stop() # stop all data
    mywin.close()


def show_instructions(duration):

    instruction_text = """
    Welcome to the SSVEP experiment! 
 
    Stay still, focus on the centre of the screen, and try not to blink. 

    This block will run for %s seconds.

    Press spacebar to continue.

    Warning: This experiment contains flashing lights and may induce a seizure. Discretion is advised.
    
    """
    instruction_text = instruction_text % duration

    # graphics
    mywin = visual.Window([1600, 900], monitor="testMonitor", units="deg", fullscr=True)

    mywin.mouseVisible = False

    # Instructions
    text = visual.TextStim(win=mywin, text=instruction_text, color=[-1, -1, -1])
    text.draw()
    mywin.flip()
    event.waitKeys(keyList="space")

    mywin.mouseVisible = True
    mywin.close()

def show_instructions_block_pause():

    instruction_text = "Take a brake of 1 minute before the experiment  continues.\
         Press spacebar to continue after the 1 min break."


    # graphics
    mywin = visual.Window([1600, 900], monitor="testMonitor", units="deg", fullscr=True)

    mywin.mouseVisible = False

    # Instructions
    text = visual.TextStim(win=mywin, text=instruction_text, color=[-1, -1, -1])
    text.draw()
    mywin.flip()
    event.waitKeys(keyList="space")

    mywin.mouseVisible = True
    mywin.close()