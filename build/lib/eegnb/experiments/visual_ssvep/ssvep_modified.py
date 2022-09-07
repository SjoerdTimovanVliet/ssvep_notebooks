import os
from this import d
from time import time
from glob import glob
from random import choice

import numpy as np
from pandas import DataFrame
from psychopy import visual, core, event

from eegnb import generate_save_fn
import pdb
__title__ = "Visual SSVEP"


def present(duration=120, eeg=None, save_fn=None, mode='2 stimuli'):
    n_trials = 2010
    iti = 0.5 # inter-trial interval
    soa = 10.0 # seconds
    jitter = 0.2 # jitter the SOA

    predicted_n_trials = duration/(soa+0.5)
    
    record_duration = np.float32(duration)
    if mode == '2 stimuli':
        markernames = [1, 2]
    elif mode == '7 stimuli':
        markernames = [1, 2, 3, 4, 5, 6, 7]
        
    elif mode == '8 stimuli':
        markernames = [1, 2, 3, 4, 5, 6, 7, 8]
  
    # Setup trial 
    stim_freq = np.array(np.random.uniform(0, len(markernames)-1, n_trials), dtype=np.int)
    pdb.set_trace()
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
            init_flicker_stim(frame_rate, freqs[12.0], soa), # 12.5Hz
            init_flicker_stim(frame_rate, freqs[15.0], soa), # 15Hz
            init_flicker_stim(frame_rate, freqs[20.0], soa), # 20Hz
            init_flicker_stim(frame_rate, freqs[30.0], soa), # 30Hz
            init_flicker_stim(frame_rate, freqs[60.0], soa), # 60Hz
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
        core.wait(iti + np.random.rand() * jitter) # Wait for the intertrial interval (iti) plus a random jitter

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

        # Present flickering stim
        for _ in range(int(stim_patterns[ind]["n_cycles"])): # Iterate through stimulation frequencies
            grating.setAutoDraw(True) # Draw the grating
            for _ in range(int(stim_patterns[ind]["cycle"][0])): # Iterate through on periods
                mywin.flip() # Flip the screen
            grating.setAutoDraw(False)  # Turn off grating
            grating_neg.setAutoDraw(True) # Draw the negative grdating
            for _ in range(stim_patterns[ind]["cycle"][1]): # Iterate through off periods
                mywin.flip() # Flip the screen
            grating_neg.setAutoDraw(False)

        # offset
        mywin.flip()
        if len(event.getKeys()) > 0 or (time() - start) > record_duration:
            break
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
