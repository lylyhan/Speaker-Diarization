import numpy as np
import os
import librosa
import muda
import scipy 
import jams
import random
import soundfile as sf
import sys


# Parse input arguments.
args = sys.argv[1:]
SRC_path = args[0]
ir_path = args[1]
bg_path = args[4]
idx_start = int(args[2])
idx_end = int(args[3])


#should downsampling equivalent to low passing?
def augmentation(SRC_path,ir_path,idx_start,idx_end):
    """
    SRC_path: path to the folder of speaker folders to augment, string
    
    augmentations: downsampling(8k/11k/16k) or no; noise(white/pink/brown) or no; clipping or no; cough or no; 
                    room acoustics (reverb) or no; drc or no
                    
            
    """

    #provide options for all the augmentations
    freqs = [8000,11000]
    #noise = ["pink","white","brownian"]
    clip = 1 #
    room = [os.path.join(ir_path,file) for file in os.listdir(ir_path)]
    #room = ["big.wav","small.wav"]
    bg = [os.path.join(bg_path,file) for file in os.listdir(bg_path)] 
    augmentations = {"acoustics":room,
                    #"noise":noise, 
                    "clipping":clip,
                    "downsample":freqs,
                    "background":bg}

    #make union 
    steps = []
    
    for aug,value in augmentations.items():
        if aug == "downsample":
            filtering = muda.deformers.Filter(btype="low",cutoff=value) #only one output
            steps.append(('downsample', filtering))
        elif aug == "noise":
            colorednoise =  muda.deformers.ColoredNoise(n_samples=1, color = value, weight_max=0.3) #noise of different weights
            steps.append(('colorednoise', colorednoise))
        elif aug == "clipping":
            clipping = muda.deformers.LinearClipping(n_samples=value) # clipping of different limits
            steps.append(('clipping', clipping))
        elif aug == "acoustics":
            acoustics = muda.deformers.IRConvolution(value) #one output
            steps.append(('acoustics', acoustics))
        elif aug == "background":
            bgnoise =  muda.deformers.BackgroundNoise(n_samples=1, files=value) #noise of different weights
            steps.append(('background', bgnoise))


    #this is in total 4 * 4 * 3 * 2 = 96 augmentations per audio file!
    union = muda.Union(steps)
    #ensure distribution between all the different deforms or equal distribution among recordings
    total_deform = len(freqs) + clip + len(room) + len(bg) #save 100 files in the new folder
 

    output_dir = '/scratch/hh2263/VCTK/VCTK-Corpus/wav48_muda_2' 
    jam_path = '/home/hh2263/Speaker-Diarization/general.jams'
    wavDir = os.listdir(SRC_path)[idx_start:idx_end]

  
    for j,spkDir in enumerate(wavDir):   # Each speaker's directory
        spk = spkDir    # speaker name
        print("making ",spkDir)
        wavPath = os.path.join(SRC_path, spkDir)
        outPath = os.path.join(output_dir,spkDir)
        os.makedirs(outPath, exist_ok=True)

        for i, wav in enumerate(os.listdir(wavPath)): # all wavfiles included
            if os.path.exists(os.path.join(outPath,wav[:-4]+'_g00.wav')):
                pass
            else:
                print("make",wav[:-4])
                utter_path = os.path.join(wavPath, wav)
            #read file
                y_orig, sr = sf.read(utter_path)
                existing_jams = jams.load(jam_path) # do we have to create a jams file for each recording??
            #empty_jam = jams.JAMS()
                j_orig = muda.jam_pack(existing_jams, _audio=dict(y=y_orig, sr=sr))
            #pick random deformation to deform
            
            #save deformed file and jam to output directory
                for i, jam_out in enumerate(union.transform(j_orig)):
                    muda.save(os.path.join(outPath,wav[:-4]+'_g{:02d}.wav'.format(i)),
                        os.path.join(outPath,wav[:-3]+'_g{:02d}.jams'.format(i)),
                        jam_out)
    
                

        #if(j>100):
        #    break



#SRC_path='/scratch/hh2263/VCTK/VCTK-Corpus/wav48'
#ir_path = '/home/hh2263/Speaker-Diarization/ir_files/'
augmentation(SRC_path,ir_path,idx_start,idx_end)

    


        
        
