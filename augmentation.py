import numpy as np
import os
import librosa
import muda
import scipy 
import jams
import soundfile as sf
import random



#should downsampling equivalent to low passing?
def augmentation(SRC_PATH,ir_path):
    """
    SRC_path: path to the folder of speaker folders to augment, string
    
    augmentations: downsampling(8k/11k/16k) or no; noise(white/pink/brown) or no; clipping or no; cough or no; 
                    room acoustics (reverb) or no; drc or no
                    
            
    """

    #provide options for all the augmentations
    freqs = [8000,11000,16000]
    noise = ["pink","white","brownian"]
    clip = 3 #
    room = [os.path.join(ir_path,file) for file in os.listdir(ir_path)]
    #room = ["big.wav","small.wav"]
  
    augmentations = {"acoustics":room,
                   # "noise":noise, 
                    "clipping":clip,
                    "downsample":freqs}

    #make union 
    steps = []
    
    for aug,value in augmentations.items():
        if aug == "downsample":
            filtering = muda.deformers.Filter(btype="low",cutoff=value) #only one output
            steps.append(('downsample', muda.deformers.Bypass(filtering)))
        elif aug == "noise":
            colorednoise =  muda.deformers.ColoredNoise(n_samples=1, color = value, weight_max=0.3) #noise of different weights
            steps.append(('colorednoise', muda.deformers.Bypass(colorednoise)))
        elif aug == "clipping":
            clipping = muda.deformers.LinearClipping(n_samples=value) # clipping of different limits
            steps.append(('clipping', clipping))
        elif aug == "acoustics":
            acoustics = muda.deformers.IRConvolution(value) #one output
            steps.append(('acoustics', acoustics))
        


    #this is in total 4 * 4 * 3 * 2 = 96 augmentations per audio file!
    union = muda.Union(steps)
    #ensure distribution between all the different deforms or equal distribution among recordings
    total_deform = (len(freqs)+1) + clip + len(room) #save 100 files in the new folder
    indices = np.arange(0,total_deform,1)
    random.shuffle(indices)

    output_dir = '/scratch/hh2263/VCTK/VCTK-Corpus/wav48_muda' 
    jam_path = '/home/hh2263/Speaker-Diarization/general.jams'
    

    wavDir = os.listdir(SRC_PATH)
    for j,spkDir in enumerate(wavDir):   # Each speaker's directory
        spk = spkDir    # speaker name
        wavPath = os.path.join(SRC_PATH, spkDir)
        outPath = os.path.join(output_dir,spkDir)
        os.makedirs(outPath, exist_ok=True)
        for i, wav in enumerate(os.listdir(wavPath)): # all wavfiles included
            utter_path = os.path.join(wavPath, wav)
            #read file
            y_orig, sr = sf.read(utter_path)
            existing_jams = jams.load(jam_path) # do we have to create a jams file for each recording??
            j_orig = muda.jam_pack(existing_jams, _audio=dict(y=y_orig, sr=sr))
            #pick random deformation to deform
            j_new = list(union.transform(j_orig))[indices[i % total_deform]]
            #save deformed file and jam to output directory
            print("saved ",outPath,wav)	
	    muda.save(os.path.join(outPath,wav), 
                os.path.join(outPath,wav[:-4])+'.jams',
                j_new)

        if(j>100):
            break


if __name__ == '__main__':
    SRC_path='/scratch/hh2263/VCTK/VCTK-Corpus/wav48'
    ir_path = '/home/hh2263/Speaker-Diarization/ir_files/'
    augmentation(SRC_path,ir_path)
  
    


        
        
