
"""A demo script showing how to DIARIZATION ON WAV USING UIS-RNN."""

import numpy as np
import soundfile as sf
import uisrnn
import librosa
import sys
sys.path.append('ghostvlad')
sys.path.append('visualization')
import toolkits
import model as spkModel
import os
from viewer import PlotDiar
import pickle
from collections import defaultdict
 
# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default=r'ghostvlad/pretrained/weights.h5', type=str)
parser.add_argument('--data_path', default='4persons', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)

parser.add_argument('--save_wavpath', type=str)
parser.add_argument('--save_pklpath', type=str)


global args
args_all = parser.parse_args()

args = argparse.Namespace(**{k: v for k, v in args_all._get_kwargs()
                              if not k.startswith("save_")})

#print(args,args_all)

#SAVED_MODEL_NAME = 'pretrained/saved_model.uisrnn_benchmark'
SAVED_MODEL_NAME = '/scratch/hh2263/VCTK/saved_model_vctk_muda_updated.uisrnn_benchmark'

def append2dict(speakerSlice, spk_period):
    key = list(spk_period.keys())[0]
    value = list(spk_period.values())[0]
    timeDict = {}
    timeDict['start'] = int(value[0]+0.5)
    timeDict['stop'] = int(value[1]+0.5)
    if(key in speakerSlice):
        speakerSlice[key].append(timeDict)
    else:
        speakerSlice[key] = [timeDict]

    return speakerSlice

def arrangeResult(labels, time_spec_rate): # {'1': [{'start':10, 'stop':20}, {'start':30, 'stop':40}], '2': [{'start':90, 'stop':100}]}
    lastLabel = labels[0]
    speakerSlice = {}
    j = 0
    for i,label in enumerate(labels):
        if(label==lastLabel):
            continue
        speakerSlice = append2dict(speakerSlice, {lastLabel: (time_spec_rate*j,time_spec_rate*i)})
        j = i
        lastLabel = label
    speakerSlice = append2dict(speakerSlice, {lastLabel: (time_spec_rate*j,time_spec_rate*(len(labels)))})
    return speakerSlice

def genMap(intervals):  # interval slices to maptable
    slicelen = [sliced[1]-sliced[0] for sliced in intervals.tolist()]
    mapTable = {}  # vad erased time to origin time, only split points
    idx = 0
    for i, sliced in enumerate(intervals.tolist()):
        mapTable[idx] = sliced[0]
        idx += slicelen[i]
    mapTable[sum(slicelen)] = intervals[-1,-1]

    keys = [k for k,_ in mapTable.items()]
    keys.sort()
    return mapTable, keys

def fmtTime(timeInMillisecond):
    millisecond = timeInMillisecond%1000
    minute = timeInMillisecond//1000//60
    second = (timeInMillisecond-minute*60*1000)//1000
    time = '{}:{:02d}.{}'.format(minute, second, millisecond)
    return time

def load_wav(in_file,file_length, current_time,block_sec, sr): #in_file as a soundfile object, block_sec is chunks in seconds
    if current_time + block_sec >= file_length: #in seconds, preview if reading 2 min is too much
        wav = np.frombuffer(in_file.buffer_read(-1, dtype='float32'),dtype="float32")
        current_time = file_length
    else:
        wav = np.frombuffer(in_file.buffer_read(sr*block_sec, dtype='float32'),dtype="float32") #load 2 minutes of audio at a time
        current_time += block_sec
    #wav, _ = librosa.load(in_file, sr=sr)
    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
      wav_output.extend(wav[sliced[0]:sliced[1]])
    return np.array(wav_output), (intervals/sr*1000).astype(int),current_time

def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T


# 0s        1s        2s                  4s                  6s
# |-------------------|-------------------|-------------------|
# |-------------------|
#           |-------------------|
#                     |-------------------|
#                               |-------------------|
def load_data(in_file, file_length, current_time, win_length=400, sr=16000, hop_length=160, n_fft=512, embedding_per_second=0.5, overlap_rate=0.5,block_sec=120):
    #get current block's audio, and update file's read pointer
    wav, intervals,current_time = load_wav(in_file, file_length, current_time,block_sec, sr=sr) 
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    spec_mag = mag_T

    spec_len = sr/hop_length/embedding_per_second
    spec_hop_len = spec_len*(1-overlap_rate)

    cur_slide = 0.0
    utterances_spec = []

    while(True):  # slide window.
        if(cur_slide + spec_len > time):
            break
        spec_mag = mag_T[:, int(cur_slide+0.5) : int(cur_slide+spec_len+0.5)]
        
        # preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)
        utterances_spec.append(spec_mag)

        cur_slide += spec_hop_len

    return utterances_spec, intervals, current_time

def main(wav_path,saved_file_pkl, embedding_per_second=1.0, overlap_rate=0.5):
    print("executing program??",wav_path)
    # gpu configuration
    toolkits.initialize_GPU(args)
    #print("1")

    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }
    
    #loading the pretrained embeddings model (a resnet) 
    network_eval = spkModel.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=args)
    
    network_eval.load_weights(args.resume, by_name=True)

    #print("2")
    model_args, _, inference_args = uisrnn.parse_arguments()

    model_args.observation_dim = 512
    uisrnnModel = uisrnn.UISRNN(model_args)
    #print("3")
    #load the retrained uisrnn speaker diarization model
    uisrnnModel.load(SAVED_MODEL_NAME,'cpu')

    #open the input file
    #print("before open file")
    in_file = sf.SoundFile(wav_path) 
    #print("after open file")
    file_length = sf.info(wav_path).duration
    #only output dictionary of speakers and their start/stop time (speakerSlice)
    out_file=open(saved_file_pkl, 'wb')
    current_time = 0

    all_speakerslice = defaultdict(list)
    all_scores = defaultdict(dict)
    num_loop = 0
    beam_set = None
    iters = 0
    #print("before while loop")
    while True:
        start_time = current_time
        #print("in while loop")
        #print(current_time,file_length)
        #load data, get the most updated current time
        specs, intervals,current_time = load_data(in_file, file_length, current_time, embedding_per_second=embedding_per_second, overlap_rate=overlap_rate)
       
        mapTable, keys = genMap(intervals)
        #obtain features of these data, predicted via resnet model - have to find streaming option here as well
        feats = []
        #print(len(specs))
        for spec in specs:
            #print(spec.shape)
            spec = np.expand_dims(np.expand_dims(spec, 0), -1)
            v = network_eval.predict(spec)
            #print(v.shape,spec.shape)
            feats += [v]
        #print(len(feats))
        if feats==[]:
            break
	
        feats = np.array(feats)[:,0,:].astype(float)  # [splits, embedding dim]

        #predict with uisrnn the final diarization result - replace here with streaming option
        if num_loop == 0:
            predicted_label,current_beam,predicted_score,avg_predicted_score,last_iter,seg_score,seg_iters = uisrnnModel.online_predict(feats, beam_set,iters, True,inference_args)
            beam_set = current_beam
            iters = last_iter
        else:
            predicted_label,current_beam,predicted_score,avg_predicted_score,last_iter,seg_score,seg_iters = uisrnnModel.online_predict(feats, beam_set, iters,False, inference_args)
            beam_set = current_beam
            iters = last_iter
        all_scores[num_loop] = {'score':predicted_score,'score_normalized':avg_predicted_score,'seg_score':seg_score,'seg_iters':seg_iters}
        
        #not reusing labels
        #predicted_label = uisrnnModel.predict(feats,inference_args)

        time_spec_rate = 1000*(1.0/embedding_per_second)*(1.0-overlap_rate) # speaker embedding every ?ms
        center_duration = int(1000*(1.0/embedding_per_second)//2)
        speakerSlice = arrangeResult(predicted_label, time_spec_rate)

      

        for spk,timeDicts in speakerSlice.items():    # time map to orgin wav(contains mute)
            for tid,timeDict in enumerate(timeDicts):
                s = 0
                e = 0
                for i,key in enumerate(keys):
                    if(s!=0 and e!=0):
                        break
                    if(s==0 and key>timeDict['start']):
                        offset = timeDict['start'] - keys[i-1]
                        s = mapTable[keys[i-1]] + offset 
                    if(e==0 and key>timeDict['stop']):
                        offset = timeDict['stop'] - keys[i-1]
                        e = mapTable[keys[i-1]] + offset

                speakerSlice[spk][tid]['start'] = s + start_time*1000
                speakerSlice[spk][tid]['stop'] = e + start_time*1000

        for key in speakerSlice.keys():
            #print(key)
            if key not in all_speakerslice: 
                all_speakerslice.setdefault(key, [])
                all_speakerslice[key] = all_speakerslice[key]+speakerSlice[key]
            else:
                all_speakerslice[key] = all_speakerslice[key]+speakerSlice[key]
        num_loop += 1       
        #keep writing to the dictionary
        #print(speakerSlice)
      

        if current_time >= file_length: #break when there's nothing else to read from the file
            break
    print(out_file,' finished!')
    pickle.dump([all_speakerslice,all_scores], out_file, pickle.HIGHEST_PROTOCOL)
    in_file.close()
    out_file.close()

if __name__ == '__main__':
    #main('./wavs/atwood_trimmed2_enhanced.wav', 'atwood_trimmed2_pretrain_0.8emb_0.1over.pkl',embedding_per_second=0.5, overlap_rate=0.5)
    #print("???")
    main(args_all.save_wavpath,args_all.save_pklpath,embedding_per_second=0.5, overlap_rate=0.5)

