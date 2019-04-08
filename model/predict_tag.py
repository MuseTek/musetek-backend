'''
Given one audio clip, output what the network thinks
'''
from __future__ import print_function
import numpy as np
import librosa
import os
from os.path import isfile
from .panotti.models import *
from .panotti.datautils import *
import pickle
from sklearn import preprocessing
from audioread import NoBackendError
from .panotti.metrics import shomik_tag_score
import pdb
def get_canonical_shape(signal):
    if len(signal.shape) == 1:
        return (1, signal.shape[0])
    else:
        return signal.shape

def find_max_shape(path, mono=False, sr=None, dur=None, clean=False):
    if (mono) and (sr is not None) and (dur is not None):   # special case for speedy testing
        return [1, int(sr*dur)]
    shapes = []
    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if not (filename.startswith('.') or ('.csv' in filename)):    # ignore hidden files & CSVs
                filepath = os.path.join(dirname, filename)
                try:
                    signal, sr = librosa.load(filepath, mono=mono, sr=sr)
                except NoBackendError as e:
                    continue
                    print("Could not open audio file {}".format(filepath))
                    raise e
                if (clean):                           # Just take the first file and exit
                    return get_canonical_shape(signal)
                shapes.append(get_canonical_shape(signal))

    return (max(s[0] for s in shapes), max(s[1] for s in shapes))

def predict_one(signal, sr, model, expected_melgram_shape):# class_names, model)#, weights_file="weights.hdf5"):
    X = make_layered_melgram(signal,sr,mels=96)
    print("signal.shape, melgram_shape, sr = ",signal.shape, X.shape, sr)

    if (X.shape[1:] != expected_melgram_shape):   # resize if necessary, pad with zeros
        Xnew = np.zeros([1]+list(expected_melgram_shape))
        min1 = min(  Xnew.shape[1], X.shape[1]  )
        min2 = min(  Xnew.shape[2], X.shape[2]  )
        min3 = min(  Xnew.shape[3], X.shape[3]  )
        Xnew[0,:min1,:min2,:min3] = X[0,:min1,:min2,:min3]  # truncate
        X = Xnew
    max_shape = find_max_shape('/hdd/datasets/MuseTek/Data/Loops/Apple',sr=sr,mono=True,dur=6)
    print(max_shape)
    shape = get_canonical_shape(signal)
    if (shape != signal.shape):             # this only evals to true for mono
        signal = np.reshape(signal, shape)
    print(max_shape)
    #layers = make_layered_melgram(padded_signal, sr, mels=96, phase=phase)
    padded_signal = np.zeros(max_shape)
    use_shape = list(max_shape[:]) 
    #import pdb
    #pdb.set_trace()
    use_shape[0] = min(shape[0], max_shape[0] )
    use_shape[1] = min(shape[1], max_shape[1] )
    padded_signal[:use_shape[0], :use_shape[1]] = signal[:use_shape[0], :use_shape[1]] 
    X = make_layered_melgram(padded_signal, sr, mels=128)
    #data = np.load('/projects/MuseTek/preprocess/tag-process/Preproc/Test/Groovy Electric Bass 07.wav.npz')
    #X = data['melgram']
    return model.predict(X,batch_size=1,verbose=False)[0]

def get_tags(weights_file, dur, resample, mono,label_encoder_path, audio_file):
    label_encoder = pickle.load(open(label_encoder_path,'rb'))
    #load model
    model, class_names = load_model_ext(weights_file)
    if model is None:
        print("no weights file found, Aborting")
        exit(1)
    num_tags= 63
    expected_melgram_shape = model.layers[0].input_shape[1:]
    pdb.set_trace()
    #check if the file exists
    if os.path.isfile(audio_file):
        signal, sr = load_audio(audio_file,mono=mono,sr=resample)
        y_proba = predict_one(signal,sr,model,expected_melgram_shape)
        answer = label_encoder.inverse_transform(np.argwhere(y_proba>.25))
        return answer.tolist()
    else:
        return None

def main(weights_file, dur, resample, mono,label_encoder_path,audio_path):
    np.random.seed(1)
    #weights_file=args.weights
    #dur = args.dur
    #resample = args.resample
    #mono = args.mono
    #label_encoder_path = args.label_encoder_path
    
    label_encoder = pickle.load(open(label_encoder_path,'rb'))
    # Load the model
    model, class_names = load_model_ext(weights_file)
    shomik_scores = get_shomik_scores(model)
    pdb.set_trace()
    if model is None:
        print("No weights file found.  Aborting")
        exit(1)

    #model.summary()

    #TODO: Keras load_models is spewing warnings about not having been compiled. we can ignore those,
    #   how to turn them off?  Answer: can invoke with python -W ignore ...

    #class_names = get_class_names(args.classpath) # now encoding names in model weights file
    num_tags = 63
    expected_melgram_shape = model.layers[0].input_shape[1:]
    print("Expected_melgram_shape = ",expected_melgram_shape)
    file_count = 0
    json_file = open("data.json", "w")
    json_file.write('{\n"items":[')

    idnum = 0
    #numfiles = len(args.file)
    numfiles = 1
    files = [audio_file]
    print("Reading",numfiles,"files")
    
    
    for infile in files:
        if os.path.isfile(infile):
            file_count += 1
            print("File",infile,":",end="")

            signal, sr = load_audio(infile, mono=mono, sr=resample)

            y_proba = predict_one(signal, sr, model, expected_melgram_shape) # class_names, model, weights_file=args.weights)
            pdb.set_trace()
            for i in range(63):
                print( label_encoder.inverse_transform([i]),": ",y_proba[i],", ",end="",sep="")
            answer = label_encoder.inverse_transform(np.argwhere(y_proba>.05))
            print("--> ANSWER:", answer)
            outstr = ""
            print(answer)
            #outstr = '\n  {\n   "id": "'+str(idnum)+'",\n      "name":"'+infile+'",\n      "tags":[\n   "'+answer+'"]\n  }'
            if (idnum < numfiles-1):
                outstr += ','
            json_file.write(outstr)
            json_file.flush()     # keep json file up to date
        else:
            pass #print(" *** File",infile,"does not exist.  Skipping.")
        idnum += 1

    json_file.write("]\n}\n")
    json_file.close()

    return

def get_shomik_scores(model,batch_size=1,threshold = .5):
    '''
    TODO: create a test generator. for each sample feed the prediction and actual tags into the the shomik score
    from metrics.py
    '''
    test_gen = build_free_sounds_dataset('/projects/MuseTek/preprocess/tag-process/Preproc/Validate/Hip Reverse Hat Beat 01.mp3.npz','/projects/MuseTek/tag-extraction/numpy_tags.p',path='/projects/MuseTek/preprocess/tag-process/Preproc/Test',batch_size=1)
    #total number of samples to test on
    limit = 1000
    shomik_scores = []
    for i in range(limit):
        #pdb.set_trace()
        X_test, Y_test = next(test_gen)
        X_pred = model.predict(X_test,batch_size=1)[0]
        shomik_scores.append(shomik_tag_score(Y_test,X_pred,threshold))
    return shomik_scores
 
  

#if __name__ == '__main__':
    #shomik_scores = get_shomik_scores()
    #import pdb
    #pdb.set_trace()
    #import argparse
    #parser = argparse.ArgumentParser(description="predicts which class file(s) belong(s) to")
    #parser.add_argument('-w', '--weights', #nargs=1, type=argparse.FileType('r'),
    #    help='weights file in hdf5 format', default="weights.hdf5")
    #parser.add_argument('-c', '--classpath', #type=argparse.string, help='directory with list of classes', default="Preproc/Test/")
    #parser.add_argument("-m", "--mono", help="convert input audio to mono",action="store_true")
    #parser.add_argument("-r", "--resample", type=int, default=44100, help="convert input audio to mono")
    #parser.add_argument('-d', "--dur",  type=float, default=None,   help='Max duration (in seconds) of each clip')
    #parser.add_argument("-l","--label_encoder_path")
    #parser.add_argument('-f','--file', help="file(s) to classify", nargs='+')
    #args = parser.parse_args()
    #test_result = get_tags("weights.hdf5",6,44100,True,'/projects/MuseTek/tag-extraction/label_encoder.p','/hdd/datasets/MuseTek/Data/Landr copy/Havoc_Infamous Classic Kit/Vox Indictment_85bpm.wav.mp3')
    #pdb.set_trace()
    #main(args)
