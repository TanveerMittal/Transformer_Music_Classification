import os
import glob
import pickle
import numpy as np
import librosa as lb
from sklearn.model_selection import train_test_split

def path_to_arr(path, sample_rate=4000, num_split=6, padded_len=122598):
    '''Converts the wav file at specified path to vector representations'''
    x, sr = lb.load(path, sr=sample_rate)
    # pad song to max length that is divisible by 6
    x = lb.util.fix_length(x, padded_len)
    # split song into 5 second clips
    clips = np.split(x, num_split)
    # transform each clip into a (513, 79) array
    vecs = [lb.stft(clip, n_fft=1024) for clip in clips]
    return vecs

def build_dataset():
    '''Builds train and validation set from data directory'''
    subdirs = next(os.walk("data/"))[1]
    label = 0
    clips = []
    labels = []
    genres = []
    for sub in subdirs:
        folder = os.path.join("data", sub, "*")
        paths = glob.glob(folder)
        for path in paths:
            vecs = path_to_arr(path)
            clips += vecs
            labels += [label for i in range(len(vecs))]`
        genres.append(sub)
        label += 1
    x_train, x_test, y_train, y_test = train_test_split(clips, labels, test_size=0.33, random_state=25)
    pickle.dump((x_train, y_train), open(os.path.join("data", "train.pkl"), "wb"))
    pickle.dump((x_test, y_test), open(os.path.join("data", "test.pkl"), "wb"))

if __name__ == "__main__":
    build_dataset()
