import os
import pickle as pkl
import numpy as np

file = "/home/michael/Documents/VLP/counts.pickle"

data = pkl.load(open(file, 'rb'))

# words per sentence
wps = np.array(data['0'])
wps_mean = np.mean(wps)
wps_std = np.std(wps)

# sentences per video
spv = np.array(data['1'])
spv_mean = np.mean(spv)
spv_std = np.std(spv)

# sentence durations
sdur = np.array(data['2'])
sdur_mean = np.mean(sdur)
sdur_std = np.std(sdur)

