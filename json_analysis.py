import os
import re
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np

# path = "/home/michael/Documents/VLP/"
path = "/home/michael/PycharmProjects/untitled/"
json_file = "dataset_howto100m.json"
with open(path + json_file) as f:
    data = json.load(f)


captions_list = data['videos']

min_duration = 0.05
max_duration = 100
word_max = 1000

min_duration_list = []
max_duration_list = []
word_max_list = []

durations = np.zeros(len(captions_list))
lengths = np.zeros(len(captions_list))
wordlength = np.zeros(len(captions_list))

for i, item in enumerate(captions_list):

    file_ban_list = [r'v_fkx1fvl9u6A.en.vtt', r'v_TuY9VET9aqY.en.vtt', r'v_Q3PX-oE9d-Q.en.vtt']

    filename = item['filename']
    ban_match = False
    for file in file_ban_list:
        if re.match(file, filename):
            ban_match = True
            break

    if ban_match:
        continue

    start, end = item['segment']
    start = float(start)
    end = float(end)
    duration = end - start

    caption = item['sentences']['raw_nltk']
    length = len(word_tokenize(caption))

    if duration < min_duration:
        min_duration_list.append(item)

    if duration > max_duration:
        max_duration_list.append(item)

    if length > word_max:
        print('this was evaluated')
        word_max_list.append(item)

    durations[i] = duration
    lengths[i] = length
    wordlength[i] = length
