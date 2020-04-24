import os
import re
import json
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import pickle as pkl
from tqdm import tqdm


def remove_markup(filepath):
    """ removes vtt markup """

    text = open(filepath).readlines()

    # remove header
    text = text[4:]

    regexps = [r'</c>',
                r'<c(\.color\w+)?>',
                r'<\d{2}:\d{2}:\d{2}\.\d{3}>',
                r'<c>',
                r'(&amp;)*',
                r'(&gt;)*',
                r'^( - )',
                r'\[([a-zA-Z]+\s)*[a-zA-Z]+\]',
                ]

    # look for and remove instances that match the regular expressions
    for i, line in enumerate(text):
        for pattern in regexps:
            line = re.sub(pattern, '', line)
        text[i] = line.strip()  # remove preceding and trailing whitespace

    return text


def split_examples(text, spacy_nlp, minTextLength=3, count=False, minduration=0.5, maxduration=30, wordlimit=100):
    """split preprocessed text into individual timestamp and caption pairs

        This code assumes the following structure for the text file:

        Timestamp 1:
            caption line 1
            caption line 2
            ...
        Timestamp 2:
            caption line 1
            ...
        ...
    """

    # use regexp with capture groups to find the start and end times
    # spacy_nlp = spacy.load('en_core_web_sm')
    timestamps = r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})'
    mystring = ""
    examples = []
    below_min_duration = 0
    above_max_duration = 0
    below_min_wordlim = 0
    above_max_wordlim = 0
    len_examples = 0

    if count:
        counts = [[], 0, []]

    # iterate throught the text in reversed order
    for i, line in enumerate(reversed(text)):

        # check if line is a timestamp
        if re.match(timestamps, line):
            # skip captions with fewer than minTextLength words
            #    (includes empty captions)

            matches = re.match(timestamps, line)
            start = convert_timestamp(matches.groups()[0])
            end = convert_timestamp(matches.groups()[1])
            duration = end - start

            if duration > maxduration:
                mystring = ""  # reset string
                below_min_duration += 1
                continue

            if duration < minduration:
                mystring = ""  # reset string
                above_max_duration += 1
                continue

            # remove stopwords
            raw = mystring

            # nltk
            mystring_nltk = word_tokenize(mystring)
            mystring_nltk = " ".join(w for w in mystring_nltk if not w in stop_words)


            # spacy
            mystring_spacy = spacy_nlp(mystring)
            mystring_spacy = " ".join(token.text for token in mystring_spacy if not token.is_stop)

            length_nltk = len(word_tokenize(mystring_nltk))
            length_spacy = len(word_tokenize(mystring_spacy))

            if (length_nltk or length_spacy) > wordlimit:
                mystring = ""  # reset string
                above_max_wordlim += 1
                continue

            if (length_nltk and length_spacy) > minTextLength:
                # if you are keeping statistics
                if count:
                    counts[0].append(length_nltk)
                    counts[2].append(duration)

                examples.append([start, end, raw, mystring_nltk, mystring_spacy])

            else:
                below_min_wordlim += 1

            mystring = ""  # reset string

        # colate text in between timestamps
        else:
            if mystring == "":
                mystring = line.lower() + mystring

            # add space between lines if string is not empty
            else:
                mystring = line.lower() + " " + mystring

    examples.reverse()

    if count:
        counts[1] = len(examples)
        return examples, counts
    else:
        len_examples = len(examples)
        breakdown = {"below_min_duration": below_min_duration, "above_max_duration": above_max_duration,
                     "below_min_wordlim": below_min_wordlim, "above_max_wordlim": above_max_wordlim,
                     "len_examples": len_examples}
        return examples, breakdown


def save_stats(counts, path="", bins=(50, 50, 100), show=False):
    # counts is a dictionary of the different statistics, Keys 0-2
    # 0) # words per sentence
    # 1) # segments per video
    # 2) segment duration

    # words per sentence
    plt.hist(counts['0'], bins=bins[0])
    plt.xlabel("words per sentence")
    plt.ylabel("frequency")
    plt.title("Words Per Sentence")
    plt.savefig(path + 'words_per_caption_hist.png')
    if show:
        plt.show()
    plt.close()

    # segments per video
    plt.hist(counts['1'], bins=bins[1])
    plt.xlabel("segments per video")
    plt.ylabel("frequency")
    plt.title("Segments Per Video")
    plt.savefig(path + 'segments_per_video.png')

    if show:
        plt.show()
    plt.close()

    # segment duration
    plt.hist(counts['2'], bins=bins[2])
    plt.xlabel("segment duration (s)")
    plt.ylabel("frequency")
    plt.title("Segment Duration")
    plt.savefig(path + 'segment_durations.png')
    if show:
        plt.show()
    plt.close()
    return


def remove_redundant(files):
    """ file names are structured as (name).(origin).vtt
        I want to capture the (name) and (origin) sections to remove redundant captions, and
        impose structure (i.e. en > en-US > en-GB > en-CA)"""

    # pattern to match and capture different sections of filename
    pattern = r'([^\.]+)\.([^\.]+)\.vtt'
    ordering = ['en', 'en-US', 'en-GB', 'en-IE', 'en-CA']
    temp = {}

    for file in files:
        match = re.match(pattern, file)

        if match:
            name = match.groups()[0]
            ext = match.groups()[1]
        else:
            print(file)
            continue

        try:
            current_ext = temp[name]

            # check if the current extension has higher priority
            if ordering.index(ext) < ordering.index(current_ext):
                temp[name] = ext

        # add name to dictionary if it does not exist yet
        except KeyError:
            temp[name] = ext

    # generate files from the dictionary of names and extensions
    filelist = []
    for (key, value) in temp.items():
        filelist.append(key + '.' + value + '.vtt')

    return filelist


def convert_timestamp(time):
    timestamp_match = r'(\d{2}):(\d{2}):(\d{2}\.\d{3})'
    matches = re.match(timestamp_match, time)
    time_secs = 3600 * float(matches.groups()[0]) + 60 * float(matches.groups()[1]) + float(matches.groups()[2])
    time_secs = float("%0.2f" % time_secs)  # keep only two decimals
    return time_secs


########################################################################################################################
########################################################################################################################
if __name__ == "__main__":

    # root = "/home/michael/Documents/VLP/"
    # root = "/home/michael/Documents/dat/"
    root = "/y/luozhou/HowTo100M_yc2/dat/"

    subfolder = "subtitles/"
    path = root + subfolder
    filenames = os.listdir(path)

    data = {}
    data['dataset'] = 'HowTo100M-YouCook2'
    data['videos'] = []
    count = False
    counts = {str(i): [] for i in range(3)}
    breakdown_dict = {"below_min_duration": 0, "above_max_duration": 0,
                 "below_min_wordlim": 0, "above_max_wordlim": 0,
                 "len_examples": 0}

    k = 0  # number for sentid and vidid

    # set nltk stop words
    stop_words = set(stopwords.words('english'))

    # setup spacy
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])

    # remove redundant filenames
    filenames = remove_redundant(filenames)

    # pattern for splitting names into (name).(en).vtt
    pattern = r'([^\.]+)\.([^\.]+)\.vtt'

    for i in tqdm(range(len(filenames))):

        file = filenames[i]

        # use try block in case we get some funky characters we can't read
        try:
            # remove the extra garbage from the vtt file
            text = remove_markup(path + file)
        except:
            break

        # split the text into individual examples
        if count:
            text, stats = split_examples(text, nlp, count=count)
            counts['0'].extend(stats[0])
            counts['1'].append(stats[1])
            counts['2'].extend(stats[2])
        else:
            text, breakdown = split_examples(text, nlp)

            # breakdown_dict and breakdown have the same keys
            for key in breakdown.keys():
                breakdown_dict[key] += breakdown[key]

        for j, entry in enumerate(text):
            # add dictionary of pertinent information for each entry
            # entry is formatted as [start, end, caption]
            mydict = {}
            matches = re.match(pattern, file)
            mydict['filename'] = matches.groups()[0] + "_segment_" + str(j).zfill(2)
            mydict['filepath'] = ""
            mydict['sentids'] = [k]
            mydict['vidid'] = k
            mydict['split'] = "train"
            mydict['segment'] = [entry[0], entry[1]]
            mydict['sentences'] = []
            sentdict = {'raw_wstopword': entry[2], 'raw_nltk': entry[3], 'raw_spacy': entry[4], 'sentid': k, 'vidid': k}
            mydict['sentences'].append(sentdict)


            # append new entry to the dictionary
            data['videos'].append(mydict)

            # update the sentid and vidid number
            k += 1

    if count:
        a = 0
        # make histograms if you want to save the statistics
        # save_stats(counts)
        # with open('counts.pickle', 'wb') as f:
        #     pkl.dump(counts, f)
    else:
        with open('breakdown.pickle', 'wb') as f:
            pkl.dump(breakdown_dict, f)

    # with open('dataset_howto100m.json', 'w') as fp:
    #     json.dump(data, fp)
