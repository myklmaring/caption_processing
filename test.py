import os
import re
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

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
                r'align:start position:0%',
                r'^( - )',
                r'\[([a-zA-Z]+\s)*[a-zA-Z]+\]',
                ]

    # look for and remove instances that match the regular expressions
    for i, line in enumerate(text):
        for pattern in regexps:
            line = re.sub(pattern, '', line)
        text[i] = line.strip()  #remove preceding and trailing whitespace

    return text


def split_examples(text, minTextLength=3):
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
    timestamps = r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})'
    mystring = ""
    examples = []

    # iterate throught the text in reversed order
    for i, line in enumerate(reversed(text)):

        # check if line is a timestamp
        if re.match(timestamps, line):
            # skip captions with fewer than minTextLength words
            #    (includes empty captions)

            matches = re.match(timestamps, line)
            start = matches.groups()[0]
            end = matches.groups()[1]

            # remove stopwords
            mystring = word_tokenize(mystring)
            mystring = " ".join(w for w in mystring if not w in stop_words)

            if len(mystring.split()) > minTextLength:
                examples.append([start, end, mystring])

            mystring = ""  # reset string

        # colate text in between timestamps
        else:
            if mystring == "":
                mystring = line + mystring

            # add space between lines if string is not empty
            else:
                mystring = line + " " + mystring

    examples.reverse()

    return examples




root = "/home/michael/Documents/dat/"
subfolder = "subtitles/"
path = root + subfolder
filenames = os.listdir(path)

text = remove_markup(path + filenames[0])
text = split_examples(text)
