import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

file = '/home/michael/Documents/VLP/counts.pickle'
path = ""
show = True
counts = pkl.load(open(file, 'rb'))
bins = [50, 100, 100]

# # words per sentence
# plt.hist(counts['0'], bins=bins[0], range=(0, 50))
# plt.xlabel("words per sentence")
# plt.ylabel("frequency")
# plt.title("Words Per Sentence")
# plt.savefig(path + 'words_per_sentence_hist.png')
# if show:
#     plt.show()
# plt.close()
#
# # segments per video
# plt.hist(counts['1'], bins=bins[1], range=(0, 800))
# plt.xlabel("segments per video")
# plt.ylabel("frequency")
# plt.title("Segments Per Video")
# plt.savefig(path + 'segments_per_video.png')
#
# if show:
#     plt.show()
# plt.close()
#
# # segment duration
# plt.hist(counts['2'], bins=bins[2], range=(0, 30))
# plt.xlabel("segment duration (s)")
# plt.ylabel("frequency")
# plt.title("Segment Duration")
# plt.savefig(path + 'segment_durations.png')
# if show:
#     plt.show()
# plt.close()

# words per sentence vs segment duration
inds = np.random.randint(0, 2, size=len(counts['0']), dtype=bool)
plt.plot(np.array(counts['0'])[inds], np.array(counts['2'])[inds], 'rx')
plt.xlabel("Words per Sentence")
plt.ylabel("Segment Duration (s)")
plt.title("Words Per Sentence vs. Segment Duration (thresholded)")
plt.savefig(path + 'wps_v_sdur_no_outliers.png')
if show:
    plt.show()
plt.close()
