import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

path = "/home/michael/Documents/git-repos/VLP/create-json-file/"
file = "breakdown.pickle"

breakdown = pkl.load(open(path + file, 'rb'))

breakdown_dict = {"below_min_duration": 0, "above_max_duration": 0,
                  "below_min_wordlim": 0, "above_max_wordlim": 0,
                  "len_examples": 0}

below_min_duration = breakdown['below_min_duration']
above_max_duration = breakdown['above_max_duration']
below_min_wordlim = breakdown['below_min_wordlim']
above_max_wordlim = breakdown['above_max_wordlim']
len_examples = breakdown['len_examples']

fig, ax = plt.subplots()
# plt.rcParams['text.color'] = '#909090'
# plt.rcParams['axes.labelcolor'] = '#909090'
# plt.rcParams['xtick.color'] = '#909090'
# plt.rcParams['ytick.color'] = '#909090'
# plt.rcParams['font.size'] = 8
explode = (0.02, 0.02, 0.02, 0.02, 0.02)
# color_palette_list = ['#009ACD', '#ADD8E6', '#63D1F4', '#0EBFE9',
#                       '#C1F0F6', '#0099CC']

labels = ['Below Min Duration', 'Above Max Duration', 'Below Min WordLim', 'Above Max WordLim', 'Examples']
values = list(breakdown.values())
p, tx, autotexts = ax.pie(values, explode=explode, labels=labels,
       # colors=color_palette_list[0:5],
       autopct='%1.2f%%',
       shadow=False,
       startangle=0,
       # pctdistance=1.2,
       radius=1.25,
       labeldistance=1.05)

ax.axis('equal')
ax.set_title("Make-Up of all Segments in the HowTo100M Dataset")
ax.legend(frameon=False, bbox_to_anchor=(0.25, 0.6))

for i, a in enumerate(autotexts):
    a.set_text("{}".format(values[i]))

plt.savefig('segment-breakdown.png')
plt.show()

