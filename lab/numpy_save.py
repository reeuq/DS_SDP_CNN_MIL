import numpy as np
import os
import torch


labels = np.load(os.path.join('./', 'labels.npy'))
x = np.load(os.path.join('./', 'bags_feature.npy'))
x = zip(x, labels)

for idx, (data, label_set) in enumerate(x):
    # label = [l[0] for l in label_set]
    # label = torch.LongTensor(label)

    for idx, bag in enumerate(data):
        insNum = bag[1]
        label = labels[idx]
        max_ins_id = 0
        if insNum > 1:
            data = map(lambda i: torch.LongTensor(i), bag)
