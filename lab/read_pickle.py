import os
import numpy as np

root_path = './../new_dataset/original/FilterNYT/'
path = os.path.join(root_path, 'train/')

x = np.load(root_path + 'w2v.npy')

print()