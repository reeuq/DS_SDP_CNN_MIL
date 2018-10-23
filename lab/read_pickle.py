from six.moves import cPickle as pickle
import copy

f = open("./../dataset/sen2sdp_result/FilterNYT/train/train_sdp.pickle", 'rb')
f_data = pickle.load(f)
data = f_data["data"]
f.close()

f = open("./../dataset/sen2sdp_result_final/FilterNYT/train/train_sdp_1.pickle", 'rb')
f_data = pickle.load(f)
data_new = f_data["data"]
f.close()

times = 0
for item in data_new:
    if not isinstance(item[2][0], str):
        times += 1
print(times)

