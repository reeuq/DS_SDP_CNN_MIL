from six.moves import cPickle as pickle

# f = open("./../dataset/sen2sdp_result/FilterNYT/train/train_sdp.pickle", 'rb')
# f_data = pickle.load(f)
# data = f_data["data"]
# f.close()

f = open("./../dataset/sen2sdp_result_final/FilterNYT/test/test_sdp.pickle", 'rb')
f_data = pickle.load(f)
data_new = f_data["data"]
f.close()

label = set()
for item in data_new:
    label.add(item[3])

print()

