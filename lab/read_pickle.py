from six.moves import cPickle as pickle

f = open("./../dataset/sen2sdp_result_final/FilterNYT/train/train_sdp.pickle", 'rb')
f_data = pickle.load(f)
data = f_data["data"]
f.close()

f = open("./../dataset/sen2sdp_result_final/FilterNYT/train/train_sdp_final.pickle", 'rb')
f_data = pickle.load(f)
data_new = f_data["data"]
f.close()

times = 0
for i in range(len(data)):
    if data[i] != data_new[i]:
        print(data[i])
        print(data_new[i])
        times += 1

print(times)

