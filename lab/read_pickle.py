from six.moves import cPickle as pickle

f = open("./../dataset/sen2sdp_result_final/FilterNYT/test/test_sdp_final_final.pickle", 'rb')
f_data = pickle.load(f)
data = f_data["data"]
f.close()

f = open("./../dataset/sen2sdp_result_final/FilterNYT/train/train_sdp_final_final.pickle", 'rb')
f_data = pickle.load(f)
data_new = f_data["data"]
f.close()

times = 0
for i in range(len(data)):
    assert type(data[i][3]) == list

for i in range(len(data_new)):
    assert type(data_new[i][3]) == list

print()
