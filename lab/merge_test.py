from six.moves import cPickle as pickle

data = []

f = open("./../dataset/sen2sdp_result/FilterNYT/test/0/test_sdp.pickle", 'rb')
p_data = pickle.load(f)
each_data = p_data['data']
data.extend(each_data)
f.close()

f = open("./../dataset/sen2sdp_result/FilterNYT/test/1/test_sdp.pickle", 'rb')
p_data = pickle.load(f)
each_data = p_data['data']
data.extend(each_data)
f.close()

with open('./../dataset/sen2sdp_result/FilterNYT/test/test_sdp.pickle', 'wb') as f:
    result = {
        'data': data,
    }
    pickle.dump(result, f, protocol=2)

print('end')
