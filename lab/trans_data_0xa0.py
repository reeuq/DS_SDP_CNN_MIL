import codecs
from six.moves import cPickle as pickle

f = codecs.open('./../dataset/original/FilterNYT/dict_temp.txt', 'r', encoding='utf-8')

words = []
for word in f.readlines():
    words.append(word.strip())

with open('./../dataset/sen2sdp_result_final/FilterNYT/train/train_sdp.pickle', 'rb') as f:
    f_data = pickle.load(f)
    train_data = f_data["data"]
    del f_data

with open('./../dataset/sen2sdp_result_final/FilterNYT/test/test_sdp.pickle', 'rb') as f:
    f_data = pickle.load(f)
    test_data = f_data["data"]
    del f_data

for word in words:
    new_word = word.replace(' ', '_')
    for data in train_data:
        for i in range(data[1]):
            line = data[2][i]
            data[2][i] = line.replace(word, new_word)

    for data in test_data:
        for i in range(data[1]):
            line = data[2][i]
            data[2][i] = line.replace(word, new_word)

fw = open("./../dataset/sen2sdp_result_final/FilterNYT/train/train_sdp_final.pickle", 'wb')
result = {
    'data': train_data,
}
pickle.dump(result, fw, protocol=2)
fw.close()

fw = open("./../dataset/sen2sdp_result_final/FilterNYT/test/test_sdp_final.pickle", 'wb')
result = {
    'data': test_data,
}
pickle.dump(result, fw, protocol=2)
fw.close()
