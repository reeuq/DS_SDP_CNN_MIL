import numpy as np
from six.moves import cPickle as pickle
import codecs


def get_sentence_id(sen, dic):
    sen_ids = []
    max_len = 0
    for each_sen in sen:
        sen_id = []
        for each_word in each_sen.split():
            sen_id.append(dic.get(each_word))
        sen_ids.append(sen_id)
        if len(each_sen.split()) > max_len:
            max_len = len(each_sen.split())
    return sen_ids, max_len


if __name__ == "__main__":
    with open('./sen2sdp_result_final/FilterNYT/train/train_sdp_final.pickle', 'rb') as f:
        f_data = pickle.load(f)
        train_data = f_data["data"]
        del f_data

    with open('./sen2sdp_result_final/FilterNYT/test/test_sdp_final.pickle', 'rb') as f:
        f_data = pickle.load(f)
        test_data = f_data["data"]
        del f_data

    f = codecs.open('./original/FilterNYT/dict_new.txt', 'r', encoding='utf-8')
    dict = {}
    word_number = 1
    for word in f.readlines():
        dict[word.strip()] = word_number
        word_number += 1
    f.close()

    max_length = 0
    for data in train_data:
        data[2], temp_max_length = get_sentence_id(data[2], dict)
        data[3] = (np.arange(27) == data[3]).astype(np.int32)
        if temp_max_length > max_length:
            max_length = temp_max_length

    for data in test_data:
        data[2], temp_max_length = get_sentence_id(data[2], dict)
        data[3] = (np.arange(27) == data[3]).astype(np.int32)
        if temp_max_length > max_length:
            max_length = temp_max_length

    f = open('./original/FilterNYT/vector.txt', 'r')
    all_lines = f.readlines()
    f.close()

    Wv = np.zeros((len(all_lines) + 38, 50))
    i = 1
    for line in all_lines:
        Wv[i, :] = list(map(float, line.split(' ')))
        i += 1

    rng = np.random.RandomState(3435)

    for j in range(i, (len(all_lines) + 38)):
        Wv[j, :] = rng.uniform(low=-0.5, high=0.5, size=(1, 50))

    with open('./final_dataset/data.pickle', 'wb') as f:
        parameter = {
            'dictionary': dict,
            'wordEmbedding': Wv,
        }
        train = {
            'data': train_data,
        }
        test = {
            'data': test_data,
        }
        pickle.dump(parameter, f, protocol=2)
        pickle.dump(train, f, protocol=2)
        pickle.dump(test, f, protocol=2)
    print("train max len: ", max_length)
