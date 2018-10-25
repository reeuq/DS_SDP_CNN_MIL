from gensim.models import word2vec
import numpy as np
from six.moves import cPickle as pickle


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


def reformat(labels):
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(27) == labels[:, None]).astype(np.int32)
    return labels


if __name__ == "__main__":
    with open('./sen2sdp_result_final/FilterNYT/train/train_sdp.pickle', 'rb') as f:
        f_data = pickle.load(f)
        train_data = f_data["data"]
        del f_data






    train_labels = reformat(np.array(train_labels))
    test_labels = reformat(np.array(test_labels))

    words = set()
    for each_sentence in train_sdp:
        words.update(each_sentence.split())
    for each_sentence in test_sdp:
        words.update(each_sentence.split())

    dictionary = dict()
    wordEmbedding = [np.zeros(300, dtype=np.float32)]
    for i, word in enumerate(words):
        dictionary[word] = i + 1
        try:
            wordVec = model.wv[word]
            wordEmbedding.append(wordVec)
        except Exception:
            wordEmbedding.append(-1 + 2 * np.random.random_sample(300))
    wordEmbedding = np.array(wordEmbedding)

    train_sdp_id, train_max_len = get_sentence_id(train_sdp, dictionary)
    test_sdp_id, test_max_len = get_sentence_id(test_sdp, dictionary)

    with open('./../resource/generated/input.pickle', 'wb') as f:
        parameter = {
            'dictionary': dictionary,
            'wordEmbedding': wordEmbedding,
        }
        train = {
            'train_sdp_id': train_sdp_id,
            'train_labels': train_labels,
        }
        test = {
            'test_sdp_id': test_sdp_id,
            'test_labels': test_labels,
        }
        pickle.dump(parameter, f, protocol=2)
        pickle.dump(train, f, protocol=2)
        pickle.dump(test, f, protocol=2)
    print("train max len: ", train_max_len)
    print("test max len: ", test_max_len)
