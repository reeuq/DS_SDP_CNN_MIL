from six.moves import cPickle as pickle


with open("./train_sdp1.pickle", 'rb') as f:
    data = pickle.load(f)
    data = data["data"]
    print()