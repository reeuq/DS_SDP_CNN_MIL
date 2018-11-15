from six.moves import cPickle as pickle
import codecs


def trans(original_path, original_path_data, result_path):
    f = codecs.open(original_path, "r", encoding="utf-8")
    fr = open(original_path_data, "rb")
    fw = open(result_path, "wb")

    f_data = pickle.load(fr)
    data = f_data["data"]
    fr.close()

    times = 0
    error_times = 0
    while 1:
        line = f.readline()
        if not line:
            break
        entities = line.split(' ')
        line = f.readline()
        bagLabel = line.split(' ')
        labels = list(map(int, bagLabel[:-1]))
        num = int(bagLabel[-1])

        if entities[0] == data[times][0][0] and entities[1] == data[times][0][1]:
            data[times][3] = labels
        else:
            error_times += 1

        for i in range(0, num):
            f.readline()

        times += 1
    print("error times: ", error_times)
    result = {
        'data': data,
    }
    pickle.dump(result, fw, protocol=2)
    fw.close()
    f.close()


print("Start trans test")
trans("./../dataset/trans2word_result/FilterNYT/test/test.txt",
      "./../dataset/sen2sdp_result_final/FilterNYT/test/test_sdp_final.pickle",
      "./../dataset/sen2sdp_result_final/FilterNYT/test/test_sdp_final_final.pickle")
print("Start trans train")
trans("./../dataset/trans2word_result/FilterNYT/train/train_process.txt",
      "./../dataset/sen2sdp_result_final/FilterNYT/train/train_sdp_final.pickle",
      "./../dataset/sen2sdp_result_final/FilterNYT/train/train_sdp_final_final.pickle")
print("end")