wordlist = ['BLANK']
wordlist.extend([word.strip('\n') for word in open("./original/FilterNYT/dict.txt")])
wordlist.append('UNK')


def trans(original_path, result_path):
    f = open(original_path, "r")
    fw = open(result_path, "w")
    while 1:
        line = f.readline()
        if not line:
            break
        entities = list(map(int, line.split(' ')))
        for entity in entities:
            fw.write(wordlist[entity])
            fw.write(" ")
        fw.write("\n")

        line = f.readline()
        fw.write(line)

        bagLabel = line.split(' ')
        num = int(bagLabel[-1])
        for i in range(0, num):
            sent = f.readline().split(' ')
            for index in list(map(int, sent[2:])):
                fw.write(wordlist[index])
                fw.write(" ")
            fw.write("\n")
    fw.close()
    f.close()


print("Start trans test")
trans("./original/FilterNYT/test/test.txt", "./trans2word_result/FilterNYT/test/test.txt")
print("Start trans train")
trans("./original/FilterNYT/train/train.txt", "./trans2word_result/FilterNYT/train/train.txt")
print("end")
