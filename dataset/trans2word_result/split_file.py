import codecs


if __name__ == "__main__":
    f = codecs.open("./FilterNYT/train/train.txt", 'r', encoding='utf-8')
    pdqm = 0
    fw = codecs.open("./FilterNYT/train/train" + str(pdqm // 33000) + ".txt", 'w', encoding='utf-8')
    while 1:
        if pdqm % 33000 == 0 and pdqm != 0:
            fw.close()
            fw = codecs.open("./FilterNYT/train/train"+str(pdqm // 33000)+".txt", 'w', encoding='utf-8')

        line = f.readline()
        if not line:
            break
        fw.write(line)

        line = f.readline()
        fw.write(line)

        bagLabel = line.split(' ')
        num = int(bagLabel[-1])

        for i in range(0, num):
            sent = f.readline()
            fw.write(sent)
        pdqm += 1

    fw.close()
    f.close()
