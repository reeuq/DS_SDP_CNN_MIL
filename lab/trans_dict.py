import codecs

f = codecs.open('./../dataset/original/FilterNYT/dict_new.txt', 'r', encoding='utf-8')
fw = codecs.open('./../dataset/original/FilterNYT/dict_new_final.txt', 'w', encoding='utf-8')

for word in f.readlines():
    new_word = word.replace('\/', '/')
    fw.write(new_word)

f.close()
fw.close()
