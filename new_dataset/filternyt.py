# -*- coding: utf-8 -*-

import os
import numpy as np
import codecs


class FilterNYTLoad(object):
    '''
    load and preprocess data
    '''
    def __init__(self, root_path, max_len=80, limit=50, pos_dim=5, pad=1):

        self.max_len = max_len
        self.limit = limit
        self.root_path = root_path
        self.pos_dim = pos_dim
        self.pad = pad

        self.w2v_path = os.path.join(root_path, 'vector.txt')
        self.word_path = os.path.join(root_path, 'dict.txt')
        self.train_path = os.path.join(root_path, 'train', 'train.txt')
        self.test_path = os.path.join(root_path, 'test', 'test.txt')

        print('loading start....')
        self.w2v, self.word2id, self.id2word = self.load_w2v()
        np.save(os.path.join(self.root_path, 'w2v.npy'), self.w2v)

        print("parsing train text...")
        self.bags_feature = self.parse_sen(self.train_path)
        np.save(os.path.join(self.root_path, 'train', 'bags_feature.npy'), self.bags_feature)

        print("parsing test text...")
        self.bags_feature = self.parse_sen(self.test_path)
        np.save(os.path.join(self.root_path, 'test', 'bags_feature.npy'), self.bags_feature)
        print('save finish!')

    def load_w2v(self):
        '''
        reading from vec.bin
        add two extra tokens:
            : UNK for unkown tokens
            : BLANK for the max len sentence
        '''
        wordlist = []
        vecs = []

        wordlist.append('BLANK')
        wordlist.extend([word.strip('\n') for word in codecs.open(self.word_path, encoding='utf-8')])

        for line in open(self.w2v_path):
            line = line.strip('\n').split()
            vec = list(map(float, line))
            vecs.append(vec)

        dim = len(vecs[0])
        vecs.insert(0, np.zeros(dim))
        wordlist.append('UNK')

        vecs.append(np.random.uniform(low=-1.0, high=1.0, size=dim))
        # rng = np.random.RandomState(3435)
        # vecs.append(rng.uniform(low=-0.5, high=0.5, size=dim))
        word2id = {j: i for i, j in enumerate(wordlist)}
        id2word = {i: j for i, j in enumerate(wordlist)}

        return np.array(vecs, dtype=np.float32), word2id, id2word

    def parse_sen(self, path):
        '''
        parse the records in data
        '''
        all_sens =[]
        f = codecs.open(path, encoding='utf-8')
        while 1:
            line = f.readline()
            if not line:
                break
            entities = list(map(int, line.split(' ')))
            line = f.readline()
            bagLabel = line.split(' ')

            rel = list(map(int, bagLabel[0:-1]))
            num = int(bagLabel[-1])
            sentences = []
            for i in range(0, num):
                sent = f.readline().split(' ')
                sentences.append(list(map(int, sent[2:-1])))
            bag = [entities, num, sentences, rel]
            all_sens += [bag]
        f.close()
        bags_feature = self.get_sentence_feature(all_sens)

        return bags_feature

    def get_sentence_feature(self, bags):
        '''
        : word embedding
        : postion embedding
        return:
        sen list
        pos_left
        pos_right
        '''
        update_bags = []

        for bag in bags:
            es, num, sens, rel = bag
            new_sen = []
            for idx, sen in enumerate(sens):
                new_sen.append(self.get_pad_sen(sen))
            update_bags.append([es, num, new_sen, rel])

        return update_bags

    def get_pad_sen(self, sen):
        '''
        padding the sentences
        '''
        sen.insert(0, self.word2id['BLANK'])
        if len(sen) < self.max_len + 2 * self.pad:
            sen += [self.word2id['BLANK']] * (self.max_len +2 * self.pad - len(sen))
        else:
            sen = sen[: self.max_len + 2 * self.pad]

        return sen


if __name__ == "__main__":
    data = FilterNYTLoad('./original/FilterNYT/')
