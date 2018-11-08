# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
from six.moves import cPickle as pickle
import shutil
import time


# train
# 一批传入多少数据
batch_size = 128
# 存储模型个数
num_checkpoints = 5
# 初始学习率
starter_learning_rate = 0.005
# 训练轮数
epochs = 15
# k交叉验证
k_fold = 5
# 获取sdp特征最大长度
sdp_max_len = 82
# sdp特征embedding维度
sdp_dim = 50
# 输入数据channel个数
input_channel_num = 1
# 每一种filter的个数
each_filter_num = 230
# 类别个数
n_class = 27
# l2正则项系数
l2_loss_beta = 0.05


class Model(object):
    def __init__(self, is_training, embedding_vector, scope=None):
        # placeholder
        self.sdp_ids = tf.placeholder(tf.int32, [None, sdp_max_len], name='sdp_ids')
        self.labels = tf.placeholder(tf.float32, [None, n_class], name='labels')

        # Embedding Layer
        # get embedding
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            # embedding_matrix = tf.get_variable('embedding_matrix', shape=embedding_vector.shape, dtype=tf.float32,
            #                                    trainable=False, initializer=tf.constant_initializer(embedding_vector))
            # sdp_embedded = tf.nn.embedding_lookup(embedding_matrix, self.sdp_ids, name='sdp_embedded')

            embedding_matrix_train = tf.get_variable('embedding_matrix_train', shape=embedding_vector.shape,
                                                     dtype=tf.float32, trainable=True,
                                                     initializer=tf.constant_initializer(embedding_vector))
            sdp_embedded_train = tf.nn.embedding_lookup(embedding_matrix_train, self.sdp_ids, name='sdp_embedded_train')

            # self.input = tf.concat([tf.reshape(sdp_embedded, [-1, sdp_max_len, sdp_dim, input_channel_num]),
            #                        tf.reshape(sdp_embedded_train, [-1, sdp_max_len, sdp_dim, input_channel_num])], 3)

            self.input = tf.reshape(sdp_embedded_train, [-1, sdp_max_len, sdp_dim, input_channel_num])

        with tf.name_scope('cnn'):
            # filter_1 = tf.get_variable("filter_1", shape=[3, sdp_dim, input_channel_num*2, each_filter_num],
            #                            initializer=tf.truncated_normal_initializer())
            # bias_1 = tf.get_variable("bias_1", shape=[each_filter_num], initializer=tf.zeros_initializer)
            # filter_2 = tf.get_variable("filter_2", shape=[4, sdp_dim, input_channel_num*2, each_filter_num],
            #                            initializer=tf.truncated_normal_initializer())
            # bias_2 = tf.get_variable("bias_2", shape=[each_filter_num], initializer=tf.zeros_initializer)
            filter_3 = tf.get_variable("filter_3", shape=[3, sdp_dim, input_channel_num, each_filter_num],
                                       initializer=tf.truncated_normal_initializer())
            bias_3 = tf.get_variable("bias_3", shape=[each_filter_num], initializer=tf.zeros_initializer)

            # conv_1 = tf.nn.relu(tf.nn.conv2d(self.sdp_embedded_train, filter_1, [1, 1, 1, 1], padding='VALID') + bias_1)
            # conv_2 = tf.nn.relu(tf.nn.conv2d(self.sdp_embedded_train, filter_2, [1, 1, 1, 1], padding='VALID') + bias_2)
            conv_3 = tf.nn.relu(tf.nn.conv2d(self.input, filter_3, [1, 1, 1, 1], padding='VALID') + bias_3)

        with tf.name_scope('pooling'):
            # max_pool_1 = tf.nn.max_pool(conv_1, ksize=[1, sdp_max_len-2, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
            # max_pool_2 = tf.nn.max_pool(conv_2, ksize=[1, sdp_max_len-3, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
            max_pool_3 = tf.nn.max_pool(conv_3, ksize=[1, sdp_max_len-2, 1, 1], strides=[1, 1, 1, 1], padding='VALID')

            # max_pool = tf.concat([max_pool_1, max_pool_2, max_pool_3], axis=3)

            if is_training:
                max_pool = tf.nn.dropout(max_pool_3, keep_prob=0.5)
            else:
                max_pool = max_pool_3

        with tf.name_scope("full_connect"):
            shape = max_pool.get_shape().as_list()
            reshape_result = tf.reshape(max_pool, [-1, shape[1]*shape[2]*shape[3]])
            W2 = tf.get_variable("weight_2", shape=[each_filter_num, n_class],
                                 initializer=tf.truncated_normal_initializer())
            b2 = tf.get_variable("b_2", shape=[n_class], initializer=tf.zeros_initializer())
            logits = tf.nn.xw_plus_b(reshape_result, W2, b2)

        with tf.name_scope("loss"):
            l2_loss = tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2)
            # + tf.nn.l2_loss(filter_2) + tf.nn.l2_loss(bias_2)\
            # + tf.nn.l2_loss(filter_3) + tf.nn.l2_loss(bias_3)
            # + tf.nn.l2_loss(filter_1) + tf.nn.l2_loss(bias_1)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))
                        # + l2_loss_beta * l2_loss
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope("prediction"):
            self.prob = tf.nn.softmax(logits)

        # with tf.name_scope("accuracy"):
        #     correct_pred = tf.equal(tf.argmax(self.prob, axis=1), tf.argmax(self.labels, axis=1))
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #     tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope))
        # 记录全局步数
        self.global_step = tf.get_variable('global_step', shape=[], trainable=False,
                                           initializer=tf.constant_initializer(0), dtype=tf.int32)
        # 选择使用的优化器
        if is_training:
            self.train_op = tf.train.AdadeltaOptimizer(starter_learning_rate)\
                .minimize(self.loss, global_step=self.global_step)


def select_instance(data_list, model, session):
    sents = [data[2] for data in data_list]
    nums = [data[1] for data in data_list]
    labels = [data[3] for data in data_list]

    select_sent = []
    select_label = []

    for idx, num in enumerate(nums):
        max_ins_id = 0
        label = labels[idx][0]
        if num > 1:
            batch_sents = np.array(sents[idx])
            feed_dictory = {model.sdp_ids: batch_sents}

            batch_pre = session.run(model.prob, feed_dict=feed_dictory)

            max_ins_id = np.argmax(batch_pre[:, label], 0)

        select_sent.append(sents[idx][max_ins_id])
        select_label.append((np.arange(27) == label).astype(np.int32))
    return list(map(lambda x: np.array(x), [select_sent, select_label]))


def predict(data_list, model, session):
    true_y = []
    pred_y = []
    pred_p = []

    sents = [data[2] for data in data_list]
    nums = [data[1] for data in data_list]
    labels = [data[3] for data in data_list]

    for idx, num in enumerate(nums):
        true_y.append(labels)

        batch_sents = np.array(sents[idx])
        feed_dictory = {model.sdp_ids: batch_sents}

        batch_pre = session.run(model.prob, feed_dict=feed_dictory)

        max_ins_label = np.argmax(batch_pre, 1)
        max_ins_prob = [batch_pre[i][max_ins_label[i]] for i in range(len(max_ins_label))]

        tmp_prob = -1.0
        tmp_NA_prob = -1.0
        pred_label = 0
        pos_flag = False

        for i in range(num):
            if pos_flag and max_ins_label[i] < 1:
                continue
            else:
                if max_ins_label[i] > 0:
                    pos_flag = True
                    if max_ins_prob[i] > tmp_prob:
                        tmp_prob = max_ins_prob[i]
                        pred_label = max_ins_label[i]
                else:
                    if max_ins_prob[i] > tmp_NA_prob:
                        tmp_NA_prob = max_ins_prob[i]

        if pos_flag:
            pred_p.append(tmp_prob)
        else:
            pred_p.append(tmp_NA_prob)

        pred_y.append(pred_label)

    return true_y, pred_y, pred_p


def eval_metric(true_y, pred_y, pred_p):
    assert len(true_y) == len(pred_p)
    positive_num = len([i for i in true_y if i[0] > 0])
    index = np.argsort(pred_p)[::-1]

    tp = 0
    fp = 0
    fn = 0
    all_pre = [0]
    all_rec = [0]

    for idx in range(len(true_y)):
        i = true_y[index[idx]]
        j = pred_y[index[idx]]

        if i[0] == 0:  # NA relation
            if j > 0:
                fp += 1
        else:
            if j == 0:
                fn += 1
            else:
                for k in i:
                    if k == -1:
                        break
                    if k == j:
                        tp += 1
                        break
        if fp + tp == 0:
            precision = 1.0
        else:
            precision = tp * 1.0 / (tp + fp)
        recall = tp * 1.0 / positive_num
        if precision != all_pre[-1] or recall != all_rec[-1]:
            all_pre.append(precision)
            all_rec.append(recall)

    print("tp={}; fp={}; fn={}; positive_num={}".format(tp, fp, fn, positive_num))
    return all_pre[1:], all_rec[1:]


def save_pr(out_dir, name, epoch, pre, rec, fp_res=None, opt=None):
    if opt is None:
        out = open('{}/{}_{}_PR.txt'.format(out_dir, name, epoch + 1), 'w')
    else:
        out = open('{}/{}_{}_{}_PR.txt'.format(out_dir, name, opt, epoch + 1), 'w')

    if fp_res is not None:
        fp_out = open('{}/{}_{}_FP.txt'.format(out_dir, name, epoch + 1), 'w')
        for idx, r, p in fp_res:
            fp_out.write('{} {} {}\n'.format(idx, r, p))
        fp_out.close()

    for p, r in zip(pre, rec):
        out.write('{} {}\n'.format(p, r))

    out.close()


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    print('load data........')

    wordEmbedding = np.load('./../new_dataset/original/FilterNYT/w2v.npy')
    train_data = np.load('./../new_dataset/original/FilterNYT/train/bags_feature.npy')
    test_data = np.load('./../new_dataset/original/FilterNYT/test/bags_feature.npy')

    np.random.seed(3435)
    if train_data.shape[0] % batch_size > 0:
        extra_data_num = batch_size - train_data.shape[0] % batch_size
        rand_train = np.random.permutation(train_data)
        extra_data = rand_train[:extra_data_num]
        new_train_data = np.append(train_data, extra_data, axis=0)
    else:
        new_train_data = train_data

    new_train_data = np.random.permutation(new_train_data)
    n_train_batches = new_train_data.shape[0] // batch_size

    print('Training set: ', new_train_data.shape)
    print('Testing set: ', test_data.shape)

    # 删除日志文件
    try:
        train_dir = os.path.abspath('./../dataset/summary/train')
        valid_dir = os.path.abspath('./../dataset/summary/valid')
        shutil.rmtree(train_dir)
        shutil.rmtree(valid_dir)
    except Exception:
        pass

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        with tf.name_scope("train") as train_scope:
            with tf.variable_scope("model", reuse=None):
                train_model = Model(True, wordEmbedding, train_scope)
        with tf.name_scope("test") as test_scope:
            with tf.variable_scope("model", reuse=True):
                eval_model = Model(False, wordEmbedding, test_scope)

        train_summary_writer = tf.summary.FileWriter('./../dataset/summary/train', sess.graph)
        valid_summary_writer = tf.summary.FileWriter('./../dataset/summary/valid')

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)
        sess.run(tf.global_variables_initializer())

        max_pre = -1.0
        max_rec = -1.0
        for e in range(epochs):
            total_loss = 0
            for mini_batch_index in np.random.permutation(list(range(n_train_batches))):
                batch_data = new_train_data[mini_batch_index * batch_size: (mini_batch_index+1) * batch_size]
                batch_train_data = select_instance(batch_data, eval_model, sess)

                feed_dict = {train_model.sdp_ids: batch_train_data[0], train_model.labels: batch_train_data[1]}

                batch_loss, _, train_summary, global_step = sess.run([train_model.loss, train_model.train_op,
                                                                      train_model.merged, train_model.global_step],
                                                                     feed_dict=feed_dict)
                total_loss += batch_loss

                train_summary_writer.add_summary(train_summary, global_step)

            true_y, pred_y, pred_p = predict(test_data, eval_model, sess)
            all_pre, all_rec = eval_metric(true_y, pred_y, pred_p)

            last_pre, last_rec = all_pre[-1], all_rec[-1]
            # if last_pre > 0.24 and last_rec > 0.24:
            save_pr('./../dataset/result', 'test', e, all_pre, all_rec)
            print('{} Epoch {} save pr'.format(now(), e + 1))
            if last_pre > max_pre and last_rec > max_rec:
                print("save model")
                max_pre = last_pre
                max_rec = last_rec
                saver.save(sess, './../dataset/model/classifier.ckpt')

            print('{} Epoch {}/{}: train loss: {}; test precision: {}, test recall {}'.format(now(), e + 1, epochs,
                                                                                              total_loss, last_pre,
                                                                                              last_rec))
