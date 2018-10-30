# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import tensorflow as tf
from keras.preprocessing import sequence
import sys
import os
from six.moves import cPickle as pickle
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from evaluation.Evaluation import precision_each_class
# from evaluation.Evaluation import recall_each_class
# from evaluation.Evaluation import f1_each_class_precision_recall
# from evaluation.Evaluation import f1_each_class
# from evaluation.Evaluation import class_label_count
# from evaluation.Evaluation import print_out


# train
# 一批传入多少数据
batch_size = 64
# 存储模型个数
num_checkpoints = 5
# 初始学习率
starter_learning_rate = 0.0005
# 训练轮数
epochs = 15
# k交叉验证
k_fold = 5
# 获取sdp特征最大长度
sdp_max_len = 49
# sdp特征embedding维度
sdp_dim = 50
# 输入数据channel个数
input_channel_num = 1
# 每一种filter的个数
each_filter_num = 128
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
            embedding_matrix = tf.get_variable('embedding_matrix', shape=embedding_vector.shape, dtype=tf.float32,
                                               trainable=False, initializer=tf.constant_initializer(embedding_vector))
            sdp_embedded = tf.nn.embedding_lookup(embedding_matrix, self.sdp_ids, name='sdp_embedded')

            embedding_matrix_train = tf.get_variable('embedding_matrix_train', shape=embedding_vector.shape,
                                                     dtype=tf.float32, trainable=True,
                                                     initializer=tf.constant_initializer(embedding_vector))
            sdp_embedded_train = tf.nn.embedding_lookup(embedding_matrix_train, self.sdp_ids, name='sdp_embedded_train')

            self.input = tf.concat([tf.reshape(sdp_embedded, [-1, sdp_max_len, sdp_dim, input_channel_num]),
                                   tf.reshape(sdp_embedded_train, [-1, sdp_max_len, sdp_dim, input_channel_num])], 3)

        with tf.name_scope('cnn'):
            filter_1 = tf.get_variable("filter_1", shape=[3, sdp_dim, input_channel_num*2, each_filter_num],
                                       initializer=tf.truncated_normal_initializer())
            bias_1 = tf.get_variable("bias_1", shape=[each_filter_num], initializer=tf.zeros_initializer)
            filter_2 = tf.get_variable("filter_2", shape=[4, sdp_dim, input_channel_num*2, each_filter_num],
                                       initializer=tf.truncated_normal_initializer())
            bias_2 = tf.get_variable("bias_2", shape=[each_filter_num], initializer=tf.zeros_initializer)
            filter_3 = tf.get_variable("filter_3", shape=[5, sdp_dim, input_channel_num*2, each_filter_num],
                                       initializer=tf.truncated_normal_initializer())
            bias_3 = tf.get_variable("bias_3", shape=[each_filter_num], initializer=tf.zeros_initializer)

            conv_1 = tf.nn.relu(tf.nn.conv2d(self.input, filter_1, [1, 1, 1, 1], padding='VALID') + bias_1)
            conv_2 = tf.nn.relu(tf.nn.conv2d(self.input, filter_2, [1, 1, 1, 1], padding='VALID') + bias_2)
            conv_3 = tf.nn.relu(tf.nn.conv2d(self.input, filter_3, [1, 1, 1, 1], padding='VALID') + bias_3)

        with tf.name_scope('pooling'):
            max_pool_1 = tf.nn.max_pool(conv_1, ksize=[1, sdp_max_len-2, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
            max_pool_2 = tf.nn.max_pool(conv_2, ksize=[1, sdp_max_len-3, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
            max_pool_3 = tf.nn.max_pool(conv_3, ksize=[1, sdp_max_len-4, 1, 1], strides=[1, 1, 1, 1], padding='VALID')

            max_pool = tf.concat([max_pool_1, max_pool_2, max_pool_3], axis=3)

            if is_training:
                max_pool = tf.nn.dropout(max_pool, keep_prob=0.5)

        with tf.name_scope("full_connect"):
            shape = max_pool.get_shape().as_list()
            reshape_result = tf.reshape(max_pool, [-1, shape[1]*shape[2]*shape[3]])
            W2 = tf.get_variable("weight_2", shape=[each_filter_num * 3, n_class],
                                 initializer=tf.truncated_normal_initializer())
            b2 = tf.get_variable("b_2", shape=[n_class], initializer=tf.zeros_initializer())
            logits = tf.nn.xw_plus_b(reshape_result, W2, b2)

        with tf.name_scope("loss"):
            l2_loss = tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2)
            # + tf.nn.l2_loss(filter_2) + tf.nn.l2_loss(bias_2)\
            # + tf.nn.l2_loss(filter_3) + tf.nn.l2_loss(bias_3)
            # + tf.nn.l2_loss(filter_1) + tf.nn.l2_loss(bias_1)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)) + \
                l2_loss_beta * l2_loss
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope("prediction"):
            self.prob = tf.nn.softmax(logits)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.prob, axis=1), tf.argmax(self.labels, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope))
        # 记录全局步数
        self.global_step = tf.get_variable('global_step', shape=[], trainable=False,
                                           initializer=tf.constant_initializer(0), dtype=tf.int32)
        # 选择使用的优化器
        if is_training:
            self.train_op = tf.train.AdamOptimizer(starter_learning_rate)\
                .minimize(self.loss, global_step=self.global_step)


def get_batches(*args):
    n_batches = (len(args[0])-1) // batch_size + 1
    new_args = []
    if len(args[0]) % n_batches != 0:
        for x in args:
            new_args.append(np.concatenate((x, x[:n_batches*batch_size - len(args[0])]), axis=0))
    else:
        new_args.extend(args)
    for i in range(0, len(new_args[0]), batch_size):
        data_batch = []
        for x in new_args:
            data_batch.append(x[i: i + batch_size])
        yield data_batch


def randomize(dataset):
    permutation = np.random.permutation(len(dataset))
    shuffled_dataset = dataset[permutation]
    return shuffled_dataset


# def trans_label(label):
#     count = class_label_count(label)
#     count_sum = float(sum(count))
#     weight = list(map(lambda x: count_sum/(6*x), count))
#     weight_matrix = np.zeros(label.shape)
#     for i in range(6):
#         weight_matrix[:, i] = weight[i]
#     return np.multiply(label, weight_matrix)


if __name__ == '__main__':
    print('load data........')
    with open('./../dataset/final_dataset/data.pickle', 'rb') as f:
        parameter = pickle.load(f)
        wordEmbedding = parameter['wordEmbedding']
        del parameter
        train = pickle.load(f)
        train_data = train['data']
        del train
        test = pickle.load(f)
        test_data = test['data']
        del test

    np.random.seed(3435)
    if len(train_data) % batch_size > 0:
        extra_data_num = batch_size - len(train_data) % batch_size
        rand_train = np.random.permutation(train_data)
        extra_data = rand_train[:extra_data_num]
        new_train_data = np.append(train_data, extra_data, axis=0)
        new_test_data = np.array(test_data)
    else:
        new_train_data = np.array(train_data)
        new_test_data = np.array(test_data)

    print('Training set: ', len(new_train_data))
    print('Testing set: ', len(new_test_data))

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

        max_f1 = 0
        for e in range(epochs):
            shuffle_train_data = randomize(train_data)
            for batch_dataset in get_batches(train_dataset, train_labels):
                    batch_sdp_id, batch_labels = batch_dataset

                    feed_dict = {train_model.sdp_ids: batch_sdp_id,
                                 train_model.labels: batch_labels}

                    batch_accuracy, global_step, train_summary, _, batch_loss, batch_predictions = \
                        sess.run([train_model.accuracy, train_model.global_step, train_model.merged,
                                  train_model.train_op, train_model.loss, train_model.prob], feed_dict=feed_dict)
                    train_fold_loss.append(batch_loss)
                    train_fold_acc.append(batch_accuracy * 100)
                    train_fold_avg_f1.append(np.mean(f1_each_class(batch_predictions, batch_labels)))

                    train_summary_writer.add_summary(train_summary, global_step)
                train_epoch_loss.append(np.mean(train_fold_loss))
                train_epoch_accuracy.append(np.mean(train_fold_acc))
                train_epoch_f1.append(np.mean(train_fold_avg_f1))
                print("train:epoch={}, k={}, step={}, loss={}, acc={}, f1={}"
                      .format(e, k, global_step, np.mean(train_fold_loss), np.mean(train_fold_acc),
                              np.mean(train_fold_avg_f1)))

                feed_dic = {eval_model.sdp_ids: valid_dataset,
                            eval_model.labels: valid_labels}

                valid_accuracy, global_step, valid_summary, valid_loss, valid_predictions = \
                    sess.run([eval_model.accuracy, eval_model.global_step, eval_model.merged, eval_model.loss,
                              eval_model.prob], feed_dict=feed_dic)

                valid_summary_writer.add_summary(valid_summary, global_step)
                predict_f1 = np.mean(f1_each_class(valid_predictions, valid_labels))

                valid_epoch_loss.append(valid_loss)
                valid_epoch_accuracy.append(valid_accuracy * 100)
                valid_epoch_f1.append(predict_f1)
                print("valid:loss={}, acc={}, f1={}".format(valid_loss, (valid_accuracy*100), predict_f1))
            print("evaluation:epoch={}, loss={}, acc={}, f1={}"
                  .format(e, np.mean(valid_epoch_loss), np.mean(valid_epoch_accuracy), np.mean(valid_epoch_f1)))
            print()
            if np.mean(valid_epoch_f1) > max_f1:
                saver.save(sess, './../resource/model/classifier.ckpt')

        # 测试阶段
        model_file = tf.train.latest_checkpoint('./../resource/model/')
        saver.restore(sess, model_file)
        feed_dic = {eval_model.sdp_ids: test_sdp_id_padding,
                    eval_model.labels: test_labels}
        test_accuracy, test_predictions = sess.run([eval_model.accuracy, eval_model.prob], feed_dict=feed_dic)

        print('---------------------------------------------------------------------------')
        test_precision = precision_each_class(test_predictions, test_labels)
        test_recall = recall_each_class(test_predictions, test_labels)
        test_f1 = f1_each_class_precision_recall(test_precision, test_recall)
        test_count = class_label_count(test_labels)
        print("test accuracy: %.1f%%" % (test_accuracy * 100))
        print_out(test_precision, test_recall, test_f1, test_count)
