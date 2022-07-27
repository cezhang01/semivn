#import tensorflow as tf
import numpy as np
import sys
import time
from evaluation import *


class Model():

    def __init__(self, args, data):

        self.parse_args(args, data)
        self.show_config()
        self.generate_placeholders()
        self.generate_variables()
        if self.call_interactive_interface == 1:
            self.call_interactive_interface_without_training()
            sys.exit()

    def parse_args(self, args, data):

        self.data = data
        self.dataset_name = args.dataset_name
        self.num_datapoints = self.data.num_datapoints
        self.num_tokens = self.data.num_tokens
        self.num_links = len(self.data.total_links)
        self.num_labels = self.data.num_labels

        self.learning_rate = args.learning_rate
        self.num_epoch = args.num_epoch
        self.num_neg = args.num_neg
        self.num_topics = args.num_topics
        self.visualization_dimensions = args.visualization_dimensions
        self.minibatch_size = args.minibatch_size
        if self.minibatch_size == 0:
            self.minibatch_size = self.num_links
        self.regularizer = args.regularizer
        self.call_interactive_interface = args.call_interactive_interface

        self.temperature = 1
        self.temperature_min = 0.1
        self.anealing_rate = 0.00003
        self.label_smoothing = 0
        self.labeling_ratio = args.labeling_ratio
        self.label_depth = self.data.label_depth

    def show_config(self):

        print('******************************************************')
        print('numpy version:', np.__version__)
        print('tensorflow version:', tf.__version__)

        print('dataset name:', self.dataset_name)
        print('#data points:', self.num_datapoints)
        print('#links:', self.num_links)
        print('#tokens:', self.data.num_tokens)
        print('#labels:', self.num_labels)

        print('learning rate:', self.learning_rate)
        print('#epoch:', self.num_epoch)
        print('#negative samples:', self.num_neg)
        print('#topics:', self.num_topics)
        print('visualization dimensions:', self.visualization_dimensions)
        print('minibatch size:', self.minibatch_size)
        print('labeling ratio:', self.labeling_ratio)
        # print('label depth:', self.label_depth)
        print('******************************************************')

    def generate_placeholders(self):

        self.sampling_links = tf.placeholder('int32', [None, 2])
        self.sampling_neg_links = tf.placeholder('int32', [self.minibatch_size * self.num_neg])
        self.sampling_labels = tf.placeholder('int32', [None, self.label_depth])
        self.sampling_labels_mask = tf.placeholder('bool', [None, self.label_depth])
        self.sampling_attribute = tf.placeholder('float64', [self.minibatch_size, self.num_tokens])
        self.attribute = tf.placeholder('float64', [self.num_datapoints, self.num_tokens])
        self.alpha = tf.placeholder('float64', [self.minibatch_size])
        self.learning_rate_adaptive = tf.placeholder('float64', [])
        self.vertex_id_per_label = tf.placeholder('int32', [None])
        self.label_id_per_label = tf.placeholder('int32', [None])

    def generate_variables(self):

        self.visual_coor = tf.Variable(tf.random_normal([self.num_datapoints, self.visualization_dimensions], dtype='float64'), dtype='float64')
        self.label_coor = tf.Variable(tf.random_normal([self.num_labels + 1, self.visualization_dimensions], dtype='float64'), dtype='float64')
        self.topic_coor = tf.Variable(tf.random_normal([self.num_topics, self.visualization_dimensions], dtype='float64'), dtype='float64')

        self.topic_word = tf.Variable(tf.random_normal([self.num_topics, self.num_tokens], dtype='float64'), dtype='float64')
        self.bias = tf.Variable(tf.random_normal([self.num_tokens], dtype='float64'), dtype='float64')

    def evaluate_diff_numerator(self, a, b, size_a, size_b):

        diff = tf.reshape(tf.expand_dims(a, 1) - tf.expand_dims(b, 0), [-1, self.visualization_dimensions])  # this is gaussian distribution
        squared_distance = tf.reshape(tf.reduce_sum(tf.square(diff), axis=1), [-1, size_b])
        numerator = -0.5 * squared_distance

        return numerator

    def encoder(self):

        self.i_coor = tf.gather(self.visual_coor, self.sampling_links[:, 0])
        self.j_coor = tf.gather(self.visual_coor, self.sampling_links[:, 1])
        self.j_coor_context = tf.gather(self.visual_coor, self.sampling_links[:, 1])
        self.avg_coor = 0.5 * (self.i_coor + self.j_coor)
        self.sampling_labels_coor = {}
        for depth in range(self.label_depth):
            self.sampling_labels_coor[depth] = tf.gather(self.label_coor, self.sampling_labels[:, depth])

        q_numerator_vertex = self.evaluate_diff_numerator(self.avg_coor, self.topic_coor, self.minibatch_size, self.num_topics)
        q_numerator_label = 0
        for depth in range(self.label_depth):
            q_numerator_label_tmp = self.evaluate_diff_numerator(self.sampling_labels_coor[depth], self.topic_coor, self.minibatch_size, self.num_topics)
            q_numerator_label += tf.multiply(q_numerator_label_tmp, tf.tile(tf.expand_dims(tf.cast(self.sampling_labels_mask[:, depth], 'float64'), axis=1), [1, self.num_topics]))
        q_numerator = q_numerator_vertex + q_numerator_label
        self.q = tf.nn.softmax(q_numerator)

        return self.q

    def sample_topic_from_q(self):

        g = tf.constant(np.random.gumbel(loc=0., scale=1., size=[self.minibatch_size, self.num_topics]), dtype='float64')
        self.z_soft = tf.nn.softmax((tf.log(self.q + 1e-20) + g) / self.temperature)
        self.z_sampled_topic_coor = tf.matmul(self.z_soft, self.topic_coor)

        return self.z_sampled_topic_coor

    def decoder(self):

        numerator = self.evaluate_diff_numerator(self.z_sampled_topic_coor, self.visual_coor, self.minibatch_size, self.num_datapoints)
        numerator = tf.log(numerator + 1e-20)

        decoding_labels = tf.one_hot(indices=self.sampling_links[:, 1], depth=self.num_datapoints, dtype='float64')

        reconstruction_loss = tf.nn.softmax_cross_entropy_with_logits(labels=decoding_labels, logits=numerator)

        return reconstruction_loss

    def decoder_neg(self):

        self.j_coor_neg = tf.gather(self.visual_coor, self.sampling_neg_links)
        self.j_coor_neg_context = tf.gather(self.visual_coor, self.sampling_neg_links)

        self.pos_term = tf.nn.sigmoid(-0.5 * tf.reduce_sum(tf.square(self.z_sampled_topic_coor - self.j_coor_context), axis=1))  # this is gaussian distribution
        neg_term = tf.nn.sigmoid(-0.5 * tf.reduce_sum(tf.square(tf.reshape(tf.tile(self.z_sampled_topic_coor, [1, self.num_neg]), [self.minibatch_size * self.num_neg, self.visualization_dimensions]) - self.j_coor_neg_context), axis=1))
        neg_term = tf.reshape(neg_term, [self.minibatch_size, self.num_neg])

        reconstruction_loss = - tf.log(self.pos_term + 1e-20) - tf.reduce_sum(tf.log(1 - neg_term + 1e-20), axis=1)

        return reconstruction_loss

    def evaluate_conditional_prior(self):  # evaluate p(t|w, l(w))

        i_numerator_vertex = self.evaluate_diff_numerator(self.i_coor, self.topic_coor, self.minibatch_size, self.num_topics)
        i_numerator_label = 0
        for depth in range(self.label_depth):
            i_numerator_label_tmp = self.evaluate_diff_numerator(self.sampling_labels_coor[depth], self.topic_coor, self.minibatch_size, self.num_topics)
            i_numerator_label += tf.multiply(i_numerator_label_tmp, tf.tile(tf.expand_dims(tf.cast(self.sampling_labels_mask[:, depth], 'float64'), axis=1), [1, self.num_topics]))
        i_numerator = i_numerator_vertex + i_numerator_label
        self.conditional_prior_i = tf.nn.softmax(i_numerator)

        return self.conditional_prior_i

    def evaluate_kl_divergence(self):

        self.evaluate_conditional_prior()
        kld_loss = tf.reduce_sum(tf.multiply(self.q, tf.log(self.q + 1e-20) - tf.log(self.conditional_prior_i + 1e-20)), axis=1)

        return kld_loss

    def evaluate_log_label(self):  # evaluate log(p(l_w|w)), this is for single-label or multi-label classification

        self.i_numerator_vertex = self.evaluate_diff_numerator(self.i_coor, self.label_coor, self.minibatch_size, self.num_labels + 1)[:, :-1]
        sampling_labels_one_hot = 0
        for depth in range(self.label_depth):
            sampling_labels_one_hot_tmp = tf.one_hot(indices=self.sampling_labels[:, depth], depth=(self.num_labels + 1), dtype='float64') + 1e-20
            sampling_labels_one_hot += sampling_labels_one_hot_tmp[:, :-1]
        label_reconstruction_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.boolean_mask(sampling_labels_one_hot, mask=self.sampling_labels_mask[:, 0]),
                                                                            logits=tf.boolean_mask(self.i_numerator_vertex, mask=self.sampling_labels_mask[:, 0]))
        #labels = tf.boolean_mask(sampling_labels_one_hot, mask=self.sampling_labels_mask[:, 0])
        #logits = tf.boolean_mask(self.i_numerator_vertex, mask=self.sampling_labels_mask[:, 0])
        #label_reconstruction_loss = - tf.reduce_sum(tf.multiply(labels, tf.log(tf.nn.sigmoid(logits) * 2 + 1e-20)) + tf.multiply(1 - labels, tf.log(1 - tf.nn.sigmoid(logits) * 2 + 1e-20)), axis=1)
        label_reconstruction_loss = tf.reduce_mean(label_reconstruction_loss)

        return label_reconstruction_loss

    def decoder_content(self):

        output_logits = tf.add(tf.matmul(self.conditional_prior_i, self.topic_word), self.bias)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_logits, labels=self.sampling_attribute))

        return loss

    def label_smoothness(self):

        self.j_numerator_vertex = self.evaluate_diff_numerator(self.j_coor, self.label_coor, self.minibatch_size, self.num_labels + 1)[:, :-1]

        #reg = tf.reduce_mean(tf.multiply(tf.reduce_mean(tf.square(self.i_numerator_vertex - self.j_numerator_vertex), axis=1), self.alpha))
        #reg = tf.reduce_mean(tf.multiply(tf.reduce_sum(tf.square(tf.nn.softmax(self.i_numerator_vertex) - tf.nn.softmax(self.j_numerator_vertex)), axis=1), self.alpha))
        reg = tf.reduce_mean(tf.multiply(tf.reduce_sum(tf.multiply(tf.nn.softmax(self.i_numerator_vertex), tf.log(tf.nn.softmax(self.i_numerator_vertex) + 1e-20) - tf.log(tf.nn.softmax(self.j_numerator_vertex) + 1e-20)), axis=1), self.alpha))

        return reg

    def construct_model(self):

        self.encoder()
        self.sample_topic_from_q()
        loss = self.decoder_neg()
        loss += self.evaluate_kl_divergence()
        loss = tf.reduce_mean(loss)
        loss += self.decoder_content()
        loss += self.evaluate_log_label()
        loss += self.regularizer * self.label_smoothness()

        return loss

    def generate_feed_dict(self):

        self.feed_dict = {}
        self.feed_dict[self.sampling_links] = self.data.sampling_links
        self.feed_dict[self.sampling_neg_links] = self.data.sampling_neg_links
        self.feed_dict[self.sampling_labels] = self.data.sampling_labels
        self.feed_dict[self.sampling_labels_mask] = self.data.sampling_labels_mask
        self.feed_dict[self.sampling_attribute] = self.data.sampling_attribute
        self.feed_dict[self.alpha] = self.data.alpha
        self.feed_dict[self.attribute] = self.data.attribute
        self.feed_dict[self.learning_rate_adaptive] = self.learning_rate

        return self.feed_dict

    def testing(self):

        topic_topic_dist = tf.nn.softmax(self.evaluate_diff_numerator(self.topic_coor, self.topic_coor, self.num_topics, self.num_topics))
        self.topic_word_inference = tf.add(tf.matmul(topic_topic_dist, self.topic_word), self.bias)

        i_coor = tf.gather(self.visual_coor, self.vertex_id_per_label)
        sampling_labels_coor = tf.gather(self.label_coor, self.label_id_per_label)
        i_numerator_vertex = self.evaluate_diff_numerator(i_coor, self.topic_coor, -1, self.num_topics)
        i_numerator_label = self.evaluate_diff_numerator(sampling_labels_coor, self.topic_coor, -1, self.num_topics)
        self.label_topic_dist = tf.expand_dims(tf.reduce_mean(tf.nn.softmax(i_numerator_vertex + i_numerator_label), axis=0), axis=0)
        self.label_word_inference = tf.squeeze(tf.add(tf.matmul(self.label_topic_dist, self.topic_word), self.bias))
        self.label_dist = tf.nn.softmax(self.evaluate_diff_numerator(self.visual_coor, self.label_coor, -1, self.num_labels + 1)[:, :-1])

    def call_interactive_interface_without_training(self):

        visual_coor = np.loadtxt('./results/' + self.dataset_name + '/' + self.dataset_name + '_' + str(self.num_topics) + '_' + str(self.labeling_ratio) + '_vertex_coor.txt')
        topic_coor = np.loadtxt('./results/' + self.dataset_name + '/' + self.dataset_name + '_' + str(self.num_topics) + '_' + str(self.labeling_ratio) + '_topic_coor.txt')
        label_coor = np.loadtxt('./results/' + self.dataset_name + '/' + self.dataset_name + '_' + str(self.num_topics) + '_' + str(self.labeling_ratio) + '_label_coor.txt')
        topic_top_words, label_top_words = [], []
        with open('./results/' + self.dataset_name + '/' + self.dataset_name + '_' + str(self.num_topics) + '_' + str(self.labeling_ratio) + '_topic_top_words.txt') as file:
            for line in file:
                topic_top_words_one_topic = []
                line = line.split()
                for word in line:
                    topic_top_words_one_topic.append(word)
                topic_top_words.append(topic_top_words_one_topic)
        with open('./results/' + self.dataset_name + '/' + self.dataset_name + '_' + str(self.num_topics) + '_' + str(self.labeling_ratio) + '_label_top_words.txt') as file:
            for line in file:
                label_top_words_one_label = []
                line = line.split()
                for word in line:
                    label_top_words_one_label.append(word)
                label_top_words.append(label_top_words_one_label)
        topic_top_words, label_top_words = np.array(topic_top_words), np.array(label_top_words)
        top_words = np.concatenate([topic_top_words, label_top_words], axis=0)
        visualization(self.dataset_name, visual_coor, label_coor, topic_coor,
                      np.concatenate([np.squeeze(self.data.label), [np.amax(self.data.label) + 1] * len(label_coor), [np.amax(self.data.label) + 2] * len(topic_coor)], axis=0),
                      self.data.total_links,
                      self.data.test_indices,
                      top_words)

    def train(self):

        loss = self.construct_model()
        self.testing()
        self.label_topic_words, self.topic_top_words = [], []
        optimizer = tf.train.AdamOptimizer(self.learning_rate_adaptive).minimize(loss)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            num_minibatch = int(np.ceil(self.num_links / self.minibatch_size))
            t = time.time()
            one_epoch_loss = 0

            for epoch_index in range(1, self.num_epoch + 1):
                for minibatch_index in range(1, num_minibatch + 1):
                    self.data.prepare_minibatch(num_minibatch, minibatch_index)
                    self.generate_feed_dict()
                    _, one_epoch_loss = sess.run([optimizer, loss], feed_dict=self.feed_dict)

                if epoch_index % 20 == 0:
                    self.temperature = np.maximum(self.temperature * np.exp(-self.anealing_rate * epoch_index), self.temperature_min)
                    self.learning_rate = self.learning_rate * 0.95

                if epoch_index % 25 == 0 or epoch_index == 1:
                    print('******************************************************')
                    print('Time: %ds' % (time.time() - t), '\tEpoch: %d/%d' % (epoch_index, self.num_epoch), '\tLoss: %f' % one_epoch_loss)

                    visual_coor = sess.run(self.visual_coor)
                    topic_coor = sess.run(self.topic_coor)
                    label_coor = sess.run(self.label_coor)[:-1]

                    X_train = visual_coor[self.data.label_mask[:, 0]]
                    X_test = visual_coor[self.data.test_indices]

                    topic_word = sess.run(self.topic_word_inference)
                    label_word = []
                    for idx in range(self.num_labels):
                        label_word.append(sess.run(self.label_word_inference, feed_dict={self.vertex_id_per_label: self.data.vertex_id_per_label[idx],
                                                                                         self.label_id_per_label: self.data.label_id_per_label[idx]}))
                    label_word = np.array(label_word)
                    label_dist = sess.run(self.label_dist)

                    self.topic_top_words = output_top_words(topic_word, 20, self.data.voc)
                    self.label_top_words = output_top_words(label_word, 20, self.data.voc)
                    self.top_words = np.concatenate([self.topic_top_words, self.label_top_words], axis=0)
                    classification_knn(X_train=X_train, X_test=X_test, Y_train=np.squeeze(self.data.training_label), Y_test=np.squeeze(self.data.test_label))
                    np.savetxt('./results/' + self.dataset_name + '/' + self.dataset_name + '_' + str(self.num_topics) + '_' + str(self.labeling_ratio) + '_topic_word.txt', topic_word, delimiter='\t')
                    np.savetxt('./results/' + self.dataset_name + '/' + self.dataset_name + '_' + str(self.num_topics) + '_' + str(self.labeling_ratio) + '_label_word.txt', label_word, delimiter='\t')
                    np.savetxt('./results/' + self.dataset_name + '/' + self.dataset_name + '_' + str(self.num_topics) + '_' + str(self.labeling_ratio) + '_vertex_coor.txt', visual_coor, delimiter='\t')
                    np.savetxt('./results/' + self.dataset_name + '/' + self.dataset_name + '_' + str(self.num_topics) + '_' + str(self.labeling_ratio) + '_topic_coor.txt', topic_coor, delimiter='\t')
                    np.savetxt('./results/' + self.dataset_name + '/' + self.dataset_name + '_' + str(self.num_topics) + '_' + str(self.labeling_ratio) + '_label_coor.txt', label_coor, delimiter='\t')
                    with open('./results/' + self.dataset_name + '/' + self.dataset_name + '_' + str(
                            self.num_topics) + '_' + str(self.labeling_ratio) + '_topic_top_words.txt', 'w') as file:
                        for line in self.topic_top_words:
                            for word in line:
                                word = word.replace('\n', '')
                                if len(word) > 0:
                                    file.write(word)
                                    file.write(' ')
                            file.write('\n')
                    with open('./results/' + self.dataset_name + '/' + self.dataset_name + '_' + str(
                            self.num_topics) + '_' + str(self.labeling_ratio) + '_label_top_words.txt', 'w') as file:
                        for line in self.label_top_words:
                            for word in line:
                                word = word.replace('\n', '')
                                if len(word) > 0:
                                    file.write(word)
                                    file.write(' ')
                            file.write('\n')

            print('Finish training! Training time:', time.time() - t)
            print('******************************************************')
            print('Keywords of each label:')
            print(self.label_top_words)
            print('******************************************************')
            print('Keywords of each topic:')
            print(self.topic_top_words)
            self.top_words = np.concatenate([self.topic_top_words, self.label_top_words], axis=0)

            visual_coor = sess.run(self.visual_coor)
            topic_coor = sess.run(self.topic_coor)
            label_coor = sess.run(self.label_coor)[:-1]
            visualization(self.dataset_name, visual_coor, label_coor, topic_coor,
                          np.concatenate([np.squeeze(self.data.label), [np.amax(self.data.label) + 1] * len(label_coor), [np.amax(self.data.label) + 2] * len(topic_coor)], axis=0),
                          self.data.total_links,
                          self.data.test_indices,
                          self.top_words)