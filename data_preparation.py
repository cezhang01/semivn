import numpy as np


class Data():

    def __init__(self, args):

        self.parse_args(args)
        self.load_data()

    def parse_args(self, args):

        self.dataset_name = args.dataset_name
        self.minibatch_size = args.minibatch_size
        self.labeling_ratio = args.labeling_ratio
        self.num_neg = args.num_neg

    def load_data(self):

        self.attribute = self.attribute_preprocessing(np.loadtxt('./data/' + self.dataset_name + '/attribute.txt'))
        self.num_datapoints = len(self.attribute)
        self.num_tokens = len(self.attribute[0])
        self.load_labels()
        self.split_labels()
        #self.adjacency_matrix = self.generate_symmetric_adjacency_matrix(np.loadtxt('./data/' + self.dataset_name + '/adjacency_matrix.txt'))
        #self.avg_degree = np.mean(np.sum(self.adjacency_matrix, axis=1))
        #self.total_links = self.generate_symmetric_links(self.generate_links(self.adjacency_matrix))
        self.total_links = self.generate_symmetric_links(np.loadtxt('./data/' + self.dataset_name + '/total_links.txt', dtype=int))
        np.random.shuffle(self.total_links)
        self.voc = np.genfromtxt('./data/' + self.dataset_name + '/voc.txt', dtype=str)
        self.generate_vertex_id_per_label()
        if self.minibatch_size == 0:
            self.minibatch_size = len(self.total_links)

    def attribute_preprocessing(self, attribute):

        attribute_preprocessed = []
        for row in attribute:
            max_row = np.log(1 + np.max(row))
            attribute_preprocessed.append(np.log(1 + row) / max_row)

        return np.asarray(attribute_preprocessed, dtype='float64')

    def load_labels(self):

        self.label = []
        self.label_depth = 0
        self.num_labels = 0
        with open('./data/' + self.dataset_name + '/label.txt') as file:
            for line in file:
                line = list(map(int, line.strip().split()))
                if len(line) > self.label_depth:
                    self.label_depth = len(line)
                if max(line) > self.num_labels:
                    self.num_labels = max(line)
                self.label.append(line)
        self.num_labels += 1  # placeholder label is not counted

        for idx, line in enumerate(self.label):
            if len(line) < self.label_depth:
                self.label[idx].extend([self.num_labels] * (self.label_depth - len(line)))
        self.label = np.asarray(self.label, dtype=int)

    def split_labels(self):

        self.test_indices = np.random.choice(self.num_datapoints, int(self.num_datapoints * (1 - self.labeling_ratio)), replace=False)
        self.label_mask = np.full([self.num_datapoints, self.label_depth], True)
        for depth in range(self.label_depth):
            self.label_mask[:, depth] = self.label[:, depth] != self.num_labels
        self.label_mask[self.test_indices] = False

        self.training_label = self.label[self.label_mask[:, 0]]
        self.test_label = self.label[self.test_indices]

    def generate_symmetric_adjacency_matrix(self, adjacency_matrix):

        adjacency_matrix_symm = np.zeros([len(adjacency_matrix), len(adjacency_matrix)])
        for row_idx in range(len(adjacency_matrix)):
            for col_idx in range(len(adjacency_matrix)):
                if adjacency_matrix[row_idx, col_idx] == 1:
                    adjacency_matrix_symm[row_idx, col_idx] = 1
                    adjacency_matrix_symm[col_idx, row_idx] = 1

        return adjacency_matrix_symm

    def generate_links(self, adjacency_matrix):

        links = []
        for row_idx in range(len(adjacency_matrix)):
            for col_idx in range(len(adjacency_matrix)):
                if adjacency_matrix[row_idx, col_idx] != 0:
                    links.append([row_idx, col_idx])

        return np.asarray(links)

    def generate_symmetric_links(self, total_links):

        total_links_symm = []
        for link in total_links:
            #if link[0] == link[1]:
                #continue
            total_links_symm.append([link[0], link[1]])
            total_links_symm.append([link[1], link[0]])
            total_links_symm.append([link[0], link[0]])
            total_links_symm.append([link[1], link[1]])
        total_links_symm = np.unique(total_links_symm, axis=0)

        return total_links_symm

    def prepare_minibatch(self, num_minibatch, minibatch_index):

        self.sampling_links = self.sample_minibatch_links(num_minibatch, minibatch_index)
        self.sampling_neg_links = self.sample_minibatch_neg_links()
        self.sampling_labels, self.sampling_labels_mask = self.sample_minibatch_labels()
        self.sampling_attribute = self.attribute[self.sampling_links[:, 0]]
        self.alpha = self.evaluate_alpha()

    def sample_minibatch_links(self, num_minibatch, minibatch_index):

        if minibatch_index == num_minibatch:
            sampling_links = self.total_links[self.minibatch_size * (minibatch_index - 1):]
            if self.minibatch_size - len(sampling_links) != 0:
                indices = np.random.choice(len(self.total_links), self.minibatch_size - len(sampling_links), replace=False)
                sampling_links = np.concatenate((sampling_links, self.total_links[indices]), axis=0)
        else:
            sampling_links = self.total_links[self.minibatch_size * (minibatch_index - 1):self.minibatch_size * minibatch_index]

        return sampling_links

    def sample_minibatch_neg_links(self):

        sampling_neg_links = []
        for sampling_link in self.sampling_links:
            for idx in range(self.num_neg):
                neg_node_id = np.random.choice(self.num_datapoints, 1)
                while neg_node_id == sampling_link[0] or neg_node_id == sampling_link[1]:
                    neg_node_id = np.random.choice(self.num_datapoints, 1)
                sampling_neg_links.append(neg_node_id)

        return np.squeeze(np.asarray(sampling_neg_links, dtype=int))

    def sample_minibatch_labels(self):

        sampling_labels = self.label[self.sampling_links[:, 0]]
        sampling_labels_mask = self.label_mask[self.sampling_links[:, 0]]

        return sampling_labels, sampling_labels_mask

    def evaluate_alpha(self):

        alpha = []
        for sampling_link in self.sampling_links:
            neighbors_i = self.total_links[self.total_links[:, 0] == sampling_link[0]]
            neighbors_j = self.total_links[self.total_links[:, 0] == sampling_link[1]]
            numerator = len(np.intersect1d(neighbors_i, neighbors_j))
            denominator = len(np.union1d(neighbors_i, neighbors_j))
            alpha.append(numerator / denominator)

        return alpha

    def generate_vertex_id_per_label(self):

        self.vertex_id_per_label, self.label_id_per_label = [], []
        for idx in range(self.num_labels):
            vertex_id_per_label_mask = np.logical_or.reduce(np.logical_and(self.label == idx, self.label_mask), axis=1)
            self.vertex_id_per_label.append(np.arange(self.num_datapoints)[vertex_id_per_label_mask])
            self.label_id_per_label.append(np.array([idx] * len(self.vertex_id_per_label[idx])))

        return self.vertex_id_per_label, self.label_id_per_label

    def softmax(self, x):

        return np.exp(x) / np.sum(np.exp(x))