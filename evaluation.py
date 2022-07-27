import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import wordcloud
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import sklearn
import numpy as np


def on_click(event, visual_coor, label_coor, topic_coor, raw_content, top_words, ax):

    if event.button == 1:
        click_coor = np.array([event.xdata, event.ydata])
        distance = np.sum(np.square(visual_coor - click_coor), axis=1)
        idx = np.argmin(distance)
        print('\n\n\n\n\n\n\n\n\n\n\n\n\n')
        print(raw_content[idx])
    else:
        click_coor = np.array([event.xdata, event.ydata])
        topic_and_label_coor = np.concatenate([topic_coor, label_coor], axis=0)
        distance = np.sum(np.square(topic_and_label_coor - click_coor), axis=1)
        idx = np.argmin(distance)
        keywords = ''
        for word in top_words[idx, :15]:
            keywords += (word + ' ')
        wordcloud = WordCloud(background_color='white', max_font_size=80, color_func=lambda *args, **kwargs: 'crimson', prefer_horizontal=1).generate_from_text(keywords)
        imagebox = OffsetImage(wordcloud, zoom=0.5)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, xy=(topic_and_label_coor[idx, 0], topic_and_label_coor[idx, 1]),
                            xybox=(150., 0),
                            xycoords='data',
                            pad=0.7,
                            boxcoords='offset points',
                            arrowprops=dict(connectionstyle="arc3,rad=0.", shrinkA=0, shrinkB=10, arrowstyle='-|>', ls='-', linewidth=2))
        ax.add_artist(ab)
        plt.show()


def visualization(dataset_name, visual_coor, label_coor, topic_coor, labels, links, test_indices, top_words):

    # print('networkx version:', nx.__version__)
    # print('matplotlib version:', matplotlib.__version__)
    # print('wordcloud version', wordcloud.__version__)
    # print('sklearn version', sklearn.__version__)

    if dataset_name == 'coronavirus':
        raw_content = np.load('./data/coronavirus/raw_content.npy')
        fig, ax = plt.subplots()
        fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, visual_coor, label_coor, topic_coor, raw_content, top_words, ax))

    coordinates = np.concatenate([visual_coor, label_coor, topic_coor], axis=0)
    G = nx.Graph()
    node_id = [idx for idx in range(len(coordinates))]
    G.add_nodes_from(node_id)
    for idx, pos in enumerate(coordinates):
        G.nodes[idx]['pos'] = pos
    G.add_edges_from(links)

    if dataset_name == 'ds':
        color_cycle = ['b', '#7f7f7f', '#ff7f0e', '#d62728', '#17becf', 'm', '#2ca02c', '#9467bd', '#8c564b', '#bcbd22', 'k', '']  # ds_tuan
        label_name = ['parallel', 'computational_geometry', 'quantum_computing', 'randomized', 'formal_languages', 'hashing', 'computational_complexity', 'logic', 'sorting', 'label', 'topic']  # ds_tuan
    elif dataset_name == 'coronavirus':
        color_cycle = ['b', '#2ca02c', '#7f7f7f', '#9467bd', '#ff7f0e', 'k', '']  # coronavirus
        label_name = ['economy, business, and finance', 'education', 'health', 'labour', 'sports', 'label', 'topic']  # coronavirus

    label_to_color = {}
    for idx, label in enumerate(np.unique(labels)):
        label_to_color[label] = color_cycle[idx]
    node_color = [label_to_color[l] for l in labels]
    edge_color = [label_to_color[labels[link[0]]] for link in G.edges]
    node_shape = ['^'] + ['o'] + ['o' for _ in range(max(labels) - 1)]
    test_mask = np.array([False] * len(labels))
    test_mask[test_indices] = True

    node_list, node_list_mask = [], []
    for idx, label in enumerate(np.unique(labels)):
        node_list.append(np.arange(len(labels))[labels == idx])
        node_list_mask.append(test_mask[labels == idx])

    for idx in range(max(labels)):  # max - 1 == label, max == topic
        if idx != max(labels) and idx != max(labels) - 1:
            #nx.draw_networkx_nodes(G, pos=coordinates, node_color=color_cycle[idx], alpha=1, node_size=115, nodelist=node_list[idx][~node_list_mask[idx]], node_shape='p', label=label_name[idx], edgecolors='white', linewidths=0.3)
            #nx.draw_networkx_nodes(G, pos=coordinates, node_color=color_cycle[idx], alpha=1, node_size=115, nodelist=node_list[idx][node_list_mask[idx]], node_shape='o', label=label_name[idx], edgecolors='white', linewidths=0.3)
            nx.draw_networkx_nodes(G, pos=coordinates, node_color=color_cycle[idx], alpha=1, node_size=115, nodelist=node_list[idx], node_shape='o', label=label_name[idx], edgecolors='white', linewidths=0.3)
    nx.draw_networkx_nodes(G, pos=coordinates, node_color='none', alpha=1, node_size=135, nodelist=node_list[-1], node_shape='o', label=label_name[-1], edgecolors='k')
    nx.draw_networkx_nodes(G, pos=coordinates, node_color='k', alpha=1, node_size=135, nodelist=node_list[-2], node_shape='^', label=label_name[-2], edgecolors='k', linewidths=1)
    #nx.draw_networkx_edges(G, pos=coordinates, edge_color='grey', alpha=0.1)  # 0.1
    plt.legend(markerscale=1, prop={'size': 10}, edgecolor='black')
    plt.axis('off')
    title = 'dataset: ' + dataset_name + '        #topics: ' + str(len(topic_coor))
    if dataset_name == 'coronavirus':
        title += '\n\nRight click topics and labels to show word clouds\nLeft click documents to show specific content at control window\n\n(Note: It is possible to show article inside the plot, but due to its long description, we show it in a separate window for clarity.)'
        plt.title(title)
    plt.show()


def classification_knn(X_train, X_test, Y_train, Y_test):

    result = []
    for k in [20, 40, 60, 80, 100]:
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(X_train, Y_train)
        prediction_label_test = classifier.predict(X_test)
        result.append(f1_score(Y_test, prediction_label_test, average='micro'))
        print('Test accuracy %d: %.4f' % (k, accuracy_score(Y_test, prediction_label_test)))

    return result


def output_top_words(topic_word, num_top_words, voc):

    index = np.flip(np.argsort(topic_word)[:, -num_top_words:], axis=1)
    words = voc[index]

    return words