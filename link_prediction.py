import pandas as pd
import numpy as np
import random
import networkx as nx
from tqdm import tqdm
import re
import os
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from node2vec import Node2Vec

import time
import pickle


SHOW_PLOT = False

init_file = "data/amherst_558_0.25_nw_init"
dynamic_file = 'data/amherst_558_0.25_nw_dynamic'

out_dir = 'output/'
os.makedirs(out_dir, exist_ok=True)


def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()

    return lines

def split_list(l, val=''):
    idx_list = [idx + 1 for idx, val in
            enumerate(l) if val == '']

    splits = [l[i: j-1] for i, j in zip([0] + idx_list, idx_list + ([len(l)] if idx_list[-1] != len(l) else []))]
    return splits

def clean_lists(lists):
    final_list = []
    for l in lists:
        final_list.append([x.replace('\t',',') for x in l][1:])
    return final_list

def get_data(graph):

    # captture nodes in 2 separate lists
    node_list_1 = []
    node_list_2 = []

    for i in tqdm(graph):
      node_list_1.append(i.split(',')[0])
      node_list_2.append(i.split(',')[1])

    graph_df = pd.DataFrame({'node_1': node_list_1, 'node_2': node_list_2})

    print(graph_df.head())

    # create graph
    G = nx.from_pandas_edgelist(graph_df, "node_1", "node_2", create_using=nx.Graph())

    # plot graph
    plt.figure(figsize=(10,10))

    pos = nx.random_layout(G, seed=23)
    nx.draw(G, with_labels=True,  pos = pos, node_size = 40, alpha = 0.6, width = 0.7)

    if SHOW_PLOT:
        plt.show()

    # combine all nodes in a list
    node_list = node_list_1 + node_list_2

    # remove duplicate items from the list
    node_list = list(dict.fromkeys(node_list))

    # build adjacency matrix
    adj_G = nx.to_numpy_matrix(G, nodelist = node_list)
    print(f"Adj Matrix Shape: {adj_G.shape}")


    # get unconnected node-pairs
    all_unconnected_pairs = []

    # Traverse adjacency matrix
    offset = 0
    for i in tqdm(range(adj_G.shape[0])):
        for j in range(offset,adj_G.shape[1]):
            if i != j:
                if nx.has_path(G, str(i),str(j)):
                    if nx.shortest_path_length(G, str(i), str(j)) <=2:
                        if adj_G[i,j] == 0:
                            all_unconnected_pairs.append([node_list[i],node_list[j]])
                # else:
                #   print(f"No path between {i} and {j}")

        offset = offset + 1

    print(f"All Unconnected Pairs: {len(all_unconnected_pairs)}")

    node_1_unlinked = [i[0] for i in all_unconnected_pairs]
    node_2_unlinked = [i[1] for i in all_unconnected_pairs]

    data = pd.DataFrame({'node_1':node_1_unlinked, 
                         'node_2':node_2_unlinked})

    # add target variable 'link'
    data['link'] = 0

    print("Data Dataframe:")
    print(data.head())


    initial_node_count = len(G.nodes)
    initial_connected_comps = nx.number_connected_components(G)

    graph_df_temp = graph_df.copy()

    # empty list to store removable links
    omissible_links_index = []

    for i in tqdm(graph_df.index.values):
      
        # remove a node pair and build a new graph
        G_temp = nx.from_pandas_edgelist(graph_df_temp.drop(index = i), "node_1", "node_2", create_using=nx.Graph())

        # check there is no spliting of graph and number of nodes is same
        if (nx.number_connected_components(G_temp) == initial_connected_comps) and (len(G_temp.nodes) == initial_node_count):
            # print("HERE")
            omissible_links_index.append(i)
            graph_df_temp = graph_df_temp.drop(index = i)

    print(f"Omissible Links: {len(omissible_links_index)}")


    # create dataframe of removable edges
    graph_df_ghost = graph_df.loc[omissible_links_index]

    # add the target variable 'link'
    graph_df_ghost['link'] = 1

    data = data.append(graph_df_ghost[['node_1', 'node_2', 'link']], ignore_index=True)

    print(data['link'].value_counts())

    ## FEATURE EXTRACTION
    # drop removable edges
    graph_df_partial = graph_df.drop(index=graph_df_ghost.index.values)

    # build graph
    G_data = nx.from_pandas_edgelist(graph_df_partial, "node_1", "node_2", create_using=nx.Graph())

    return G_data, data

def get_embeddings(G_data, data):

    # Generate walks
    node2vec = Node2Vec(G_data, dimensions=100, walk_length=16, num_walks=50)

    # train node2vec model
    n2w_model = node2vec.fit(window=7, min_count=1)
    x = [(n2w_model[str(i)]+n2w_model[str(j)]) for i,j in zip(data['node_1'], data['node_2'])]

    return x, data


def test_embeddings(x, data):
    xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), data['link'], 
                                                    test_size = 0.3, 
                                                    random_state = 35)

    print("Running Logistics Regression on the embeddings...")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    lr.fit(xtrain, ytrain)
    predictions = lr.predict_proba(xtest)

    # print("Running random forest on the embeddings...")
    # model = RandomForestClassifier(max_depth=5, random_state=0, n_estimators=50)
    # model.fit(xtrain, ytrain)
    # preds = model.predict_proba(xtest)

    auc_score = roc_auc_score(ytest, predictions[:,1])
    print(f"AUC ROC score for LogisticRegression: {auc_score}")
    return auc_score
    # print(f"AUC ROC score for RandomForest: {roc_auc_score(ytest, preds[:,1])}")

g_init = read_file(init_file)
g_dyn = read_file(dynamic_file)

dynamic_splits = split_list(g_dyn, val='')

num_nodes = g_init[0]
g_init = g_init[1:]

g_init = clean_lists([g_init])[0]
dynamic_splits = clean_lists(dynamic_splits)

print(len(dynamic_splits))
print(len(dynamic_splits)/200)

print(f"No. of Nodes: {num_nodes}, No. of links: {len(g_init)}")
auc_scores = []
times = {}


G_data, data = get_data(g_init)
start_time = time.time()
x, data = get_embeddings(G_data, data)
end_time = time.time()
auc_score = test_embeddings(x, data)

print(f"Initial Graph with {num_nodes} nodes took {end_time - start_time} ms for embeddings...")

times[num_nodes] = end_time - start_time
OFFSET = 2

START = 0
END = START + OFFSET


NUM_SNAPSHOTS = 9
new_graph = g_init
current_snapshot = 1

auc_scores.append(auc_score)

while END < NUM_SNAPSHOTS * OFFSET:#len(dynamic_splits):
    new_splits = dynamic_splits[START:END]
    new_edges = [item for sublist in new_splits for item in sublist]

    new_graph += new_edges

    print(f"Snapshot: {current_snapshot}")
    print(f"No. of Nodes: {int(num_nodes)+END}, No. of links: {len(new_graph)}")

    current_snapshot += 1

    G_data, data = get_data(new_graph)
    start_time = time.time()
    x, data = get_embeddings(G_data, data)
    end_time = time.time()

    print(f"Dynamic Graph with {int(num_nodes)+END} nodes took {end_time - start_time} ms for embeddings...")
    times[int(num_nodes)+END] = end_time - start_time

    auc_score = test_embeddings(x, data)
    auc_scores.append(auc_score)

    START = END
    END = START + OFFSET


print(f"AUC scores: {auc_scores}")
print(f"Times Dict: {times}")

with open(os.path.join(out_dir, 'auc_scores.pkl'), 'wb') as handle:
    pickle.dump(auc_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(os.path.join(out_dir, 'times_dict.pkl'), 'wb') as handle:
    pickle.dump(times, handle, protocol=pickle.HIGHEST_PROTOCOL)
