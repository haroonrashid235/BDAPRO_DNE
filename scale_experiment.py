import pandas as pd
import numpy as np
import random
import math
import networkx as nx
from tqdm import tqdm
import re
import os
import sys
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from node2vec import Node2Vec
import json
import time
import pickle

from collections import Counter

########### Config ##########
BINARY_CLASSIFICATION=False
SHOW_PLOT = False
#############################
OFFSET_LIST = [2000, 4000, 8000, 16000, 32000]


init_file = "NCI1/NCI1.nw_init"
dynamic_file = "NCI1/NCI1.nw_dynamic"

ground_truth_file="NCI1/NCI1.node_labels"

out_dir = 'output/'
os.makedirs(out_dir, exist_ok=True)


def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()

    return lines


def load_json_from_file(filename):
    return json.loads(open(filename).read())

def load_ground_truth(file_path):
    lst = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split(',')
            lst.append([int(float(i)) for i in items])
    lst.sort()
    l=[]
    if BINARY_CLASSIFICATION:
        l = [1 if i[1]>0 else 0 for i in lst]
    else:
        l = [i[1] for i in lst]

    count_dict = Counter(l)
    l = [i if count_dict[i] >= 1500 else 2 for i in l]
    return l
###########################

def split_list(l, val=''):
    idx_list = [idx + 1 for idx, val in
                enumerate(l) if val == '']

    splits = [l[i: j - 1] for i, j in zip([0] + idx_list, idx_list + ([len(l)] if idx_list[-1] != len(l) else []))]
    return splits


def clean_lists(lists):
    final_list = []
    for l in lists:
        final_list.append([x.replace('\t', ',') for x in l][1:])
    return final_list


def get_data(edge_list):
    # captture nodes in 2 separate lists
    node_list_1 = []
    node_list_2 = []

    for i in tqdm(edge_list):
        node_list_1.append(i.split(' ')[0])
        node_list_2.append(i.split(' ')[1])

    graph_df = pd.DataFrame({'node_1': node_list_1, 'node_2': node_list_2})
    # print(graph_df.head())

    # create graph
    G = nx.from_pandas_edgelist(graph_df, "node_1", "node_2", create_using=nx.Graph())

    # combine all nodes in a list
    node_list = node_list_1 + node_list_2

    # remove duplicate items from the list
    node_list = list(dict.fromkeys(node_list))
    all_labels = load_ground_truth(ground_truth_file)
    all_labels += [2,2,2,2]

    labels = [all_labels[int(i)] for i in node_list]
    data = pd.DataFrame({'node':node_list, 'label':labels})
    return G, data


def get_embeddings(G_data, data):
    # Generate walks
    node2vec = Node2Vec(G_data, dimensions=20, walk_length=16, num_walks=50, workers=8)

    # train node2vec model
    n2w_model = node2vec.fit(window=7, min_count=1)
    x = [n2w_model[str(i)] for i in data['node']]

    return x, data


def test_embeddings(x, data):
    scaler = preprocessing.StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    xtrain, xtest, ytrain, ytest = train_test_split(np.array(x_scaled), data['label'],
                                                    test_size=0.3,
                                                    random_state=35,stratify=data['label'])

    print("Running Logistics Regression on the embeddings...")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    lr.fit(xtrain, ytrain)
    predictions = lr.predict_proba(xtest)
    # print("Running random forest on the embeddings...")
    # model = RandomForestClassifier(max_depth=5, random_state=0, n_estimators=50)
    # model.fit(xtrain, ytrain)
    # preds = model.predict_proba(xtest)

    auc_score = roc_auc_score(ytest, predictions, multi_class='ovr')
    print(f"AUC ROC score for LogisticRegression: {auc_score}")
    return auc_score
    # print(f"AUC ROC score for RandomForest: {roc_auc_score(ytest, preds[:,1])}")

def run_experiment(OFFSET):

    g_init = read_file(init_file)
    g_dyn = read_file(dynamic_file)

    dynamic_splits = split_list(g_dyn, val='')

    num_nodes = g_init[0]
    g_init = g_init[1:]

    g_init = clean_lists([g_init])[0]
    dynamic_splits = clean_lists(dynamic_splits)


    NUM_SNAPSHOTS = math.ceil(len(dynamic_splits) / OFFSET)
    
    print(f"NUM_SNAPSHOTS: {NUM_SNAPSHOTS}, OFFSET: {OFFSET}")

    print(f"No. of Nodes: {num_nodes}, No. of links: {len(g_init)}")
    auc_scores = []
    times = {}

    G_data, data = get_data(g_init)

    start_time = time.time()

    x, data = get_embeddings(G_data, data)

    end_time = time.time()
    auc_score = test_embeddings(x, data)

    print(f"Initial Graph with {num_nodes} nodes took {end_time - start_time} s for embeddings...")

    times[num_nodes] = end_time - start_time


    START = 0
    END = START + OFFSET

    
    new_graph = g_init
    current_snapshot = 1

    auc_scores.append(auc_score)

    while END < (NUM_SNAPSHOTS * OFFSET) + OFFSET:  # len(dynamic_splits):
        new_splits = dynamic_splits[START:END]
        new_edges = [item for sublist in new_splits for item in sublist]

        new_graph += new_edges

        print(f"Snapshot: {current_snapshot}")
        print(f"No. of Nodes: {int(num_nodes) + END}, No. of links: {len(new_graph)}")

        current_snapshot += 1

        G_data, data = get_data(new_graph)
        start_time = time.time()
        x, data = get_embeddings(G_data, data)

        end_time = time.time()

        print(f"Dynamic Graph with {int(num_nodes) + END} nodes took {end_time - start_time} s for embeddings...")
        times[int(num_nodes) + END] = end_time - start_time

        auc_score = test_embeddings(x, data)
        auc_scores.append(auc_score)

        START = END
        END = START + OFFSET

    print(f"AUC scores: {auc_scores}")
    print(f"Times Dict: {times}")

    return auc_scores, times


auc_scores_list = []
times = []
for offset in OFFSET_LIST:
    auc_score, times_dict = run_experiment(offset)
    auc_scores_list.append(auc_score)
    times.append(times_dict)


with open(os.path.join(out_dir, 'auc_scores.pkl'), 'wb') as handle:
        pickle.dump(auc_scores_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(out_dir, 'times_dict.pkl'), 'wb') as handle:
    pickle.dump(times, handle, protocol=pickle.HIGHEST_PROTOCOL)


print("Experiment Finished...")
print(f"AUC scores List: {auc_scores_list}")
print(f"Times Dict: {times}")
