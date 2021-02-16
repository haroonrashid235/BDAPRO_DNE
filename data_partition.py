############### CONFIG ##############
init_dataset_size = 20000
snapshot_size = 10000
split_on = ',' # Change the splitter character to what is suitable (e.g., '\t' )
SWAPPED_COLUMNS = True # Switch to True, if the slowly changing column is to the right (should be the left one)
######################################

import networkx as nx
from tqdm import tqdm
file_path = 'NCI1/NCI1.edges'     #'data/Duke14.dat'#'data/amherst_nw.dat'
new_init_file_path=''
new_snap_file_path=''
nodes =[]


def gen_paths():
    global new_init_file_path
    global new_snap_file_path
    path_without_suffix = file_path[:file_path.find('.')]
    new_init_file_path = path_without_suffix+'.nw_init'
    new_snap_file_path = path_without_suffix+'.nw_dynamic'

def read_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    return lines[1:]

def build_graph(edges:list):
    # edges = read_file(file_path)
    nodes_set=set()
    g = nx.DiGraph()
    for e in edges:
        (a, b) = e.strip().split(split_on)
        a,b = float(a), float(b)
        nodes_set.add(int(a))
        nodes_set.add(int(b))
        if SWAPPED_COLUMNS:
            g.add_edge(int(b), int(a))
        else:
            g.add_edge(int(a), int(b))

    global nodes
    nodes = list(nodes_set)
    return g

def partition_dataset(g:nx.Graph):
    f1=open(new_init_file_path,"a+")
    f1.truncate(0)
    print(init_dataset_size, file=f1)
    print('Generating initial dataset file..')
    for n in tqdm(nodes[:init_dataset_size]):
        for m in g.neighbors(n):
            if m < init_dataset_size:
                print(str(n)+' '+str(m) , file = f1)
            else:
                g.add_edge(m, n)
    for n in nodes[:init_dataset_size]:
        print(str(n) + ' ' + str(n), file=f1)
    f1.close()
    print('Done!')
    f2 = open(new_snap_file_path, "a+")
    f2.truncate(0)
    print('Generating dynamic nodes file..')
    for n in tqdm(nodes[init_dataset_size:]):
        neighbors_list = []
        for x in g.neighbors(n):
            if x < n:
                neighbors_list.append(x)
            else:
                g.add_edge(x, n)
        neighbors_list.append(n)
        print(str(n)+' '+str(len(neighbors_list)), file=f2)
        for m in neighbors_list:
            print(str(m)+' '+str(n), file = f2)
        print('', file=f2)
    f2.close()
############ main ################
gen_paths()
e=read_file(file_path)
partition_dataset(build_graph(e))
print('Done!')