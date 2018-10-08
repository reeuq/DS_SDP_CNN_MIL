from nltk.parse.stanford import StanfordDependencyParser
import os
import networkx as nx
from six.moves import cPickle as pickle
import codecs
import time
import logging

os.environ['STANFORD_PARSER'] = './../lib/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = './../lib/stanford-parser-3.9.1-models.jar'


logger = logging.getLogger()
logger.setLevel(logging.INFO)

rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_path = os.path.dirname(os.getcwd()) + '/logs/'
log_name = log_path + rq + '.log'
logfile = log_name
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)


def get_sentence_sdp(dependency_trees, relation_entity):
    sdps = []
    for dep_graphs in dependency_trees:
        for parse in dep_graphs:
            edges = []
            for item in parse.triples():
                item_head = item[0][0]
                item_tail = item[2][0]
                edges.append((item_head, item_tail))
            graph = nx.Graph(edges)

            # pos = nx.spring_layout(graph, iterations=5000)
            # plt.figure()
            # nx.draw(graph, pos)

            shortest_path = nx.shortest_path(graph, source=relation_entity[0], target=relation_entity[1])
            sdp = ''
            for index in range(len(shortest_path) - 1):
                for item in parse.triples():
                    if item[0][0] == shortest_path[index] and item[2][0] == shortest_path[index+1]:
                        sdp = sdp + shortest_path[index] + ' >>> ' + item[1] + ' >>> '
                    elif item[0][0] == shortest_path[index+1] and item[2][0] == shortest_path[index]:
                        sdp = sdp + shortest_path[index] + ' <<< ' + item[1] + ' <<< '
            sdp = sdp + shortest_path[index + 1]
            sdps.append(sdp)
    return sdps


def sen2SDP(path):
    f = codecs.open(path, "r", encoding="utf-8")
    data = []
    pqdm = 0
    fw = codecs.open("./sen2sdp_result/FilterNYT/train/train_sdp" + str(pqdm / 100) + ".pickle", 'wb')
    while 1:
        if pqdm % 100 == 0 and pqdm != 0:
            result = {
                'data': data,
            }
            pickle.dump(result, fw, protocol=2)
            fw.close()
            data = []
            fw = codecs.open("./sen2sdp_result/FilterNYT/train/train_sdp" + str(pqdm / 100) + ".pickle", 'wb')
            print(pqdm)

        line = f.readline()
        if not line:
            break
        entities = line.split(' ')[:-1]

        line = f.readline()
        bagLabel = line.split(' ')
        label = int(bagLabel[0])
        num = int(bagLabel[-1])

        sentences = []
        for i in range(0, num):
            sent = f.readline().split(' ')[:-1]
            sentences.append(sent)

        try:
            bag_dependency_tree = dep_parser.parse_sents(sentences)
            bag_sdps = get_sentence_sdp(bag_dependency_tree, entities)
            data.append([entities, num, bag_sdps, label])
        except Exception:
            data.append([entities, num, sentences, label])
            logger.error(entities, exc_info=True)
        pqdm += 1

    result = {
        'data': data,
    }
    pickle.dump(result, fw, protocol=2)
    fw.close()
    f.close()


if __name__ == "__main__":
    dep_parser = StanfordDependencyParser(model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')
    print("-------------test data start--------------")
    sen2SDP("./trans2word_result/FilterNYT/train/train1.txt")
    print("--------------test data end---------------")

