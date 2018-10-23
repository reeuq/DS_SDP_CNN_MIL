from six.moves import cPickle as pickle
import codecs
from dataset import sen2SDP
from nltk.parse.stanford import StanfordDependencyParser
import os
import logging
import time

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


# 读取pickle文件
f = open("./sen2sdp_result/FilterNYT/train/train_sdp.pickle", 'rb')
f_data = pickle.load(f)
data = f_data["data"]
f.close()

# 读取日志文件
f = codecs.open('./sen2sdp_result/FilterNYT/train/201810011422.log', 'r', encoding='utf-8')
lines = set(f.readlines())
f.close()
f = codecs.open('./sen2sdp_result/FilterNYT/train/201810011444.log', 'r', encoding='utf-8')
lines.update(f.readlines())
f.close()

# 获取出错的实体对
error_entity = []
for line in lines:
    index = line.find('- ERROR: [u')
    if index != -1:
        temp_contain_entity = line[index:]
        index_wrong_code = temp_contain_entity.find('\\x')
        while index_wrong_code != -1:
            wrong_code = temp_contain_entity[index_wrong_code:index_wrong_code + 4]
            right_code = chr(int(wrong_code.replace('\\', '0'), 16))
            temp_contain_entity = temp_contain_entity.replace(wrong_code, right_code)
            index_wrong_code = temp_contain_entity.find('\\x')

        separate_index = temp_contain_entity.find("', u'")
        entity_1 = temp_contain_entity[12:separate_index]
        entity_2 = temp_contain_entity[separate_index+5:-3]

        error_entity.append((entity_1, entity_2))

dep_parser = StanfordDependencyParser(model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')

times = 0
for entity in error_entity:
    for item in data:
        if entity[0] == item[0][0] and entity[1] == item[0][1]:
            try:
                bag_dependency_tree = dep_parser.parse_sents(item[2])
                bag_sdps = sen2SDP.get_sentence_sdp(bag_dependency_tree, entity)
                item[2] = bag_sdps
                times += 1
            except Exception:
                logger.error(entity, exc_info=True)

fw = codecs.open("./sen2sdp_result_final/FilterNYT/train/train_sdp.pickle", 'wb')
result = {
    'data': data,
}
pickle.dump(result, fw, protocol=2)
fw.close()

print("error entity number is: " + str(len(error_entity)))
print("times is: " + str(times))
