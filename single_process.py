import re
import ipdb
import json
import jieba
import socket
import torch
import threading
from path import Path
from string import punctuation
from gensim.models import Word2Vec
from configure import ConfigParser
from net.full_model import FullModel

config = ConfigParser()
path = Path(config)

# Parameters
batch_size = 1
max_seq_len = config.getint("preprocess", "max_seq_len")
use_gpu = config.getboolean("train", "use_gpu")
device_num = config.getint("train", "gpu_device")
cuda = torch.device("cuda:%d" % device_num)
cpu = torch.device("cpu")
device = cuda if use_gpu else cpu

# Filter
sub_list = ["经[\w]*审理查明",
            "公诉机关指控",
            "[\w]+检察院指控",
            "[\w]+起诉书指控",
            "[\d]+年[\d]+月[\d]+日[\w]+[\d]+[时]+[左右|许]+"
            "[应]*[依照]*《中华人民共和国刑法》[\w]+[的|之]*[规定]*",
            "认为被告人[\w]+的行为已构成[×|X|＊]+罪"
            "[提请本院依法惩处|要求依法判处|诉请本院依法判处]",
            "[足以|据此]认定",
            "上述[指控|事实]",
            "[×|X|＊]+",
            "\r\n"]
add_punc = "，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥"
all_punc = (punctuation + add_punc).split()
with open(path.stop_word_path, "r", encoding="UTF-8") as file:
    stop_words = [word.strip() for word in file.readlines()]
filter = all_punc + stop_words

# Word2Vec
model = Word2Vec.load(path.w2v_model_path)
vocab = model.wv.vocab
dictionary = dict()
embed_matrix = []
for word in vocab.keys():
    dictionary[word] = len(dictionary)
    embed_matrix.append(model.wv[word].tolist())

# Model
config.set("train", "batch_size", batch_size)
model = FullModel()
model_path = path.model_path
model_pretrain_path = model_path + "model-%d.pkl" % config.getint("train", "last_model")
model.load_state_dict(torch.load(model_pretrain_path))
model.to(device)
model.eval()

def predict(fact):
    for sub in sub_list:
        sub = re.compile(sub)
        fact = re.sub(sub, "", fact)
    fact_cut = " ".join(jieba.cut(fact.strip()))
    fact_embed = []
    fact_len = 0
    for word in fact_cut:
        if fact_len == max_seq_len:
            break
        if word not in filter:
            try:
                fact_embed.append(dictionary[word])
                fact_len += 1
            except Exception:
                continue
    input_fact = torch.tensor(fact_embed, device=device, dtype=torch.int64)
    input_len = torch.tensor([fact_len], device=device, dtype=torch.int64)
    _, _, _, tags_accu, tags_article, tags_imprison = model(input_fact, input_len)
    output = dict()
    output['accusation'] = tags_accu.tolist()
    output['articles'] = tags_article.tolist()
    output['imprison'] = tags_imprison.tolist()
    return output

def server_start():
    serversocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    host = socket.gethostname()
    port = 12345
    serversocket.bind((host,port))
    serversocket.listen(5)
    myaddr = serversocket.getsockname()
    print("Server Address:%s"%str(myaddr))
    while True:
        clientsocket,addr = serversocket.accept()
        print("Link Address:%s" % str(addr))
        try:
            t = ServerThreading(clientsocket)
            t.start()
            pass
        except Exception as identifier:
            print(identifier)
            pass
        pass
    serversocket.close()
    pass



class ServerThreading(threading.Thread):
    def __init__(self,clientsocket,recvsize=1024*1024,encoding="gbk"):
        threading.Thread.__init__(self)
        self._socket = clientsocket
        self._recvsize = recvsize
        self._encoding = encoding
        pass

    def run(self):
        print("Start threading.....")
        try:
            msg = ''
            while True:
                rec = self._socket.recv(self._recvsize)
                msg += rec.decode(self._encoding)
                if msg.strip().endswith('over'):
                    msg=msg[:-4]
                    break
            re = json.loads(msg)
            res = predict(re['content'])
            sendmsg = json.dumps(res)
            self._socket.send(("%s"%sendmsg).encode(self._encoding))
            pass
        except Exception as identifier:
            self._socket.send("500".encode(self._encoding))
            print(identifier)
            pass
        finally:
            self._socket.close()
        print("Mission Completed.....")
        pass

    def __del__(self):
        pass

if __name__ == "__main__":
    server_start()