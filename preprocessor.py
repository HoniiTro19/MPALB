import re
import os
import ipdb
import json
import time
import jieba
from path import Path
from demo import Demo
from string import punctuation
from gensim.models import Word2Vec
from configure import ConfigParser
from utils import chinese2number, load_list_dict

class Preprocessor:
    def __init__(self):
        self.config = ConfigParser()
        self.path = Path(self.config)
        self.raw_max_len = 0
        self.raw_min_len = 99999
        self.seg_max_len = 0
        self.seg_min_len = 99999
        self.max_seq_len = self.config.getint("preprocess", "max_seq_len")
        self.interval = self.config.getint("preprocess", "seq_len_interval")
        self.raw_seq_len = list()
        self.seg_seq_len = list()
        self.intervals = [6, 9, 12, 12 * 2, 12 * 3, 12 * 5, 12 * 7, 12 * 10, 12*9999]
        self.imprison_interval = list()
        self.demo = Demo()
        self.is_remove_unfrequent = self.config.getboolean("preprocess", "is_remove_unfrequent")
        self.set_type = self.config.get("train", "train_set")
        self.split_train_num = self.config.getint("preprocess", "split_train_num")
        self.split_test_num = self.config.getint("preprocess", "split_test_num")

    def preprocess(self):
        print("extracting articles")
        self.extract_law()

        print("formalizing datasets")
        if self.set_type == "small":
            self.dataset_formal(self.path.train_small_path, self.path.train_small_seg_path)
            self.dataset_formal(self.path.valid_small_path, self.path.valid_small_seg_path)
            self.dataset_formal(self.path.test_small_path, self.path.test_small_seg_path)
        elif self.set_type == "large":
            self.split_large()
            data_large_path = self.path.data_large_path
            preprocess_path = self.path.preprocess_path
            raw_train_data = [data_large_path + "train-%d.json" % idx for idx in range(self.split_train_num)]
            seg_train_data = [preprocess_path + "train_large_seg-%d.json" % idx for idx in range(self.split_train_num)]
            raw_test_data = [data_large_path + "test-%d.json" % idx for idx in range(self.split_test_num)]
            seg_test_data = [preprocess_path + "test_large_seg-%d.json" % idx for idx in range(self.split_test_num)]
            for src_path, tar_path in zip(raw_train_data, seg_train_data):
                self.dataset_formal(src_path, tar_path)
            for src_path, tar_path in zip(raw_test_data, seg_test_data):
                self.dataset_formal(src_path, tar_path)
        else:
            raise Exception("Cannot recognize the set type %s, suppose to be 'small/large'" % self.set_type)

        print("filtering unfrequent types")
        self.filter_frequent(self.set_type)

        self.article_formal()
        print("Max sentence length in raw data %d" % self.raw_max_len)
        print("Min sentence length in raw data %d" % self.raw_min_len)
        self.imprison_list()
        self.demo.imprison_plot(self.imprison_interval)

        self.word2vec(self.set_type)
        self.demo.seq_len_histplot(self.raw_seq_len, self.seg_seq_len)
        print("Max sentence length after segmentation %d" % self.seg_max_len)
        print("Min sentence length after segmentation %d" % self.seg_min_len)

    def filter_frequent(self, set_type):
        with open(self.path.meta_accu_path, "r", encoding="UTF-8") as file:
            accus = [accu.strip("\n") for accu in file.readlines()]
        with open(self.path.meta_law_path, "r", encoding="UTF-8") as file:
            articles = [law.strip("\n") for law in file.readlines()]
        
        dataset = list()
        if set_type == "small":
            tar_accu_path = self.path.meta_accu_frequent_small_path
            tar_law_path = self.path.meta_law_frequent_small_path
            with open(self.path.train_small_seg_path, "r", encoding="UTF-8") as file:
                dataset.extend(json.loads(file.read()))
            with open(self.path.valid_small_seg_path, "r", encoding="UTF-8") as file:
                dataset.extend(json.loads(file.read()))
            with open(self.path.test_small_seg_path, "r", encoding="UTF-8") as file:
                dataset.extend(json.loads(file.read()))
        else:
            tar_accu_path = self.path.meta_accu_frequent_large_path
            tar_law_path = self.path.meta_law_frequent_large_path
            preprocess_path = self.path.preprocess_path
            seg_train_data = [preprocess_path + "train_large_seg-%d.json" % idx for idx in range(self.split_train_num)]
            seg_test_data = [preprocess_path + "test_large_seg-%d.json" % idx for idx in range(self.split_test_num)]
            for seg_path in seg_train_data + seg_test_data:
                with open(seg_path, "r", encoding="UTF-8") as file:
                    dataset.extend(json.loads(file.read()))

        accu_frequency = {accu: 0 for accu in accus}
        article_frequency = {article: 0 for article in articles}
        for data in dataset:
            accu_frequency[data["accusation"]] += 1
            article_frequency[data["article"]] += 1
        accu_frequent = [accu for accu in accus if accu_frequency[accu] > 120]
        article_frequent = [article for article in articles if article_frequency[article] > 120]

        with open(tar_accu_path, "w", encoding="UTF-8") as file:
            for accu in accu_frequent:
                file.write(accu + "\n")
        with open(tar_law_path, "w", encoding="UTF-8") as file:
            for article in article_frequent:
                file.write(article + "\n")

    def extract_law(self):
        with open(self.path.meta_penallaw_path, "r", encoding="UTF-8") as file:
            raw_article = file.read()

        article_list = list()
        article_dict = dict()
        part_pattern = re.compile(r"第[零一二三四五六七八九十百]+编.*")
        chapter_pattern = re.compile(r"第[零一二三四五六七八九十百]+章.*")
        raw_article = re.sub(part_pattern, "", raw_article)
        raw_article = re.sub(chapter_pattern, "", raw_article).strip()
        article_pattern = re.compile(r"第([零一二三四五六七八九十百]+)条(之[零一二三四五六七八九十百]+)?\s")
        articles = re.split(article_pattern, raw_article)
        article_id = 0
        is_subarticle = False
        for idx in range(1, len(articles)):
            if idx % 3 == 2:
                continue
            article = articles[idx]
            if idx % 3 == 1:
                article_id = chinese2number(article)
                if article_id == article:
                    raise Exception("Article Id Error")
                if article_id in article_list:
                    is_subarticle = True
                else:
                    is_subarticle = False
                    article_list.append(article_id)
            else:
                article = re.split(r"\n", article)[1]
                if is_subarticle:
                    article_dict[article_id] += article
                else:
                    article_dict[article_id] = article
        with open(self.path.article_dict_path, "w", encoding="UTF-8") as file:
            file.write(json.dumps(article_dict))

    def dataset_formal(self, src_path, tar_path):
        with open(src_path, "r", encoding="UTF-8") as file:
            lines = file.readlines()
        formal_data_list = list()
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

        for line in lines:
            formal_data = dict()
            line = json.loads(line.strip("\n"))
            fact = line["fact"]
            relevant_articles = line["meta"]["relevant_articles"]
            accusation = line["meta"]["accusation"]
            imprisonment = line["meta"]["term_of_imprisonment"]

            if len(accusation) > 1 or len(relevant_articles) > 1 or "二审" in fact:
                continue

            if len(fact) > self.raw_max_len:
                self.raw_max_len = len(fact)
            if len(fact) < self.raw_min_len:
                self.raw_min_len = len(fact)
            self.raw_seq_len.append(len(fact) // self.interval)
            formal_data["article"] = str(relevant_articles[0])
            formal_data["accusation"] = accusation[0].replace("[", "").replace("]", "")

            for sub in sub_list:
                sub = re.compile(sub)
                fact = re.sub(sub, "", fact)

            fact_cut = " ".join(jieba.cut(fact.strip()))
            formal_data["fact"] = fact_cut

            # Divide imprisonment inteval
            if imprisonment["death_penalty"] == True or imprisonment["life_imprisonment"] == True:
                formal_data["imprison"] = len(self.intervals) + 1
                self.imprison_interval.append("DP_LI")
            elif imprisonment["imprisonment"] == 0:
                formal_data["imprison"] = 0
                self.imprison_interval.append("0")
            else:
                for idx, interval in enumerate(self.intervals):
                    if imprisonment["imprisonment"] <= interval:
                        formal_data["imprison"] = idx + 1
                        if idx != len(self.intervals) - 1:
                            self.imprison_interval.append("%d" % self.intervals[idx])
                        else:
                            self.imprison_interval.append("%d-" % self.intervals[idx-1])
                        break
            formal_data_list.append(formal_data)
        with open(tar_path, "w", encoding="UTF-8") as file:
            file.write(json.dumps(formal_data_list))

    def article_formal(self):
        with open(self.path.article_dict_path, "r", encoding="UTF-8") as file:
            articles = file.read()
        articles = json.loads(articles)
        criminal_basis_ids = [id for id in self.config.get("preprocess", "criminal_basis_ids").split(",")]
        penalty_basis_ids = [id for id in self.config.get("preprocess", "penalty_basis_ids").split(",")]
        article_ids, _ = load_list_dict(self.path.meta_law_path)
        article_ids_small, _ = load_list_dict(self.path.meta_law_frequent_small_path)
        article_ids_large, _ = load_list_dict(self.path.meta_law_frequent_large_path)

        criminal_basis_segs = list()
        penalty_basis_segs = list()
        article_segs = list()
        article_segs_small = list()
        article_segs_large = list()
        for idx in article_ids:
            article_seg = dict()
            article_seg["fact"] = " ".join(jieba.cut(articles[idx].strip()))
            article_segs.append(article_seg)
        for idx in article_ids_small:
            article_seg = dict()
            article_seg["fact"] = " ".join(jieba.cut(articles[idx].strip()))
            article_segs_small.append(article_seg)
        for idx in article_ids_large:
            article_seg = dict()
            article_seg["fact"] = " ".join(jieba.cut(articles[idx].strip()))
            article_segs_large.append(article_seg)
        for idx in criminal_basis_ids:
            criminal_basis_seg = dict()
            criminal_basis_seg["fact"] = " ".join(jieba.cut(articles[idx].strip()))
            criminal_basis_segs.append(criminal_basis_seg)
        for idx in penalty_basis_ids:
            penalty_basis_seg = dict()
            penalty_basis_seg["fact"] = " ".join(jieba.cut(articles[idx].strip()))
            penalty_basis_segs.append(penalty_basis_seg)

        with open(self.path.criminal_basis_seg_path, "w", encoding="UTF-8") as file:
            file.write(json.dumps(criminal_basis_segs))
        with open(self.path.penalty_basis_seg_path, "w", encoding="UTF-8") as file:
            file.write(json.dumps(penalty_basis_segs))
        with open(self.path.article_seg_path, "w", encoding="UTF-8") as file:
            file.write(json.dumps(article_segs))
        with open(self.path.article_seg_small_path, "w", encoding="UTF-8") as file:
            file.write(json.dumps(article_segs_small))
        with open(self.path.article_seg_large_path, "w", encoding="UTF-8") as file:
            file.write(json.dumps(article_segs_large))

    def imprison_list(self):
        meta_imprison_path = self.path.meta_imprison_path
        with open(meta_imprison_path, "w", encoding="UTF-8") as file:
            file.write("0" + "\n")
            for interval in self.intervals:
                file.write(str(interval) + "\n")
            file.write("DP_LI")

    def word2vec(self, set_type):
        print("word2vec()")
        print("collecting data")
        if set_type == "small":
            tar_fact_list = [self.path.train_small_w2v_path,
                             self.path.test_small_w2v_path,
                             self.path.valid_small_w2v_path]
            src_fact_list = [self.path.train_small_seg_path,
                             self.path.test_small_seg_path,
                             self.path.valid_small_seg_path]
        else:
            preprocess_path = self.path.preprocess_path
            src_train_list = [preprocess_path + "train_large_seg-%d.json" % idx for idx in range(self.split_train_num)]
            src_test_list = [preprocess_path + "test_large_seg-%d.json" % idx for idx in range(self.split_test_num)]
            tar_train_list = [preprocess_path + "train_large_w2v-%d.json" % idx for idx in range(self.split_train_num)]
            tar_test_list = [preprocess_path + "test_large_w2v-%d.json" % idx for idx in range(self.split_test_num)]
            tar_fact_list = tar_train_list + tar_test_list
            src_fact_list = src_train_list + src_test_list

        tar_article_list = [self.path.article_w2v_path,
                            self.path.criminal_basis_w2v_path,
                            self.path.penalty_basis_w2v_path]

        src_article_list = [self.path.article_seg_path,
                            self.path.criminal_basis_seg_path,
                            self.path.penalty_basis_seg_path]

        if set_type == "small":
            tar_article_list.append(self.path.article_w2v_small_path)
            src_article_list.append(self.path.article_seg_small_path)
        else:
            tar_article_list.append(self.path.article_w2v_large_path)
            src_article_list.append(self.path.article_seg_large_path)

        add_punc = "，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥"
        all_punc = (punctuation + add_punc).split()
        with open(self.path.stop_word_path, "r", encoding="UTF-8") as file:
            stop_words = [word.strip() for word in file.readlines()]
        filter = all_punc + stop_words
        cases = []
        w2v_model_path = self.path.w2v_model_path
        embed_size = self.config.getint("preprocess", "embed_size")
        if not self.config.getboolean("preprocess", "is_train_wordvec") and os.path.exists(w2v_model_path):
            model = Word2Vec.load(self.path.w2v_model_path)
        else:
            for src_path in src_fact_list:
                with open(src_path, "r", encoding="UTF-8") as file:
                    lines = json.loads(file.read())
                idx = 0
                for line in lines:
                    idx += 1
                    fact = line["fact"].split()
                    fact = [word for word in fact if word not in filter]
                    if len(fact) > self.seg_max_len:
                        self.seg_max_len = len(fact)
                    if len(fact) < self.seg_min_len:
                        self.seg_min_len = len(fact)
                    cases.append(fact)
                    if idx % 10000 == 0:
                        print(line["fact"])
            print("training gensim Word2Vec")
            word2vec_epoch = self.config.getint("preprocess", "word2vec_epoch")
            model = Word2Vec(cases, workers=4, min_count=25, sg=1, iter=word2vec_epoch, size=embed_size)
            model.save(w2v_model_path)
        vocab = model.wv.vocab
        dictionary = dict()
        embed_matrix = []
        for word in vocab.keys():
            dictionary[word] = len(dictionary)
            embed_matrix.append(model.wv[word].tolist())

        # BLANK
        embed_matrix.append([0] * embed_size)
        with open(self.path.embed_matrix_path, "w", encoding="UTF-8") as file:
            file.write(json.dumps(embed_matrix))

        # Multi Layer
        # for src_path, tar_path in zip(src_data_list, tar_data_list):
        #     file = open(src_path, "r")
        #     lines = json.loads(file.read())
        #     file.close()
        #     for line in lines:
        #         facts = line["fact"].split("。")
        #         fact_embed = []
        #         fact_len = []
        #         for fact in facts:
        #             if len(fact) == 0:
        #                 pass
        #             if len(fact_len) == self.max_doc_len:
        #                 break
        #             line_embed = []
        #             fact_len.append(0)
        #             for word in fact:
        #                 if word not in filter:
        #                     if fact_len[-1] < self.max_sent_len:
        #                         try:
        #                             line_embed.append(dictionary[word])
        #                         except Exception:
        #                             line_embed.append(len(dictionary))
        #                         fact_len[-1] += 1
        #             line_embed += [len(dictionary) + 1] * (self.max_sent_len - fact_len[-1])
        #             fact_embed.append(line_embed)
        #         fact_len_sum = sum(fact_len)
        #         self.seg_seq_len.append(fact_len_sum // self.interval)
        #         pad_embed = [len(dictionary) + 1] * self.max_sent_len
        #         pad_doc_num = self.max_doc_len - len(fact_len)
        #         for _ in range(pad_doc_num):
        #             fact_embed.append(pad_embed)
        #             fact_len.append(0)
        #         line["fact"] = fact_embed
        #         line["len"] = fact_len
        #     file = open(tar_path, "w")
        #     file.write(json.dumps(lines))
        #     file.close()

        print("embedding the context")
        if self.is_remove_unfrequent:
            law_path = self.path.meta_law_frequent_small_path
            accu_path = self.path.meta_accu_frequent_small_path
        else:
            law_path = self.path.meta_law_path
            accu_path = self.path.meta_accu_path

        law_list, law_dict = load_list_dict(law_path)
        accu_list, accu_dict = load_list_dict(accu_path)
        for src_path, tar_path in zip(src_fact_list, tar_fact_list):
            with open(src_path, "r") as file:
                lines = json.loads(file.read())
            out = list()
            for line in lines:
                fact = line["fact"]
                if line["accusation"] not in accu_list or line["article"] not in law_list:
                    continue
                fact_embed = []
                fact_len = 0
                for word in fact:
                    if fact_len == self.max_seq_len:
                        break
                    if word not in filter:
                        try:
                            fact_embed.append(dictionary[word])
                            fact_len += 1
                        except Exception:
                            continue
                            # fact_embed.append(len(dictionary))

                if fact_len == 0:
                    continue
                self.seg_seq_len.append(fact_len // self.interval)
                fact_embed += [len(dictionary)] * (self.max_seq_len - fact_len)
                line["fact"] = fact_embed
                line["accusation"] = accu_dict[line["accusation"]]
                line["article"] = law_dict[line["article"]]
                line["len"] = fact_len
                out.append(line)
            with open(tar_path, "w") as file:
                file.write(json.dumps(out))

        for src_path, tar_path in zip(src_article_list, tar_article_list):
            with open(src_path, "r") as file:
                lines = json.loads(file.read())
            out = list()
            for line in lines:
                fact = line["fact"]
                fact_embed = []
                fact_len = 0
                for word in fact:
                    if fact_len == self.max_seq_len:
                        break
                    if word not in filter:
                        try:
                            fact_embed.append(dictionary[word])
                            fact_len += 1
                        except Exception:
                            continue
                            # fact_embed.append(len(dictionary))
                if fact_len == 0:
                    continue
                fact_embed += [len(dictionary)] * (self.max_seq_len - fact_len)
                line["fact"] = fact_embed
                line["len"] = fact_len
                out.append(line)
            with open(tar_path, "w") as file:
                file.write(json.dumps(out))

    def split_large(self):
        data_large_path = self.path.data_large_path
        train_large_data = self.path.train_large_path
        with open(train_large_data, "r", encoding="UTF-8") as file:
            lines = file.readlines()
        total_train_num = len(lines)
        interval = total_train_num // self.split_train_num
        for file_id in range(self.split_train_num):
            file_name = data_large_path + "train-%s.json" % file_id
            start = interval * file_id
            end = min(interval * (file_id + 1), total_train_num)
            with open(file_name, "w", encoding="UTF-8") as file:
                for idx in range(start, end):
                    file.write(lines[idx])

        test_large_data = self.path.test_large_path
        with open(test_large_data, "r", encoding="UTF-8") as file:
            lines = file.readlines()
        total_test_num = len(lines)
        interval = total_test_num // self.split_test_num
        for file_id in range(self.split_test_num):
            file_name = data_large_path + "test-%s.json" % file_id
            start = interval * file_id
            end = min(interval * (file_id + 1), total_test_num)
            with open(file_name, "w", encoding="UTF-8") as file:
                for idx in range(start, end):
                    file.write(lines[idx])
                    
if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.preprocess()