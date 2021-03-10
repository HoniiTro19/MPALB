import os
import json
import ipdb
import numpy as np
from path import Path
from collections import Counter
import matplotlib.pyplot as plt
from configure import ConfigParser

config = ConfigParser()
path = Path(config)

def count(file_path):
    accu, accu_amount, law, law_amount, imprison_time = [], [], [], [], []
    file = open(file_path, "r", encoding="UTF-8")
    lines = file.readlines()
    for line in lines:
        line = json.loads(line)
        meta = line["meta"]
        for idx in range(len(meta["accusation"])):
           accu.append(meta["accusation"][idx].replace("[", "").replace("]", ""))
        accu_amount.append(len(meta["accusation"]))
        for idx in range(len(meta["relevant_articles"])):
            law.append(str(meta["relevant_articles"][idx]))
        law_amount.append(len(meta["relevant_articles"]))
        if meta["term_of_imprisonment"]["death_penalty"] == True or meta["term_of_imprisonment"]["life_imprisonment"] == True:
            imprison_time.append("Death or Life Imprisonment")
        else:
            imprison_time.append(meta["term_of_imprisonment"]["imprisonment"])
    return accu, accu_amount, law, law_amount, imprison_time

def imprison_count(imprison_time_list):
    death_life_prison = 0
    total_prison_time = 0
    for imprison_time in imprison_time_list:
        if imprison_time == "Death or Life Imprisonment":
            death_life_prison += 1
        else:
            total_prison_time += imprison_time
    avg_prison_time = total_prison_time / (len(imprison_time_list) - death_life_prison)
    return avg_prison_time, death_life_prison

def amount_histplot(accu_amount_list, law_amount_list, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False, tight_layout=True)
    accu_amount_max = max(accu_amount_list)
    law_amount_max = max(law_amount_list)
    ax1.set_xlabel("Accusation Number")
    ax1.set_ylabel("Occurrence Number")
    ax1.set_xticks(np.arange(1, accu_amount_max, step=2))
    ax1.hist(accu_amount_list, bins=accu_amount_max)
    ax2.set_xlabel("Law number")
    ax2.set_ylabel("Occurrence Number")
    ax2.set_xticks(np.arange(1, law_amount_max, step=2))
    ax2.hist(law_amount_list, bins=np.arange(1, law_amount_max))
    fig.savefig(save_path)

def most_least_common_num(counter):
    nums = counter.most_common()
    most_common = [temp[1] for temp in nums[:10]]
    least_common = [temp[1] for temp in nums[-10:]]
    most_common_sum = sum(most_common)
    least_commom_sum = sum(least_common)
    return most_common_sum, least_commom_sum


class Demo:
    def __init__(self):
        pass

    """
        Visualize the statistics information of CAIL2018 dataset
        1. Number of documents in train/valid/test set in small/large dataset
        2. Number of accusation/law types occurred in all sets
        3. Most/Least frequent accusation/law types
        4. The average imprisonment time in each case in all sets, including death penalty and life imprisonment 
        5. The amount of related laws and accusations in each case in all sets
    """
    def statistics(self):
        meta_accu, meta_law = list(), list()
        # Accusations and laws downloaded from https://github.com/thunlp/CAIL/tree/master/meta
        file = open(path.meta_accu_path, "r", encoding="UTF-8")
        lines = file.readlines()
        file.close()

        for line in lines:
            meta_accu.append(line.strip())
        assert len(meta_accu) == len(set(meta_accu)), "Duplication in accu.txt."
        file = open(path.meta_law_path, "r", encoding="UTF-8")
        lines = file.readlines()
        file.close()

        for line in lines:
            meta_law.append(line.strip())
        assert len(meta_law) == len(set(meta_law)), "Duplication in law.txt."

        # Accusations, laws, and imprisonment extracted from dataset https://cail.oss-cn-qingdao.aliyuncs.com/CAIL2018_ALL_DATA.zip
        accu_trainsmall, accu_amount_trainsmall, law_trainsmall, law_amount_trainsmall, imprison_time_trainsmall = count(path.train_small_path)
        accu_validsmall, accu_amount_validsmall, law_validsmall, law_amount_validsmall, imprison_time_validsmall = count(path.valid_small_path)
        accu_testsmall, accu_amount_testsmall, law_testsmall, law_amount_testsmall, imprison_time_testsmall = count(path.test_small_path)
        accu_trainlarge, accu_amount_trainlarge, law_trainlarge, law_amount_trainlarge, imprison_time_trainlarge = count(path.train_large_path)
        accu_testlarge, accu_amount_testlarge, law_testlarge, law_amount_testlarge, imprison_time_testlarge = count(path.test_large_path)
        accu_rest, accu_amount_rest, law_rest, law_amount_rest, imprison_time_rest = count(path.data_rest_path)
        accu_finaltest, accu_amount_finaltest, law_finaltest, law_amount_finaltest, imprison_time_finaltest = count(path.data_finaltest_path)

        accu = list(set(accu_trainsmall + accu_validsmall + accu_testsmall + accu_trainlarge + accu_testlarge + accu_rest + accu_finaltest))
        law = list(set(law_trainsmall + law_validsmall + law_testsmall + law_trainlarge + law_testlarge + law_rest + law_finaltest))

        if Counter(accu) != Counter(meta_accu):
            file = open(path.accu_diff_log_path, "w", encoding="UTF-8")
            for each_accu in accu:
                if each_accu not in meta_accu:
                    file.write(each_accu + "\n")
            file.close()

        if Counter(law) != Counter(meta_law):
            file = open(path.law_diff_log_path, "w", encoding="UTF-8")
            for each_law in law:
                if each_law not in meta_law:
                    file.write(str(each_law) + "\n")
            file.close()
        assert Counter(accu) == Counter(meta_accu), "Extracted accus cannot match accu.txt."
        assert Counter(law) == Counter(meta_law), "Extracted laws cannot match law.txt."

        accu_small = accu_trainsmall + accu_validsmall + accu_testsmall
        accu_large = accu_trainlarge + accu_testlarge
        accu_amount_small = accu_amount_trainsmall + accu_amount_validsmall + accu_amount_testsmall
        accu_amount_large = accu_amount_trainlarge + accu_amount_testlarge
        law_small = law_trainsmall + law_validsmall + law_testsmall
        law_large = law_trainlarge + law_testlarge
        law_amount_small = law_amount_trainsmall + law_amount_validsmall + law_amount_testsmall
        law_amount_large = law_amount_trainlarge + law_amount_testlarge
        imprison_time_small = imprison_time_trainsmall + imprison_time_validsmall + imprison_time_testsmall
        imprison_time_large = imprison_time_trainlarge + imprison_time_testlarge

        assert len(accu_amount_small) == len(law_amount_small) == len(imprison_time_small), "Extracted number of accu, law and imprisionment cannot match in small dataset."
        assert len(accu_amount_large) == len(law_amount_large) == len(imprison_time_large), "Extracted number of accu, law and imprisionment cannot match in large dataset."

        avg_prison_small, death_life_prison_small = imprison_count(imprison_time_small)
        avg_prison_large, death_life_prison_large = imprison_count(imprison_time_large)

        # Record statistics information
        accu_small_counter = Counter(accu_small)
        accu_large_counter = Counter(accu_large)
        law_small_counter = Counter(law_small)
        law_large_counter = Counter(law_large)
        accu_small_most_common, accu_small_least_common = most_least_common_num(accu_small_counter)
        accu_large_most_common, accu_large_least_common = most_least_common_num(accu_large_counter)
        law_small_most_common, law_small_least_common = most_least_common_num(law_small_counter)
        law_large_most_common, law_large_least_common = most_least_common_num(law_large_counter)

        file = open(path.demo_statistics_path, "w", encoding="UTF-8")
        # 1
        file.write("%d cases in exricise_contest/data_train.json" % len(accu_amount_trainsmall) + "\n")
        file.write("%d cases in exricise_contest/data_valid.json" % len(accu_amount_validsmall) + "\n")
        file.write("%d cases in exricise_contest/data_test.json" % len(accu_amount_testsmall) + "\n")
        file.write("%d cases in first_stage/train.json" % len(accu_amount_trainlarge) + "\n")
        file.write("%d cases in first_stage/test.json" % len(accu_amount_testlarge) + "\n")
        file.write("%d cases in restData/rest_data.json" % len(accu_amount_rest) + "\n")
        file.write("%d cases in final_test.json" % len(accu_amount_finaltest) + "\n" + "\n")
        # 2
        file.write("%d accusation types in all datasets" % len(accu) + "\n")
        file.write("%d law types in all datasets" % len(law) + "\n" + "\n")
        # 4
        file.write("average prison time in exercise_contest/ is %d months" % avg_prison_small + "\n")
        file.write("totally %d death or life imprison cases in exeicise_contest/" % death_life_prison_small + "\n")
        file.write("average prison time in first_stage/ is %d months" % avg_prison_large + "\n")
        file.write("totally %d death or life imprison cases in first_stage/" % death_life_prison_large + "\n")

        file.write("top 10 most/least common cases number in exercise_contest/ is %d/%d" % (accu_small_most_common, accu_small_least_common) + "\n")
        file.write("top 10 most/least common cases number in first_stage/ is %d/%d" % (accu_large_most_common, accu_large_least_common) + "\n")
        file.write("top 10 most/least common laws number in exercise_contest/ is %d/%d" % (law_small_most_common, law_small_least_common) + "\n")
        file.write("top 10 most/least common laws number in first_stage/ is %d/%d" % (law_large_most_common, law_large_least_common) + "\n")
        file.close()

        # 3
        file = open(path.demo_distribution_accu_path, "w", encoding="UTF-8")
        file.write(json.dumps(dict(accu_small_counter)) + "\n" + json.dumps(dict(accu_large_counter)))
        file.close()
        file = open(path.demo_distribution_article_path, "w", encoding="UTF-8")
        file.write(json.dumps(dict(law_small_counter)) + "\n" + json.dumps(dict(law_large_counter)))
        file.close()

        # 5
        if not os.path.isdir(path.demo_figure_path):
            os.mkdir(path.demo_figure_path)
        amount_histplot(accu_amount_small, law_amount_small, path.demo_figure_path + "amount_small.png")
        amount_histplot(accu_amount_large, law_amount_large, path.demo_figure_path + "amount_large.png")

    def seq_len_histplot(self, raw_seq_len, seg_seq_len):
        seq_len_interval = config.getint("preprocess", "seq_len_interval")
        raw_seq_len = [temp * seq_len_interval for temp in raw_seq_len]
        seg_seq_len = [temp * seq_len_interval for temp in seg_seq_len]
        seg_seq_max = max(seg_seq_len)
        save_path = path.demo_figure_path + "seq_len.png"
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False, tight_layout=True)
        ax1.set_xlabel("Raw Sequence Length")
        ax1.set_ylabel("Occurrence Number")
        ax1.hist(raw_seq_len, bins=np.arange(0, 2000, step=100), range=(0, 2000))
        ax2.set_xlabel("Segmented Sequence Length")
        ax2.set_ylabel("Occurrence Number")
        ax2.hist(seg_seq_len, bins=np.arange(0, seg_seq_max, step=100), range=(0, seg_seq_max))
        fig.savefig(save_path)

    def imprison_plot(self, imprison_interval):
        save_path = path.demo_figure_path + "imprison.png"
        intervals = ["0", "6", "9", "12", "24", "36", "60", "84", "120", "120-", "DP_LI"]
        interval_count = {interval: 0 for interval in intervals}
        for interval in imprison_interval:
            interval_count[interval] += 1
        plt.figure()
        plt.bar(np.arange(len(intervals)), interval_count.values())
        plt.xlabel("Imprisonment Length")
        plt.ylabel("Occurrence Number")
        plt.xticks(np.arange(len(intervals)), intervals)
        plt.savefig(save_path)
        plt.close()

    def train_valid_record(self, name, train_loss, valid_loss):
        epoch_num = len(train_loss)
        save_path = path.demo_figure_path + name
        fig, ax = plt.subplots()
        ax.plot(train_loss, label="Train Loss")
        ax.plot(valid_loss, label="Valid Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_xticks(np.arange(epoch_num))
        ax.legend()
        fig.savefig(save_path)

    def optimize_plot(self, name, accu_f1, article_f1, imprison_f1, xticks):
        x = np.arange(len(xticks))
        save_path = path.demo_figure_path + name
        plt.figure()
        plt.plot(x, accu_f1, label="Accusation F1")
        plt.plot(x, article_f1, label="Article F1")
        plt.plot(x, imprison_f1, label="Imprison F1")
        plt.xlabel("Value")
        plt.ylabel("Percentage")
        plt.xticks(x, xticks)
        plt.legend()
        plt.margins(0.2)
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(save_path)
        plt.close()

    def display_imprison(self, cm, name):
        classes = ["0", "6", "9", "12", "24", "36", "60", "84", "120", "120-", "DP_LI"]
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        str_cm = cm.astype(np.str).tolist()

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if int(cm[i, j] * 100 + 0.5) == 0:
                    cm[i, j] = 0

        fig, ax = plt.subplots()

        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               ylabel='Actual',
               xlabel='Predicted')

        ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
        ax.tick_params(which="minor", bottom=False, left=False)

        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if int(cm[i, j] * 100 + 0.5) > 0:
                    ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        save_path = path.demo_figure_path + name
        plt.savefig(save_path, dpi=300)

if __name__ == "__main__":
    path = Path(config)
    demo = Demo()
    demo.statistics()
