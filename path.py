import os

class Path:
    def __init__(self, config):
        self.parent_dir = os.path.abspath(os.path.dirname(__file__))

        # data
        self.data_small_path = self.parent_dir + config.get("data", "data_small_path")
        self.data_large_path = self.parent_dir + config.get("data", "data_large_path")
        self.data_rest_path = self.parent_dir + config.get("data", "data_rest_path")
        self.data_finaltest_path = self.parent_dir + config.get("data", "data_finaltest_path")
        self.train_small_path = self.data_small_path + "data_train.json"
        self.test_small_path = self.data_small_path + "data_test.json"
        self.valid_small_path = self.data_small_path + "data_valid.json"
        self.train_large_path = self.data_large_path + "train.json"
        self.test_large_path = self.data_large_path + "test.json"

        # meta
        self.meta_path = self.parent_dir + config.get("data", "meta_path")
        self.meta_accu_path = self.meta_path + "accu.txt"
        self.meta_accu_frequent_small_path = self.meta_path + "accu_frequent_small.txt"
        self.meta_accu_frequent_large_path = self.meta_path + "accu_frequent_large.txt"
        self.meta_law_path = self.meta_path + "law.txt"
        self.meta_law_frequent_small_path = self.meta_path + "law_frequent_small.txt"
        self.meta_law_frequent_large_path = self.meta_path + "law_frequent_large.txt"
        self.meta_imprison_path = self.meta_path + "imprison.txt"
        self.meta_penallaw_path = self.meta_path + "penalLaw.txt"

        # log
        self.log_dir = self.parent_dir + config.get("log", "log_dir")
        self.accu_diff_log_path = self.log_dir + "accu_extracted_diff.txt"
        self.law_diff_log_path = self.log_dir + "law_extracted_diff.txt"

        # demo
        self.demo_path = self.parent_dir + config.get("demo", "demo_path")
        self.demo_statistics_path = self.demo_path + "statistics.txt"
        self.demo_distribution_accu_path = self.demo_path + "distribution_accu.txt"
        self.demo_distribution_article_path = self.demo_path + "distribution_article.txt"
        self.demo_figure_path = self.parent_dir + config.get("demo", "demo_figure_path")

        # preprocess
        self.preprocess_path = self.parent_dir + config.get("preprocess", "preprocess_path")
        self.stop_word_path = self.preprocess_path + "stop_word.txt"
        self.article_dict_path = self.preprocess_path + "article_dict.json"
        self.article_seg_path = self.preprocess_path + "article_seg.json"
        self.article_seg_small_path = self.preprocess_path + "article_seg_small.json"
        self.article_seg_large_path = self.preprocess_path + "article_seg_large.json"
        self.legal_basis_seg_path = self.preprocess_path + "legal_basis_seg.json"
        self.criminal_basis_seg_path = self.preprocess_path + "criminal_basis_seg.json"
        self.penalty_basis_seg_path = self.preprocess_path + "penalty_basis_seg.json"
        self.train_small_seg_path = self.preprocess_path + "train_small_seg.json"
        self.test_small_seg_path = self.preprocess_path + "test_small_seg.json"
        self.valid_small_seg_path = self.preprocess_path + "valid_small_seg.json"

        self.train_small_w2v_path = self.preprocess_path + "train_small_w2v.json"
        self.test_small_w2v_path = self.preprocess_path + "test_small_w2v.json"
        self.valid_small_w2v_path = self.preprocess_path + "valid_small_w2v.json"
        self.article_w2v_path = self.preprocess_path + "article_w2v.json"
        self.legal_basis_w2v_path = self.preprocess_path + "legal_basis_w2v.json"
        self.criminal_basis_w2v_path = self.preprocess_path + "criminal_basis_w2v.json"
        self.penalty_basis_w2v_path = self.preprocess_path + "penalty_basis_w2v.json"

        self.w2v_model_path = self.preprocess_path + "word2vec.model"
        self.embed_matrix_path = self.preprocess_path + "embed_matrix.json"

        # model
        self.model_path = self.parent_dir + config.get("train", "model")
