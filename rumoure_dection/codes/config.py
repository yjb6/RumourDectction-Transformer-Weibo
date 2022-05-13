import os
import json

__author__ = "Serena Khoo, yjb"


class Config():

  def __init__(self):
    # GPU settings
    self.gpu = True

    # GPU settings
    self.gpu_idx = [1]  # Do note that number of GPU < batch_size
    self.main_gpu = [1]

    # Training
    self.num_epoch = 300
    self.batch_size = 1
    self.batch_size_test = 1
    self.num_classes = 2

    # Word Embedding settings
    self.glove_directory = r"rumoure_dection\data"
    self.glove_file = "sgns.weibo.bigram.txt"
    self.vector_path = ""
    self.vocab_path = ""
    self.max_vocab = 50000
    self.emb_dim = 300
    self.num_structure_index = 5

    # Versioning of methods
    self.include_key_structure = True
    self.include_val_structure = True
    self.word_module_version = 4  # {0: max_pooling, 1: average_pooling, 2: max_pooling_w_attention, 3: average_pooling_w_attention, 4: attention}
    self.post_module_version = 3  # {0: average_pooling, 1: condense_into_fix_vector, 2: first_vector, 3: attention}

    # Word Embedding training
    self.train_word_emb = False
    self.train_pos_emb = False

    # Time interval embedding
    self.size = 100  # Number of bins
    self.interval = 10  # Time lapse for each interval
    self.include_time_interval = True

    # Word padding settings
    self.max_length = 18  # Pad the content to 35 words
    self.max_tweets = 512  # Based on the data, 339 is the largest for twitter15 and 270 is the largest for twitter16

    # data paths
    self.extension = "json"
    self.data_set = "train_pheme_test_sg"
    self.data_folder = "data/weibo_small/"
    self.train_file_path = "train.json"
    self.test_1_file_path = "4753064223048549_test.json"
    self.test_2_file_path = "4753064223048549.json"
    self.data_folder_test=r"rumoure_dection\data\test"
    self.test_path=''

    # JSON keys --> All the keys in the JSON need to be present
    self.keys_order = {"post_id": "id_", "label": "label", "content": "tweets", "time_delay": "time_delay",
                       "structure": "structure"}

    # Logs paths
    self.dataset_name = "full_pheme"
    self.experiment_name = "HiT_0"
    self.log_folder = "../logs/"
    self.record_file = "record"
    self.interval = 100

    # Model parameters settings
    self.d_model = 300
    self.dropout_rate = 0.3

    # <------------------------ WORD LEVEL ------------------------>
    self.ff_word = True
    self.num_emb_layers_word = 2  # Model parameters settings (To encode query, key and val)
    self.n_mha_layers_word = 2  # Number of Multihead Attention layers
    self.n_head_word = 2  # Number of MHA heads

    # <------------------------ POST LEVEL ------------------------>
    self.ff_post = True
    self.num_emb_layers = 2  # Model parameters settings (To encode query, key and val)
    self.n_mha_layers = 12  # Number of Multihead Attention layers
    self.n_head = 2  # Number of MHA heads

    # Model parameters settings (For feedforward network)
    self.d_feed_forward = 600

    # Learning rate
    self.learning_rate = 0.01
    self.beta_1 = 0.90
    self.beta_2 = 0.98
    self.n_warmup_steps = 6000
    self.vary_lr = True

    #best model path
    self.besr_model_floder=r"rumoure_dection\bestmodel"
    self.model_path=r'\best_model_accuracy_test.pt'
    self.time_delay_encoder_path=r'\best_model_time_delay_encoder_accuracy_test.pt'
    self.word_encoder_path=r'\best_model_word_encoder_accuracy_test.pt'
    self.word_pos_encoder_path=r'\best_model_word_pos_encoder_accuracy_test.pt'

  def __repr__(self):
    return str(vars(self))


config = Config()
