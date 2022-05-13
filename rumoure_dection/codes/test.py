import os
import sys
import torch

print(torch.cuda.is_available())
from rumoure_dection.codes.config import config

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in vars(config)["gpu_idx"])
# import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torch

import numpy as np

from tqdm import tqdm
from datetime import datetime

# <------- GPU optimization code -------->
from rumoure_dection.codes.utils.parallel import DataParallelModel, DataParallelCriterion

# <------- Self defined classes -------->
from rumoure_dection.codes.DataLoader import DataLoader
import sys

sys.path.append('./Transformer/')
sys.path.append('Layers/')
sys.path.append('./SubModules/')
sys.path.append('./Encoder/')
from rumoure_dection.codes.Transformer import HierarchicalTransformer
from rumoure_dection.codes.Encoder import WordEncoder
from rumoure_dection.codes.Encoder import PositionEncoder
from rumoure_dection.codes.Optimizer import Optimizer
import pynvml
from rumoure_dection.codes.utils.utils import *

__author__ = "Serena Khoo"

def gpuinformation():

  pynvml.nvmlInit()
  # 这里的0是GPU id
  handle = pynvml.nvmlDeviceGetHandleByIndex(0)
  meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
  print("已使用的GPU：",meminfo.used/1024/1024)
  print("空余的GPU：",meminfo.free/1024/1024)
gpuinformation()
class Test():

  def __init__(self, dataloader, hierarchical_transformer, config, i):

    self.iter = i
    self.config = config
    self.cpu = torch.device("cpu")
    self.multi_gpu = len(self.config.gpu_idx) > 1

    self.dataloader = dataloader
    self.word_encoder = WordEncoder.WordEncoder(config, self.dataloader)
    self.word_pos_encoder = PositionEncoder.PositionEncoder(config, self.config.max_length)
    self.time_delay_encoder = PositionEncoder.PositionEncoder(config, self.config.size)

    # <----------- Check for GPU setting ----------->
    if self.config.gpu:

      self.hierarchical_transformer = DataParallelModel(hierarchical_transformer.cuda())
      self.criterion = DataParallelCriterion(nn.NLLLoss())

    else:
      self.hierarchical_transformer = hierarchical_transformer
      self.criterion = nn.NLLLoss()

    # <----------- load best model ----------->
    self.best_model_floder=self.config.besr_model_floder
    self.model_path=self.best_model_floder+self.config.model_path
    self.time_delay_encoder_path = self.best_model_floder+self.config.time_delay_encoder_path
    self.word_encoder_path = self.best_model_floder+self.config.word_encoder_path
    self.word_pos_encoder_path = self.best_model_floder+self.config.word_pos_encoder_path

    self.hierarchical_transformer.load_state_dict(state_dict=torch.load(self.model_path))
    # a=torch.load(self.word_encoder_path)
    # print(a.keys())
    # print(self.word_encoder)
    # self.word_encoder.load_state_dict(state_dict=a)
    self.word_pos_encoder.load_state_dict(state_dict=torch.load(self.word_pos_encoder_path))
    self.time_delay_encoder.load_state_dict(state_dict=torch.load(self.time_delay_encoder_path))

    self.adam_optimizer = optim.Adam(self.hierarchical_transformer.parameters(), np.power(self.config.d_model, - 0.5),
                                     betas=(self.config.beta_1, self.config.beta_2))
    self.optimizer = Optimizer.Optimizer(self.config, self.adam_optimizer)
  def test(self):
      self.hierarchical_transformer.eval()  # Make sure that it is on eval mode first,让BN和drop out用训练好的值
      predicted_y_lst=[]
      with torch.no_grad():

          for X,y,  word_pos, time_delay, structure, attention_mask_word, attention_mask_post in self.dataloader.get_data(
                  "test"):

              # <-------- Casting as a variable --------->
              X = Variable(X)
              print("X",X.size())
              word_pos = Variable(word_pos)
              time_delay = Variable(time_delay)
              structure = Variable(structure)
              attention_mask_word = Variable(attention_mask_word)
              attention_mask_post = Variable(attention_mask_post)

              # <-------- Encode content -------------->
              X = self.word_encoder(X)
              word_pos = self.word_pos_encoder(word_pos)
              time_delay = self.time_delay_encoder(time_delay)

              # <-------- Move to GPU -------------->
              if self.config.gpu:
                  X = X.cuda()
                  word_pos = word_pos.cuda()
                  time_delay = time_delay.cuda()
                  print("timedelay",time_delay.size())
                  structure = structure.cuda()
                  attention_mask_word = attention_mask_word.cuda()
                  attention_mask_post = attention_mask_post.cuda()

              # <--------- Getting the predictions --------->
              predicted_y = self.hierarchical_transformer(X, word_pos, time_delay, structure,
                                                          attention_mask_word=attention_mask_word,
                                                          attention_mask_post=attention_mask_post)

              # predicted_y, self_atten_output_post, self_atten_weights_dict_word, self_atten_weights_dict_post = self.hierarchical_transformer(X, word_pos, time_delay)
              # self_atten_weights_dict_word = merge_attention_dict(self_atten_weights_dict_word, self.config, "word")
              # self_atten_weights_dict_post = merge_attention_dict(self_atten_weights_dict_post, self.config, "post")

              if self.multi_gpu:
                  predicted_y = torch.cat(list(predicted_y), dim=0)

              # <------- to np array ------->

              predicted_y = predicted_y.cpu().numpy()


              # print("test", predicted_y)

              # <------- Appending it to the master list ------->
              predicted_y_lst.extend(predicted_y)
              # <--------- Free up the GPU -------------->
              del X
              del y
              del predicted_y
              del word_pos
              del time_delay
              del structure

          # <------- Get scores ------->
          predicted_y_lst = np.array(predicted_y_lst)
          predicted_y_lst = get_labels(predicted_y_lst)
          return predicted_y_lst[0]



def main(wbid):
    # config.test_path=config.data_folder_test+r'/'+str(wbid)+'_test.json'
    config.test_path =  str(wbid) + '_test.json'
    loader = DataLoader.DataLoader(config, 0, type_='test')
    hierarchical_transformer = HierarchicalTransformer.HierarchicalTransformer(config)
    tester = Test(loader, hierarchical_transformer, config, 0)
    res=tester.test()

    del loader
    del hierarchical_transformer
    del tester
    return res
if __name__ == "__main__":
    config.test_path = config.data_folder_test + r'/' + str(4713907509792628) + '_test'
    loader = DataLoader.DataLoader(config, 0,type_='test')
    hierarchical_transformer = HierarchicalTransformer.HierarchicalTransformer(config)
    tester = Test(loader, hierarchical_transformer, config, 0)
    tester.test()

    del loader
    del hierarchical_transformer
    del tester
