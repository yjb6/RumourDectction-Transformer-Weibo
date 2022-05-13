import torch
import torch.nn as nn
import torch.nn.functional as F
from rumoure_dection.codes.Transformer import Transformer
import sys

sys.path.append('../SubModules/')

from rumoure_dection.codes.SubModules import WordModule
from rumoure_dection.codes.SubModules import PostModule

# from SubModules import WordModule
# from SubModules import PostModule

__author__ = "Serena Khoo"


class HierarchicalTransformer(nn.Module):

  @staticmethod
  def init_weights(layer):
    if type(layer) == nn.Linear:
      nn.init.xavier_normal_(layer.weight)

  def __init__(self, config):

    super(HierarchicalTransformer, self).__init__()

    # <----------- Config ----------->
    self.config = config

    # <----------- Both word and post modules ----------->
    self.word_module = WordModule.WordModule(config)
    self.post_module = PostModule.PostModule(config)

  def forward(self, X, word_pos, time_delay, structure, attention_mask_word=None, attention_mask_post=None,
              return_attention=False):

    # <----------- Getting the dimensions ----------->
    batch_size, num_posts, num_words, emb_dim = X.shape

    # <----------- Passing through word module ----------->
    X_word, self_atten_weights_dict_word = self.word_module(X, word_pos, attention_mask=attention_mask_word)

    # <----------- Passing through post module ----------->
    output, self_atten_output_post, self_atten_weights_dict_post = self.post_module(X_word, time_delay, batch_size,
                                                                                    num_posts, emb_dim,
                                                                                    structure=structure,
                                                                                    attention_mask=attention_mask_post)
    '''获取attention值最大的post下标:'''

    res_dict={}
    importent_attention = self_atten_output_post[0][0].tolist()  # b转为list，调用函数index(max())求最大值下标
    print(len(importent_attention))
    print(importent_attention[128])
    most_importent = importent_attention.index(max(importent_attention))
    print('v:', most_importent)
    for index, attention in self_atten_weights_dict_post.items():
        attention=attention.squeeze(0)
        for val in attention:
            val=val[most_importent].tolist()
            most_relevent=val.index(max(val))
            if most_relevent in res_dict.keys():
                res_dict[most_relevent]+=1
            else:
                res_dict[most_relevent]=1
    res_dict = sorted(res_dict.items(), key=lambda e: e[1], reverse=True)
    print(res_dict)
    self_atten_weights_word=self_atten_weights_dict_word[1]
    self_atten_weights_word=self_atten_weights_word.squeeze(0)
    print(self_atten_weights_word.size())
    for i in self_atten_weights_word:
        print(i[80][0:8,0:8])
    # <--------- Clear the memory -------------->
    torch.cuda.empty_cache()

    if return_attention:
      return output, self_atten_output_post, self_atten_weights_dict_word, self_atten_weights_dict_post

    # <-------- Delete the attention weights if not returning it ---------->
    del self_atten_weights_dict_word
    del self_atten_weights_dict_post
    del self_atten_output_post
    torch.cuda.empty_cache()

    return output

def loadmodel():
    sys.path.append('./codes/Transformer/')
    dic=torch.load('../../bestmodel/best_model_accuracy_test.pt')
    print(dic)

    hierarchical_transformer = HierarchicalTransformer
    print(hierarchical_transformer)
    a=hierarchical_transformer.load_state_dict(state_dict=dic)


# loadmodel()