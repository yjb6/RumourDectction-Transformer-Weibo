# RumourDectction-Transformer-Weibo
智探云平台源码，基于Transformer的微博谣言检测，包含前后端开发实现  
基于Python3.8+Pytorchh==1.10.2+cuda==11.1+torchtext==0.11.2+Django  
模型训练搭建基于[Interpretable Rumor Detection in Microblogs by Attending to User Interactions](https://github.com/serenaklm/rumor_detection)  
数据来源于[Github公开数据集](https://github.com/thunlp/Chinese_Rumor_Dataset)  
[权重以及预训练模型、测试数据](https://www.dropbox.com/s/6maphepdv5sxdin/rumour_detection_data.zip?dl=0)  
启动项目:python manage.py runserver
### Weibo rumour detection
* Using deep learning models to detect a large number of false information on Internet social media can detect the authenticity of false information in the early stage of dissemination,And block the dissemination of false information, reduce the harm of false information to the society.
* Read papers in related fields and use the Pytorch framework to reproduce model codes for Transformer, LSTM, CNN and other models.
* Data preprocessing is performed on [Github's public Weibo Chinese rumor data set](https://github.com/thunlp/Chinese_Rumor_Dataset), and features such as text information, time information, and structural information of events are constructed.
* Complete the model training, and deeply explore the self-attention mechanism to visualize the feasibility of the model.
<div align=center>
<img src="https://user-images.githubusercontent.com/71499139/166968881-6b283782-8ca0-4f0b-8d46-e345e0819e32.png"/>
 <img src="https://user-images.githubusercontent.com/71499139/166968653-2eb81b1a-c574-4b57-9a3e-a5d6dbfea492.png"/>
</div>
<div align=center>
<img src="https://user-images.githubusercontent.com/71499139/166969059-ceaf0e6c-7489-4cff-bc73-ac622dc14ad5.png"/>
 <img src="https://user-images.githubusercontent.com/71499139/166969246-dbf56a64-d436-45b1-8deb-035917d53b16.png"/>
</div>
* Complete the front-end and back-end development of the system. Realize the text and structure information of the microblog propagation path constructed by the crawler according to the input microblog ID. and judged by the model and visualize the analysis results.
<div align=center>
<img src="https://user-images.githubusercontent.com/71499139/166971642-a56289c1-38a3-4811-9bbb-923dd9bf32cc.png"/>
</div>
<div align=center>
<img src="https://user-images.githubusercontent.com/71499139/166972002-fbd500dc-552f-4b8a-97ef-ab15fbe85103.png"/>
</div>
