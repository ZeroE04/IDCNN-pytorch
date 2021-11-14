# IDCNN-pytorch(中文)<English readme is below>
用pytorch实现的IDCNN，实测在精度和BiLSTM一致的情况下，速度大幅提升

### 前言
本工作初衷是提供一个**极其精简**的且可以直接跑起来代码，并对文本序列进行标注
1. 预处理：先把train.txt，valid.txt放到data/raw里面，修改config.py里面其路径，然后python preprocess.py，随后会在data/processed/里面生成准备好的训练数据
2. 训练：python train.py，随后会按照config.py里面设置的路径进行model保存；
3. 推理：python predict.py 今天北京的天气怎么样

#### 常见问题
 * torch==1.2.0
 * 原论文的IDCNN中间的CNN是并行的，我这里改成了串行四个block，详细看ner.py
 * 如果你要用预训练的embedding，可以在config.py内指定EMBEDDING_FILE路径，并把ner.py内的get_embedding函数中的nil参数改成False
 
#### 训练自己的多分类网络
 * 直接把train.txt和valid.txt换成你自己的就行了，仅仅是对字符级别进行分类(即序列标注)，所以不限制BIEOS还是BIO等。

# IDCNN-pytorch(English)
Pytorch implementation of IDCNN

### preface
The original intention of this work is to provide a **extremely compact** code of IDCNN-pytorch
1. process: put your train.txt/valid.txt in data/raw, and run "python preprocess.py", than the prepared training_data will generated in data/processed/
2. train: python train.py, model will save in /model, you can change the save_dir and model_name in config.py
3. inference: python predict.py 今天北京的天气怎么样
#### notes
 * torch==1.2.0
 * In the original of IDCNN, cnn blocks is parallel, and I changed it to four serial blocks here. See ner.py in detail.
 * If you want to use pre-trained embedding, specify the EMBEDDING_FILE path in config.py and set the nil parameter in the get_embedding function in ner.py to False

#### train your data
 * put your-self train.txt/valid.txt in /data/raw, there are no restrictions on BIEOS or BIO, etc.

