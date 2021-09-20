---
layout: post
title: ELMO、BERT、ERINE、GPT的李宏毅视频学习笔记
categories: DL
description: ELMO、BERT、ERINE、GPT的李宏毅视频学习笔记
keywords: ELMO,BERT,ERINE,GPT
---
## 背景

机器是如何理解我们的文字的呢？最早的技术是1-of-N encoding，把每个词汇表示成一个向量，每一个向量只有一个地方为1，其他地方为0。但是这么做词汇词汇之间的<u>关联</u>没有考虑，因为不同词之间的距离都是一样的。

所以，接下来有了word class的概念，举例说dog、cat和bird都是动物，它们应该是同类。但是动物之间也是有区别的，如dog和cat是<u>哺乳类</u>动物，和鸟类还是有区别的。

后来有了更进阶的想法，称作word embedding，我们用一个向量表示一个单词，相近的词汇距离较近，如cat和dog。那word embedding怎么训练呢？比较熟知的就是word2vec方法。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/EBEG_01.png)

但是，同一个词可能有不同的意思，如下图的bank，前两个指<u>银行</u>，后两个指河堤：

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/EBEG_02.png)

尽管有不同的意思，但使用**传统的word embedding的方法，相同的单词都会对应同样的embedding**。但我们希望针对不同的意思bank，可以给出不同的embedding表示。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/EBEG_03.png)

根据上下文语境的不同，同一个单词bank我们希望得到不同的embedding，如果bank的意思是银行，我们期望它们之间embedding能够相近，同时能够与何地意思的bank相距较远。

基于这个思想，首先有了ELMO。
## 一、ELMO

ELMO是通过基于RNN来预测词向量的，如下图所示，对于“潮水退了就知道谁没穿裤子”这句话里面的“潮水”这个词，通过正向RNN和逆向RNN都会产生一个词向量，然后把这两个词向量进行加权得到最后的词向量。其中加权的权重参数是从下游任务里面学习到的。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/ELMO_01.png)

当然，我们可以搞很多层：

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/ELMO_02.png)

这么多层的RNN，内部每一层输出都是单词的一个表示，那我们取哪一层的输出来代表单词的embedding呢？ELMO的做法就是我全都要。

在ELMO中，一个单词会得到多个embedding，对不同的embedding进行加权求和，可以得到最后的embedding用于下游任务。要说明一个这里的embedding个数，下图中只画了两层RNN输出的hidden state，其实输入到RNN的原始embedding也是需要的，所以你会看到说右下角的图片中，包含了三个embedding。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/ELMO_03.png)

但不同的权重是基于下游任务<u>学习</u>出来的，上图中右下角给了5个不同的任务，其得到的embedding权重各不相同。

## 二、BERT

Bert是Bidirectional Encoder Representations from Transformers的缩写，它也是芝麻街的人物之一。Transformer中的Encoder就是Bert预训练的架构。**李宏毅老师特别提示：如果是中文的话，可以把字作为单位，而不是词**。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/BERT_01.png)


BERT 的训练过程有两种方式，一种是Masked LM，另外一种是预测下一句话的方法。

### 1. Masked LM

文献中给出了两种训练的方法，第一个称为Masked LM，做法是随机把一些单词变为Mask，让模型去猜测盖住的地方是什么单词。假设输入里面的第二个词汇是被盖住的，把其对应的embedding输入到一个多分类模型中，来预测被盖住的单词。

Masked LM是通过随机遮蔽15%的词，然后对这15%的词来进行预测。预测的时候将MASK位置产生的向量通过一个线性多分类器来得到是哪个词。如果两个词填在同一个地方没有违和感，那么这两个词就有相似的embedding。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/BERT_02.png)

### 2.Next Sentence Prediction

另一种方法是预测下一个句子，这里，先把两句话连起来，中间加一个[SEP]作为两个句子的分隔符。而在两个句子的开头，放一个[CLS]标志符，将其得到的embedding输入到二分类的模型，输出两个句子是不是接在一起的。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/BERT_03.png)

这个是通过输入两句话（用SEP分隔），来预测输出这两句话是不是连续的。通过这样的方式来学习语言模型。通常来说，方法1和方法2是同时被使用的。

在ELMO中，训练好的embedding是不会参与下游训练的，下游任务会训练不同embedding对应的权重，但在Bert中，Bert是和下游任务一起训练的：

如果是分类任务，在句子前面加一个标志，将其经过Bert得到的embedding输出到二分类模型中，得到分类结果。二分类模型从头开始学，而Bert在预训练的基础上进行微调（fine-tuning）。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/BERT_04.png)

文中还有很多其他的应用，如单词分类：

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/BERT_05.png)

如自然语言推理任务，给定一个前提 / 假设，得到推论是否正确：

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/BERT_06.png)


最后一个例子是抽取式QA，抽取式的意思是输入一个原文和问题，输出两个整数start和end，代表答案在原文中的起始位置和结束位置，两个位置中间的结果就是答案。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/BERT_07.png)

具体怎么解决刚才的QA问题呢？把问题 - 分隔符 - 原文输入到BERT中，每一个单词输出一个黄颜色的embedding，这里还需要学习两个（一个橙色一个蓝色）的向量，这两个向量分别与原文中每个单词对应的embedding进行点乘，经过softmax之后得到输出最高的位置。正常情况下start <= end，但如果start > end的话，说明是矛盾的case，此题无解。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/BERT_08.png)

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/BERT_09.png)

扩展：Bert 一出来就在各项比赛中崭露头角

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/BERT_10.png)



<!-- BERT不只是可以用来产生词向量供下游服务，同时也可以直接用来做很多任务，比如

```
(a) 句子关系判断（句对匹配）
(b) 文本分类
(c) 机器问答
(d) 序列标注，如命名实体识别(NER)等
``` -->
<!-- 
![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/BERT_03.png) -->

## 三、ERINE

这里李宏毅老师还举例了百度提出的ERNIE，ERNIE也是芝麻街的人物，而且还是Bert的好朋友，这里没有细讲，感兴趣的话大家可以看下原文。

ERINE和BERT不同的地方在于，BERT是**随机遮蔽的字**，而ERINE是**随机遮蔽的词**。这样能更好地捕捉到中文里面词与词的边界关系。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/ERINE_01.png)


Bert学到了什么呢？可以看下下面两个文献（给大伙贴出来：<https://arxiv.org/abs/1905.05950> 和<https://openreview.net/pdf?id=SJzSgnRcKX>

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/ERIGE_02.png)


## 四、GPT

GPT是Generative Pre-Training 的简称，但GPT不是芝麻街的人物。GPT-2的模型非常巨大，它其实是Transformer的Decoder。


![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/GPT_01.png)

GPT-2是Transformer的Decoder部分，输入一个句子中的上一个词，我们希望模型可以得到句子中的下一个词。

GPT是生成式的预训练语言模型，其内部是通过self-attention实现的，“退了”这个词和前面的词作self-attention然后产生“就”。由于self-attention和词的远近是**没有关系**的，因此不需要逆向再操作一遍。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/GPT_02.png)

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/GPT_03.png)


由于GPT-2的模型非常巨大，它在很多任务上都达到了惊人的结果，甚至可以做到zero-shot learning（简单来说就是模型的迁移能力非常好），如阅读理解任务，不需要任何阅读理解的训练集，就可以得到很好的结果。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/GPT_04.png)

GPT-2也可以自己进行写作，写得还是不错的！

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/GPT_05.png)