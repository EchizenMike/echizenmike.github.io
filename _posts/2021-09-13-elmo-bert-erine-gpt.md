---
layout: post
title: ELMO、BERT、ERINE、GPT的李宏毅视频学习笔记
categories: DL
description: ELMO、BERT、ERINE、GPT的李宏毅视频学习笔记
keywords: ELMO,BERT,ERINE,GPT
---

## 一、ELMO

ELMO是通过基于RNN来预测词向量的，如下图所示，对于“潮水退了就知道谁没穿裤子”这句话里面的“潮水”这个词，通过正向RNN和逆向RNN都会产生一个词向量，然后把这两个词向量进行加权得到最后的词向量。其中加权的权重参数是从下游任务里面学习到的。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/ELMO_01.png)

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/ELMO_02.png)

## 二、BERT

BERT 的训练过程有两种方式，一种是Masked LM，另外一种是预测下一句话的方法。

### 1. Masked LM

Masked LM是通过随机遮蔽15%的词，然后对这15%的词来进行预测。预测的时候将MASK位置产生的向量通过一个线性多分类器来得到是哪个词。如果两个词填在同一个地方没有违和感，那么这两个词就有相似的embedding。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/BERT_01.png)

### 2.Next Sentence Prediction

这个是通过输入两句话（用SEP分隔），来预测输出这两句话是不是连续的。通过这样的方式来学习语言模型。通常来说，方法1和方法2是同时被使用的。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/BERT_02.png)

BERT不只是可以用来产生词向量供下游服务，同时也可以直接用来做很多任务，比如

```
(a) 句子关系判断（句对匹配）
(b) 文本分类
(c) 机器问答
(d) 序列标注，如命名实体识别(NER)等
```

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/BERT_03.png)

## 三、ERINE

ERINE和BERT不同的地方在于，BERT是**随机遮蔽的字**，而ERINE是**随机遮蔽的词**。这样能更好地捕捉到中文里面词与词的边界关系。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/ERINE_01.png)

## 四、GPT

GPT是生成式的预训练语言模型，其内部是通过self-attention实现的，“退了”这个词和前面的词作self-attention然后产生“就”。由于self-attention和词的远近是**没有关系**的，因此不需要逆向再操作一遍。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/GPT_01.png)

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/GPT_02.png)
