---
layout: post
title: ELMO、BERT、ERINE、GPT的李宏毅视频学习笔记
categories: Meachine-Learning
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