---
layout: post
title: ELMO、BERT、ERINE、GPT的李宏毅视频学习笔记
categories: MeachineLearning
description: ELMO、BERT、ERINE、GPT的李宏毅视频学习笔记
keywords: ELMO,BERT,ERINE,GPT
---

## ELMO

ELMO是通过基于RNN来预测词向量的，如下图所示，对于“潮水退了就知道谁没穿裤子”这句话里面的“潮水”这个词，通过正向RNN和逆向RNN都会产生一个词向量，然后把这两个词向量进行加权得到最后的词向量。其中加权的权重参数是从下游任务里面学习到的。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/ELMO_01.png)