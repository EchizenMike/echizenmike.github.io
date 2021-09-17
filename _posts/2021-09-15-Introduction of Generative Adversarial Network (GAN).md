---
layout: post
title: Introduction of Generative Adversarial Network (GAN)
categories: DL
description: Introduction of Generative Adversarial Network (GAN)
keywords: GAN
---

有很多种不同类型的GAN,可以在这里查看<https://github.com/hindupuravinash/the-gan-zoo>

## Generation

对于Image Generation，要实现的是输入一个vector，输出一个image；而对于Sentence Generation，则实现的是输入一个vector，输出为以恶搞sentence。那么GAN就是用来实现这个中间的NN Generator。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_01.png)

## Basic Idea of GAN
### Generator vs Discriminator

这个Generator可以是一个<u>神经网络</u>，也可以是一个<u>函数f</u>

输入的vector表示我们要generate图像的某种特征，比如vector的第一维如果代表头发的长度，我们现在把这个值设得很大，那么就会generate一章头发很长的图像；如果我们改变了vector倒数第二(头发为蓝色)的值，可以发现generate的新图头发变蓝了，由于此时只改变了头发的颜色，其它特征都是类似的，所以只有头发的颜色发生了变化。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_02.png)

Discriminator可以是一个神经网络，也可以是一个函数f。输入是一张image，输出为scalar，数值越大，表示这个image越接近真实图像。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_03.png)

在大自然中，某种鸟类以枯叶蝶为食。枯叶蝶必须不断地进化，使其看起来越来越像一个枯叶；它的天敌也在不断地进化，如果枯叶蝶看起来并不像叶子，那么它就会被捕食。其中枯叶蝶就相当于一个generator，天敌就相当于discriminator。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_04.png)

* 第一代generator不知道怎样产生二次元的头像，所以会产生一些看起来很像是杂讯的图像，再把这个图像输入discriminator，来判断这是不是一张真实的图像，第一代discriminator可以根据图像是否有颜色，来正确分辨真实图像和生成的图像。
* 那么第二代的generator的目标就是想办法<u>骗过</u>第一代的discriminator，生成<u>有颜色</u>的图像，随之discriminator也会发生进化，学习了真实图像和生成图像之间的差异（真实图像是有嘴的）
* 第三代的generator生成的图像可以骗过第二代的discriminator，然后discriminator也会继续进化，...
* generator 和 discriminator 都会不断地进化，因此generator会产生越来越像真实图片的图像。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_05.png)

这个过程可以看作是一个对抗的过程，也可以用一个较为和平的比喻来进行解释。generator相当于一个学生，discriminator相当于一个老师，学生并不知道真实的图像长什么样，但老师看过很多真实的图像，就知道真实的图像长的什么样子。

第一代的generator相当于一年级学生，重复着上述的过程。学生会画的越来越好，老师也会越来越严格，那么学生最后就可以画出很像二次元的头像了。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_06.png)


### Algorithm
首先需要随机初始化G和D参数，

**Step 1：**我们需要先调整D的参数，就必须先把G的参数固定。首先把随机产生的vector输入G(fix),生成新的图像后从database中sampled出来的图像进行比较。要实现D如果输入真实图像，就会得高分，与1越接近越好，如果输入生成的图像，就会得低分，与0越接近越好。有了这个标准，我们就可以来训练这个discriminator D;
![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_07.png)
**Step 2:**训练好discriminator D之后，我们就可以fix D，来调整generator G。一个vector输入第一代G之后，会生成一张图像，再输入D(fix)，就可以得到一个很低的分数(0.13)。那么G训练的目标就是使生成图片可以“骗”过D，即生成的图片使D给出一个比较高的分数。由于D看过真实的图像，如果给出了很高的分数，就可以说明G生成的图像和真实图像是很接近的。

在真实的代码实现中，我们通常会把generator和discriminator当成是一个大的network，其中generator的输出就可以看作是一个hidden layer，discriminator所在的参数是fix的，不用调整，只需要根据整个网络的输出来调整generator的参数。

由于我们希望使discriminator的输出分数值越大越好，因此这里使用了梯度上升算法**Gradient Ascent**，也就是梯度下降法前面多乘了一个负号。
![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_08.png)

现在来叙述一些总的算法流程，![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/theta_d.png),![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/theta_g.png) 分别表示discriminator和generator的参数。

Learning D：首先从数据库中取出m个真实图片，再根据一个分布随机产生m个vector作为输入![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_08_1.png),此时fix G的参数，得到G生成的图像![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_08_2.png),再输入discriminator D，不断调整![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/theta_d.png),使得得到的分数越大越好，公式：

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_10.png)

其中![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/D_xi.png)表示真实图像得到的分数，D的目标就是使真实图像获得的分数越大越好；而![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/D_xi_h.png)表示G生成的图像所得到的分数，应该越小越好，所以前面加了负号。为了方便求梯度，在式子前面加入了log，求出梯度![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/Delta_d.png),再更新![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/theta_d.png)的值，

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_11.png)

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_09.png)

Learning G：把D训练好之后，我们就可以fix D，来训练generator G的参数。首先也需要从分布中随机生成一些噪声z，再输入G，G(zi)再输入D，得到相对应的分数，G的目标是想办法骗过D，不断调整参数θg，使生成的图像所得到的分数越高越好，


$$\widetilde{V} = \frac{1}{m}\sum_1^m logD(G(z^i))$$

求出梯度$\Delta\widetilde{V}(\theta_g)$,再更新$\theta_g$的值，

$$\theta_g\leftarrow \theta_g+ \eta\Delta\widetilde(\theta_g)$$

在每个iteration里，都会进行这个步骤，先训练discriminator，再训练generator；这两个步骤会反复进行。

### Anime Face Generation
结果展示
![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_12_1.png)
![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_12_2.png)
![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_12_3.png)
![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_12_4.png)

#### Structed learning
![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_13.png)
![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_14.png)
![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_15.png)

### Why Structured Learning Challenging?

One-shot/Zero-shot Learning，如果有的类别都没有范例，或者只有很少一部分的范例。

而structured learning是一种极端的One-shot learning，由于output为一个structure，比如一个句子，可能这些句子在training data中从来没出现过，那么如何学习去输出一个从来没看到的structure，machine必须学会去创造。
![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_16.png)

machine还必须学会如何去planing，有全局观；比如sentence generation中，如果只看第一句话，会认为是负面的，但如果你把整句话都看完，就会发现这整句话在表达一个正面的意思。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_17.png)

### Structured Learning Approach
structured learning有两套方法：
* Bottom up，机器在生成一个部件时，会先生成多个component，这种方法一个很大的问题就是容易失去大局观；

* Top down，产生一个完整的物件之后，再去从整体上看产生物件好不好。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_18.png)

把这两种方法结合起来就是Generator

### Can Generator learn by itself?

#### Generator
对于Generator，首先输入不同的vector，就可以输出不同的图片。如果我们现在输入1对应的vector，generator会生成一张image，目标是使image和真实的图像越接近越好，这个真实图像现在generator能看到，那么这不就和一般的supervised learning一模一样了吗？
![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_19.png)

那我们怎么知道输入的那些vector的数值呢？

我们可以用一个Encoder来表示，把image输入这个NN Encoder，就会输出对应的特征，把图像的特征用vector来表示即可。
![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_20.png)

#### Auto-encoder

Auto-encoder分为<u>encoder</u>和<u>decoder</u>。对于28x28图像，先用encoder使得输入的图像变成<u>code</u>,decoder把这个code再恢复成原来的图像，这两者会一起进行学习。

![image](https://raw.githubusercontent.com/EchizenMike/echizenmike.github.io/master/images/ml/dl/gan_21.png)

![image](https://github.com/EchizenMike/echizenmike.github.io/blob/master/images/ml/dl/ELMO_01.png)