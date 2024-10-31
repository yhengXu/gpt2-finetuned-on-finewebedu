# GPT-2 124M模型复现

本项目代码复现自https://github.com/karpathy/build-nanogpt，基于huggingface的gpt-2预训练权重，在FineWeb-Edu 10B数据集上进行微调。

## train_gpt2_finewebedu.py

模型的训练代码，包含模型定义、训练文本载入和训练过程。在训练过程中，每执行1000步，会保存该检查点上的模型参数。

为了加快训练、推理速度，此实现使用了bf16数据格式。训练方法上，使用了带预热的余弦衰减learning rate方法优化学习率、使用weight decay强制optimizer优化更多权重。考虑到个人GPU不能加载大batch数据的问题，使用了梯度累计方法，以串行方式模拟任意大小的batch。最后，为使用多个GPU的分布式训练方法进行了适配。

基于huggingface给出的GPT-2预训练权重，模型在AutoDL平台上使用4*4090GPU在FineWeb-Edu 10B数据集上进行了1个epoch的微调。

## fineweb.py

下载https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu数据集，使用GPT-2的tokenizer将其转换为numpy数组并分片存储。其中，第一个分片作为评估集，在训练过程中计算模型输出和标准输出的交叉熵。剩余分片作为训练集。

## log.txt

记录模型训练期间交叉熵的变化

## model_weight.pt

模型权重

## run_gpt2.py

加载模型权重进行推理。实现及如下：

给定Prefill单词序列

```
tokens = enc.encode("Hello, I'm a language model,")
```

模型返回num_return_sequences个扩写的句子，每句长度为max_length

```
num_return_sequences = 10
max_length = 30
```

形如：
```
> Hello, I'm a language model, and I know that you might like to add stuff to that, such as this:
> Hello, I'm a language model, a code in a computer program, if you have nothing other than this word "computer," that is exactly what
> Hello, I'm a language model, a language user. I need to make a machine-learning framework, but I'm not sure I can do
> Hello, I'm a language model, I really like how you can make it look complex.
Do you use any mathematical formulas for this article?
> Hello, I'm a language model, so I can give you a bit about the theory and the history of human language in this world. I've
> Hello, I'm a language model, I don't want to make any changes to what you write!
In this case, "Html"
> Hello, I'm a language model, or object oriented abstraction for modelling and building digital applications. I believe in an abstract learning model, but at the
> Hello, I'm a language model, not just a language. So what is it, and I find it fascinating, that in the word 'computer
> Hello, I'm a language model, a programming language. I've been going for several years with a lot of friends all about C++. I
> Hello, I'm a language model, but in the past I've always thought of it simply as a computer language developed by programmers.
```