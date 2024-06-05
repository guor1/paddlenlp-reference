# 写在前面
本项目是基于百度paddle框架，复现NLP领域的一些任务，仅供学习交流使用。对于每一个任务，我都会先用传统机器学习思路实现，然后再用transformers实现，加深对原理的理解。

# 为什么用paddle不用pytorch？
1. 因为那一堵墙，huggingface、pytorch、tensorflow很多资源下载很慢，还下载不下来。
2. 飞浆AI Studio提供了免费的算力资源，甚至还可以免费使用GPU，不过现在开始推广会员收费了。
3. 在语法方面，其实paddle和pytorch的语法差异很小，基本都是换了包名，确实有差异的读一下api文档也能快速迁移。

# 任务列表
1. [中文情感分析.ipynb](./classification/中文情感分析.ipynb)
2. [中文机器阅读理解.ipynb](./question_answering/中文机器阅读理解.ipynb)

# 文本分类
1. TextCNN
2. BiLSTM
3. FastText