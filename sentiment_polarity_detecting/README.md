## 数据预处理

1. 将数据文件`rt-polaritydata/`中的文件编码修改为`UTF-8`编码（原来是`Windows 1252`）
2. 读取`rt-polarity.pos`和`rt-polarity.neg`，将其随机打乱，并且按一定比例（默认为 8 : 1 : 1）划分为训练、验证、测试集。其中句子和标签之间用"\t"分割，正样本标签为"pos"，负样本标签为"neg"
3. 将数据集文件的每一行，进行分析转化，作为一个样本。一个样本包含token组成的list和一个标签（"pos"或"neg"）

## 使用的模型结构

### (1) 未使用单词文件`subjclueslen1-HLTEMNLP05.tff`
Embedding层（256维） + Self Attention + 双向LSTM（隐层状态128维） + 单层全连接网络（输出维度：2）

### (2) 使用了单词文件`subjclueslen1-HLTEMNLP05.tff`
根据该文件，对原始句子中的每一个token都打上标签，作为token序列输入。

标签分为5种：
- strong_pos: 该单词出现在文件中，并且type=strongsubj，priorpolarity=positive
- weak_pos: 该单词出现在文件中，并且type=weaksubj，priorpolarity=positive
- neutral: 该单词**未出现**在文件中
- weak_neg: 该单词出现在文件中，并且type=weaksubj，priorpolarity=negative
- strong_neg: 该单词出现在文件中，并且type=strongsubj，priorpolarity=negative
将这五个标签经过一个单独的Embedding权重矩阵（32维），再与原始的Embedding层（224维）拼接，最后构成256维的总的Embedding层

模型其余部分与`(1)`相同。

## 如何运行

### 所需依赖环境
- python: 3.6.5
- numpy: 1.16.4
- pytorch: 1.1.0
- allennlp: 0.8.4

### 训练
运行：
```
python3 train.py
```

### 预测
运行：
```
python3 predict.py
```

预测结果示例：
![预测结果示例](https://i.loli.net/2019/07/15/5d2c3372a241263339.png)

## 评价指标和结果

### 评价指标
评价指标采用准确率（accuracy）

### 结果

#### 模型(1)
| 训练集accuracy | 验证集accuracy |
| --- | --- |
| 0.974909133544378 | 0.7636022514071295 |

#### 模型(2)
| 训练集accuracy | 验证集accuracy |
| --- | --- |
| 0.9234376831984993 | 0.774859287054409 |
