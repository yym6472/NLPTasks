## 数据预处理

1. 将原始文件`1998-01-105-带音.txt`编码修改为`UTF-8`编码（原来是`GB2313`），并将换行符修改为`LF`（原来是`LRLF`），保存到`data/source.txt`中
2. 将`source.txt`中19980101的数据取出构成一个稍小的数据集`source_small.txt`，用来测试数据集(`CWSDatasetReader`)能否正确读取数据
3. 将`source.txt`随机划分，并且按一定比例（默认为 8 : 1 : 1）划分为训练、验证、测试集
4. 将数据集文件的每一行，进行分析转化，作为一个样本（代码见`assets.py`中的`CWSDatasetReader._parse_line()`方法），注意其中用方括号`[...]`括起来的实体将被作为整个词
5. 根据数据集中的分词结果，使用 (B, M, E, S) 对每个中文字符打上标签，一个示例如下：
   ```
   原始句子：      一年有三百六十五天
   分词句子：      一年  有  三百六十五  天
   标签化后：      B E   S   B M M M E   S
   ```

## 使用的模型结构

Embedding层（256维） + 双向LSTM（隐层状态128维） + 单层全连接网络（输出维度：4）

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
![预测结果示例](https://i.loli.net/2019/07/11/5d271d6bd838855723.png)

## 评价指标和结果

### 评价指标
评价指标采用conll使用的，基于Span的F1值评价方法，见[Conlleval](https://www.clips.uantwerpen.be/conll2003/ner/bin/conlleval)。

### 结果
| 训练集F1 | 验证集F1 |
| --- | --- |
| 0.9679108556217378 | 0.9225928541677485 |
