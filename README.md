# text_summary_lab

这是ECNU 当代人工智能第四次实验。

## Setup

本项目基于pytho3.7实现， 使用如下版本的依赖：

- torch==1.8.0+cu111
- transformers==4.30.2
- nltk==3.4.5
- numpy==1.18.1
- pandas==1.0.1
- torch==1.13.1  （运行transformer架构时用）

你可以直接运行以下命令来安装：

```python
pip install -r requirements/requirements.txt
```

## Repository structure
以下是项目中重要的文件结构：

```python
text_summary_lab/
├─data/         # 原始数据集
├─fituning.py   # 预训练模型微调训练
├─log/          # 所有训练日志
├─main.py
├─model/        # 所有模型文件
│ ├─gru2gru.py
│ ├─gruatt2.py
│ ├─lstm2lstm.py
│ ├─lstmatt.py
│ ├─t5_base/        # huggingface 下载的模型
│ ├─t5_base.py
│ ├─t5_medical/     # huggingface 下载的模型
│ ├─t5_medical.py
│ └─transformer/    # 手动实现的transformer架构
├─myutils/     # 日志、数据预处理、画目录结构的代码
├─res/         # 模型checkpoint, 结果图片
├─sh/          # 运行脚本
└─train_val_test.py   # 训练框架，所有模型共用（transformer除外）
```

## 运行基于LSTM或GRU的Seq2Seq
```python
bash runvanilla.sh
```

或者你也可以：

```python
python main.py --model lstm2lstm --version v1 --batch_size 32 --lr 0.1 --num_layers 2
python main.py --model lstmatt --version v1 --batch_size 28 --lr 0.05 
python main.py --model gruatt --version v1 --batch_size 28 --lr 0.01 
```
更多可选参数请看config.yaml


## 运行预训练微调
```python
bash runfituning.sh
```

或者你也可以：

```python
python main.py --model t5-base --version v1 --batch_size 8 --device cpu
python main.py --model t5-medical --version v1 --batch_size 8 --device cpu
```
更多可选参数请看config.yaml


## 运行Transformer架构

```python
pip install torch==1.13.1  # 更换版本
bash runtrainsformer.sh
```

或者你也可以：

```python
pip install torch==1.13.1  # 更换版本
python main.py --model transformer --version v1 --batch_size 64 --device cpu 
```
更多可选参数请看config.yaml


**运行的日志会保存在log文件夹下，得到的checkpoint和生成的折线图会保存在res文件夹下。
这些自动保存的内容都会以 模型名称+ [--version] 选项命名，--version 默认为运行时刻的时间。**

