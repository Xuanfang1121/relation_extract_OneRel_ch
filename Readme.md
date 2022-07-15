### 关系抽取——OneRel
系抽取OneRel在中文数据集上的应用。这里采用transformers的tokenizer，中文数据处理取消了[unuesd1]


#### 依赖
```
torch==1.8.1
transformers==4.10.1
numpy==1.19.2
```

#### 代码结构
```
 .
 |__common
 |__config
 |__data
 |__models
 |__utils
 |__train.py
 |__predict.py
```

#### 数据结构
```
{
    "text": "《邪少兵王》是冰火未央写的网络小说连载于旗峰天下",
    "triple_list": [
      [
        "邪少兵王",
        "作者",
        "冰火未央"
      ]
    ]
  }
```

#### 代码说明
1.如果目录下没有文件夹log，新建文件夹log <br>
2.修改config/config.ini中的参数 <br>
3.模型训练<br>
4.模型推理<br>

####模型结果
百度关系抽取2020数据集训练
```
precision:0.7313706008471454, recall:0.7400391555108716, f1 :0.7356793434879514
```

####参考文献
[1][https://github.com/ssnvxia/OneRel]
