# 如何判断机器学习算法的性能

## 问题
    - 模型很差怎么办？真实损失
    - 真实环境难以拿到真实label？
    
## 改进
    - 训练数据和测试数据分离 (train test split)
    - 通过测试数据直接判断模型好坏在模型进入真实环境前改进模型

# 超参数和模型参数

## 超参数
    - 在算法运行前需要决定的参数
        - kNN算法中的k是典型的超参数
        
    - 寻找好的超参数 -> 调参
        - 领域知识
        - 经验数值
        - 实验搜索
        
## 网格搜索
    from sklearn.model_selection import GridSearchCV
        
# 距离
    - 欧拉距离 
        ![avatar](https://bkimg.cdn.bcebos.com/formula/87a52feb423631405eb499ddaec6941d.svg)
    - 曼哈顿距离
    - 明科夫斯基距离
        ![avatar](https://bkimg.cdn.bcebos.com/formula/199c95d9914c4b851533ce7e82bf8ecb.svg)
    - 向量空间余弦相似度 Cosine Similarity
    - 调整余弦相似度 Adjuested Cosine Similarity
    - 皮尔森相关系数 Pearson Correlation Coefficient
    - Jaccard相似系数 Jaccard Coefficient


## 模型参数
    - 算法过程中学习的参数
        - knn算法没有模型参数

