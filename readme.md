# 财务舞弊识别集成学习系统
## 项目概述
本项目构建了一个基于集成学习的财务舞弊识别系统，通过结合多个先进的梯度提升模型，实现高精度的财务舞弊检测。系统采用了完整的数据预处理流程和多种集成学习策略，显著提高了检测的准确性和可靠性。

### 核心功能
- 完整的数据预处理流程（集成、清洗、变换、归约）
- 混合采样技术处理类别不平衡问题
- 基于XGBoost、LightGBM和CatBoost的梯度提升模型
- Optuna贝叶斯超参数优化
- 加权软投票和堆叠集成方法
- 全面的模型评估和可视化分析
## 目录结构
```
├── Dataset/             # 原始数据集
目录
│   ├── 偿债能力/         # 偿债能力指
标数据
│   ├── 发展能力/         # 发展能力指
标数据
│   ├── 披露财务指标/     # 披露财务指
标数据
│   ├── 每股指标/         # 每股指标数
据
│   ├── 盈利能力/         # 盈利能力指
标数据
│   ├── 经营能力/         # 经营能力指
标数据
│   ├── 股利分配/         # 股利分配数
据
│   ├── 违规信息总表/     # 违规信息数
据
│   └── 风险水平/         # 风险水平指
标数据
├── data_integration.py  # 数据集成脚
本
├── data_cleaning.py     # 数据清洗脚
本
├── data_transformation.py # 数据变换
脚本
├── data_reduction.py    # 数据归约脚
本
├── financial_fraud_ensemble_final.
ipynb # 集成学习主Notebook
├── integrated_data.csv  # 集成后的数
据
├── cleaned_data.csv     # 清洗后的数
据
├── transformed_data.csv # 变换后的数
据
├── reduced_data.csv     # 归约后的数
据
├── 
integrated_model_performance_report.
csv # 模型性能报告
└── model_performance_comparison.
csv # 模型性能对比
```
## 系统架构
### 数据处理流程
1. 数据集成 ：将分散在多个Excel文件中的财务指标数据与违规信息整合
2. 数据清洗 ：处理缺失值、异常值，提高数据质量
3. 数据变换 ：通过特征工程创建衍生特征，进行标准化和编码
4. 数据归约 ：通过特征选择和PCA降维减少数据维度
### 模型架构
- 基础模型层 ：
  
  - XGBoost：高性能梯度提升框架，具有强大的正则化能力
  - LightGBM：基于直方图的高效梯度提升框架
  - CatBoost：针对类别特征优化的梯度提升框架
- 集成策略层 ：
  
  - 加权软投票：根据模型性能分配权重
  - 堆叠集成：使用逻辑回归或ExtraTrees作为元模型
## 技术栈
- 数据处理 ：Python, Pandas, NumPy, Scikit-learn
- 数据采样 ：imbalanced-learn (SMOTE, RandomUnderSampler)
- 机器学习模型 ：XGBoost, LightGBM, CatBoost
- 超参数优化 ：Optuna
- 评估与可视化 ：Matplotlib, Seaborn, Scikit-learn metrics
- 开发环境 ：Jupyter Notebook
## 核心实现
### 1. 数据预处理 数据集成
- 将多维度财务指标与违规信息整合
- 构建违规标签（0/1）
- 生成 integrated_data.csv 包含56个特征，119,060条记录 数据清洗与变换
- 处理缺失值和异常值
- 时间特征提取和财务比率计算
- 特征标准化和编码
- 通过PCA降维生成24个主成分，保留96.22%的方差
### 2. 不平衡数据处理
采用混合采样策略，结合SMOTE过采样和RandomUnderSampler欠采样，有效平衡数据集，提高少数类识别能力。

### 3. 模型训练与优化
每个基础模型通过Optuna进行贝叶斯超参数优化，优化目标为交叉验证AUC值：

- XGBoost优化参数 ：n_estimators, max_depth, learning_rate, gamma等
- LightGBM优化参数 ：n_estimators, max_depth, learning_rate, num_leaves等
- CatBoost优化参数 ：n_estimators, max_depth, learning_rate, l2_leaf_reg等
### 4. 集成学习实现 加权软投票集成
- 基于模型AUC表现自动分配权重
- 支持专家经验调整权重
- 使用sklearn的VotingClassifier实现 堆叠集成
- 自定义StackingEnsemble类实现分层K折交叉验证堆叠
- 第一层：XGBoost、LightGBM、CatBoost基础模型
- 第二层：逻辑回归或ExtraTrees元模型
## 性能结果
各模型测试集AUC表现：

模型 AUC 加权集成 0.8039 VotingClassifier 0.8039 堆叠集成(逻辑回归) 0.8023 LightGBM 0.7917 XGBoost 0.7892 CatBoost 0.7881 堆叠集成(ExtraTrees) 0.7791

集成方法相比最佳单模型（LightGBM）提升约1.54%。

## 使用指南
### 环境配置
```
pip install -r requirements.txt
```
### 工作流程
1. 数据准备 ：将原始数据放入 Dataset 目录
2. 数据预处理 ：按顺序运行数据预处理脚本
   ```
   python data_integration.py
   python data_cleaning.py
   python data_transformation.py
   python data_reduction.py
   ```
3. 模型训练与评估 ：运行Jupyter Notebook
   ```
   jupyter notebook 
   financial_fraud_ensemble_final.
   ipynb
   ```
## 实际应用建议
1. 实时监控与更新 ：建立定期模型更新机制，跟踪模型性能变化
2. 人机结合决策 ：模型提供风险评分，专家进行最终审核
3. 解释性增强 ：使用SHAP值等工具解释模型决策
4. 合规性考虑 ：确保模型符合金融监管要求，维护决策记录
## 未来改进方向
1. 动态特征工程 ：开发更复杂的时间序列特征
2. 深度学习集成 ：将深度学习模型纳入集成框架
3. 主动学习 ：减少标注成本
4. 联邦学习 ：保护数据隐私的同时进行模型训练
5. 因果推断 ：探索财务指标与舞弊行为之间的因果关系
## 总结
本项目通过完整的数据预处理流程和先进的集成学习技术，构建了一个高性能的财务舞弊识别系统。加权软投票集成方法在测试中表现最佳，相比单一模型有显著性能提升。系统的模块化设计使其易于扩展和维护，可根据实际业务需求进行调整和优化。