#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据归约脚本
用于对变换后的数据进行特征选择、降维和数据抽样等操作
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 设置显示选项
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 10)

# 获取脚本所在目录的绝对路径
script_dir = Path(__file__).parent.absolute()
# 设置工作目录为脚本所在目录
os.chdir(script_dir)

def load_transformed_data():
    """
    加载变换后的数据
    返回数据框
    """
    input_file = script_dir / 'transformed_data.csv'
    print(f"正在加载变换后的数据: {input_file.relative_to(script_dir)}")
    
    try:
        df = pd.read_csv(input_file, encoding='utf-8-sig')
        print(f"变换后数据形状: {df.shape}")
        print(f"关键字段检查:")
        for col in ['Stkcd', 'Accper', 'Typrep ', 'isviolation']:
            print(f"  - {col}: {'存在' if col in df.columns else '不存在'}")
        
        # 显示前几行数据
        if len(df) > 0:
            print(f"\n数据预览:")
            print(df.head(2))
        
        return df
    except Exception as e:
        print(f"加载变换后的数据时出错: {e}")
        return pd.DataFrame()

def feature_selection(df):
    """
    特征选择
    - 移除低方差特征
    - 相关性分析移除高度相关特征
    - 保留关键字段
    """
    print("\n执行特征选择...")
    
    # 复制数据框
    df_reduced = df.copy()
    
    # 定义关键字段（必须保留）
    key_fields = ['Stkcd', 'Accper', 'Typrep ', 'isviolation', 'year', 'quarter', 'Typrep_encoded']
    
    # 获取数值型列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 排除关键字段和目标变量
    feature_cols = [col for col in numeric_cols if col not in key_fields and col != 'isviolation']
    
    print(f"  总特征数量: {len(feature_cols)}")
    
    # 1. 移除低方差特征
    try:
        # 计算方差
        var_selector = VarianceThreshold(threshold=0.01)  # 移除方差小于0.01的特征
        var_selector.fit(df[feature_cols])
        
        # 获取保留的特征
        selected_features = df[feature_cols].columns[var_selector.get_support()].tolist()
        removed_features = [col for col in feature_cols if col not in selected_features]
        
        print(f"  通过方差阈值选择的特征数量: {len(selected_features)}")
        if removed_features and len(removed_features) <= 5:
            print(f"  移除的低方差特征: {removed_features}")
        elif removed_features:
            print(f"  移除的低方差特征数量: {len(removed_features)}")
    except Exception as e:
        print(f"  方差选择失败: {e}，使用所有特征")
        selected_features = feature_cols
    
    # 2. 移除高度相关的特征
    try:
        if len(selected_features) > 10:
            corr_matrix = df[selected_features].corr().abs()
            # 创建相关性上三角矩阵
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            # 找出相关系数大于0.9的特征对
            to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
            
            if to_drop:
                print(f"  移除的高度相关特征数量: {len(to_drop)}")
                selected_features = [col for col in selected_features if col not in to_drop]
    except Exception as e:
        print(f"  相关性分析失败: {e}，跳过高度相关特征移除")
    
    # 3. 使用SelectKBest进一步筛选最重要的特征
    try:
        # 定义要选择的特征数量（最多30个）
        k = min(30, len(selected_features))
        if k > 10 and 'isviolation' in df.columns:
            print(f"  使用SelectKBest选择前{k}个最重要特征")
            
            # 准备特征和目标变量
            X = df[selected_features].fillna(0)
            y = df['isviolation']
            
            # 创建选择器并拟合
            selector = SelectKBest(f_classif, k=k)
            selector.fit(X, y)
            
            # 获取选择的特征
            best_features = X.columns[selector.get_support()].tolist()
            
            # 显示最重要的5个特征的得分
            scores = selector.scores_
            feature_scores = pd.DataFrame({'Feature': selected_features, 'Score': scores})
            feature_scores = feature_scores.sort_values('Score', ascending=False)
            print(f"  最重要的5个特征:")
            for i, (_, row) in enumerate(feature_scores.head(5).iterrows()):
                print(f"    {i+1}. {row['Feature']}: {row['Score']:.2f}")
            
            selected_features = best_features
    except Exception as e:
        print(f"  SelectKBest选择失败: {e}，使用当前特征集")
    
    print(f"  最终选择的特征数量: {len(selected_features)}")
    
    # 确保关键字段被保留
    final_cols = [col for col in key_fields if col in df.columns] + selected_features
    # 确保没有重复列名
    final_cols = list(dict.fromkeys(final_cols))
    
    # 选择最终的列
    df_reduced = df[final_cols].copy()
    
    return df_reduced

def dimension_reduction(df):
    """
    维度规约（可选）
    - 对数值特征进行PCA降维
    """
    print("\n执行维度规约...")
    
    # 复制数据框
    df_reduced = df.copy()
    
    # 定义关键字段（不参与降维）
    key_fields = ['Stkcd', 'Accper', 'Typrep ', 'isviolation', 'year', 'quarter', 'Typrep_encoded']
    
    # 获取数值型列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 排除关键字段和目标变量
    pca_cols = [col for col in numeric_cols if col not in key_fields and col != 'isviolation']
    
    # 如果数值特征数量大于20，则进行PCA
    if len(pca_cols) > 20:
        print(f"  对{len(pca_cols)}个特征进行PCA降维")
        
        try:
            # 准备数据
            X = df[pca_cols].fillna(0)
            
            # 标准化数据
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 执行PCA，保留解释95%方差的主成分
            pca = PCA(n_components=0.95, random_state=42)
            principal_components = pca.fit_transform(X_scaled)
            
            # 创建主成分列名
            n_components = pca.n_components_
            pca_columns = [f'pca_{i+1}' for i in range(n_components)]
            
            # 创建主成分数据框
            pca_df = pd.DataFrame(data=principal_components, columns=pca_columns, index=df.index)
            
            # 合并主成分到原始数据框
            df_reduced = pd.concat([df.drop(columns=pca_cols), pca_df], axis=1)
            
            print(f"  PCA降维完成，生成{n_components}个主成分")
            print(f"  解释的方差比例: {sum(pca.explained_variance_ratio_):.2%}")
            
        except Exception as e:
            print(f"  PCA降维失败: {e}，跳过降维步骤")
    else:
        print(f"  特征数量较少({len(pca_cols)}个)，无需PCA降维")
    
    return df_reduced

def data_sampling(df, sampling_ratio=1.0):
    """
    数据抽样（可选）
    - 如果数据量太大，可以进行抽样
    """
    print("\n执行数据抽样...")
    
    total_samples = len(df)
    print(f"  原始数据量: {total_samples}")
    
    # 如果设置了抽样比例且数据量较大，则进行抽样
    if sampling_ratio < 1.0 and total_samples > 50000:
        sample_size = int(total_samples * sampling_ratio)
        print(f"  抽样比例: {sampling_ratio}, 抽样数量: {sample_size}")
        
        # 分层抽样，保持违规样本的比例
        if 'isviolation' in df.columns:
            # 分层抽样确保类别平衡
            from sklearn.model_selection import train_test_split
            _, df_sampled = train_test_split(df, test_size=sample_size/total_samples, 
                                           stratify=df['isviolation'], random_state=42)
            print(f"  分层抽样完成，保持违规样本比例")
        else:
            # 简单随机抽样
            df_sampled = df.sample(n=sample_size, random_state=42)
            print(f"  简单随机抽样完成")
        
        return df_sampled
    else:
        print(f"  数据量适中或抽样比例为1.0，无需抽样")
        return df

def save_reduced_data(df):
    """
    保存归约后的数据
    """
    if df is None or len(df) == 0:
        print("错误: 没有数据可保存")
        return False
    
    output_file = script_dir / 'reduced_data.csv'
    print(f"\n保存归约后的数据到: {output_file.relative_to(script_dir)}")
    
    try:
        # 保存为CSV格式，使用utf-8-sig编码以支持中文
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"数据保存成功，形状: {df.shape}")
        print(f"特征数量: {len(df.columns)}")
        
        # 显示归约后的数据摘要
        print("\n归约后数据摘要:")
        print(f"- 总记录数: {len(df)}")
        print(f"- 特征总数: {len(df.columns)}")
        print(f"- 关键字段: {[col for col in ['Stkcd', 'Accper', 'Typrep ', 'isviolation'] if col in df.columns]}")
        
        # 统计违规样本比例
        if 'isviolation' in df.columns:
            violation_rate = df['isviolation'].mean() * 100
            print(f"- 违规样本比例: {violation_rate:.2f}%")
        
        # 显示前10个特征
        print(f"\n保留的主要特征:")
        main_features = [col for col in df.columns if col not in ['Stkcd', 'Accper', 'Typrep ', 'isviolation']][:10]
        for i, feature in enumerate(main_features):
            print(f"  {i+1}. {feature}")
        
        return True
    except Exception as e:
        print(f"保存数据时出错: {e}")
        return False

def main():
    """
    主函数
    """
    print("="*60)
    print("数据归约开始")
    print("="*60)
    
    # 1. 加载变换后的数据
    df = load_transformed_data()
    
    if df.empty:
        print("错误: 未能加载变换后的数据")
        return
    
    # 2. 数据归约步骤
    print("\n" + "="*40)
    print("开始执行数据归约步骤")
    print("="*40)
    
    # 2.1 特征选择
    df_reduced = feature_selection(df)
    
    # 2.2 维度规约（可选）
    df_reduced = dimension_reduction(df_reduced)
    
    # 2.3 数据抽样（可选，这里不进行抽样以保留全部数据）
    df_reduced = data_sampling(df_reduced, sampling_ratio=1.0)
    
    # 3. 保存归约后的数据
    save_reduced_data(df_reduced)
    
    print("\n" + "="*60)
    print("数据归约完成")
    print("="*60)

if __name__ == '__main__':
    main()