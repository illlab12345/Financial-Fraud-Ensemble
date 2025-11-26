#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据变换脚本
用于对清洗后的数据进行特征工程、标准化、编码等操作
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# 设置显示选项
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 10)

# 获取脚本所在目录的绝对路径
script_dir = Path(__file__).parent.absolute()
# 设置工作目录为脚本所在目录
os.chdir(script_dir)

def load_cleaned_data():
    """
    加载清洗后的数据
    返回数据框
    """
    input_file = script_dir / 'cleaned_data.csv'
    print(f"正在加载清洗后的数据: {input_file.relative_to(script_dir)}")
    
    try:
        df = pd.read_csv(input_file, encoding='utf-8-sig')
        print(f"清洗后数据形状: {df.shape}")
        print(f"关键字段检查:")
        for col in ['Stkcd', 'Accper', 'Typrep ', 'isviolation']:
            print(f"  - {col}: {'存在' if col in df.columns else '不存在'}")
        
        # 显示前几行数据
        if len(df) > 0:
            print(f"\n数据预览:")
            print(df.head(2))
        
        # 提取年份信息用于后续处理
        if 'Accper' in df.columns:
            df['year'] = pd.to_datetime(df['Accper']).dt.year
            print(f"已从Accper提取年份信息")
        
        return df
    except Exception as e:
        print(f"加载清洗后的数据时出错: {e}")
        return pd.DataFrame()

def calculate_basic_financial_ratios(df):
    """
    计算基本财务比率（简化版）
    """
    print("\n计算基本财务比率...")
    
    # 复制数据框以避免修改原始数据
    df_transformed = df.copy()
    
    # 1. 获取各类财务指标列
    financial_groups = {
        'profitability': [col for col in df.columns if col.startswith('F05')],  # 盈利能力
        'operation': [col for col in df.columns if col.startswith('F04')],      # 经营能力
        'debt': [col for col in df.columns if col.startswith('F02')],           # 偿债能力
        'growth': [col for col in df.columns if col.startswith('F08')],         # 发展能力
        'risk': [col for col in df.columns if col.startswith('F07')],           # 风险水平
        'per_share': [col for col in df.columns if col.startswith('F09')]       # 每股指标
    }
    
    # 2. 为每个财务指标类别创建综合分数
    for group_name, cols in financial_groups.items():
        if cols and all(col in df.columns for col in cols[:min(3, len(cols))]):  # 只使用前3个指标避免计算过重
            score_col = f"{group_name}_score"
            df_transformed[score_col] = df[cols[:3]].mean(axis=1)
            print(f"  已创建{group_name}综合分数: {score_col}")
    
    # 3. 创建一个整体财务健康分数
    score_cols = [col for col in df_transformed.columns if col.endswith('_score')]
    if score_cols:
        df_transformed['financial_health_score'] = df_transformed[score_cols].mean(axis=1)
        print(f"  已创建整体财务健康分数")
    
    print(f"  新增特征数量: {len(df_transformed.columns) - len(df.columns)}")
    return df_transformed

def encode_categorical_features(df):
    """
    对分类特征进行编码（简化版）
    """
    print("\n编码分类特征...")
    
    df_transformed = df.copy()
    
    # 1. 报表类型（Typrep）的编码
    if 'Typrep ' in df.columns:
        print("  对报表类型进行编码")
        # 创建简化的编码
        typrep_map = {value: idx for idx, value in enumerate(df['Typrep '].unique())}
        df_transformed['Typrep_encoded'] = df['Typrep '].map(typrep_map)
        print(f"  报表类型映射: {typrep_map}")
    
    # 2. 提取季度特征
    if 'Accper' in df.columns:
        print("  提取季度特征")
        # 从日期中提取月份并创建季度列
        df_transformed['quarter'] = pd.to_datetime(df['Accper']).dt.quarter
        print(f"  季度分布: {df_transformed['quarter'].value_counts().to_dict()}")
    
    return df_transformed

def normalize_key_features(df):
    """
    对关键特征进行标准化（简化版）
    """
    print("\n对关键特征进行标准化...")
    
    # 获取关键字段
    key_cols = ['Stkcd', 'Accper', 'Typrep ', 'isviolation', 'year', 'quarter']
    # 获取数值型列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 排除关键字段和目标变量
    cols_to_normalize = [col for col in numeric_cols if col not in key_cols and col.endswith('_score')]
    
    # 如果没有score列，则选择一些重要的财务指标
    if not cols_to_normalize:
        # 选择重要的财务指标前缀
        important_prefixes = ['F05', 'F04', 'F02']  # 盈利能力、经营能力、偿债能力
        cols_to_normalize = []
        for prefix in important_prefixes:
            prefix_cols = [col for col in numeric_cols if col.startswith(prefix)]
            cols_to_normalize.extend(prefix_cols[:2])  # 每个类别选择前2个指标
        cols_to_normalize = list(set(cols_to_normalize))[:10]  # 最多选择10个指标
    
    print(f"  需要标准化的特征数量: {len(cols_to_normalize)}")
    
    if not cols_to_normalize:
        print("  没有需要标准化的特征")
        return df
    
    # 创建标准化器
    scaler = StandardScaler()
    
    # 创建数据框副本
    df_transformed = df.copy()
    
    # 对整个数据集进行标准化（简化处理）
    try:
        # 填充NaN值
        df_filled = df[cols_to_normalize].fillna(0)
        # 拟合和转换
        scaled_data = scaler.fit_transform(df_filled)
        # 将标准化后的数据放回原数据框
        for i, col in enumerate(cols_to_normalize):
            norm_col = f"{col}_norm"
            df_transformed[norm_col] = scaled_data[:, i]
        
        print(f"  成功创建{len(cols_to_normalize)}个标准化特征")
    except Exception as e:
        print(f"  标准化过程中出错: {e}")
    
    return df_transformed

def handle_missing_values(df):
    """
    处理缺失值（简化版）
    """
    print("\n处理缺失值...")
    
    # 复制数据框
    df_transformed = df.copy()
    
    # 获取数值型列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 计算缺失值比例
    missing_ratio = df[numeric_cols].isnull().mean()
    high_missing_cols = missing_ratio[missing_ratio > 0.5].index.tolist()
    
    if high_missing_cols:
        print(f"  删除高缺失列 ({len(high_missing_cols)}): {', '.join(high_missing_cols[:5])}...")
        df_transformed = df_transformed.drop(columns=high_missing_cols)
    
    # 对剩余数值列的缺失值进行填充
    remaining_numeric_cols = [col for col in numeric_cols if col not in high_missing_cols]
    for col in remaining_numeric_cols:
        if col in df_transformed.columns:
            # 使用均值填充
            df_transformed[col] = df_transformed[col].fillna(df_transformed[col].mean())
    
    print(f"  缺失值处理完成")
    return df_transformed

def save_transformed_data(df):
    """
    保存变换后的数据
    """
    if df is None or len(df) == 0:
        print("错误: 没有数据可保存")
        return False
    
    output_file = script_dir / 'transformed_data.csv'
    print(f"\n保存变换后的数据到: {output_file.relative_to(script_dir)}")
    
    try:
        # 保存为CSV格式，使用utf-8-sig编码以支持中文
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"数据保存成功，形状: {df.shape}")
        print(f"特征数量: {len(df.columns)}")
        
        # 显示变换后的数据摘要
        print("\n变换后数据摘要:")
        print(f"- 总记录数: {len(df)}")
        print(f"- 特征总数: {len(df.columns)}")
        print(f"- 关键字段: {[col for col in ['Stkcd', 'Accper', 'Typrep ', 'isviolation'] if col in df.columns]}")
        
        # 统计违规样本比例
        if 'isviolation' in df.columns:
            violation_rate = df['isviolation'].mean() * 100
            print(f"- 违规样本比例: {violation_rate:.2f}%")
        
        return True
    except Exception as e:
        print(f"保存数据时出错: {e}")
        return False

def main():
    """
    主函数
    """
    print("="*60)
    print("数据变换开始")
    print("="*60)
    
    # 1. 加载清洗后的数据
    df = load_cleaned_data()
    
    if df.empty:
        print("错误: 未能加载清洗后的数据")
        return
    
    # 2. 数据变换步骤（简化版）
    print("\n" + "="*40)
    print("开始执行数据变换步骤")
    print("="*40)
    
    # 2.1 处理缺失值
    df = handle_missing_values(df)
    
    # 2.2 计算基本财务比率
    df = calculate_basic_financial_ratios(df)
    
    # 2.3 编码分类特征
    df = encode_categorical_features(df)
    
    # 2.4 标准化关键特征
    df = normalize_key_features(df)
    
    # 3. 保存变换后的数据
    save_transformed_data(df)
    
    print("\n" + "="*60)
    print("数据变换完成")
    print("="*60)

if __name__ == '__main__':
    main()