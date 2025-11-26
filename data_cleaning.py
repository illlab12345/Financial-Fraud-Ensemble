#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据清洗脚本
用于对集成后的数据进行清洗，包括处理缺失值、移除重复记录、处理异常值等
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)

# 获取脚本所在目录的绝对路径
script_dir = Path(__file__).parent.absolute()
# 设置工作目录为脚本所在目录
os.chdir(script_dir)

def load_integrated_data():
    """
    加载集成后的数据
    返回数据框
    """
    input_file = script_dir / 'integrated_data.csv'
    print(f"正在加载集成数据: {input_file.relative_to(script_dir)}")
    
    try:
        df = pd.read_csv(input_file, encoding='utf-8-sig')
        print(f"集成数据形状: {df.shape}")
        print(f"集成数据列数: {len(df.columns)}")
        print(f"关键字段检查:")
        for col in ['Stkcd', 'Accper', 'Typrep ', 'isviolation']:
            print(f"  - {col}: {'存在' if col in df.columns else '不存在'}")
        
        # 显示前几行数据以便了解结构
        if len(df) > 0:
            print(f"\n数据预览:")
            print(df.head(3))
        
        return df
    except Exception as e:
        print(f"加载集成数据时出错: {e}")
        return pd.DataFrame()

def remove_unit_rows(df):
    """
    移除包含单位信息的行（如"没有单位'"）
    """
    print("\n移除单位信息行...")
    
    # 识别单位行（通常第二行和第三行是单位行）
    # 检查前几行是否包含"没有单位"或列名相关的字符串
    rows_to_remove = []
    
    for i in range(min(5, len(df))):  # 检查前5行
        row_data = df.iloc[i].astype(str).tolist()
        # 检查是否有多个列包含"没有单位"或与列名相似的字符串
        unit_count = sum(1 for val in row_data if '没有单位' in val or val.endswith("'"))
        # 如果超过30%的列包含单位信息，则认为这是单位行
        if unit_count > len(df.columns) * 0.3:
            rows_to_remove.append(i)
            print(f"  发现单位行: 第{i+1}行")
    
    if rows_to_remove:
        original_shape = df.shape
        df = df.drop(rows_to_remove).reset_index(drop=True)
        print(f"  移除了 {len(rows_to_remove)} 行单位信息")
        print(f"  移除后数据形状: {df.shape}")
    else:
        print("  未发现明显的单位信息行")
    
    return df

def clean_column_names(df):
    """
    清理列名，移除不需要的字符
    """
    print("\n清理列名...")
    
    original_columns = df.columns.tolist()
    new_columns = {}
    
    for col in original_columns:
        # 移除引号、空格等
        clean_col = col.replace("'", "").strip()
        # 处理重复的空格
        clean_col = ' '.join(clean_col.split())
        # 确保Typrep列名带空格
        if clean_col == 'Typrep':
            clean_col = 'Typrep '
        elif clean_col == 'isviolation':
            # 保持isviolation列名不变
            pass
        elif clean_col.startswith('Stkcd') or clean_col == '股票代码':
            # 确保股票代码列名为Stkcd
            clean_col = 'Stkcd'
        elif clean_col.startswith('Accper') or clean_col == '截止日期':
            # 确保会计期间列名为Accper
            clean_col = 'Accper'
        
        new_columns[col] = clean_col
    
    # 重命名列
    df = df.rename(columns=new_columns)
    
    # 检查是否有重复列名
    if df.columns.duplicated().any():
        duplicated_cols = df.columns[df.columns.duplicated()].tolist()
        print(f"警告: 发现重复列名: {duplicated_cols}")
        # 为重复列添加后缀
        cols = []
        count = {}
        for col in df.columns:
            if col in count:
                count[col] += 1
                cols.append(f"{col}_{count[col]}")
            else:
                count[col] = 0
                cols.append(col)
        df.columns = cols
        print(f"  已为重复列添加后缀")
    
    print(f"  清理后的列名（前10个）: {df.columns.tolist()[:10]}...")
    return df

def clean_stock_code(df):
    """
    清理股票代码列
    """
    print("\n清理股票代码...")
    
    if 'Stkcd' not in df.columns:
        print("  未找到股票代码列")
        return df
    
    # 转换为字符串类型
    df['Stkcd'] = df['Stkcd'].astype(str).str.strip()
    
    # 移除引号和特殊字符
    df['Stkcd'] = df['Stkcd'].str.replace("'", "").str.strip()
    
    # 填充空值
    before_count = df['Stkcd'].isna().sum()
    df['Stkcd'] = df['Stkcd'].replace('', np.nan)
    
    # 检查并移除无效的股票代码
    valid_stock_codes = df['Stkcd'].str.match(r'^\d{6}$').fillna(False)
    invalid_count = len(df) - valid_stock_codes.sum()
    
    if invalid_count > 0:
        print(f"  发现 {invalid_count} 条无效股票代码记录")
        df = df[valid_stock_codes].reset_index(drop=True)
        print(f"  移除后数据形状: {df.shape}")
    
    # 统计唯一股票代码数量
    unique_stocks = df['Stkcd'].nunique()
    print(f"  唯一股票代码数量: {unique_stocks}")
    
    return df

def clean_accounting_period(df):
    """
    清理会计期间列
    """
    print("\n清理会计期间...")
    
    if 'Accper' not in df.columns:
        print("  未找到会计期间列")
        return df
    
    # 转换为字符串类型
    df['Accper'] = df['Accper'].astype(str).str.strip()
    
    # 移除引号和特殊字符
    df['Accper'] = df['Accper'].str.replace("'", "").str.strip()
    
    # 标准化日期格式
    def standardize_date(date_str):
        try:
            # 处理YYYY-MM-DD格式
            if '-' in date_str:
                parts = date_str.split('-')
                if len(parts) == 3:
                    return date_str
                elif len(parts[0]) == 4:
                    return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
            # 处理YYYYMMDD格式
            elif date_str.isdigit() and len(date_str) == 8:
                return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            # 处理只包含年份的情况
            elif date_str.isdigit() and len(date_str) == 4:
                return f"{date_str}-12-31"  # 默认设为年末
            return date_str
        except:
            return date_str
    
    # 应用标准化函数
    df['Accper'] = df['Accper'].apply(standardize_date)
    
    # 检查日期格式
    date_pattern = r'^\d{4}-\d{2}-\d{2}$'
    valid_dates = df['Accper'].str.match(date_pattern).fillna(False)
    invalid_count = len(df) - valid_dates.sum()
    
    if invalid_count > 0:
        print(f"  发现 {invalid_count} 条无效日期格式记录")
        print(f"  无效日期示例: {df[~valid_dates]['Accper'].head(5).tolist()}")
    
    # 提取年份信息用于后续分析
    try:
        df['year'] = df['Accper'].str[:4].astype(int)
        print(f"  年份范围: {df['year'].min()} - {df['year'].max()}")
    except:
        print("  无法提取年份信息")
    
    return df

def clean_typerep(df):
    """
    清理报表类型列
    """
    print("\n清理报表类型...")
    
    if 'Typrep ' not in df.columns:
        # 检查是否有不带空格的Typrep
        if 'Typrep' in df.columns:
            df = df.rename(columns={'Typrep': 'Typrep '})
        else:
            print("  未找到报表类型列，创建默认值")
            df['Typrep '] = 'A'  # 默认值
            return df
    
    # 转换为字符串类型
    df['Typrep '] = df['Typrep '].astype(str).str.strip()
    
    # 移除引号和特殊字符
    df['Typrep '] = df['Typrep '].str.replace("'", "").str.strip()
    
    # 清理常见值
    type_mapping = {
        'A': 'A', '1': 'A', 'annual': 'A', '年报': 'A',
        'B': 'B', '2': 'B', 'semiannual': 'B', '半年报': 'B',
        'Q': 'Q', '3': 'Q', 'quarterly': 'Q', '季报': 'Q'
    }
    
    # 标准化报表类型
    df['Typrep '] = df['Typrep '].map(type_mapping).fillna(df['Typrep '])
    
    # 统计报表类型分布
    type_counts = df['Typrep '].value_counts()
    print(f"  报表类型分布: {type_counts.to_dict()}")
    
    # 对于非标准值，设为默认值
    standard_types = ['A', 'B', 'Q']
    non_standard_count = (~df['Typrep '].isin(standard_types)).sum()
    
    if non_standard_count > 0:
        print(f"  发现 {non_standard_count} 条非标准报表类型记录")
        # 将非标准值设为默认值'A'
        df.loc[~df['Typrep '].isin(standard_types), 'Typrep '] = 'A'
        print("  已将非标准报表类型设为'A'")
    
    return df

def clean_violation_label(df):
    """
    清理违规标签列
    """
    print("\n清理违规标签...")
    
    if 'isviolation' not in df.columns:
        print("  未找到违规标签列，创建默认值")
        df['isviolation'] = 0
        return df
    
    # 转换为数值类型
    df['isviolation'] = pd.to_numeric(df['isviolation'], errors='coerce').fillna(0).astype(int)
    
    # 确保只包含0和1
    df['isviolation'] = df['isviolation'].apply(lambda x: 1 if x >= 1 else 0)
    
    # 统计违规分布
    violation_counts = df['isviolation'].value_counts()
    print(f"  违规标签分布: {violation_counts.to_dict()}")
    
    violation_rate = df['isviolation'].mean()
    print(f"  违规比例: {violation_rate:.2%}")
    
    return df

def remove_duplicate_records(df):
    """
    移除重复记录
    """
    print("\n移除重复记录...")
    
    # 检查完全重复的记录
    exact_duplicates = df.duplicated().sum()
    if exact_duplicates > 0:
        print(f"  发现 {exact_duplicates} 条完全重复记录")
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"  移除后数据形状: {df.shape}")
    else:
        print("  未发现完全重复记录")
    
    # 检查股票代码+会计期间+报表类型的重复记录
    key_cols = ['Stkcd', 'Accper', 'Typrep ']
    if all(col in df.columns for col in key_cols):
        key_duplicates = df.duplicated(subset=key_cols).sum()
        if key_duplicates > 0:
            print(f"  发现 {key_duplicates} 条股票代码+会计期间+报表类型重复记录")
            # 对于重复的记录，保留第一条
            df = df.drop_duplicates(subset=key_cols, keep='first').reset_index(drop=True)
            print(f"  移除后数据形状: {df.shape}")
    
    return df

def handle_missing_values(df):
    """
    处理缺失值
    """
    print("\n处理缺失值...")
    
    # 计算总体缺失率
    total_missing = df.isna().sum().sum()
    total_cells = df.size
    overall_missing_rate = total_missing / total_cells
    
    print(f"  总体缺失值数量: {total_missing}")
    print(f"  总体缺失率: {overall_missing_rate:.2%}")
    
    # 计算每列的缺失率
    col_missing_rates = df.isna().mean()
    high_missing_cols = col_missing_rates[col_missing_rates > 0.9].index.tolist()
    medium_missing_cols = col_missing_rates[(col_missing_rates > 0.5) & (col_missing_rates <= 0.9)].index.tolist()
    
    if high_missing_cols:
        print(f"  高缺失率列（>90%）数量: {len(high_missing_cols)}")
        print(f"  示例: {high_missing_cols[:5]}...")
        # 可以选择删除高缺失率的列
        # df = df.drop(columns=high_missing_cols)
        # print(f"  已删除{len(high_missing_cols)}个高缺失率列")
    
    # 处理关键字段的缺失值
    key_cols = ['Stkcd', 'Accper', 'Typrep ', 'isviolation']
    for col in key_cols:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                print(f"  关键字段 {col} 缺失 {missing_count} 条")
                # 对于关键字段的缺失值，直接删除该行
                df = df.dropna(subset=[col]).reset_index(drop=True)
                print(f"  已移除关键字段{col}缺失的记录")
    
    # 对数值列的缺失值进行填充（可选）
    # 这里不进行填充，保留原始缺失状态
    
    print(f"  处理后数据形状: {df.shape}")
    return df

def clean_numeric_columns(df):
    """
    清理数值列，转换为合适的数值类型
    """
    print("\n清理数值列...")
    
    # 识别数值列（排除关键字段）
    exclude_cols = ['Stkcd', 'Accper', 'Typrep ', 'isviolation', 'year']
    numeric_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"  数值列数量: {len(numeric_cols)}")
    
    converted_cols = 0
    conversion_errors = 0
    
    for col in numeric_cols:
        try:
            # 先尝试直接转换
            original_dtype = df[col].dtype
            
            # 处理字符串类型的数值
            if df[col].dtype == 'object':
                # 移除特殊字符
                df[col] = df[col].astype(str).str.replace("'", "").str.strip()
                # 空字符串转为NaN
                df[col] = df[col].replace('', np.nan)
                # 转换为数值类型
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 检查是否成功转换
            if df[col].dtype.kind in 'ifc':  # integer, float, complex
                converted_cols += 1
            else:
                conversion_errors += 1
        except Exception as e:
            conversion_errors += 1
    
    print(f"  成功转换的数值列: {converted_cols}")
    if conversion_errors > 0:
        print(f"  转换失败的数值列: {conversion_errors}")
    
    return df

def save_cleaned_data(df):
    """
    保存清洗后的数据
    """
    if df is None or len(df) == 0:
        print("错误: 没有数据可保存")
        return False
    
    output_file = script_dir / 'cleaned_data.csv'
    print(f"\n保存清洗后的数据到: {output_file.relative_to(script_dir)}")
    
    try:
        # 保存为CSV格式，使用utf-8-sig编码以支持中文
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"数据保存成功，形状: {df.shape}")
        print(f"列数: {len(df.columns)}")
        
        # 显示清洗后的数据摘要
        print("\n清洗后数据摘要:")
        print(f"- 总记录数: {len(df)}")
        print(f"- 唯一股票代码数: {df['Stkcd'].nunique()}" if 'Stkcd' in df.columns else "- 股票代码信息: 不可用")
        print(f"- 违规样本数: {df['isviolation'].sum()}" if 'isviolation' in df.columns else "- 违规信息: 不可用")
        print(f"- 违规比例: {df['isviolation'].mean():.2%}" if 'isviolation' in df.columns else "- 违规比例: 不可用")
        print(f"- 数据年份范围: {df['year'].min()} - {df['year'].max()}" if 'year' in df.columns else "- 年份信息: 不可用")
        
        return True
    except Exception as e:
        print(f"保存数据时出错: {e}")
        return False

def main():
    """
    主函数
    """
    print("="*60)
    print("数据清洗开始")
    print("="*60)
    
    # 1. 加载集成数据
    df = load_integrated_data()
    
    if df.empty:
        print("错误: 未能加载集成数据")
        return
    
    # 2. 数据清洗步骤
    print("\n" + "="*40)
    print("开始执行数据清洗步骤")
    print("="*40)
    
    # 2.1 移除单位信息行
    df = remove_unit_rows(df)
    
    # 2.2 清理列名
    df = clean_column_names(df)
    
    # 2.3 清理股票代码
    df = clean_stock_code(df)
    
    # 2.4 清理会计期间
    df = clean_accounting_period(df)
    
    # 2.5 清理报表类型
    df = clean_typerep(df)
    
    # 2.6 清理违规标签
    df = clean_violation_label(df)
    
    # 2.7 移除重复记录
    df = remove_duplicate_records(df)
    
    # 2.8 处理缺失值
    df = handle_missing_values(df)
    
    # 2.9 清理数值列
    df = clean_numeric_columns(df)
    
    # 3. 保存清洗后的数据
    save_cleaned_data(df)
    
    print("\n" + "="*60)
    print("数据清洗完成")
    print("="*60)

if __name__ == '__main__':
    main()