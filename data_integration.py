#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集成脚本
用于将多个财务数据表和违规信息表集成到一个文件中
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

def load_violation_data():
    """
    加载违规信息总表
    返回违规信息数据框
    """
    violation_file = script_dir / 'Dataset' / '违规信息总表' / 'STK_Violation_Main.xlsx'
    print(f"正在加载违规信息数据: {violation_file.relative_to(script_dir)}")
    
    try:
        df_violation = pd.read_excel(violation_file)
        print(f"违规信息数据形状: {df_violation.shape}")
        print(f"违规信息列名: {df_violation.columns.tolist()}")
        
        # 显示前几行数据以便了解结构
        if len(df_violation) > 0:
            print(f"违规信息前3行:")
            print(df_violation.head(3))
        
        return df_violation
    except Exception as e:
        print(f"加载违规信息数据时出错: {e}")
        return pd.DataFrame()

def load_financial_data():
    """
    加载所有财务数据文件
    返回字典，键为数据类型名称，值为数据框
    """
    data_files = {
        '偿债能力': script_dir / 'Dataset' / '偿债能力' / 'FI_T1.xlsx',
        '经营能力': script_dir / 'Dataset' / '经营能力' / 'FI_T4.xlsx',
        '盈利能力': script_dir / 'Dataset' / '盈利能力' / 'FI_T5.xlsx',
        '风险水平': script_dir / 'Dataset' / '风险水平' / 'FI_T7.xlsx',
        '发展能力': script_dir / 'Dataset' / '发展能力' / 'FI_T8.xlsx',
        '每股指标': script_dir / 'Dataset' / '每股指标' / 'FI_T9.xlsx',
        '披露财务指标': script_dir / 'Dataset' / '披露财务指标' / 'FI_T2.xlsx',
        '股利分配': script_dir / 'Dataset' / '股利分配' / 'FI_T11.xlsx'
    }
    
    financial_data = {}
    
    for name, file_path in data_files.items():
        relative_path = file_path.relative_to(script_dir)
        print(f"\n正在加载{name}数据: {relative_path}")
        
        try:
            df = pd.read_excel(file_path)
            print(f"{name}数据形状: {df.shape}")
            print(f"{name}列名: {df.columns.tolist()}")
            
            # 显示前2行数据以便了解结构
            if len(df) > 0:
                print(f"前2行数据:")
                print(df.head(2))
            
            financial_data[name] = df
        except Exception as e:
            print(f"加载{name}数据时出错: {e}")
    
    return financial_data

def create_violation_label(df_violation, stkcd, accper):
    """
    根据违规信息总表创建违规标签
    如果该股票在该年度有违规记录，则标记为1，否则为0
    
    参数:
        df_violation: 违规信息数据框
        stkcd: 股票代码
        accper: 会计期间
    
    返回:
        int: 0或1，表示是否违规
    """
    if df_violation is None or len(df_violation) == 0:
        return 0
    
    # 提取年份
    try:
        if isinstance(accper, str):
            # 处理YYYY-MM-DD格式
            if '-' in accper:
                year = int(accper.split('-')[0])
            # 处理YYYYMMDD格式
            elif len(accper) >= 4:
                year = int(accper[:4])
            else:
                year = None
        elif isinstance(accper, (int, float)) and not pd.isna(accper):
            # 处理数字格式
            year = int(str(int(accper))[:4])
        else:
            year = None
    except:
        year = None
    
    if year is None:
        return 0
    
    # 转换stkcd为字符串以便比较
    stkcd_str = str(stkcd).strip()
    
    # 检查违规信息表中的不同列名组合
    # 尝试常见的股票代码列名
    stock_code_columns = ['Symbol', '股票代码', 'Stkcd', '股票代码_']
    year_columns = ['ViolationYear', '违规年份', 'Year', '年份']
    date_columns = ['DisposalDate', '处理日期', 'DeclareDate', '公告日期']
    
    # 检查股票代码列
    for stock_col in stock_code_columns:
        if stock_col in df_violation.columns:
            # 检查年份列
            for year_col in year_columns:
                if year_col in df_violation.columns:
                    try:
                        violations = df_violation[
                            (df_violation[stock_col].astype(str).str.strip() == stkcd_str) & 
                            (df_violation[year_col] == year)
                        ]
                        if len(violations) > 0:
                            return 1
                    except:
                        continue
            
            # 如果没有年份列，尝试从日期列中提取年份
            for date_col in date_columns:
                if date_col in df_violation.columns:
                    try:
                        relevant_rows = df_violation[
                            df_violation[stock_col].astype(str).str.strip() == stkcd_str
                        ]
                        for _, row in relevant_rows.iterrows():
                            date_value = str(row[date_col])
                            # 尝试从日期字符串中提取年份
                            for i in range(len(date_value) - 3):
                                if date_value[i:i+4].isdigit():
                                    violation_year = int(date_value[i:i+4])
                                    if violation_year == year:
                                        return 1
                    except:
                        continue
    
    return 0

def integrate_data(financial_data, df_violation):
    """
    数据集成：合并所有财务数据
    
    参数:
        financial_data: 字典，包含所有财务数据
        df_violation: 违规信息数据框
    
    返回:
        pd.DataFrame: 集成后的数据框
    """
    print("\n" + "="*60)
    print("开始数据集成")
    print("="*60)
    
    # 确定主表（选择记录数最多的表或指定的表）
    main_df = None
    main_name = None
    
    # 首先尝试使用经营能力数据作为主表（通常包含关键字段）
    if '经营能力' in financial_data and len(financial_data['经营能力']) > 0:
        main_df = financial_data['经营能力'].copy()
        main_name = '经营能力'
    else:
        # 寻找包含最多记录的表
        max_rows = 0
        for name, df in financial_data.items():
            if len(df) > max_rows:
                max_rows = len(df)
                main_df = df.copy()
                main_name = name
    
    print(f"使用{main_name}数据作为主表，形状: {main_df.shape}")
    
    # 创建列名映射字典
    column_mapping = {
        '股票代码': 'Stkcd',
        '股票代码_': 'Stkcd',
        '股票代码_1': 'Stkcd',
        '截止日期': 'Accper',
        '截止日期_': 'Accper',
        '报表类型': 'Typrep ',  # 注意示例中有空格
        '报表类型编码': 'Typrep '
    }
    
    # 处理主表列名
    new_columns = {}
    found_stock_code = False
    
    for col in main_df.columns:
        clean_col = col.replace("'", "").strip()
        
        # 使用映射字典
        if clean_col in column_mapping:
            new_columns[col] = column_mapping[clean_col]
            if column_mapping[clean_col] == 'Stkcd':
                found_stock_code = True
        # 关键词匹配
        elif '股票代码' in clean_col:
            new_columns[col] = 'Stkcd'
            found_stock_code = True
        elif not found_stock_code and '代码' in clean_col and '行业' not in clean_col:
            new_columns[col] = 'Stkcd'
            found_stock_code = True
        elif any(keyword in clean_col for keyword in ['截止日期', '日期', '时间']):
            new_columns[col] = 'Accper'
        elif any(keyword in clean_col for keyword in ['报表类型', '类型']):
            new_columns[col] = 'Typrep '
    
    # 重命名主表列
    if new_columns:
        main_df = main_df.rename(columns=new_columns)
        print(f"主表列名映射完成: {new_columns}")
    
    # 确保关键字段存在
    required_cols = ['Stkcd', 'Accper']
    missing_cols = [col for col in required_cols if col not in main_df.columns]
    
    if missing_cols:
        print(f"警告: 主表缺少关键字段: {missing_cols}")
        return None
    
    print(f"主表关键字段确认: {'Stkcd' in main_df.columns}, {'Accper' in main_df.columns}")
    
    # 标准化Accper字段为年份格式
    print("\n标准化日期字段...")
    try:
        # 函数：安全地转换日期
        def safe_convert_date(val):
            try:
                if isinstance(val, str):
                    # 处理YYYY-MM-DD格式
                    if '-' in val:
                        return val.split('-')[0]  # 返回年份
                    # 处理YYYYMMDD格式
                    elif val.isdigit() and len(val) >= 4:
                        return val[:4]
                elif isinstance(val, (int, float)) and not pd.isna(val):
                    return str(int(val))[:4]
                return str(val)
            except:
                return str(val)
        
        # 创建年份列用于违规标签
        main_df['_year'] = main_df['Accper'].apply(safe_convert_date)
        print(f"日期字段处理示例: {main_df[['Accper', '_year']].head(3)}")
    except Exception as e:
        print(f"处理日期字段时出错: {e}")
        main_df['_year'] = main_df['Accper'].astype(str)
    
    # 合并其他财务数据
    merge_keys = ['Stkcd', 'Accper']
    
    for name, df in financial_data.items():
        if name == main_name:
            continue
        
        print(f"\n合并{name}数据...")
        df_copy = df.copy()
        
        # 处理当前数据框的列名映射
        df_mapping = {}
        found_stock_code = False
        
        for col in df_copy.columns:
            clean_col = col.replace("'", "").strip()
            
            if clean_col in column_mapping:
                df_mapping[col] = column_mapping[clean_col]
                if column_mapping[clean_col] == 'Stkcd':
                    found_stock_code = True
            elif '股票代码' in clean_col:
                df_mapping[col] = 'Stkcd'
                found_stock_code = True
            elif not found_stock_code and '代码' in clean_col and '行业' not in clean_col:
                df_mapping[col] = 'Stkcd'
                found_stock_code = True
            elif any(keyword in clean_col for keyword in ['截止日期', '日期', '时间']):
                df_mapping[col] = 'Accper'
            elif any(keyword in clean_col for keyword in ['报表类型', '类型']):
                df_mapping[col] = 'Typrep '
        
        # 重命名列
        if df_mapping:
            df_copy = df_copy.rename(columns=df_mapping)
            print(f"{name}列名映射: {df_mapping}")
        
        # 检查合并键
        available_keys = [key for key in merge_keys if key in df_copy.columns]
        
        if available_keys:
            print(f"使用合并键: {available_keys}")
            
            # 检查重复列
            common_cols = [col for col in df_copy.columns 
                          if col in main_df.columns and col not in available_keys]
            
            if common_cols:
                print(f"注意: 发现{len(common_cols)}个重复列，为其添加后缀")
                # 为重复列添加后缀
                for col in common_cols:
                    df_copy = df_copy.rename(columns={col: f"{col}_{name}"})
            
            # 执行合并
            try:
                main_df = pd.merge(main_df, df_copy, on=available_keys, how='left')
                print(f"合并后形状: {main_df.shape}")
            except Exception as e:
                print(f"合并{name}数据时出错: {e}")
        else:
            print(f"警告: {name}数据缺少合并键，跳过")
    
    # 创建违规标签
    print("\n创建违规标签...")
    try:
        # 首先创建一个股票-年份的映射字典，避免重复计算
        violation_dict = {}
        
        def get_violation_label(stkcd, accper):
            # 创建缓存键
            key = f"{stkcd}_{accper}"
            if key not in violation_dict:
                violation_dict[key] = create_violation_label(df_violation, stkcd, accper)
            return violation_dict[key]
        
        # 应用函数创建违规标签
        main_df['isviolation'] = main_df.apply(lambda row: 
                                            get_violation_label(row['Stkcd'], row['Accper']), 
                                            axis=1)
        
        # 统计违规数量
        violation_count = main_df['isviolation'].sum()
        print(f"违规样本数: {violation_count}")
        print(f"总样本数: {len(main_df)}")
        print(f"违规比例: {violation_count / len(main_df) * 100:.2f}%")
        
    except Exception as e:
        print(f"创建违规标签时出错: {e}")
        # 如果出错，默认设置为0
        main_df['isviolation'] = 0
    
    # 删除临时列
    if '_year' in main_df.columns:
        main_df = main_df.drop(columns=['_year'])
    
    # 确保Typrep列名与示例一致（带空格）
    if 'Typrep' in main_df.columns and 'Typrep ' not in main_df.columns:
        main_df = main_df.rename(columns={'Typrep': 'Typrep '})
    
    # 确保关键字段在前面
    cols = main_df.columns.tolist()
    priority_cols = ['Stkcd', 'Accper', 'Typrep ', 'isviolation']
    
    # 移动优先列到前面
    for col in reversed(priority_cols):
        if col in cols:
            cols.remove(col)
            cols.insert(0, col)
    
    main_df = main_df[cols]
    
    print(f"\n集成完成，最终形状: {main_df.shape}")
    print(f"列名顺序: {main_df.columns.tolist()[:10]}...")
    
    return main_df

def save_integrated_data(df):
    """
    保存集成后的数据
    
    参数:
        df: 集成后的数据框
    """
    if df is None or len(df) == 0:
        print("错误: 没有数据可保存")
        return False
    
    output_file = script_dir / 'integrated_data.csv'
    print(f"\n保存集成数据到: {output_file.relative_to(script_dir)}")
    
    try:
        # 保存为CSV格式，使用utf-8-sig编码以支持中文
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"数据保存成功，形状: {df.shape}")
        print(f"列数: {len(df.columns)}")
        
        # 显示前几行数据
        print("\n集成数据预览:")
        print(df.head(3))
        
        return True
    except Exception as e:
        print(f"保存数据时出错: {e}")
        return False

def main():
    """
    主函数
    """
    print("="*60)
    print("数据集成开始")
    print("="*60)
    
    # 1. 加载违规信息数据
    df_violation = load_violation_data()
    
    # 2. 加载财务数据
    financial_data = load_financial_data()
    
    if not financial_data:
        print("错误: 未能加载任何财务数据")
        return
    
    # 3. 数据集成
    integrated_df = integrate_data(financial_data, df_violation)
    
    if integrated_df is None:
        print("错误: 数据集成失败")
        return
    
    # 4. 保存结果
    save_integrated_data(integrated_df)
    
    print("\n" + "="*60)
    print("数据集成完成")
    print("="*60)

if __name__ == '__main__':
    main()