#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek R1 × LangChain 智能数据分析系统
"""

# 导入所需的库
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import CSVLoader
import matplotlib.pyplot as plt
import json
import re
import os
from dotenv import load_dotenv

# 导入PythonREPL
from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from langchain_experimental.tools import PythonREPLTool

# 加载环境变量
load_dotenv()

# 初始化LangChain的ChatOpenAI客户端
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "sk-your-api-key"),  # 替换为你的DeepSeek API密钥
    base_url="https://api.deepseek.com",  # DeepSeek API端点
    model="deepseek-reasoner",  # 指定模型名称
    temperature=0.2,  # 降低随机性，使结果更稳定
    verbose=True  # 输出详细日志
)

def load_data(file_path):
    """智能加载并预处理CSV数据
    
    Args:
        file_path: CSV文件路径
        
    Returns:
        返回处理后的数据内容（前3000字符，防止超出上下文窗口限制）
    """
    documents = CSVLoader(file_path).load()
    data = [ doc.page_content for doc in documents ]
    return data

# 更新分析指令模板，明确要求处理所有月份的数据
ANALYSIS_TEMPLATE = """
你是一个专业数据分析引擎，负责解析以下CSV数据片段：
{data}

请执行以下分析任务：
1. 按月份统计销售总额（返回JSON数组，每项包含month和sales字段）
   - 必须统计CSV中出现的所有月份，从1月到6月
   - 确保数据按月份排序

2. 计算各产品销售占比（返回JSON对象，数值保留2位小数）

3. 生成可视化月度销售趋势的Matplotlib代码
   - 图表必须显示所有月份的数据
   - 确保x轴标签包含所有月份（例如：2024-01到2024-06）
   - 使用合适的颜色和标记增强可视化效果

严格按照以下JSON格式输出结果：
{{
    "trend": [
        {{"month": "2024-01", "sales": 150000}},
        {{"month": "2024-02", "sales": 160000}},
        ...  # 所有月份的数据
    ],
    "product_ratio": {{"产品A": 0.45, "产品B": 0.55, ...}},
    "visual_code": "import matplotlib.pyplot as plt\\n..."
}}
"""

def analyze_data(data_sample):
    """使用DeepSeek模型执行数据分析
    
    Args:
        data_sample: 待分析的数据样本
        
    Returns:
        包含分析结果的Python字典
    """
    # 构建消息
    messages = [
        SystemMessage(content="你是一个严谨的数据分析专家，擅长统计计算和数据可视化"),
        HumanMessage(content=ANALYSIS_TEMPLATE.format(data=data_sample))
    ]
    
    # 调用大模型获取响应
    response = llm.invoke(messages)
    
    # 安全解析响应内容为Python字典
    try:
        # 尝试从可能的代码块中提取JSON
        json_match = re.search(r'```(?:json)?\s*(.*?)```', response.content, re.DOTALL)
        if json_match:
            content = json_match.group(1).strip()
        else:
            content = response.content
            
        # 尝试用json解析
        result = json.loads(content)
        
        # 验证结果
        if not validate_result(result):
            print("⚠️ 结果验证失败，重新尝试分析...")
            # 如果需要，这里可以添加重试逻辑
            
        return result
    except json.JSONDecodeError:
        # 如果json解析失败，谨慎使用eval (仅用于教学目的)
        try:
            result = eval(content)
            if not validate_result(result):
                print("⚠️ 结果验证失败，但将继续处理...")
            return result
        except:
            print("❌ 响应解析失败，请检查prompt")
            return None

def validate_result(result):
    """验证分析结果是否包含所有月份的数据"""
    if not result:
        return False
        
    # 确保趋势数据存在且至少有1条记录
    if "trend" not in result or not result["trend"]:
        print("⚠️ 警告: 未找到趋势数据")
        return False
        
    # 提取CSV中出现的所有月份
    import pandas as pd
    try:
        df = pd.read_csv("sales_data.csv")
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.strftime('%Y-%m')
        actual_months = sorted(df['month'].unique())
        
        # 检查结果中是否包含所有月份
        result_months = [item["month"] for item in result["trend"]]
        missing_months = [m for m in actual_months if m not in result_months]
        
        if missing_months:
            print(f"⚠️ 警告: 结果中缺少以下月份: {missing_months}")
            return False
            
    except Exception as e:
        print(f"⚠️ 验证数据时出错: {str(e)}")
        
    return True

def render_plot(code_block):
    """使用LangChain的PythonREPL工具执行AI生成的可视化代码
    
    Args:
        code_block: 包含matplotlib代码的字符串
    """
    try:
        # 清理代码（删除可能存在的代码块包装符号）
        code_block = re.sub(r'^```python\s+|\s+```$', '', code_block, flags=re.DOTALL)
        
        # 添加保存图表的代码
        if "plt.savefig" not in code_block:
            code_block += "\nplt.savefig('sales_analysis.png')\n"
        
        # 使用PythonREPL安全执行代码
        repl = PythonREPLTool()
        result = repl.run(code_block)
        
        print("✅ 可视化图表已生成")
        return result
    except Exception as e:
        print(f"❌ 可视化生成失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 大数据智能采样策略
def smart_sample(file_path, sample_size=500):
    """对大型数据集进行智能采样
    
    Args:
        file_path: CSV文件路径
        sample_size: 采样行数
        
    Returns:
        代表性样本数据文本
    """
    import pandas as pd
    
    df = pd.read_csv(file_path)
    
    # 确保采样均匀覆盖时间维度(如果存在)
    if 'date' in df.columns or 'time' in df.columns:
        date_col = 'date' if 'date' in df.columns else 'time'
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)
        
        # 分层抽样
        step = max(1, len(df) // sample_size)
        sampled = df.iloc[::step].head(sample_size)
    else:
        # 随机抽样
        sampled = df.sample(min(sample_size, len(df)))
    
    return sampled.to_string(index=False)

def competitive_analysis(product_data, market_data, competitor_data):
    """
    整合产品、市场和竞争对手数据，生成全面竞争分析
    """
    # 构建分析指令
    analysis_prompt = f"""
    分析三个数据源:
    1. 我方产品数据: {product_data}
    2. 市场总体数据: {market_data}
    3. 主要竞争对手数据: {competitor_data}
    
    提供以下洞察:
    - 市场份额对比与变化趋势
    - 价格定位与利润率分析
    - 产品差异化机会点
    - 战略建议与行动方案
    """
    
    # 调用DeepSeek R1执行分析
    messages = [
        SystemMessage(content="你是一个专业的市场分析师，擅长竞争情报分析"),
        HumanMessage(content=analysis_prompt)
    ]
    response = llm.invoke(messages)
    print(response)
    # 返回分析结果
    return response.content

def main():
    # 文件路径
    file_path = "sales_data.csv"  # 确保此文件存在于当前目录
    
    try:
        # 加载数据
        print("正在加载数据...")
        data = load_data(file_path)
        
        # 执行AI分析
        print("正在分析数据...")
        result = analyze_data(data)
        
        if result:
            # 调整月度趋势输出，确保按月份排序
            if "trend" in result:
                # 确保按月份排序
                sorted_trend = sorted(result["trend"], key=lambda x: x["month"])
                result["trend"] = sorted_trend
            
            # 输出结果
            print("\n📊 月度销售趋势:")
            for month_data in result['trend']:
                print(f"  {month_data['month']}: {month_data['sales']:,}元")
            
            print("\n🔄 产品销售占比:")
            for product, ratio in result['product_ratio'].items():
                print(f"  {product}: {ratio*100:.1f}%")
            
            # 生成可视化
            print("\n🎨 生成数据可视化...")
            render_plot(result["visual_code"])
            print("✅ 分析完成! 可视化结果已保存至 sales_analysis.png")
            
            # 添加竞争分析部分
            print("\n🔍 执行市场竞争分析...")
            try:
                # 这里可以加载真实的竞争对手数据文件
                # 或者使用现有数据模拟其他数据源
                # 提取我们自己的产品数据
                product_data = data[:min(10, len(data))]
                
                # 模拟市场数据和竞争对手数据
                # 在实际应用中，应该从单独的文件加载
                market_data = "日期,总销量,平均价格\n2024-01,15000,599\n2024-02,16500,589\n2024-03,18200,579"
                competitor_data = "日期,竞品A销量,竞品A价格,竞品B销量,竞品B价格\n2024-01,5200,549,4800,649\n2024-02,5500,539,5100,639"
                
                # 执行竞争分析
                competitive_result = competitive_analysis(product_data, market_data, competitor_data)
                
                # 显示分析结果
                print("\n🏆 市场竞争分析:")
                print(competitive_result)
                
                # 可选：将竞争分析结果保存到文件
                with open("competitive_analysis.txt", "w") as f:
                    f.write(competitive_result)
                print("✅ 竞争分析报告已保存至 competitive_analysis.txt")
            except Exception as e:
                print(f"⚠️ 竞争分析过程中出错: {str(e)}")
                
        else:
            print("❌ 分析失败，请检查数据格式或API配置")
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 '{file_path}'")
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")

if __name__ == "__main__":
    main()
