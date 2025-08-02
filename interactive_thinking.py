#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式思考生成器 - 让用户自己输入问题验证动态思考功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from api import DeepSeekNumPy
import re

class InteractiveThinkingGenerator:
    """交互式思考生成器，支持用户自定义输入。"""
    
    def __init__(self, model_dir: str):
        """初始化交互式思考生成器。"""
        print("🚀 初始化交互式思考生成器...")
        self.model = DeepSeekNumPy(model_dir)
        print("✅ 初始化完成！")
        
    def _classify_question_type(self, prompt: str) -> str:
        """根据问题内容分类问题类型。"""
        prompt_lower = prompt.lower()
        
        # 数学问题
        math_keywords = ['计算', '解', '方程', '+', '-', '*', '/', '=', '数学', 'calculate', 'solve', 'math']
        if any(keyword in prompt_lower for keyword in math_keywords):
            return '数学问题'
        
        # 逻辑推理问题
        logic_keywords = ['逻辑', '推理', '如果', '那么', '因为', 'logic', 'reasoning', 'if', 'then']
        if any(keyword in prompt_lower for keyword in logic_keywords):
            return '逻辑推理'
        
        # 解释说明问题
        explain_keywords = ['解释', '什么是', '如何', '为什么', 'explain', 'what is', 'how', 'why']
        if any(keyword in prompt_lower for keyword in explain_keywords):
            return '解释说明'
        
        # 创意问题
        creative_keywords = ['创作', '写', '设计', '想象', 'create', 'write', 'design', 'imagine']
        if any(keyword in prompt_lower for keyword in creative_keywords):
            return '创意思考'
        
        # 分析问题
        analysis_keywords = ['分析', '比较', '评价', 'analyze', 'compare', 'evaluate']
        if any(keyword in prompt_lower for keyword in analysis_keywords):
            return '分析问题'
        
        return '一般问题'
    
    def _get_thinking_template(self, prompt: str, question_type: str) -> str:
        """根据问题类型生成思考模板。"""
        
        if question_type == '数学问题':
            return f"""<｜User｜>{prompt}<｜Assistant｜><think>
让我仔细分析这个数学问题：{prompt}

首先，我需要识别问题的类型和所涉及的数学概念。

步骤1：理解问题
- 问题要求什么？
- 给出了哪些已知条件？
- 需要使用什么数学方法？

步骤2：制定解题策略
- 这是什么类型的数学问题？
- 应该使用哪些公式或定理？
- 解题的逻辑顺序是什么？

步骤3：逐步求解
- 让我按照逻辑顺序一步步计算
- 每一步都要确保计算正确
- 检查中间结果是否合理

步骤4：验证答案
- 答案是否符合常理？
- 可以用其他方法验证吗？
- 单位和格式是否正确？
</think>

基于我的数学分析，让我来解决这个问题："""
        
        elif question_type == '逻辑推理':
            return f"""<｜User｜>{prompt}<｜Assistant｜><think>
这是一个逻辑推理问题：{prompt}

让我系统地分析这个逻辑问题：

步骤1：识别逻辑结构
- 前提条件是什么？
- 结论是什么？
- 逻辑关系是什么？

步骤2：分析推理过程
- 这个推理是否有效？
- 是否存在逻辑谬误？
- 前提是否支持结论？

步骤3：验证逻辑
- 用反例检验
- 考虑其他可能性
- 确认推理的严密性
</think>

通过逻辑分析，我的推理如下："""
        
        elif question_type == '解释说明':
            return f"""<｜User｜>{prompt}<｜Assistant｜><think>
用户想要了解：{prompt}

我需要提供清晰、准确的解释：

步骤1：理解核心概念
- 这个概念的本质是什么？
- 它的关键特征有哪些？
- 与其他概念的关系如何？

步骤2：组织解释结构
- 从简单到复杂
- 用具体例子说明
- 避免过于技术性的术语

步骤3：确保理解
- 解释是否清晰易懂？
- 是否涵盖了关键要点？
- 需要补充什么信息？
</think>

让我来详细解释这个问题："""
        
        else:  # 默认模板
            return f"""<｜User｜>{prompt}<｜Assistant｜><think>
让我仔细思考这个问题：{prompt}

步骤1：理解问题
- 用户真正想要什么？
- 问题的核心是什么？
- 需要考虑哪些方面？

步骤2：分析和思考
- 相关的知识和经验
- 可能的解决方案
- 需要注意的要点

步骤3：组织回答
- 如何清晰地表达？
- 重点应该放在哪里？
- 如何让回答更有帮助？
</think>

基于我的思考，我的回答是："""
    
    def generate_with_thinking(self, prompt: str, max_tokens: int = 50):
        """使用思考模式生成回答。"""
        
        # 分类问题类型
        question_type = self._classify_question_type(prompt)
        
        print(f"\n🧠 问题类型: {question_type}")
        print(f"📝 用户问题: {prompt}")
        print("=" * 60)
        print("🤔 开始思考过程...")
        
        # 获取思考模板
        thinking_template = self._get_thinking_template(prompt, question_type)
        
        # 生成回答
        response = self.model.generate(
            thinking_template,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            verbose=True
        )
        
        return response
    
    def extract_thinking_and_response(self, full_response: str):
        """从完整回答中提取思考过程和最终回答。"""
        
        # 查找 <think> 标签
        think_pattern = r'<think>(.*?)</think>'
        think_match = re.search(think_pattern, full_response, re.DOTALL)
        
        if think_match:
            thinking_process = think_match.group(1).strip()
            # 移除思考部分，获取最终回答
            final_response = re.sub(think_pattern, '', full_response, flags=re.DOTALL).strip()
        else:
            thinking_process = "未检测到思考过程"
            final_response = full_response
        
        return thinking_process, final_response

def main():
    """主函数 - 交互式思考生成器。"""
    print("🎯 DeepSeek 交互式思考生成器")
    print("=" * 60)
    print("💡 这个工具可以展示AI的思考过程")
    print("📋 支持的问题类型：数学、逻辑、解释、创意、分析等")
    print("⚡ 已优化生成速度（减少token数量）")
    print("=" * 60)
    
    # 初始化生成器
    model_dir = "../DeepSeek-R1-Distill-Qwen-1.5B"
    generator = InteractiveThinkingGenerator(model_dir)
    
    print("\n🎮 开始交互模式！")
    print("💬 请输入您的问题（输入 'quit' 或 'exit' 退出）：")
    print("-" * 60)
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n❓ 您的问题: ").strip()
            
            # 检查退出条件
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("\n👋 感谢使用！再见！")
                break
            
            if not user_input:
                print("⚠️  请输入一个问题")
                continue
            
            # 生成回答
            print("\n🔄 正在生成回答...")
            response = generator.generate_with_thinking(user_input, max_tokens=40)
            
            # 提取思考过程和最终回答
            thinking, final_answer = generator.extract_thinking_and_response(response)
            
            print("\n" + "=" * 60)
            print("🧠 AI的思考过程:")
            print("-" * 30)
            print(thinking)
            
            print("\n💡 最终回答:")
            print("-" * 30)
            print(final_answer)
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出程序")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            print("请重试或输入新问题")

if __name__ == "__main__":
    main()