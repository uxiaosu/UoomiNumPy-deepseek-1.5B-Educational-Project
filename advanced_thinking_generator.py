#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Thinking Generator - 改进版思考生成器
实现更复杂和真实的思考过程
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from api import DeepSeekNumPy
import re
import random

class AdvancedThinkingGenerator:
    """改进版思考生成器，支持动态思考模式和多步推理。"""
    
    def __init__(self, model_dir: str):
        """初始化改进版思考生成器。"""
        self.model = DeepSeekNumPy(model_dir)
        
        # 定义不同类型问题的思考模板
        self.thinking_templates = {
            'math': self._get_math_thinking_template,
            'logic': self._get_logic_thinking_template,
            'explanation': self._get_explanation_thinking_template,
            'creative': self._get_creative_thinking_template,
            'analysis': self._get_analysis_thinking_template,
            'default': self._get_default_thinking_template
        }
    
    def _classify_question_type(self, prompt: str) -> str:
        """根据问题内容分类问题类型。"""
        prompt_lower = prompt.lower()
        
        # 数学问题
        math_keywords = ['calculate', 'solve', 'equation', 'math', '计算', '解', '方程', '+', '-', '*', '/', '=', '数学']
        if any(keyword in prompt_lower for keyword in math_keywords):
            return 'math'
        
        # 逻辑推理问题
        logic_keywords = ['logic', 'reasoning', 'if', 'then', 'because', '逻辑', '推理', '如果', '那么', '因为']
        if any(keyword in prompt_lower for keyword in logic_keywords):
            return 'logic'
        
        # 解释说明问题
        explain_keywords = ['explain', 'what is', 'how does', 'why', '解释', '什么是', '如何', '为什么']
        if any(keyword in prompt_lower for keyword in explain_keywords):
            return 'explanation'
        
        # 创意问题
        creative_keywords = ['create', 'design', 'imagine', 'story', '创造', '设计', '想象', '故事']
        if any(keyword in prompt_lower for keyword in creative_keywords):
            return 'creative'
        
        # 分析问题
        analysis_keywords = ['analyze', 'compare', 'evaluate', '分析', '比较', '评估']
        if any(keyword in prompt_lower for keyword in analysis_keywords):
            return 'analysis'
        
        return 'default'
    
    def _get_math_thinking_template(self, prompt: str) -> str:
        """数学问题的思考模板。"""
        return f"""<think>
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
    
    def _get_logic_thinking_template(self, prompt: str) -> str:
        """逻辑推理问题的思考模板。"""
        return f"""<think>
这是一个逻辑推理问题：{prompt}

我需要运用逻辑思维来分析这个问题。

逻辑分析框架：
1. 前提识别
   - 题目给出了哪些前提条件？
   - 这些前提是否可靠？
   - 有没有隐含的假设？

2. 推理过程
   - 从前提可以得出什么结论？
   - 推理链条是否完整？
   - 有没有逻辑漏洞？

3. 多角度思考
   - 是否存在其他可能的解释？
   - 反例是什么？
   - 边界情况如何处理？

4. 结论验证
   - 结论是否与前提一致？
   - 逻辑是否严密？
   - 是否考虑了所有情况？
</think>

通过逻辑推理，我的分析如下："""
    
    def _get_explanation_thinking_template(self, prompt: str) -> str:
        """解释说明问题的思考模板。"""
        return f"""<think>
用户想要我解释：{prompt}

为了给出清晰的解释，我需要：

1. 概念分解
   - 这个概念的核心是什么？
   - 包含哪些关键要素？
   - 与其他概念的关系是什么？

2. 层次化解释
   - 从简单到复杂
   - 从具体到抽象
   - 从现象到本质

3. 举例说明
   - 什么样的例子最能说明问题？
   - 如何用类比让人更容易理解？
   - 反例能帮助澄清概念吗？

4. 实际应用
   - 这个概念在现实中如何体现？
   - 为什么这个概念重要？
   - 如何运用这个知识？
</think>

让我来详细解释这个概念："""
    
    def _get_creative_thinking_template(self, prompt: str) -> str:
        """创意问题的思考模板。"""
        return f"""<think>
这是一个需要创造性思维的问题：{prompt}

创意思考过程：

1. 发散思维
   - 有哪些可能的方向？
   - 不同的角度会产生什么想法？
   - 如何突破常规思维？

2. 灵感整合
   - 哪些想法最有趣？
   - 如何将不同元素结合？
   - 什么样的组合最有创意？

3. 可行性评估
   - 这些想法现实吗？
   - 需要什么条件来实现？
   - 如何优化和改进？

4. 创意表达
   - 如何生动地表达这个想法？
   - 什么样的形式最吸引人？
   - 如何让创意更有感染力？
</think>

发挥创意，我的想法是："""
    
    def _get_analysis_thinking_template(self, prompt: str) -> str:
        """分析问题的思考模板。"""
        return f"""<think>
需要分析的问题是：{prompt}

系统性分析方法：

1. 问题拆解
   - 这个问题包含哪些子问题？
   - 各部分之间的关系是什么？
   - 哪些是关键因素？

2. 多维度分析
   - 从不同角度看有什么发现？
   - 优势和劣势分别是什么？
   - 机会和威胁在哪里？

3. 深层挖掘
   - 表面现象背后的原因是什么？
   - 有什么潜在的影响？
   - 长期趋势如何？

4. 综合判断
   - 权衡各种因素后的结论是什么？
   - 不确定性在哪里？
   - 需要进一步了解什么？
</think>

经过深入分析，我的观点是："""
    
    def _get_default_thinking_template(self, prompt: str) -> str:
        """默认思考模板，适用于一般问题。"""
        thinking_styles = [
            f"""<think>
面对这个问题：{prompt}

我需要仔细思考一下。

首先，让我理解问题的核心：
- 用户真正想知道什么？
- 这个问题的背景和上下文是什么？
- 有哪些重要的细节需要注意？

然后，我来分析可能的答案：
- 有哪些不同的观点或方法？
- 每种方法的优缺点是什么？
- 哪种答案最准确和有用？

最后，我要确保回答的质量：
- 信息是否准确可靠？
- 解释是否清晰易懂？
- 是否回答了用户的真正需求？
</think>

经过思考，我的回答是：""",
            
            f"""<think>
让我深入思考这个问题：{prompt}

思考路径：

第一层思考 - 问题理解
我需要确保完全理解了问题的含义和用户的期望。

第二层思考 - 知识调用
相关的知识和经验有哪些？如何组织这些信息？

第三层思考 - 逻辑构建
如何用逻辑清晰的方式来回答这个问题？

第四层思考 - 答案优化
如何让答案更准确、更有用、更容易理解？
</think>

基于我的深入思考："""
        ]
        
        return random.choice(thinking_styles)
    
    def generate_with_advanced_thinking(self, prompt: str, max_tokens: int = 50, verbose: bool = True):
        """使用改进的思考模式生成回答。"""
        
        # 分类问题类型
        question_type = self._classify_question_type(prompt)
        
        # 获取对应的思考模板
        thinking_template = self.thinking_templates[question_type](prompt)
        
        if verbose:
            print(f"🧠 启用高级思考模式 (类型: {question_type})")
            print(f"📝 问题: {prompt}")
            print("=" * 60)
        
        # 生成回答
        response = self.model.generate(
            thinking_template,
            max_new_tokens=max_tokens,
            temperature=0.8,  # 稍高的温度增加创造性
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.15,
            verbose=verbose
        )
        
        return response
    
    def multi_step_thinking(self, prompt: str, steps: int = 3, max_tokens_per_step: int = 30, verbose: bool = True):
        """多步骤思考，逐步深入分析问题。"""
        
        if verbose:
            print(f"🔄 启用多步骤思考模式 ({steps}步)")
            print(f"📝 问题: {prompt}")
            print("=" * 60)
        
        accumulated_thinking = ""
        
        for step in range(1, steps + 1):
            if step == 1:
                step_prompt = f"""<｜User｜>{prompt}<｜Assistant｜><think>
这是一个需要深入思考的问题。让我分步骤来分析：

第{step}步思考 - 初步理解：
让我先理解这个问题的基本含义和要求...
</think>

第{step}步分析："""
            else:
                step_prompt = f"""继续深入思考这个问题：{prompt}

前面的思考：{accumulated_thinking}

<think>
第{step}步思考 - 深入分析：
基于前面的思考，我需要进一步分析...
</think>

第{step}步分析："""
            
            if verbose:
                print(f"\n🔍 第{step}步思考中...")
            
            step_response = self.model.generate(
                step_prompt,
                max_new_tokens=max_tokens_per_step,
                temperature=0.7,
                top_p=0.9,
                verbose=False
            )
            
            accumulated_thinking += f"\n第{step}步：{step_response}"
            
            if verbose:
                print(f"✅ 第{step}步完成")
        
        # 最终综合
        final_prompt = f"""基于多步骤的深入思考：{accumulated_thinking}

<think>
综合思考：
现在我已经从多个角度分析了这个问题，让我整合所有的思考，给出最终的完整回答...
</think>

综合以上分析，我的最终回答是："""
        
        if verbose:
            print(f"\n🎯 生成最终综合回答...")
        
        final_response = self.model.generate(
            final_prompt,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            verbose=verbose
        )
        
        return {
            'steps': accumulated_thinking,
            'final_answer': final_response,
            'full_process': accumulated_thinking + "\n\n" + final_response
        }

def test_advanced_thinking():
    """测试改进版思考功能。"""
    print("🚀 高级思考功能测试")
    print("=" * 60)
    
    # 初始化改进版生成器
    model_dir = "../DeepSeek-R1-Distill-Qwen-1.5B"
    generator = AdvancedThinkingGenerator(model_dir)
    
    # 测试不同类型的问题
    test_cases = [
        {
            'question': "计算 15 × 23 + 47，请详细说明计算过程",
            'type': '数学问题'
        },
        {
            'question': "解释什么是人工智能，它是如何工作的？",
            'type': '解释问题'
        },
        {
            'question': "如果所有的鸟都会飞，企鹅是鸟，那么企鹅会飞吗？请分析这个逻辑问题",
            'type': '逻辑问题'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 测试 {i}: {test_case['type']}")
        print(f"❓ 问题: {test_case['question']}")
        print("-" * 50)
        
        try:
            # 测试高级思考模式
            response = generator.generate_with_advanced_thinking(
                test_case['question'],
                max_tokens=40,
                verbose=True
            )
            
            print(f"\n🤖 回答: {response}")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    # 测试多步骤思考
    print("\n\n🔄 多步骤思考测试")
    print("=" * 60)
    
    try:
        multi_step_question = "分析一下为什么有些人学习编程很困难？"
        result = generator.multi_step_thinking(
            multi_step_question,
            steps=2,
            max_tokens_per_step=25,
            verbose=True
        )
        
        print(f"\n📊 完整思考过程:")
        print(result['full_process'])
        
    except Exception as e:
        print(f"❌ 多步骤思考错误: {e}")

if __name__ == "__main__":
    test_advanced_thinking()