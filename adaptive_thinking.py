#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应思考生成器 - AI自主生成思考模板
让AI根据问题自然地生成思考过程，而不是使用固定模板
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from api import DeepSeekNumPy
import re

class AdaptiveThinkingGenerator:
    """自适应思考生成器，让AI自主生成思考过程。"""
    
    def __init__(self, model_dir: str):
        """初始化自适应思考生成器。"""
        print("🚀 初始化自适应思考生成器...")
        self.model = DeepSeekNumPy(model_dir)
        print("✅ 初始化完成！")
    
    def generate_thinking_prompt(self, user_question: str, max_tokens: int = 30):
        """让AI自己生成思考提示词。"""
        
        # 让AI生成思考框架的提示
        meta_prompt = f"""<｜User｜>用户问了这个问题："{user_question}"

请为这个问题设计一个思考框架。你需要生成一个简短的思考提示，告诉AI应该如何思考这个问题。

要求：
1. 不要直接回答问题
2. 只生成思考的方向和步骤
3. 保持简洁，不超过3-4个要点
4. 适合这个具体问题的特点

思考框架：<｜Assistant｜>针对问题"{user_question}"，我建议按以下方式思考：

"""
        
        print("🧠 AI正在生成思考框架...")
        thinking_framework = self.model.generate(
            meta_prompt,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=False
        )
        
        return thinking_framework.strip()
    
    def generate_with_adaptive_thinking(self, user_question: str, max_tokens: int = 50):
        """使用自适应思考模式生成回答。"""
        
        print(f"\n📝 用户问题: {user_question}")
        print("=" * 60)
        
        # 第一步：让AI生成思考框架
        thinking_framework = self.generate_thinking_prompt(user_question, max_tokens=25)
        print(f"\n🎯 AI生成的思考框架:\n{thinking_framework}")
        print("-" * 40)
        
        # 第二步：基于生成的框架进行思考和回答
        adaptive_prompt = f"""<｜User｜>{user_question}<｜Assistant｜><think>
{thinking_framework}

现在让我按照这个框架来思考：

</think>

基于我的思考，我的回答是："""
        
        print("🤔 AI正在按照自己的框架思考...")
        response = self.model.generate(
            adaptive_prompt,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
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
    
    def pure_adaptive_thinking(self, user_question: str, max_tokens: int = 60):
        """纯自适应思考模式 - 完全让AI自主决定如何思考。"""
        
        print(f"\n📝 用户问题: {user_question}")
        print("=" * 60)
        print("🧠 启用纯自适应思考模式...")
        
        # 完全开放的提示，让AI自主决定思考方式
        open_prompt = f"""<｜User｜>{user_question}<｜Assistant｜><think>
让我仔细思考这个问题...

</think>

"""
        
        response = self.model.generate(
            open_prompt,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=True
        )
        
        return response

def main():
    """主函数 - 自适应思考生成器。"""
    print("🎯 DeepSeek 自适应思考生成器")
    print("=" * 60)
    print("💡 AI将自主生成思考模板和过程")
    print("🧠 支持两种模式：")
    print("   1. 框架生成模式：AI先生成思考框架，再按框架思考")
    print("   2. 纯自适应模式：AI完全自主决定思考方式")
    print("⚡ 已优化生成速度")
    print("=" * 60)
    
    # 初始化生成器
    model_dir = "../DeepSeek-R1-Distill-Qwen-1.5B"
    generator = AdaptiveThinkingGenerator(model_dir)
    
    print("\n🎮 开始交互模式！")
    print("💬 请输入您的问题（输入 'quit' 或 'exit' 退出）：")
    print("🔧 输入 'mode' 切换思考模式")
    print("-" * 60)
    
    current_mode = "adaptive"  # adaptive 或 pure
    
    while True:
        try:
            # 获取用户输入
            user_input = input(f"\n❓ 您的问题 [{current_mode}模式]: ").strip()
            
            # 检查退出条件
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("\n👋 感谢使用！再见！")
                break
            
            # 切换模式
            if user_input.lower() == 'mode':
                current_mode = "pure" if current_mode == "adaptive" else "adaptive"
                mode_name = "纯自适应模式" if current_mode == "pure" else "框架生成模式"
                print(f"\n🔄 已切换到：{mode_name}")
                continue
            
            if not user_input:
                print("⚠️  请输入一个问题")
                continue
            
            # 根据模式生成回答
            print("\n🔄 正在生成回答...")
            
            if current_mode == "adaptive":
                response = generator.generate_with_adaptive_thinking(user_input, max_tokens=40)
            else:
                response = generator.pure_adaptive_thinking(user_input, max_tokens=50)
            
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