#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thinking Generator - Enhanced version with DeepSeek-R1 thinking support
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from api import DeepSeekNumPy
import re

class ThinkingGenerator:
    """Enhanced generator with DeepSeek-R1 thinking capabilities."""
    
    def __init__(self, model_dir: str):
        """Initialize the thinking generator."""
        self.model = DeepSeekNumPy(model_dir)
    
    def generate_with_thinking(self, prompt: str, max_tokens: int = 100, verbose: bool = True):
        """Generate response with explicit thinking process."""
        
        # Format prompt to encourage thinking
        thinking_prompt = f"""<｜User｜>{prompt}<｜Assistant｜><think>
Let me think about this step by step.

The user is asking: {prompt}

I need to:
1. Understand what they're asking
2. Think through the problem
3. Provide a clear answer

Let me work through this:
</think>

Based on my thinking, here's my response:
"""
        
        if verbose:
            print("🧠 启用思考模式生成...")
            print(f"📝 原始问题: {prompt}")
            print("=" * 60)
        
        # Generate with thinking prompt
        response = self.model.generate(
            thinking_prompt,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            verbose=verbose
        )
        
        return response
    
    def chat_with_thinking(self, user_message: str, max_tokens: int = 150, verbose: bool = True):
        """Chat with explicit thinking process."""
        
        messages = [
            {
                "role": "system",
                "content": "You are DeepSeek-R1, an AI assistant that thinks step by step before responding. Always show your thinking process in <think> tags before giving your final answer."
            },
            {
                "role": "user", 
                "content": user_message
            }
        ]
        
        if verbose:
            print("💭 启用聊天思考模式...")
            print(f"👤 用户: {user_message}")
            print("=" * 60)
        
        # Use chat format with thinking
        response = self.model.chat(
            messages,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            verbose=verbose
        )
        
        return response
    
    def extract_thinking_and_response(self, full_response: str):
        """Extract thinking process and final response."""
        
        # Look for thinking patterns
        thinking_patterns = [
            r'<think>(.*?)</think>',
            r'思考过程[：:](.*?)(?=\n\n|$)',
            r'让我想想[：:](.*?)(?=\n\n|$)'
        ]
        
        thinking = ""
        response = full_response
        
        for pattern in thinking_patterns:
            match = re.search(pattern, full_response, re.DOTALL | re.IGNORECASE)
            if match:
                thinking = match.group(1).strip()
                response = full_response.replace(match.group(0), '').strip()
                break
        
        return thinking, response

def test_thinking_generation():
    """Test the thinking generation capabilities."""
    print("🧠 DeepSeek-R1 思考功能测试")
    print("=" * 60)
    
    # Initialize thinking generator
    model_dir = "../DeepSeek-R1-Distill-Qwen-1.5B"
    generator = ThinkingGenerator(model_dir)
    
    # Test questions that should trigger thinking
    test_questions = [
        "What is 2+2? Please think step by step.",
        "How do you solve a quadratic equation?",
        "Explain the concept of gravity in simple terms."
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 测试 {i}: {question}")
        print("=" * 50)
        
        try:
            # Test thinking generation
            response = generator.generate_with_thinking(
                question, 
                max_tokens=80, 
                verbose=True
            )
            
            print(f"\n🤖 完整回应:")
            print(f"'{response}'")
            
            # Extract thinking and response
            thinking, final_response = generator.extract_thinking_and_response(response)
            
            if thinking:
                print(f"\n🧠 思考过程:")
                print(f"'{thinking}'")
                print(f"\n💬 最终回答:")
                print(f"'{final_response}'")
            else:
                print("\n⚠️ 未检测到明确的思考过程")
            
            print("\n" + "=" * 50)
            
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    # Test chat with thinking
    print("\n\n💬 聊天思考模式测试")
    print("=" * 60)
    
    try:
        chat_response = generator.chat_with_thinking(
            "请解释什么是人工智能，并说明你的思考过程。",
            max_tokens=100,
            verbose=True
        )
        
        print(f"\n🤖 聊天回应:")
        print(f"'{chat_response}'")
        
    except Exception as e:
        print(f"❌ 聊天错误: {e}")

if __name__ == "__main__":
    test_thinking_generation()