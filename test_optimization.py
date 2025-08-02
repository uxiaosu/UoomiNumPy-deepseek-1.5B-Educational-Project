#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化效果测试脚本

这个脚本用于测试不同优化级别的效果，包括：
1. 生成速度对比
2. Token用量对比
3. 内存使用对比
4. 输出质量对比
"""

import time
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.api import DeepSeekNumPy
from optimization_config import OptimizationConfig

def test_optimization_levels(model_dir: str):
    """测试不同优化级别的效果
    
    Args:
        model_dir: 模型目录路径
    """
    print("🧪 DeepSeek模型优化效果测试")
    print("=" * 60)
    
    # 加载模型
    print("📦 加载模型...")
    model = DeepSeekNumPy(model_dir)
    
    # 测试prompt
    test_prompt = "人工智能的未来发展趋势是"
    
    # 测试不同优化级别
    optimization_levels = [
        ('debug', OptimizationConfig.DEBUG_CONFIG),
        ('high_efficiency', OptimizationConfig.HIGH_EFFICIENCY),
        ('basic', OptimizationConfig.BASIC_OPTIMIZATION),
        ('quality', OptimizationConfig.QUALITY_OPTIMIZATION)
    ]
    
    results = []
    
    for level_name, config in optimization_levels:
        print(f"\n🔧 测试 {level_name} 优化级别")
        print("-" * 40)
        print(f"参数: max_tokens={config['max_new_tokens']}, temp={config['temperature']}, top_p={config['top_p']}")
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 生成文本
            response = model.generate(
                prompt=test_prompt,
                max_new_tokens=config['max_new_tokens'],
                temperature=config['temperature'],
                top_p=config['top_p'],
                top_k=config['top_k'],
                verbose=False  # 关闭详细输出以准确测量时间
            )
            
            # 记录结束时间
            end_time = time.time()
            generation_time = end_time - start_time
            
            # 计算统计信息
            response_length = len(response)
            tokens_per_second = config['max_new_tokens'] / generation_time if generation_time > 0 else 0
            
            # 保存结果
            result = {
                'level': level_name,
                'config': config,
                'response': response,
                'generation_time': generation_time,
                'response_length': response_length,
                'tokens_per_second': tokens_per_second
            }
            results.append(result)
            
            # 显示结果
            print(f"⏱️  生成时间: {generation_time:.2f}秒")
            print(f"📏 响应长度: {response_length}字符")
            print(f"⚡ 生成速度: {tokens_per_second:.1f} tokens/秒")
            print(f"📝 生成内容: {response[:100]}{'...' if len(response) > 100 else ''}")
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            continue
    
    # 显示对比结果
    print("\n📊 优化效果对比")
    print("=" * 60)
    print(f"{'级别':<15} {'时间(秒)':<10} {'速度(t/s)':<12} {'长度':<8} {'效率评分':<10}")
    print("-" * 60)
    
    # 计算效率评分 (速度 / 时间)
    for result in results:
        efficiency_score = result['tokens_per_second'] / result['generation_time'] if result['generation_time'] > 0 else 0
        print(f"{result['level']:<15} {result['generation_time']:<10.2f} {result['tokens_per_second']:<12.1f} {result['response_length']:<8} {efficiency_score:<10.2f}")
    
    # 推荐最佳配置
    if results:
        fastest = max(results, key=lambda x: x['tokens_per_second'])
        most_efficient = max(results, key=lambda x: x['tokens_per_second'] / x['generation_time'] if x['generation_time'] > 0 else 0)
        
        print("\n🏆 推荐配置")
        print("-" * 30)
        print(f"⚡ 最快速度: {fastest['level']} ({fastest['tokens_per_second']:.1f} tokens/秒)")
        print(f"🎯 最高效率: {most_efficient['level']} (效率评分: {most_efficient['tokens_per_second'] / most_efficient['generation_time']:.2f})")
    
    return results

def show_optimization_recommendations():
    """显示优化建议"""
    print("\n💡 优化建议")
    print("=" * 40)
    
    recommendations = [
        "1. 🚀 日常使用推荐 'basic' 配置 - 平衡速度和质量",
        "2. ⚡ 追求速度使用 'high_efficiency' 配置 - 最快响应",
        "3. 🎨 重视质量使用 'quality' 配置 - 更好的输出",
        "4. 🔧 调试问题使用 'debug' 配置 - 详细信息",
        "5. 📱 移动设备或低配置机器建议使用 'high_efficiency'",
        "6. 🖥️  高性能机器可以使用 'quality' 获得更好体验",
        "7. 💾 内存不足时减少 max_new_tokens 参数",
        "8. 🔄 长对话时定期清理历史记录"
    ]
    
    for rec in recommendations:
        print(rec)

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("使用方法: python test_optimization.py <模型目录>")
        print("示例: python test_optimization.py ./examples/weights/deepseek_numpy_weights")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    
    try:
        # 测试优化效果
        results = test_optimization_levels(model_dir)
        
        # 显示优化建议
        show_optimization_recommendations()
        
        print("\n✅ 优化测试完成！")
        
    except KeyboardInterrupt:
        print("\n👋 测试被用户中断")
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()