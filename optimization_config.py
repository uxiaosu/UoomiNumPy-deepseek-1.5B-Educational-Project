#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化配置文件 - 用于减少token用量和提高推理效率

这个文件包含了多种优化策略：
1. 减少生成长度
2. 优化采样参数
3. 提高推理效率
4. 减少内存占用
"""

class OptimizationConfig:
    """优化配置类"""
    
    # 基础优化配置 - 适合日常使用
    BASIC_OPTIMIZATION = {
        'max_new_tokens': 10,         # 极大减少生成长度，只生成5个token
        'temperature': 0.7,          # 稍微降低随机性
        'top_p': 0.8,               # 减少候选token数量
        'top_k': 30,                # 进一步限制候选
        'repetition_penalty': 1.1,   # 减少重复
        'verbose': True              # 显示详细推理过程
    }
    
    # 高效配置 - 最大化速度
    HIGH_EFFICIENCY = {
        'max_new_tokens': 20,        # 更短的生成
        'temperature': 0.6,          # 更确定的输出
        'top_p': 0.7,               # 更少的候选
        'top_k': 20,                # 严格限制
        'repetition_penalty': 1.2,   # 强制避免重复
        'verbose': False
    }
    
    # 质量优化配置 - 平衡质量和效率
    QUALITY_OPTIMIZATION = {
        'max_new_tokens': 50,        # 适中的长度
        'temperature': 0.8,          # 保持创造性
        'top_p': 0.9,               # 较好的多样性
        'top_k': 40,                # 合理的候选数
        'repetition_penalty': 1.05,  # 轻微避免重复
        'verbose': False
    }
    
    # 调试配置 - 用于问题诊断
    DEBUG_CONFIG = {
        'max_new_tokens': 10,        # 最短生成用于测试
        'temperature': 0.5,          # 确定性输出
        'top_p': 0.6,               # 限制候选
        'top_k': 10,                # 最少候选
        'repetition_penalty': 1.0,   # 无重复惩罚
        'verbose': True              # 开启详细输出用于调试
    }

def apply_optimization(chat_session, optimization_level='basic'):
    """应用优化配置到聊天会话
    
    Args:
        chat_session: ChatSession实例
        optimization_level: 优化级别 ('basic', 'high_efficiency', 'quality', 'debug')
    """
    config_map = {
        'basic': OptimizationConfig.BASIC_OPTIMIZATION,
        'high_efficiency': OptimizationConfig.HIGH_EFFICIENCY,
        'quality': OptimizationConfig.QUALITY_OPTIMIZATION,
        'debug': OptimizationConfig.DEBUG_CONFIG
    }
    
    if optimization_level not in config_map:
        raise ValueError(f"未知的优化级别: {optimization_level}")
    
    config = config_map[optimization_level]
    
    # 更新聊天会话的默认参数
    chat_session.default_max_tokens = config['max_new_tokens']
    chat_session.default_temperature = config['temperature']
    chat_session.default_top_p = config['top_p']
    chat_session.default_top_k = config['top_k']
    chat_session.default_repetition_penalty = config['repetition_penalty']
    chat_session.verbose = config['verbose']
    
    print(f"✅ 已应用 {optimization_level} 优化配置")
    print(f"📊 参数: max_tokens={config['max_new_tokens']}, temp={config['temperature']}, top_p={config['top_p']}")

def get_memory_optimization_tips():
    """获取内存优化建议"""
    tips = [
        "💡 内存优化建议:",
        "1. 减少max_new_tokens可显著降低内存使用",
        "2. 关闭verbose模式减少输出开销",
        "3. 使用较小的top_k值减少计算量",
        "4. 定期清理对话历史",
        "5. 避免过长的输入prompt"
    ]
    return "\n".join(tips)

def get_speed_optimization_tips():
    """获取速度优化建议"""
    tips = [
        "⚡ 速度优化建议:",
        "1. 使用high_efficiency配置获得最快速度",
        "2. 降低temperature提高确定性，减少采样时间",
        "3. 减小top_p和top_k值限制候选token数量",
        "4. 使用较短的prompt和生成长度",
        "5. 关闭详细输出模式"
    ]
    return "\n".join(tips)

if __name__ == "__main__":
    print("🔧 DeepSeek模型优化配置")
    print("=" * 40)
    print(get_memory_optimization_tips())
    print("\n")
    print(get_speed_optimization_tips())