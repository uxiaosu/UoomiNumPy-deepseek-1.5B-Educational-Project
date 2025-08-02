#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Chat Example for UoomiNumPy deepseek Educational Project

This example demonstrates how to create an interactive chat session with the DeepSeek model.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from api import load_model
from optimization_config import OptimizationConfig, apply_optimization, get_memory_optimization_tips, get_speed_optimization_tips


class ChatSession:
    """Interactive chat session with DeepSeek model."""
    
    def __init__(self, model_dir: str, optimization_level: str = 'basic'):
        """Initialize chat session.
        
        Args:
            model_dir: Path to the model directory
            optimization_level: Optimization level ('basic', 'high_efficiency', 'quality', 'debug')
        """
        print("🤖 Loading DeepSeek model...")
        self.model = load_model(model_dir)
        self.conversation_history = []
        
        # 设置默认优化参数
        config_map = {
            'basic': OptimizationConfig.BASIC_OPTIMIZATION,
            'high_efficiency': OptimizationConfig.HIGH_EFFICIENCY,
            'quality': OptimizationConfig.QUALITY_OPTIMIZATION,
            'debug': OptimizationConfig.DEBUG_CONFIG
        }
        
        config = config_map.get(optimization_level, OptimizationConfig.BASIC_OPTIMIZATION)
        self.default_max_tokens = config['max_new_tokens']
        self.default_temperature = config['temperature']
        self.default_top_p = config['top_p']
        self.default_top_k = config['top_k']
        self.default_repetition_penalty = config['repetition_penalty']
        self.verbose = config['verbose']
        
        print(f"🔧 已应用 {optimization_level} 优化配置")
        print(f"📊 参数: max_tokens={self.default_max_tokens}, temp={self.default_temperature}, top_p={self.default_top_p}")
        print("✅ Model loaded successfully!")
        print("💬 Chat session started. Type 'quit', 'exit', or 'bye' to end.")
        print("🔧 Type 'help' for available commands.")
        print("=" * 60)
    
    def add_system_message(self, content: str):
        """Add a system message to set the assistant's behavior.
        
        Args:
            content: System message content
        """
        self.conversation_history.append({
            "role": "system",
            "content": content
        })
        print(f"🔧 System message added: {content}")
    
    def chat_turn(self, user_input: str, max_tokens: int = None, temperature: float = None) -> str:
        """Process one chat turn.
        
        Args:
            user_input: User's message
            max_tokens: Maximum tokens to generate (uses default if None)
            temperature: Sampling temperature (uses default if None)
            
        Returns:
            Assistant's response
        """
        # Use default values if not specified
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        if temperature is None:
            temperature = self.default_temperature
            
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate response
        try:
            response = self.model.chat(
                messages=self.conversation_history,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=self.default_top_p,
                top_k=self.default_top_k,
                verbose=self.verbose
            )
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            
            return response
            
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def show_history(self):
        """Display conversation history."""
        print("\n📜 Conversation History:")
        print("=" * 40)
        
        for i, message in enumerate(self.conversation_history, 1):
            role = message['role'].title()
            content = message['content']
            print(f"{i}. {role}: {content}")
        
        print("=" * 40)
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("🗑️  Conversation history cleared.")
    
    def show_help(self):
        """Show available commands."""
        help_text = """
🔧 Available Commands:
  help          - Show this help message
  history       - Show conversation history
  clear         - Clear conversation history
  system <msg>  - Add system message (e.g., 'system You are a helpful tutor')
  temp <value>  - Set temperature (e.g., 'temp 0.7')
  tokens <num>  - Set max tokens (e.g., 'tokens 150')
  info          - Show model information
  optimize <level> - Change optimization level (basic/high_efficiency/quality/debug)
  status        - Show current optimization settings
  tips          - Show optimization tips
  quit/exit/bye - End chat session

💡 Tips:
  - Use lower temperature (0.1-0.5) for more focused responses
  - Use higher temperature (0.8-1.2) for more creative responses
  - Adjust max tokens based on desired response length
        """
        print(help_text)
    
    def change_optimization(self, level: str):
        """Change optimization level.
        
        Args:
            level: New optimization level
        """
        config_map = {
            'basic': OptimizationConfig.BASIC_OPTIMIZATION,
            'high_efficiency': OptimizationConfig.HIGH_EFFICIENCY,
            'quality': OptimizationConfig.QUALITY_OPTIMIZATION,
            'debug': OptimizationConfig.DEBUG_CONFIG
        }
        
        if level not in config_map:
            print(f"❌ 未知的优化级别: {level}")
            print("可用级别: basic, high_efficiency, quality, debug")
            return
        
        config = config_map[level]
        self.default_max_tokens = config['max_new_tokens']
        self.default_temperature = config['temperature']
        self.default_top_p = config['top_p']
        self.default_top_k = config['top_k']
        self.default_repetition_penalty = config['repetition_penalty']
        self.verbose = config['verbose']
        
        print(f"✅ 已切换到 {level} 优化配置")
        print(f"📊 新参数: max_tokens={self.default_max_tokens}, temp={self.default_temperature}, top_p={self.default_top_p}")
    
    def show_status(self):
        """Show current optimization settings."""
        print("\n📊 当前优化设置:")
        print(f"  Max Tokens: {self.default_max_tokens}")
        print(f"  Temperature: {self.default_temperature}")
        print(f"  Top-P: {self.default_top_p}")
        print(f"  Top-K: {self.default_top_k}")
        print(f"  Repetition Penalty: {self.default_repetition_penalty}")
        print(f"  Verbose: {self.verbose}")
    
    def show_optimization_tips(self):
        """Show optimization tips."""
        print("\n" + get_memory_optimization_tips())
        print("\n" + get_speed_optimization_tips())
    
    def show_model_info(self):
        """Show model information."""
        info = self.model.get_model_info()
        print("\n📊 Model Information:")
        print(f"  Type: {info['model_type']}")
        print(f"  Parameters: {info['total_parameters']:,}")
        print(f"  Vocab Size: {info['config']['vocab_size']:,}")
        print(f"  Hidden Size: {info['config']['hidden_size']}")
        print(f"  Layers: {info['config']['num_hidden_layers']}")
    
    def run(self):
        """Run the interactive chat session."""
        # Use optimization defaults instead of hardcoded values
        temperature = self.default_temperature
        max_tokens = self.default_max_tokens
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Chat session ended.")
                    break
                
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                
                elif user_input.lower() == 'info':
                    self.show_model_info()
                    continue
                
                elif user_input.lower() == 'status':
                    self.show_status()
                    continue
                
                elif user_input.lower() == 'tips':
                    self.show_optimization_tips()
                    continue
                
                elif user_input.lower().startswith('optimize '):
                    level = user_input[9:].strip()
                    if level:
                        self.change_optimization(level)
                    else:
                        print("❌ Please provide an optimization level.")
                    continue
                
                elif user_input.lower().startswith('system '):
                    system_msg = user_input[7:].strip()
                    if system_msg:
                        self.add_system_message(system_msg)
                    else:
                        print("❌ Please provide a system message.")
                    continue
                
                elif user_input.lower().startswith('temp '):
                    try:
                        temp_value = float(user_input[5:].strip())
                        if 0.1 <= temp_value <= 2.0:
                            temperature = temp_value
                            print(f"🌡️  Temperature set to {temperature}")
                        else:
                            print("❌ Temperature should be between 0.1 and 2.0")
                    except ValueError:
                        print("❌ Invalid temperature value.")
                    continue
                
                elif user_input.lower().startswith('tokens '):
                    try:
                        token_value = int(user_input[7:].strip())
                        if 10 <= token_value <= 500:
                            max_tokens = token_value
                            print(f"🎯 Max tokens set to {max_tokens}")
                        else:
                            print("❌ Max tokens should be between 10 and 500")
                    except ValueError:
                        print("❌ Invalid token value.")
                    continue
                
                # Generate response
                print("🤖 Assistant: ", end="", flush=True)
                response = self.chat_turn(user_input, max_tokens, temperature)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print("Type 'help' for available commands or 'quit' to exit.")


def preset_conversations():
    """Run some preset conversation examples."""
    print("🎭 Preset Conversation Examples")
    print("=" * 50)
    
    model_dir = "../../DeepSeek-R1-Distill-Qwen-1.5B"
    
    # Example conversations
    conversations = [
        {
            "name": "Math Tutor",
            "system": "You are a helpful math tutor. Explain concepts clearly and provide step-by-step solutions.",
            "messages": [
                "What is the quadratic formula?",
                "Can you solve x² + 5x + 6 = 0?",
                "Explain why we use the discriminant."
            ]
        },
        {
            "name": "Creative Writer",
            "system": "You are a creative writing assistant. Help with storytelling, character development, and plot ideas.",
            "messages": [
                "Help me create a character for a sci-fi story.",
                "What's a good plot twist for a mystery novel?",
                "Describe a futuristic city in 50 words."
            ]
        },
        {
            "name": "Code Helper",
            "system": "You are a programming assistant. Help with coding questions and explain programming concepts.",
            "messages": [
                "Explain what a function is in programming.",
                "What's the difference between a list and a dictionary in Python?",
                "How do I handle errors in my code?"
            ]
        }
    ]
    
    try:
        model = load_model(model_dir)
        
        for conv in conversations:
            print(f"\n🎯 {conv['name']} Conversation:")
            print("-" * 40)
            
            # Start with system message
            messages = [{"role": "system", "content": conv['system']}]
            
            for user_msg in conv['messages']:
                print(f"\n👤 User: {user_msg}")
                
                # Add user message
                messages.append({"role": "user", "content": user_msg})
                
                # Generate response
                response = model.chat(
                    messages=messages,
                    max_new_tokens=80,
                    temperature=0.7,
                    verbose=False
                )
                
                print(f"🤖 Assistant: {response}")
                
                # Add assistant response
                messages.append({"role": "assistant", "content": response})
        
        print("\n✅ Preset conversations completed!")
        
    except Exception as e:
        print(f"❌ Error running preset conversations: {e}")


def main():
    """Main function to run chat examples."""
    print("💬 UoomiNumPy deepseek Model - Chat Examples")
    print("=" * 60)
    
    # Check command line arguments
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python chat_example.py <model_directory> [optimization_level]")
        print("Example: python chat_example.py ../DeepSeek-R1-Distill-Qwen-1.5B basic")
        print("Optimization levels: basic, high_efficiency, quality, debug")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    optimization_level = sys.argv[2] if len(sys.argv) == 3 else 'basic'
    
    print("Choose an option:")
    print("1. Interactive Chat Session")
    print("2. Preset Conversation Examples")
    print("3. Both")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            # Interactive chat
            chat = ChatSession(model_dir, optimization_level)
            chat.run()
            
        elif choice == "2":
            # Preset conversations
            preset_conversations()
            
        elif choice == "3":
            # Both
            preset_conversations()
            print("\n" + "=" * 60)
            print("Starting interactive chat session...")
            chat = ChatSession(model_dir, optimization_level)
            chat.run()
            
        else:
            print("❌ Invalid choice. Please run the script again.")
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please make sure the model directory path is correct.")
        print("Update the model_dir variable to point to your model.")
    
    except KeyboardInterrupt:
        print("\n👋 Program interrupted. Goodbye!")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()