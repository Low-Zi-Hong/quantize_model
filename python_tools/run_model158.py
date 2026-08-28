import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open

def unpack_158bit(packed_weight, gamma):
    """
    将 2-bit 压缩的 uint8 矩阵解包回 FP16 的 -1, 0, 1，并乘上 gamma
    编码: 0→0, 1→+1, 2→-1
    顺序: 低2位是第0个数，高2位是第3个数 (对应 pack_multiplier = [1,4,16,64])
    """
    out_features, packed_in_features = packed_weight.shape

    def decode(bits):
        # bits: uint8 tensor, values in {0,1,2,3}
        val = torch.zeros_like(bits, dtype=torch.int8)
        val[bits == 1] = 1
        val[bits == 2] = -1
        # bits == 0 保持 0，bits == 3 理论不该出现，也保持 0
        return val

    w0 = decode((packed_weight)       & 0b11)   # 低位 = 第0个数
    w1 = decode((packed_weight >> 2)  & 0b11)
    w2 = decode((packed_weight >> 4)  & 0b11)
    w3 = decode((packed_weight >> 6)  & 0b11)   # 高位 = 第3个数

    unpacked_ternary = torch.stack([w0, w1, w2, w3], dim=-1).view(out_features, -1).to(torch.float16)

    return unpacked_ternary * gamma.to(torch.float16)

def unpack_4bit_emb(packed_weight, scales):
    """
    将 4-bit 压缩的 uint8 词表解包回 FP16
    """
    vocab_size, packed_hidden = packed_weight.shape
    
    # 提取高 4 位和低 4 位，减去 8 (恢复为 -8 到 7)
    w0 = ((packed_weight >> 4) & 0b1111).to(torch.int8) - 8
    w1 = ((packed_weight) & 0b1111).to(torch.int8) - 8
    
    # 恢复维度 [vocab_size, hidden_size]
    unpacked_q = torch.stack([w0, w1], dim=-1).view(vocab_size, -1).to(torch.float16)
    
    # 乘上每行的缩放系数 (广播机制)
    return unpacked_q * scales.to(torch.float16)

def load_and_generate(
    original_model_path="../cropped_Qwen",
    safetensors_path="../cropped_Qwen/qwen_158.safetensors"
):
    print("⏳ 正在加载原始架构...")
    # 先加载原始的空壳架构
    tokenizer = AutoTokenizer.from_pretrained(original_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        original_model_path, 
        torch_dtype=torch.float16, 
        device_map="cpu" # 验证解包逻辑，用 CPU 即可
    )
    
    print(f"🔓 正在从 {safetensors_path} 解包注入全损权重...")
    
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        # 获取 safetensors 里所有的 key
        keys = f.keys()
        
        # 1. 恢复 4-bit Embedding
        if "model.embed_tokens.weight_packed_4bit" in keys:
            packed_emb = f.get_tensor("model.embed_tokens.weight_packed_4bit")
            scales = f.get_tensor("model.embed_tokens.scales")
            unpacked_emb = unpack_4bit_emb(packed_emb, scales)
            
            # 暴力覆盖原生模型的 Embedding
            model.model.embed_tokens.weight.data.copy_(unpacked_emb)
            
            # ✨ 核心操作：由于我们删除了 LM Head，这里将输出层直接绑定到解包后的 Embedding
            model.lm_head.weight = model.model.embed_tokens.weight
            print("✅ 4-bit 词表已解包，并成功挂载到 LM Head！")
            
        # 2. 遍历恢复所有 1.58-bit (2-bit packed) 的 Attention 和 MLP 层
        for name, module in model.named_modules():
            packed_key = f"{name}.weight_packed"
            gamma_key = f"{name}.gamma"
            bias_key = f"{name}.bias"
            weight_key = f"{name}.weight" # 原生 FP16 层
            
            # 如果是量化层
            if packed_key in keys:
                packed_w = f.get_tensor(packed_key)
                gamma = f.get_tensor(gamma_key)
                
                # 执行 2-bit 逆向解包
                unpacked_w = unpack_158bit(packed_w, gamma)
                module.weight.data.copy_(unpacked_w)
                
                if getattr(module, 'bias', None) is not None and bias_key in keys:
                    module.bias.data.copy_(f.get_tensor(bias_key))
            
            # 如果是原生保留的 FP16 层 (比如 LayerNorm)
            elif weight_key in keys and "embed_tokens" not in name and "lm_head" not in name:
                module.weight.data.copy_(f.get_tensor(weight_key))
                if getattr(module, 'bias', None) is not None and bias_key in keys:
                    module.bias.data.copy_(f.get_tensor(bias_key))

    print("🚀 解包完成！模型准备就绪，开启全损测试...\n")
    print("="*50)
    
    # 将模型放入 GPU 加速推理 (如果有的话)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # 测试 Prompt
    prompt = "Q: Who are you?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=30, 
            do_sample=False,        # 关闭采样，使用严格的 Argmax
            temperature=None,       # 贪心搜索不需要 temperature
            top_p=None              # 关闭 Top-P
            # 移除 repetition_penalty
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"用户: {prompt}")
    print(f"全损模型: {response}")
    print("="*50)

if __name__ == "__main__":
    load_and_generate()