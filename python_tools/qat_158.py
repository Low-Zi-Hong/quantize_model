import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
import os

block_size = 512              # 你的 ESP32-S3 上下文窗口长度
batch_size = 4                # 训练批次大小

class BitNet158STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        eps = 1e-8

        gamma = weight.abs().mean()

        weight_scaled = weight / (gamma + eps)

        weight_clip = torch.clamp(torch.round(weight_scaled),min = -1.0, max = 1.0)

        weight_quant = weight_clip * gamma

        ctx.save_for_backward(weight_scaled)

        return weight_quant

    @staticmethod
    def backward(ctx, grad_output):
        weight_scaled, = ctx.saved_tensors

        grad_weight = grad_output.clone()

        grad_weight[weight_scaled > 1.0] = 0.0

        grad_weight[weight_scaled < -1.0 ] = 0.0

        return grad_weight

class BitNetLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        
    def forward(self, x):
        # 将原始的浮点权重 (self.weight) 送入 STE 算子
        # 吐出 1.58-bit 的量化权重 (w_quant) 参与实际计算
        w_quant = BitNet158STE.apply(self.weight)
        return F.linear(x, w_quant, self.bias)

def replace_linear_with_bitnet(model, target_modules=["gate_proj", "up_proj", "down_proj"]):
    """
    递归遍历模型，将目标名称的 nn.Linear 替换为 BitNetLinear
    """
    for name, module in model.named_children():
        # 检查当前模块是否是我们要替换的目标，并且是 nn.Linear
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            
            # 1. 记录原层的形状和配置
            in_features = module.in_features
            out_features = module.out_features
            has_bias = module.bias is not None
            
            # 2. 实例化你写的 QAT 层
            new_module = BitNetLinear(in_features, out_features, bias=has_bias)
            
            # 3. 灵魂转移：将 Qwen 的原始 FP32/FP16 权重拷贝给 QAT 层作为幕后真身
            new_module.weight.data = module.weight.data.clone()
            if has_bias:
                new_module.bias.data = module.bias.data.clone()
                
            # 4. 执行替换
            setattr(model, name, new_module)
            print(f"Replaced {name} with BitNetLinear")
            
        else:
            # 递归进入下一层
            replace_linear_with_bitnet(module, target_modules)
            
    return model

def export_158bit_safetensors(model, export_path="qwen_158bit_packed.safetensors"):
    model.eval() 
    export_dict = {}
    
    # 强制在 CPU 上进行打包计算，避免显存溢出或设备不匹配问题
    pack_multiplier = torch.tensor([1, 4, 16, 64], dtype=torch.uint8, device="cpu")
    
    with torch.no_grad():
        for name, module in model.named_modules():
            
            if isinstance(module, BitNetLinear):
                # 将权重移动到 CPU 进行处理
                weight = module.weight.data.cpu()
                
                gamma = weight.abs().mean()
                weight_scaled = weight / (gamma + 1e-8)
                weight_ternary = torch.clamp(torch.round(weight_scaled), min=-1.0, max=1.0)   # ← 补回这行

                weight_mapped = torch.zeros_like(weight_ternary, dtype=torch.uint8)
                weight_mapped[weight_ternary == 1]  = 1
                weight_mapped[weight_ternary == -1] = 2
                
                out_features, in_features = weight_mapped.shape
                # ESP32-S3 的内存对齐通常要求维度是偶数，4的倍数是最好的
                assert in_features % 4 == 0, f"输入特征维度 {in_features} 必须能被 4 整除才能打包"
                
                weight_reshaped = weight_mapped.view(out_features, in_features // 4, 4)
                
                # 在 CPU 上完成张量乘法和求和打包
                weight_packed = (weight_reshaped * pack_multiplier).sum(dim=-1).to(torch.uint8)
                
                export_dict[f"{name}.weight_packed"] = weight_packed
                # 显式转换为 float16 供单片机读取
                export_dict[f"{name}.gamma"] = gamma.to(torch.float16)
                
                if module.bias is not None:
                    export_dict[f"{name}.bias"] = module.bias.data.cpu().to(torch.float16)
                
                print(f"✅ 打包成功: {name} | 原始尺寸: {weight.shape} -> 压缩尺寸: {weight_packed.shape}")
                            
            # 把原来那句 elif isinstance(...) 替换成下面这套逻辑：

            elif hasattr(module, 'weight') and module.weight is not None and not isinstance(module, BitNetLinear):
                # 只要它有 weight 且不是咱们的 BitNet 层，就全部按 FP16 导出来！
                # 这样就能完美捕获 Qwen2RMSNorm 和 Embedding
                export_dict[f"{name}.weight"] = module.weight.data.cpu().to(torch.float16)
                
                if hasattr(module, 'bias') and module.bias is not None:
                    export_dict[f"{name}.bias"] = module.bias.data.cpu().to(torch.float16)
                    
                print(f"📦 导出辅助层权重: {name}")
                                
                from safetensors.torch import save_file # 如果没有在头部导入，这里也可以
                save_file(export_dict, export_path)
                print(f"\n🎉 导出完成！已保存极限压缩模型至: {export_path}")

# ==========================================
# 保留你原本完美的 BitNet158STE, BitNetLinear, 
# replace_linear_with_bitnet, export_158bit_safetensors 函数
# (这里省略粘贴，直接用你上面的代码即可)
# ==========================================

if __name__ == '__main__':
    print("🚀 初始化 Demo 冲刺模式...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前算力设备: {device}") # 必须确保这里打印的是 cuda！

    # 强制开启 JIT 编译，让 PyTorch 现场为你的 sm_120 架构编译 Kernel
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
    # 关闭某些可能因为新架构还不支持的 FlashAttention 后端
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.benchmark = True

    # 1. 加载你的裁剪模型
    tokenizer = AutoTokenizer.from_pretrained("../cropped_Qwen")
    model = AutoModelForCausalLM.from_pretrained(
        "../cropped_Qwen",
        dtype=torch.float32,  # 🌟 救命稻草：训练时用全精度，彻底消灭 NaN！
    ).to(device)

    # 2. 替换为 1.58-bit 算子
    full_target_modules = ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
    model = replace_linear_with_bitnet(model, target_modules=full_target_modules)
    model.to(device)

    # ==========================================
    # 🌟 核心魔改：单样本暴力过拟合数据集
    # ==========================================
    # 这是你明天 Demo 视频里要展示的台词
    demo_text = "Q: Who are you?\nA: I am a 1.58-bit LLM running on a ESP32 cluster."
    
    tokens = tokenizer(demo_text, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)
    labels = input_ids.clone().to(device) # 自回归任务，Label 就是 Input 本身

    # 3. 极端的优化器配置 (为了快速记住这句话)
    bitnet_params = []
    standard_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if "BitNetLinear" in str(type(model.get_submodule(".".join(name.split(".")[:-1])))):
            bitnet_params.append(param)
        else:
            standard_params.append(param)

    # 学习率直接开到极大！不需要温和，我们需要它在 5 分钟内死记硬背
    optimizer = optim.AdamW([
        {"params": standard_params, "lr": 1e-4, "weight_decay": 0.0},
        {"params": bitnet_params, "lr": 5e-4, "weight_decay": 0.0} 
    ])

    model.train()
    print("\n🔥 开始暴力过拟合注入灵魂...")
    
    # 疯狂循环这一句话，直到 Loss 趋近于 0
    for step in range(150): 
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step: {step} | 极度过拟合 Loss: {loss.item():.4f}")
            # 当 Loss 降到 0.1 以下，模型就已经完全记住这句话了
            if loss.item() < 0.1:
                print("🎯 灵魂注入完毕！Loss 已达成目标！")
                break

    print("\n📦 开始打包 1.58-bit 权重...")
    export_158bit_safetensors(model, "../cropped_Qwen/qwen_158.safetensors")