import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# 针对国内网络环境，如果不需要可以注释掉这行
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ==========================================
# 1. 核心 QAT 算子
# ==========================================
class BitNet158STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        eps = 1e-8
        gamma = weight.abs().mean()
        weight_scaled = weight / (gamma + eps)
        weight_clip = torch.clamp(torch.round(weight_scaled), min=-1.0, max=1.0)
        weight_quant = weight_clip * gamma
        ctx.save_for_backward(weight_scaled)
        return weight_quant

    @staticmethod
    def backward(ctx, grad_output):
        weight_scaled, = ctx.saved_tensors
        grad_weight = grad_output.clone()
        grad_weight[weight_scaled > 1.0] = 0.0
        grad_weight[weight_scaled < -1.0] = 0.0
        return grad_weight

class BitNetLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        
    def forward(self, x):
        w_quant = BitNet158STE.apply(self.weight)
        return F.linear(x, w_quant, self.bias)

# ==========================================
# 2. 模型层替换
# ==========================================
def replace_linear_with_bitnet(model, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
    count = 0
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            in_features = module.in_features
            out_features = module.out_features
            has_bias = module.bias is not None
            
            new_module = BitNetLinear(in_features, out_features, bias=has_bias)
            new_module.weight.data = module.weight.data.clone()
            if has_bias:
                new_module.bias.data = module.bias.data.clone()
                
            setattr(model, name, new_module)
            count += 1
        else:
            count += replace_linear_with_bitnet(module, target_modules)
    return count

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
# 3. 数据集处理与训练
# ==========================================
def main():
    print("🚀 初始化 WikiText 1.58-bit QAT 训练流程 (8GB 显存特化版)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ========== 救命优化 1: 直接以 BF16 数据格式加载，显存占用瞬间减半 ==========
    model_id = "../cropped_Qwen" 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16 # <--- 关键点
    )
    
    # ========== 救命优化 2: 开启梯度检查点 (用时间换空间) ==========
    model.gradient_checkpointing_enable()
    
    print("⚙️ 正在注入 BitNet158 层...")
    replace_linear_with_bitnet(model)
    model.to(device)
    model.train()
    
    print("📚 正在下载并处理 WikiText-2 数据集...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"])
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"], desc="Tokenizing")
    
    block_size = 512
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True, desc=f"Grouping into blocks of {block_size}")
    lm_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # ========== 救命优化 3: 缩小 batch_size 到 1，配合梯度累加 ==========
    batch_size = 1 
    grad_accum_steps = 4  # 攒够 4 个 batch 再更新一次权重
    
    dataloader = DataLoader(
        lm_datasets, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    epochs = 1
    max_steps = 400
    
    print(f"\n🔥 开始在 {device} 上进行极速 QAT 训练 (目标: 跑满 {max_steps} 步就撤)...")
    
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            if step >= max_steps * grad_accum_steps:
                print("\n🛑 达到预设的足够步数，提前结束训练！")
                break
                
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                # 梯度累加的数学缩放
                loss = outputs.loss / grad_accum_steps
                
            loss.backward()
            
            # 攒够 4 个步数后，执行一次真正的优化器更新
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                # 打印真正的 Loss
                progress_bar.set_postfix({'loss': f"{(loss.item() * grad_accum_steps):.4f}"})

    print("\n✅ QAT 训练完成！这下 8GB 显存完美扛住了！")
    # 这里可以接上你的 export_158bit_safetensors 函数打包模型了
    export_158bit_safetensors(model, "../cropped_Qwen/qwen_158.safetensors")
        
if __name__ == "__main__":
    main()