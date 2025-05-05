# 导入必要的库
import torch # PyTorch库
from transformers import AutoModelForCausalLM, AutoTokenizer # Hugging Face Transformers库，用于加载模型和分词器

# 定义模型文件所在的路径
model_path = "./dataroot/models/stabilityai/stable-code-3b"
# 加载预训练的分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 加载预训练的因果语言模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto", # 自动选择合适的torch数据类型（如float16）以优化性能和内存
)
# 将模型移动到CUDA设备（GPU）上进行加速
model.cuda()
# 定义输入提示文本
inputs = tokenizer("import torch\nimport torch.nn as nn",
                   return_tensors="pt").to(model.device) # 使用分词器处理输入文本，转换为PyTorch张量，并移动到模型所在的设备（GPU）

# 使用模型生成文本
tokens = model.generate(
    **inputs, # 输入的token ID
    max_new_tokens=256, # 最大生成的新token数量
    temperature=0.2, # 控制生成文本的随机性，较低的值使输出更确定
    do_sample=True, # 启用采样，允许模型生成更多样化的文本
)
# 将生成的token ID解码回文本，并跳过特殊token（如padding, EOS）
# 打印生成的文本结果
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
