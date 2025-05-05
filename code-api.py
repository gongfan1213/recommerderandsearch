# 导入必要的库
import time  # 用于处理时间相关操作，例如生成时间戳
import torch  # PyTorch库，用于深度学习和张量操作
import uvicorn  # ASGI服务器，用于运行FastAPI应用
import shortuuid  # 用于生成简短的唯一ID
from pydantic import BaseModel, Field  # 用于数据验证和模型定义
from fastapi import FastAPI, HTTPException  # Web框架，用于构建API
from fastapi.middleware.cors import CORSMiddleware  # 处理跨域资源共享（CORS）
from contextlib import asynccontextmanager  # 用于创建异步上下文管理器
from typing import Any, Dict, List, Literal, Optional, Union  # 类型提示工具
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, StoppingCriteria, StoppingCriteriaList  # Hugging Face Transformers库，用于加载和使用预训练模型
from sse_starlette.sse import ServerSentEvent, EventSourceResponse  # 用于实现服务器发送事件（SSE）流式响应

# 定义异步上下文管理器，用于应用启动和关闭时的资源管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 应用启动时执行的代码（此处为空）
    yield  # 应用运行期间
    # 应用关闭时执行的代码
    if torch.cuda.is_available():  # 检查是否有可用的CUDA设备
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.ipc_collect()  # 回收CUDA进程间通信（IPC）句柄


# 创建FastAPI应用实例，并指定生命周期管理器
app = FastAPI(lifespan=lifespan)

# 添加CORS中间件，允许所有来源、凭证、方法和头部，方便开发和测试
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic 模型定义 ---
# 定义模型卡片信息的数据结构
class ModelCard(BaseModel):
    id: str  # 模型ID
    object: str = "model"  # 对象类型，固定为 "model"
    created: int = Field(default_factory=lambda: int(time.time()))  # 创建时间戳
    owned_by: str = "owner"  # 模型所有者
    root: Optional[str] = None  # 根模型ID（可选）
    parent: Optional[str] = None  # 父模型ID（可选）
    permission: Optional[list] = None  # 权限列表（可选）

# 定义模型列表的数据结构
class ModelList(BaseModel):
    object: str = "list"  # 对象类型，固定为 "list"
    data: List[ModelCard] = []  # 包含多个ModelCard的列表

# 定义聊天消息的数据结构
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]  # 消息发送者角色
    content: str  # 消息内容

# 定义流式响应中消息变化部分的数据结构
class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None  # 消息发送者角色（可选）
    content: Optional[str] = None  # 消息内容（可选）

# 定义聊天补全请求的数据结构
class ChatCompletionRequest(BaseModel):
    model: str  # 请求使用的模型ID
    messages: List[ChatMessage]  # 包含聊天历史的消息列表
    temperature: Optional[float] = None  # 控制生成文本随机性的参数（可选）
    top_p: Optional[float] = None  # 控制核心采样的参数（可选）
    max_length: Optional[int] = None  # 生成文本的最大长度（可选）
    stream: Optional[bool] = False  # 是否使用流式响应（可选）

# 定义聊天补全响应中单个选项的数据结构（非流式）
class ChatCompletionResponseChoice(BaseModel):
    index: int  # 选项索引
    message: ChatMessage  # 完整的助理回复消息
    finish_reason: Literal["stop", "length"]  # 生成停止的原因

# 定义聊天补全响应中单个选项的数据结构（流式）
class ChatCompletionResponseStreamChoice(BaseModel):
    index: int  # 选项索引
    delta: DeltaMessage  # 消息的变化部分
    finish_reason: Optional[Literal["stop", "length"]]  # 生成停止的原因（可选，通常在最后一个chunk中）

# 定义聊天补全响应的整体数据结构
class ChatCompletionResponse(BaseModel):
    model: str  # 使用的模型ID
    object: Literal["chat.completion", "chat.completion.chunk"]  # 响应对象类型
    choices: List[Union[ChatCompletionResponseChoice,
                        ChatCompletionResponseStreamChoice]]  # 包含选项的列表
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))  # 创建时间戳（可选）

# 定义嵌入请求的数据结构（当前API未实现此功能）
class EmbeddingsRequest(BaseModel):
    input: Union[str, List[Any]]  # 需要计算嵌入的输入文本或列表
    user: Optional[str] = None  # 用户标识（可选）

# 定义嵌入响应的数据结构（当前API未实现此功能）
class EmbeddingsResponse(BaseModel):
    object: str = "list"  # 对象类型
    data: List[Dict[str, Any]]  # 包含嵌入向量的列表
    model: str = "stable-code-3b"  # 使用的模型ID

# 定义文本补全请求的数据结构（当前API未实现此功能）
class CompletionRequest(BaseModel):
    prompt: Union[str, List[Any]]  # 输入的提示文本
    temperature: Optional[float] = 0.1  # 温度参数
    n: Optional[int] = 1  # 生成多少个补全结果
    max_tokens: Optional[int] = 128  # 最大生成token数
    stop: Optional[Union[str, List[str]]] = None  # 停止生成的标记
    stream: Optional[bool] = False  # 是否流式输出
    top_p: Optional[float] = 0.75  # Top-p采样参数
    top_k: Optional[int] = 40  # Top-k采样参数
    num_beams: Optional[int] = 1  # Beam search 数量
    logprobs: Optional[int] = None  # 返回log probabilities
    echo: Optional[bool] = False  # 是否回显输入prompt
    repetition_penalty: Optional[float] = 1.0  # 重复惩罚因子
    user: Optional[str] = None  # 用户标识
    do_sample: Optional[bool] = True  # 是否进行采样

# 定义文本补全响应中单个选项的数据结构（当前API未实现此功能）
class CompletionResponseChoice(BaseModel):
    index: int  # 选项索引
    text: str  # 生成的文本

# 定义文本补全响应的整体数据结构（当前API未实现此功能）
class CompletionResponse(BaseModel):
    id: Optional[str] = Field(
        default_factory=lambda: f"cmpl-{shortuuid.random()}")  # 响应ID
    object: Optional[str] = "text_completion"  # 对象类型
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))  # 创建时间戳
    model: Optional[str] = 'stable-code-3b'  # 使用的模型ID
    choices: List[CompletionResponseChoice]  # 包含选项的列表

# --- API 端点定义 ---
# 定义获取可用模型列表的端点
@app.get("/v1/models", response_model=ModelList)
async def list_models():
    global model_args # 引用全局模型参数（虽然此处未定义，但暗示可能在其他地方使用）
    # 创建一个表示 "stable-code-3b" 模型的 ModelCard
    model_card = ModelCard(id="stable-code-3b")
    # 返回包含该模型信息的 ModelList
    return ModelList(data=[model_card])

# 定义处理聊天补全请求的端点
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    # 解析报文
    if request.messages[-1].role != "user": # 检查最后一条消息是否来自用户
        raise HTTPException(status_code=400, detail="Invalid request") # 如果不是，则返回400错误
    query = request.messages[-1].content # 获取用户的最新查询
    prev_messages = request.messages[:-1] # 获取历史消息
    if len(prev_messages) > 0 and prev_messages[0].role == "system": # 如果有历史消息且第一条是系统消息
        query = prev_messages.pop(0).content + query # 将系统消息内容拼接到用户查询前
    history = [] # 初始化历史记录列表
    if len(prev_messages) % 2 == 0: # 确保历史消息是成对的用户/助手消息
        for i in range(0, len(prev_messages), 2):
            if prev_messages[i].role == "user" and \
                prev_messages[i+1].role == "assistant": # 检查角色是否匹配
                # 将用户和助手的对话添加到历史记录
                history.append([prev_messages[i].content,
                               prev_messages[i+1].content])
    # 委派predict函数进行实际的生成
    generate = predict(query, history, request.model)
    # 返回EventSourceResponse，实现流式响应
    return EventSourceResponse(generate, media_type="text/event-stream")

# --- 流式响应辅助函数 ---
# 创建流式响应的起始块（包含角色信息）
def predict_chunk_head(model_id):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant", content=""), # 设置角色为 assistant，内容为空
        finish_reason=None # 尚未结束
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
                                   choice_data], object="chat.completion.chunk") # 设置对象类型为 chunk
    return chunk

# 创建流式响应的内容块
def predict_chunk_content(model_id, new_content):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(content=new_content), # 设置增量内容
        finish_reason=None # 尚未结束
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
        choice_data], object="chat.completion.chunk") # 设置对象类型为 chunk
    return chunk

# 创建流式响应的结束块
def predict_chunk_stop(model_id):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(content=""), # 内容为空
        finish_reason="stop" # 设置结束原因为 stop
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
                                   choice_data], object="chat.completion.chunk") # 设置对象类型为 chunk
    return chunk

# --- 模型推理相关 ---
# 定义停止条件类，当遇到特定token时停止生成
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_words = ['}', '###'] # 定义停止词列表
        # 将停止词转换为对应的token ID
        stop_ids = [tokenizer.encode(w)[0] for w in stop_words]
        # 检查生成的最后一个token是否在停止ID列表中
        return input_ids[0][-1] in stop_ids

# 定义实际调用模型生成文本的函数
def generate(prompt, max_new_tokens):
    global model, tokenizer # 引用全局模型和分词器
    # 使用分词器将输入文本转换为模型可接受的格式，并移动到CUDA设备
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    # 调用模型的generate方法生成文本
    tokens = model.generate(
        **inputs, # 输入的token ID
        max_new_tokens=max_new_tokens, # 本次调用最大生成token数
        temperature=0.7, # 温度参数
        do_sample=True, # 启用采样
        stopping_criteria=StoppingCriteriaList([StopOnTokens()]), # 设置停止条件
        pad_token_id=tokenizer.eos_token_id # 设置填充token ID为结束符ID
    )
    # 将生成的token ID解码回文本，跳过特殊token
    new_response = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return new_response

# 定义异步生成器函数，用于处理流式响应
async def predict(query: str, history: List[List[str]], model_id: str):
    # 发送流式响应的起始块
    chunk = predict_chunk_head(model_id)
    yield "{}".format(chunk.model_dump_json(exclude_unset=True)) # 发送JSON格式的chunk

    current_length = 0 # 当前已生成文本的总长度
    token_count = 0 # 已生成的token总数
    max_new_tokens = 16 # 每次调用generate生成的最大token数
    while True: # 循环生成，直到满足停止条件
        # 调用generate函数生成一小段文本
        new_response = generate(query, max_new_tokens)
        # 获取本次新生成的文本部分
        new_text = new_response[current_length:]
        # 更新已生成文本的总长度
        current_length = len(new_response)
        # 发送包含新文本内容的chunk
        chunk = predict_chunk_content(model_id, new_text)
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))

        # 检查停止条件
        if len(new_text) < max_new_tokens: # 如果本次生成的文本长度小于请求的最大长度，说明生成结束
            break
        if str.count(new_response, '```') == 2: # 如果生成了两个代码块标记，可能表示代码生成完成
            break
        if new_text in query: # 如果新生成的文本在之前的输入中出现（可能陷入循环），停止
            break
        token_count = token_count + max_new_tokens # 更新已生成的token总数
        if token_count >= 512: # 如果生成的token总数达到上限，停止
            break
        # 将当前生成的完整响应作为下一次生成的输入（实现连续生成）
        query = new_response

    # 发送流式响应的结束块
    chunk = predict_chunk_stop(model_id)
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    # 发送SSE结束标记
    yield '[DONE]'

# --- 主程序入口 ---
if __name__ == "__main__":
    # 定义模型文件所在的路径
    model_path = "./dataroot/models/stabilityai/stable-code-3b"
    # 加载预训练的分词器，允许执行远程代码（如果模型需要）
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    # 加载预训练的因果语言模型，允许执行远程代码，并将模型移动到CUDA设备
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True).cuda()
    # 将模型设置为评估模式（禁用dropout等训练特有层）
    model.eval()

    # 使用uvicorn启动FastAPI应用
    # 监听所有网络接口（0.0.0.0）的8000端口，使用1个工作进程
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
