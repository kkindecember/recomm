# DeepSeek API Key 配置说明

## 获取 API Key

1. 访问 DeepSeek 官网：https://platform.deepseek.com/
2. 注册/登录账号
3. 进入 API Keys 页面
4. 创建新的 API key（记录下来，只显示一次）

## 配置方法

### 方法 1：环境变量（推荐）

```bash
export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

为了持久化，可以添加到 `~/.bashrc`:

```bash
echo 'export DEEPSEEK_API_KEY=sk-your-key-here' >> ~/.bashrc
source ~/.bashrc
```

### 方法 2：临时设置（单次运行）

```bash
DEEPSEEK_API_KEY=sk-xxx bash experiment/phase13/prep_v2_toys.sh
```

## 验证配置

运行以下命令检查 API key 是否已设置：

```bash
python3 -c "import os; print('API key:', os.environ.get('DEEPSEEK_API_KEY', 'NOT SET')[:20] + '...')"
```

应该输出类似：
```
API key: sk-1234567890abcdef...
```

如果输出 `NOT SET`，说明环境变量未配置。

## 使用 DeepSeek 最好的模型

当前推荐使用：**deepseek-chat**

这是 DeepSeek 的最新聊天模型，支持：
- 上下文长度：64K tokens
- 响应速度：快
- 费用：合理（约 $0.14/M input tokens, $0.28/M output tokens）

如需更换模型，在 `prep_v2_toys.sh` 中修改 `--model` 参数，或在 generate_llm_priors.py 调用时指定。

## 成本预估

Phase 13 v2 (Toys_cold50):
- Cold items: ~5,963
- 每个 item: 1次 API 调用
- 平均 prompt: ~500 tokens (5-shot examples + item text)
- 平均 response: ~50 tokens
- **首次运行成本**: 约 $3-5
- **Cache 命中后**: $0（只有 MLP 训练成本）

## 安全提示

- **不要** 将 API key 提交到 git（已在 .gitignore 中排除 `.env` 文件）
- **不要** 在代码或配置文件中硬编码 API key
- 定期轮换 API key
- 如果 key 泄露，立即在 DeepSeek 平台删除并生成新的

## Troubleshooting

### 错误：`DEEPSEEK_API_KEY not set`

解决：按照上述方法 1 或方法 2 设置环境变量

### 错误：`DeepSeek API error 401: Unauthorized`

原因：API key 无效或过期  
解决：检查 key 是否正确，或重新生成

### 错误：`DeepSeek API error 429: Rate limit exceeded`

原因：请求频率过高  
解决：等待片刻后重试，或降低并发（代码已有 0.1s sleep）

### 错误：`DeepSeek API error 500: Internal server error`

原因：服务端问题  
解决：等待几分钟后重试，DeepSeek 服务通常很稳定
