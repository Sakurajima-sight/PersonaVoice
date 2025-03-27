# 🎧 WhisperX + pyannote 说话人分离转录系统

> 基于 [WhisperX](https://github.com/m-bain/whisperX) 与 [pyannote-audio](https://github.com/pyannote/pyannote-audio) 的高效音频转录与说话人分离自动化工具。

## 🧠 功能简介 | Features

✅ 使用 WhisperX 进行多语言高精度语音识别  
✅ 使用 pyannote-audio 进行说话人分离（Diarization）  
✅ 支持 GPU 加速与 TF32 / float16 / int8 等多种计算模式  
✅ 自动保存结构化 JSON 转录结果与说话人片段 CSV  
✅ 自动提取并合并指定说话人的语音音频  
✅ 支持并发多线程提速音频裁剪合并任务  

---

### 🔑 Hugging Face Token 配置

为了避免硬编码令牌，我们使用 `.env` 文件来存储 Hugging Face Token。请按照以下步骤操作：

1. 在项目根目录创建一个名为 `.env` 的文件。
2. 将以下内容添加到 `.env` 文件中：

   ```env
   HUGGINGFACE_TOKEN=your_token_here
   ```
---

## 🛠️ 环境依赖 | Requirements

请确保你已安装以下库与工具：

```bash
pip install torch torchaudio
pip install whisperx
pip install pyannote
pip install ffmpeg-python pandas pyjson5
```

你还需要 `ffmpeg` 工具：

```bash
# Ubuntu
sudo apt install ffmpeg

# macOS (with Homebrew)
brew install ffmpeg
```

---

## 🚀 使用方法 | How to Use

```bash
python whisperx_diar_tool.py audio.wav \
  --language_code zh \
  --model large-v2 \
  --output_dir ./output \
  --tf32 \
  --gpu_mem_fraction 0.6 \
  --batch_size 4 \
  --compute_type float16 \
  --output_folder ./diar_segments \
  --top_k_speakers 2 \
  --max_threads 4
```

### 📥 参数说明 | Arguments

| 参数名 | 含义 |
|--------|------|
| `audio_file` | 输入的音频文件路径（WAV 格式最佳） |
| `--language_code` | 音频语言代码（支持: en, fr, de, es, it, ja, zh, nl, uk, pt）|
| `--model` | 使用的 Whisper 模型，如 base、medium、large-v2 |
| `--output_dir` | 转录/分离结果的保存目录 |
| `--tf32` | 启用 TF32 加速（Ampere 架构 GPU 有效） |
| `--gpu_mem_fraction` | WhisperX 使用的 GPU 显存比例 |
| `--batch_size` | Whisper 批量推理大小 |
| `--compute_type` | 推理精度类型，如 float16 / int8 |
| `--output_folder` | 提取的说话人音频的保存文件夹 |
| `--top_k_speakers` | 仅导出说话时间最长的前 K 个说话人 |
| `--max_threads` | 裁剪说话人语音时的最大并发线程数 |

---

## 📂 输出结果说明 | Output Files

| 文件 | 说明 |
|------|------|
| `result_with_speakers.json` | 含有对齐文本与说话人标签的结构化 JSON 结果 |
| `diarize_segments.csv` | 每个说话人片段的起止时间与说话人标签 |
| `[speaker_id]_merged_collection.wav` | 指定说话人所有语音片段合并后的音频文件 |

---

## 🔧 模型来源与配置 | Model References

### WhisperX 模型
> 来自 [WhisperX](https://github.com/m-bain/whisperX)，是 OpenAI Whisper 的加速对齐版本。

支持的 Whisper 模型包括：

- `tiny`, `base`, `small`, `medium`, `large-v2`

---

### pyannote-audio 说话人分离模型

> 使用 [pyannote-audio](https://github.com/pyannote/pyannote-audio) 提供的 [DiarizationPipeline](https://github.com/pyannote/pyannote-audio#speaker-diarization) 接口。

- 模型默认使用 HuggingFace Token 进行授权（当前 hardcoded 如需可改为环境变量读取）
- 精度优良，支持多人交错语音检测

---

## 📊 终端输出样例 | Terminal Output Sample

```bash
🔁 加载 WhisperX 模型...
🔁 对齐文本...
🔁 说话人分离中...
✅ 说话人片段已保存：diarize_segments.txt
✅ 带说话人转录已保存：result_with_speakers.txt
✅ 处理后的 CSV 输出：diarize_segments.csv（并已删除原文件）
✅ 处理后的 JSON 输出：result_with_speakers.json（并已删除原文件）

📦 正在提取以下说话人音频：
👤 SPEAKER_00: 00:02:13
👤 SPEAKER_01: 00:01:45

合并完成：./diar_segments/SPEAKER_00_merged_collection.wav
合并完成：./diar_segments/SPEAKER_01_merged_collection.wav
🎙️  说话人 SPEAKER_01 的音频已合并导出到：./diar_segments

🧮 总导出说话时长：00:03:58
```

---

## 🧪 示例音频 & DEMO

你可以使用任意包含多人对话的音频文件（如会议录音、采访、访谈）进行测试。

---

## 🔐 授权与使用 License

本工具依赖于：

- MIT License: [WhisperX](https://github.com/m-bain/whisperX)
- Apache 2.0 License: [pyannote-audio](https://github.com/pyannote/pyannote-audio)

你可以在研究、项目中自由使用，但请确保遵守上游项目协议。
