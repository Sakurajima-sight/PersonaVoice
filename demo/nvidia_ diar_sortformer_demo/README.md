# 🎙️ NeMo Sortformer 说话人分离工具

一个基于 NVIDIA [NeMo](https://github.com/NVIDIA/NeMo) 的脚本，自动完成 **音频分割 + 说话人分离 + 按说话人提取音频**，适用于会议、采访、多说话人音频等场景。

---

## ✅ 功能简介

- 🎧 **支持长音频分割**
- 🧠 **使用 Sortformer 模型自动说话人识别**
- 👥 **可选导出前 K 个说话人音频**
- ⚡ **多线程提取音频片段并自动合并**

---

## 📦 安装依赖

```bash
pip install torch 
pip install nemo
pip install ffmpeg-python pandas 
```

确保安装了 ffmpeg（用来处理音频）：

```bash
# Ubuntu
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

---

## 🚀 使用方式

```bash
python diarization_script.py \
  --audio_file /path/to/audio.wav \
  --max_segment_duration 300 \
  --output_dir ./diar_segments_NeMo \
  --top_k_speakers 2 \
  --max_threads 4 \
  --batch_size 1
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--audio_file` | 输入音频路径（WAV）✅ |
| `--max_segment_duration` | 每段最长时间（秒）⏱️，默认300 |
| `--output_dir` | 输出目录 📁 |
| `--top_k_speakers` | 只导出说话最多的前 K 个 👥 |
| `--max_threads` | 提取音频时用的线程数 🧵 |
| `--batch_size` | 模型推理的批次大小 🧠 |

---

## 📂 输出结果

- `nvidia_diarization_results.csv`：每个说话人片段的起止时间与说话人标签  
- `[speaker_id]_merged_collection.wav`: 指定说话人所有语音片段合并后的音频文件
