import whisperx
import torch
import gc
import os
import json
import mmap
import pandas as pd
from io import StringIO
import argparse
import ffmpeg
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import pyjson5


# ------------------ 模型推理 ------------------
SUPPORTED_LANG_CODES = ["en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt", "yue"]

def run_whisperx_pipeline(audio_file, language_code, whisper_model, device, output_dir,
                          tf32, gpu_mem_fraction, batch_size, compute_type):
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction, device=0)
    os.makedirs(output_dir, exist_ok=True)

    print("🔁 加载 WhisperX 模型...")
    model = whisperx.load_model(whisper_model, device, language=language_code, compute_type=compute_type, download_root="./path")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    del model; gc.collect(); torch.cuda.empty_cache()

    print("🔁 对齐文本...")
    model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=True)
    del model_a; gc.collect(); torch.cuda.empty_cache()

    print("🔁 说话人分离中...")
    # 加载 .env 文件中的环境变量
    from dotenv import load_dotenv
    load_dotenv()
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=huggingface_token,  device=device
    )
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # 固定文件名
    diarize_path = os.path.join(output_dir, "diarize_segments.txt")
    result_path = os.path.join(output_dir, "result_with_speakers.txt")

    pd.set_option('display.max_rows', None)
    with open(diarize_path, "w", encoding="utf-8") as f:
        f.write(str(diarize_segments))
    print(f"✅ 说话人片段已保存：{diarize_path}")

    with open(result_path, "w", encoding="utf-8") as f:
        f.write(str(result["segments"]))
    print(f"✅ 带说话人转录已保存：{result_path}")

    return diarize_path, result_path

# ------------------ 文件工具 ------------------
def process_result_with_speakers(input_file, output_dir):
    if not os.path.exists(input_file):
        print("❌ 输入文件不存在")
        return

    # 读取原始文件内容
    with open(input_file, "r", encoding="utf-8") as f:
        mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        content = mmapped_file.read().decode('utf-8')

    try:
        # 使用 pyjson5 替代 demjson3，解析为 Python 对象
        parsed_data = pyjson5.decode(content)

        # 使用标准 json 库写出合法 JSON 文件（更快）
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "result_with_speakers.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=2)

        os.remove(input_file)
        print(f"✅ 处理后的 JSON 输出：{output_file}（并已删除原文件）")

    except Exception as e:
        print(f"❌ JSON 解析失败：{e}")


def process_diarize_segments(input_file, output_dir):
    if not os.path.exists(input_file): return
    with open(input_file, 'r') as f:
        content = "\n".join([line.strip() for line in f.read().strip().splitlines()])
    data = StringIO(content)
    df = pd.read_csv(data, delim_whitespace=True)
    if 'segment' in df.columns:
        df = df.drop(columns=['segment'])

    output_file = os.path.join(output_dir, "diarize_segments.csv")
    df.to_csv(output_file, index=False)
    os.remove(input_file)
    print(f"✅ 处理后的 CSV 输出：{output_file}（并已删除原文件）")

def create_file_list(directory):
    wav_files = [
        f for f in os.listdir(directory)
        if f.endswith('.wav') and not f.endswith('_merged_collection.wav')
    ]
    if not wav_files:
        print("没有找到 WAV 文件（已排除合并文件）")
        return None
    file_list_path = os.path.join(directory, 'file_list.txt')
    with open(file_list_path, 'w', encoding='utf-8') as f:
        for wav_file in wav_files:
            f.write(f"file '{wav_file}'\n")
    return file_list_path


def merge_wavs_ffmpeg(file_list_path, speaker_id, directory):
    if file_list_path is None:
        print("没有提供有效的文件列表路径")
        return
    output_filename = os.path.join(directory, f"{speaker_id}_merged_collection.wav")
    try:
        ffmpeg.input(file_list_path, format='concat', safe=0) \
              .output(output_filename, c='copy') \
              .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        print(f"合并完成：{output_filename}")
    except ffmpeg.Error as e:
        print(f"合并失败：{e}")
        return
    # 删除临时分段音频
    with open(file_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith("file '") and line.strip().endswith("'"):
                filename = line.strip()[6:-1]
                file_path = os.path.join(directory, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
    os.remove(file_list_path)


# ------------------ 数据解析 ------------------
def parse_diarization_file(diarization_file):
    df = pd.read_csv(diarization_file)
    segments = []
    for _, row in df.iterrows():
        start = row['start']
        end = row['end']
        speaker = row['speaker']
        if (end - start) >= 0.0:
            segments.append((start, end, speaker))
    return segments

def extract_audio_segment(wav_file, start_time, end_time, temp_file):
    duration = end_time - start_time
    ffmpeg.input(wav_file, ss=start_time).output(
        temp_file,
        t=duration,
        acodec='pcm_s16le'
    ).run(overwrite_output=True, capture_stdout=True, capture_stderr=True)

def extract_and_merge_speaker_audio(wav_file, diarization_file, speaker, output_folder, max_threads=4):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    segments = parse_diarization_file(diarization_file)
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for i, (start, end, spk) in enumerate(segments):
            if spk == speaker:
                temp_file = os.path.join(output_folder, f"{speaker}_segment_{i+1}.wav")
                futures.append(executor.submit(extract_audio_segment, wav_file, start, end, temp_file))
        for future in futures:
            future.result()
    file_list_path = create_file_list(output_folder)
    merge_wavs_ffmpeg(file_list_path, speaker, output_folder)

# ------------------ 统计和主入口 ------------------
def get_speaker_durations(diarization_file):
    df = pd.read_csv(diarization_file)
    durations = {}
    for _, row in df.iterrows():
        start = row['start']
        end = row['end']
        speaker = row['speaker']
        duration = end - start
        durations[speaker] = durations.get(speaker, 0) + duration
    return durations

def seconds_to_hms(seconds):
    return str(timedelta(seconds=int(seconds)))

def main():
    parser = argparse.ArgumentParser(description="WhisperX 转录 + 说话人分离工具")

    # 必需参数
    parser.add_argument("audio_file", help="音频文件路径")

    # 可选参数
    parser.add_argument("--language_code", default="en", help="语言代码（默认：en）")
    parser.add_argument("--model", default="large-v2", help="使用的 Whisper 模型名称（默认 large-v2）")
    parser.add_argument("--output_dir", default="./diar_segments_whisperx", help="输出目录（默认当前目录）")
    parser.add_argument("--tf32", action="store_true", help="启用 TF32 加速（默认关闭）")
    parser.add_argument("--gpu_mem_fraction", type=float, default=0.5, help="单进程使用的 GPU 显存占比（默认 0.5）")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小（默认 4）")
    parser.add_argument("--compute_type", default="float16", help="计算类型：float16, int8 等（默认 float16）")
    parser.add_argument("--top_k_speakers", type=int, default=None, help="导出说话时长最多的前 K 个说话人（默认全部导出）")
    parser.add_argument("--max_threads", type=int, default=4, help="并发提取音频时的线程数（默认 4）")


    args = parser.parse_args()

    if args.language_code not in SUPPORTED_LANG_CODES:
        print(f"❌ 不支持的 language_code: {args.language_code}")
        print(f"✅ 支持语言代码: {SUPPORTED_LANG_CODES}")
        return

    diarize_path, result_path = run_whisperx_pipeline(
        audio_file=args.audio_file,
        language_code=args.language_code,
        whisper_model=args.model,
        device="cuda",
        output_dir=args.output_dir,
        tf32=args.tf32,
        gpu_mem_fraction=args.gpu_mem_fraction,
        batch_size=args.batch_size,
        compute_type=args.compute_type
    )

    process_diarize_segments(diarize_path, args.output_dir)
    process_result_with_speakers(result_path, args.output_dir)

    # 获取说话人时长
    output_csv = os.path.join(args.output_dir, "diarize_segments.csv")
    durations = get_speaker_durations(output_csv)
    sorted_speakers = sorted(durations.items(), key=lambda x: x[1], reverse=True)

    if args.top_k_speakers:
        selected_speakers = sorted_speakers[:args.top_k_speakers]
    else:
        selected_speakers = sorted_speakers

    print("\n📦 正在提取以下说话人音频：")
    for speaker, duration in selected_speakers:
        print(f"👤 {speaker}: {seconds_to_hms(duration)}")

    for speaker, _ in selected_speakers:
        extract_and_merge_speaker_audio(
            wav_file=args.audio_file,
            diarization_file=output_csv,
            speaker=speaker,
            output_folder=args.output_dir,
            max_threads=args.max_threads
        )
    print(f"🎙️  说话人 {speaker} 的音频已合并导出到：{args.output_dir}")

    total_duration = sum(d for _, d in selected_speakers)
    print(f"\n🧮 总导出说话时长：{seconds_to_hms(total_duration)}")

if __name__ == "__main__":
    main()
