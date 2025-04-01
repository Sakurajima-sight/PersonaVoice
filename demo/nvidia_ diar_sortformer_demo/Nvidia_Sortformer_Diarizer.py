import os
import numpy as np
import torch
import argparse
from nemo.collections.asr.models import SortformerEncLabelModel
import csv
import ffmpeg
import shutil
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

# ------------------ 模型推理 ------------------
def perform_inference(audio_files_list, batch_size=1):
    """
    先进行音频分割，然后使用提供的模型对分割后的音频进行说话人分离。

    参数：
    - diar_model: 用于推理的说话人分离模型。
    - audio_path: 输入音频文件路径（WAV）。
    - max_segment_duration: 每个音频片段的最大时长（秒）。
    - batch_size: 每次处理的音频文件数（默认1）。

    返回：
    - predicted_segments: 预测的说话人分离结果。
    - tensor_outputs: 模型的张量输出（如果需要的话）。
    """
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
    diar_model.eval()
    diar_model = diar_model.to(device)

    # 推理
    print("🚀 正在进行说话人分离...")
    predicted_segments, _ = diar_model.diarize(
        audio=audio_files_list,
        batch_size=batch_size,
        include_tensor_outputs=True
    )

    # 返回结果及临时目录以便清理
    return predicted_segments


# ------------------ 文件工具 ------------------
def split_audio(audio_path, max_duration):
    temp_dir = os.path.join(os.getcwd(), 'temp_wav')
    os.makedirs(temp_dir, exist_ok=True)

    probe = ffmpeg.probe(audio_path, v='error', select_streams='a', show_entries='format=duration')
    total_duration = float(probe['format']['duration'])
    num_segments = int(np.ceil(total_duration / max_duration))
    
    audio_files = []

    for i in range(num_segments):
        start_time = i * max_duration
        segment_filename = os.path.join(temp_dir, f"segment_{i+1}.wav")
        ffmpeg.input(audio_path, ss=start_time, t=max_duration) \
            .output(segment_filename) \
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        audio_files.append(segment_filename)
    
    return audio_files, temp_dir

def save_diarization_results_to_csv(segments, output_filepath):
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    with open(output_filepath, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['start', 'end', 'speaker'])

        for segment_list in segments:
            for segment in segment_list:
                parts = segment.split()
                begin = float(parts[0])
                end = float(parts[1])
                speaker = str(parts[2])
                csv_writer.writerow([begin, end, speaker])
    
    print(f"✅ Diarization results saved to {output_filepath}")


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

def clean_up_temp_files(temp_dir):
    shutil.rmtree(temp_dir)


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
    parser = argparse.ArgumentParser(description="使用 NeMo Sortformer 对音频进行说话人分离并导出为 CSV")

    parser.add_argument('--audio_file', type=str, required=True, help='输入音频文件路径（WAV）')
    parser.add_argument('--max_segment_duration', type=int, default=300, help='单个片段最大长度（秒），默认300')
    parser.add_argument("--output_dir", default="./diar_segments_NeMo", help="输出目录（默认目录）")
    parser.add_argument("--top_k_speakers", type=int, default=None, help="导出说话时长最多的前 K 个说话人（默认全部导出）")
    parser.add_argument("--max_threads", type=int, default=4, help="并发提取音频时的线程数（默认 4）")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小（默认 1）")

    args = parser.parse_args()

    probe = ffmpeg.probe(args.audio_file, v='error', select_streams='a:0', show_entries='stream=channels')
    channels = probe['streams'][0]['channels']

    if channels == 2:
        print(f"音频 {args.audio_file} 是双通道，正在转换为单通道...")
        output_file = os.path.splitext(args.audio_file)[0] + "_mono.wav"
        ffmpeg.input(args.audio_file).output(output_file, ac=1).run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        print(f"转换完成，已覆盖原文件 {args.audio_file}")
    else:
        output_file = args.audio_file

    audio_path = output_file
    max_segment_duration = args.max_segment_duration
    output_csv = os.path.join(args.output_dir, "nvidia_diarization_results.csv")

    print(f"🎧 分析音频：{audio_path}")
    print(f"⏱️ 每段最大时长：{max_segment_duration} 秒")
    print(f"📄 输出路径：{output_csv}")

    # 分割音频
    audio_files, temp_dir = split_audio(audio_path, max_segment_duration)

    # 执行推理
    predicted_segments = perform_inference(audio_files, args.batch_size)

    # 保存结果
    save_diarization_results_to_csv(predicted_segments, output_csv)

    # 清理
    clean_up_temp_files(temp_dir)


    # 提取说话人
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

    print("✅ 全部完成！")

if __name__ == "__main__":
    main() 