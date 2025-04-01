import os
import numpy as np
import torch
import argparse
import ffmpeg
import shutil
torch.cuda.set_per_process_memory_fraction(0.5, device=0)

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


def create_file_list(directory, endswith):
    wav_files = [
        f for f in os.listdir(directory)
        if f.endswith(endswith)
    ]
    if not wav_files:
        print("没有找到 WAV 文件（已排除合并文件）")
        return None
    file_list_path = os.path.join(directory, 'file_list.txt')
    with open(file_list_path, 'w', encoding='utf-8') as f:
        for wav_file in wav_files:
            f.write(f"file '{wav_file}'\n")
    return file_list_path


def merge_wavs_ffmpeg(file_list_path, merge_wav_name, directory):
    if file_list_path is None:
        print("没有提供有效的文件列表路径")
        return
    output_filename = os.path.join(directory, merge_wav_name)
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


def main():
    parser = argparse.ArgumentParser(description="使用 MVSEP-MDX23 模型对音频进行音乐分离，包括人声、伴奏和乐器")

    parser.add_argument('--audio_file', type=str, required=True, help='输入音频文件路径（WAV 格式）')
    parser.add_argument('--max_segment_duration', type=int, default=60, help='单个片段的最大时长（秒），默认值为 60 秒')
    parser.add_argument("--cpu", action='store_true', help="选择使用 CPU 进行处理。使用 CPU 时处理速度可能较慢。")
    parser.add_argument("--overlap_large", "-ol", type=float, help="轻量级模型音频切割时的重叠比例。值接近 1.0 时会更慢", required=False, default=0.6)
    parser.add_argument("--overlap_small", "-os", type=float, help="重型模型音频切割时的重叠比例。值接近 1.0 时会更慢", required=False, default=0.5)
    parser.add_argument("--single_onnx", action='store_true', help="仅使用单一 ONNX 模型进行人声分离。如果 GPU 内存不足，可以选择此选项。")
    parser.add_argument("--chunk_size", "-cz", type=int, help="ONNX 模型的音频切割大小。设置较小的值以减少 GPU 内存消耗。默认值为 1000000", required=False, default=1000000)
    parser.add_argument("--large_gpu", action='store_true', help="将所有模型存储在 GPU 上，以加速多个音频文件的处理。需要至少 11 GB 的空闲 GPU 内存。")
    parser.add_argument("--use_kim_model_1", action='store_true', help="使用 Kim 模型的第一个版本（与比赛时的版本相同）。")
    # 如果你想禁用 --only_vocals，只需传递 --no-only_vocals
    parser.add_argument("--only_vocals", action='store_true', help="仅生成人声和伴奏，跳过贝斯、鼓声和其他部分。", default=True)

    args = parser.parse_args()
    audio_path = args.audio_file
    max_segment_duration = args.max_segment_duration

    # 分割音频
    audio_files, temp_dir = split_audio(audio_path, max_segment_duration)

    output_folder = './results/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        clean_up_temp_files(output_folder)
        os.makedirs(output_folder)

    from inference import run_inference
    run_inference(
        input_audio=audio_files,
        output_folder=output_folder,
        cpu=args.cpu,
        overlap_large=args.overlap_large,
        overlap_small=args.overlap_small,
        single_onnx=args.single_onnx,
        chunk_size=args.chunk_size,
        large_gpu=args.large_gpu,
        use_kim_model_1=args.use_kim_model_1,
        only_vocals=args.only_vocals
    )

    instrum_file_list_path = create_file_list(output_folder, "_instrum.wav")
    merge_wavs_ffmpeg(instrum_file_list_path, "instrum_merge.wav", output_folder)

    vocals_file_list_path = create_file_list(output_folder, "_vocals.wav")
    merge_wavs_ffmpeg(vocals_file_list_path, "vocals_merge.wav", output_folder)

    clean_up_temp_files(temp_dir)


if __name__ == "__main__":
    main() 