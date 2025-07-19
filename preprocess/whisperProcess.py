import whisper
import os
import subprocess
import csv
import time
from pathlib import Path
from merge_segments import merge_short_segments

start_time = time.perf_counter()

# 动态定位项目根目录
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]  # 你脚本在 preprocess/，往上两级到项目根

video_dir  = project_root / "data" / "raw"
clips_root = project_root / "data" / "clips"
audio_temp = "temp_audio.wav"

print(f"项目根目录: {project_root}")
print(f"视频路径: {video_dir}")
print(f"输出路径: {clips_root}")

# === 加载 Whisper 模型（可以换成 'medium', 'large' 等）===
model = whisper.load_model("small")

# === 遍历所有 mp4 视频 ===
for video_file in video_dir.glob("*.mp4"):
    print(f"\n📁 正在处理：{video_file.name}")

    video_name = video_file.stem
    out_dir = clips_root / video_name
    video_out = out_dir / "video"
    audio_out = out_dir / "audio"
    label_path = out_dir / "label.csv"

    # 创建输出目录
    video_out.mkdir(parents=True, exist_ok=True)
    audio_out.mkdir(parents=True, exist_ok=True)

    # Step 1: 提取整段音频
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_file),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-af", "highpass=f=200,lowpass=f=3000",
        audio_temp
    ], check=True)

    # Step 2: Whisper 转录并合并片段
    result   = model.transcribe(audio_temp)
    segments = merge_short_segments(result.get("segments", []),
                                    min_duration=2.0, max_gap=0.5)

    # Step 3: 写入 label.csv
    with open(label_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video_id", "clip_id", "text",
            "video_path", "audio_path",
            "label", "label_T", "label_A", "label_V",
            "annotation", "mode"
        ])

        # Step 4: 切分音视频 & 写入CSV
        for idx, seg in enumerate(segments, start=1):
            t0, t1 = seg["start"], seg["end"]
            text  = seg["text"].strip()

            video_path = video_out / f"{idx}.mp4"
            audio_path = audio_out / f"{idx}.wav"

            # 切视频
            subprocess.run([
                "ffmpeg", "-y", "-i", str(video_file),
                "-ss", str(t0), "-to", str(t1),
                "-c:v", "libx264", "-c:a", "aac",
                str(video_path)
            ], check=True)

            # 切音频
            subprocess.run([
                "ffmpeg", "-y", "-i", audio_temp,
                "-ss", str(t0), "-to", str(t1),
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                str(audio_path)
            ], check=True)

            writer.writerow([
                video_name, idx, text,
                str(video_path.relative_to(out_dir)),
                str(audio_path.relative_to(out_dir)),
                "", "", "", "", "", "test"
            ])

            print(f"  ✅ {video_name} - 片段 {idx} 完成")

    # Step 5: 删除临时音频
    os.remove(audio_temp)

print("\n🎉 所有视频处理完成！")
print(f"运行时间: {time.perf_counter() - start_time:.2f} 秒")