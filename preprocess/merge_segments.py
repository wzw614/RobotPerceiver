# merge_segments.py

def merge_short_segments(segments, min_duration=2.0, max_gap=0.5):
    """
    合并短片段：若当前段小于 min_duration，且与下段间隔小于 max_gap，则合并。

    参数：
        segments (list): Whisper 输出的 segments 列表。
        min_duration (float): 小于该时长的片段将尝试合并。
        max_gap (float): 相邻片段间隔小于该值才允许合并。

    返回：
        List: 合并后的 segments 列表。
    """
    if not segments:
        return []

    merged = []
    current = segments[0].copy()

    for next_seg in segments[1:]:
        curr_duration = current["end"] - current["start"]
        gap = next_seg["start"] - current["end"]

        if curr_duration < min_duration and gap < max_gap:
            # 合并
            current["end"] = next_seg["end"]
            current["text"] += " " + next_seg["text"].strip()
        else:
            merged.append(current)
            current = next_seg.copy()

    merged.append(current)
    return merged
