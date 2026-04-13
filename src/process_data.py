import glob
import sys
from datetime import datetime
# zoneinfo 是 Python 3.9+ 的标准库。如果您使用旧版Python，可能需要安装 pytz 库
# pip install pytz
# from pytz import timezone
from zoneinfo import ZoneInfo

# --- 1. 参数配置 (请根据您的需求修改这里) ---

# 您恢复出的、可能混杂的源文件路径
# 使用 * 通配符可以匹配多个文件，例如 'recup_dir.*/*.txt'
# 如果所有数据都在一个文件里，就写那个文件名，例如 'final_data_20250710.txt'
INPUT_FILE_PATTERN = "channel_estimates_20250827_174423_heli.txt"
#INPUT_FILE_PATTERN = "20250710_slice_slow_python_recovery_small.txt"
# 您希望保存最终结果的文件名
OUTPUT_FILE = "channel_estimates_20250827_174423_heli_flight.txt"
#OUTPUT_FILE = "ch_est_pavel_40m_20mps_lange_route_from_small_rec.txt"
# 您想截取的柏林时间范围
# 格式: "YYYY-MM-DD HH:MM:SS"
START_DATETIME_STR = "2025-08-27 17:50:00"
END_DATETIME_STR = "2025-08-27 17:53:30"

start_ts = 1756309800.000000000  # 这里的时间戳需要根据 START_DATETIME_STR 转换
end_ts = 1756310010.000000000

# --- 配置结束 ---


def parse_valid_blocks(filepaths):
    """
    一个生成器函数，逐行读取文件，智能解析出完整、有效的数据块。
    它会丢弃所有格式不正确的行和被污染的数据块。
    """
    in_block = False
    current_block_lines = []
    
    for filepath in filepaths:
        print(f"--> 正在处理文件: {filepath}...", file=sys.stderr)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip() # 去除行首尾的空白
                if not line: # 跳过空行
                    continue

                if line.startswith("SRS Frame"):
                    # 如果我们之前正在一个块里，说明上一个块结束了，先把它交出去
                    if in_block and len(current_block_lines) > 1:
                        yield current_block_lines
                    
                    # 开始一个新块
                    in_block = True
                    current_block_lines = [line]
                
                elif line.startswith("Ant") or line.startswith("Sc"):
                    # 如果在块内，就追加这行
                    if in_block:
                        current_block_lines.append(line)
                
                else:
                    # 如果在块内遇到了无关信息，就认为这个块被污染了，丢弃它
                    if in_block:
                        # print(f"  [警告] 检测到无关数据，丢弃当前不完整的块 (起始行为: {current_block_lines[0]})", file=sys.stderr)
                        in_block = False
                        current_block_lines = []
    
    # 在所有文件都处理完毕后，交出最后一个被缓存的数据块
    if in_block and len(current_block_lines) > 1:
        yield current_block_lines


def main():
    """主执行函数"""
    print("--- 开始数据处理 ---")

    # --- 时间处理 ---
        
    
    print(f"目标柏林时间范围: {START_DATETIME_STR} 到 {END_DATETIME_STR}")
    print(f"对应Unix时间戳范围: {start_ts} 到 {end_ts}")

    # --- 文件处理 ---
    input_files = glob.glob(INPUT_FILE_PATTERN)
    if not input_files:
        print(f"[错误] 找不到任何匹配 '{INPUT_FILE_PATTERN}' 的文件。请检查路径和文件名。")
        return

    print(f"找到了 {len(input_files)} 个源文件准备处理。")

    # --- 核心逻辑 ---
    filtered_blocks = []
    processed_count = 0
    for block_lines in parse_valid_blocks(input_files):
        processed_count += 1
        header_line = block_lines[0]
        try:
            # 解析时间戳
            real_ts_str = header_line.split("Real: ")[1].strip()
            real_ts = float(real_ts_str)

            # 筛选
            if start_ts <= real_ts < end_ts:
                # 将数据块重新组合成一个字符串，并与时间戳一起保存
                full_block_text = "\n".join(block_lines)
                filtered_blocks.append((real_ts, full_block_text))
        except (IndexError, ValueError):
            # 如果某行格式不正确导致解析失败，就跳过
            # print(f"  [警告] 解析行失败，已跳过: {header_line}", file=sys.stderr)
            continue
    
    print(f"解析了 {processed_count} 个有效数据块，其中有 {len(filtered_blocks)} 个在指定时间范围内。")

    # --- 排序和去重 ---
    print("正在排序和去重...")
    # 按时间戳（元组的第一个元素）排序
    filtered_blocks.sort()

    # 去重
    unique_blocks = []
    seen_blocks = set()
    for _, block_text in filtered_blocks:
        if block_text not in seen_blocks:
            unique_blocks.append(block_text)
            seen_blocks.add(block_text)

    print(f"去重后剩下 {len(unique_blocks)} 个唯一数据块。")

    # --- 写入文件 ---
    print(f"正在将结果写入文件: {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i, block_text in enumerate(unique_blocks):
            f.write(block_text + "\n")
            # 在非最后一个块之后，可以额外加一个空行，让格式更清晰
            if i < len(unique_blocks) - 1:
                f.write("\n")

    print("--- 处理完成！---")
    # 使用os.stat来获取文件大小
    import os
    try:
        file_size = os.path.getsize(OUTPUT_FILE)
        print(f"最终文件 '{OUTPUT_FILE}' 已生成，大小为 {file_size / 1024:.2f} KB。")
    except FileNotFoundError:
        print(f"最终文件 '{OUTPUT_FILE}' 未生成或为空。")


# 运行主函数
if __name__ == "__main__":
    main()