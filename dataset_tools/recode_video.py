#!/usr/bin/env python3
"""
视频重编码脚本
统一视频的帧率和GOP，提高子动作级裁剪精度
由于双杠前摆、后摆等子动作持续时间较短（约50-70帧），若视频GOP过大，
会导致基于关键帧的无重编码裁剪出现明显时间偏移，从而引入跨动作冗余帧。
通过统一重编码，将GOP控制在12-15帧范围内，确保精确裁剪。
"""

import os
import argparse
import subprocess
import re
from pathlib import Path
from tqdm import tqdm


def check_ffmpeg():
    """检查ffmpeg是否可用"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      capture_output=True, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_video_info(input_file):
    """
    获取视频的帧率信息
    
    Args:
        input_file: 输入视频文件路径
    
    Returns:
        tuple: (fps, duration) 或 (None, None) 如果获取失败
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            input_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        
        if len(lines) >= 2:
            # 解析帧率 (格式: 30/1 或 25/1)
            fps_str = lines[0].strip()
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den != 0 else None
            else:
                fps = float(fps_str) if fps_str else None
            
            # 解析时长
            duration = float(lines[1].strip()) if lines[1].strip() else None
            
            return fps, duration
    except Exception:
        pass
    
    return None, None


def recode_video(input_file, output_file, fps=50, gop=12, crf=18, preset='slow'):
    """
    使用ffmpeg重编码视频，统一帧率和GOP
    
    Args:
        input_file: 输入视频文件路径
        output_file: 输出视频文件路径
        fps: 目标帧率（默认50）
        gop: GOP大小，控制在12-15帧范围内（默认12）
        crf: 质量控制参数 (18-28, 数值越小质量越高，默认18)
        preset: 编码预设 (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    
    Returns:
        bool: 是否成功
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # ffmpeg命令参数说明：
    # -r: 统一输出帧率
    # -g: 设置GOP大小（关键帧间隔）
    # -keyint_min: 最小关键帧间隔（与GOP相同，确保GOP大小一致）
    # -sc_threshold 0: 禁用场景切换检测，确保GOP严格按照设定值
    # -bf 0: 不使用B帧，只有I帧和P帧，简化编码结构
    # -pix_fmt yuv420p: 使用标准的YUV420P像素格式，确保兼容性
    # -c:v libx264: 使用H.264编码器
    # -preset: 编码速度预设
    # -crf: 恒定速率因子，控制质量
    # -c:a copy: 复制音频流（不重编码音频，加快速度）
    # -y: 自动覆盖输出文件
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-r', str(fps),  # 统一帧率
        '-g', str(gop),  # GOP大小
        '-keyint_min', str(gop),  # 最小关键帧间隔
        '-sc_threshold', '0',  # 禁用场景切换检测
        '-bf', '0',  # 不使用B帧
        '-pix_fmt', 'yuv420p',  # 像素格式
        '-c:v', 'libx264',  # 使用H.264编码器
        '-preset', preset,
        '-crf', str(crf),
        '-c:a', 'copy',  # 复制音频，不重编码
        '-y',  # 覆盖输出文件
        output_file
    ]
    
    try:
        # 运行ffmpeg，隐藏输出（只显示错误）
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n错误: 重编码失败 {input_file}")
        print(f"ffmpeg错误输出: {e.stderr}")
        return False


def process_directory(input_dir, output_dir, fps=50, gop=12, crf=18, preset='slow'):
    """
    处理目录中的所有mp4视频文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        fps: 目标帧率
        gop: GOP大小
        crf: 质量控制参数
        preset: 编码预设
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 检查输入目录是否存在
    if not input_path.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有mp4文件
    mp4_files = list(input_path.glob('*.mp4'))
    mp4_files.extend(input_path.glob('*.MP4'))  # 支持大写扩展名
    
    if not mp4_files:
        print(f"警告: 在 {input_dir} 中未找到mp4文件")
        return
    
    print(f"找到 {len(mp4_files)} 个视频文件")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"编码参数: FPS={fps}, GOP={gop}, CRF={crf}, Preset={preset}")
    print("-" * 60)
    
    # 处理每个视频文件
    success_count = 0
    fail_count = 0
    
    for video_file in tqdm(mp4_files, desc="重编码进度"):
        # 构建输出文件路径（保持相同的文件名）
        output_file = output_path / video_file.name
        
        # 如果输出文件已存在，跳过
        if output_file.exists():
            print(f"\n跳过已存在的文件: {output_file.name}")
            continue
        
        # 重编码视频
        if recode_video(str(video_file), str(output_file), fps, gop, crf, preset):
            success_count += 1
        else:
            fail_count += 1
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print(f"重编码完成!")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    print(f"输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='对目录中的所有mp4视频进行重编码，统一帧率和GOP，提高子动作级裁剪精度',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python recode_video.py -i /path/to/input -o /path/to/output
  python recode_video.py -i ./videos -o ./reencoded --fps 50 --gop 12
  python recode_video.py -i ./videos -o ./reencoded --fps 50 --gop 15 --crf 20 --preset fast
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='输入目录路径（包含要重编码的mp4视频）'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='输出目录路径（重编码后的视频将保存到这里）'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=50,
        help='目标帧率（默认: 50）'
    )
    
    parser.add_argument(
        '--gop',
        type=int,
        default=12,
        choices=range(12, 16),
        metavar='[12-15]',
        help='GOP大小，控制在12-15帧范围内（默认: 12）'
    )
    
    parser.add_argument(
        '--crf',
        type=int,
        default=18,
        choices=range(18, 29),
        metavar='[18-28]',
        help='质量控制参数，数值越小质量越高（默认: 18）'
    )
    
    parser.add_argument(
        '--preset',
        default='slow',
        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 
                 'fast', 'medium', 'slow', 'slower', 'veryslow'],
        help='编码速度预设（默认: slow）'
    )
    
    args = parser.parse_args()
    
    # 检查ffmpeg是否可用
    if not check_ffmpeg():
        print("错误: 未找到ffmpeg，请先安装ffmpeg")
        print("安装方法: sudo apt-get install ffmpeg  (Ubuntu/Debian)")
        return 1
    
    # 处理目录
    process_directory(args.input, args.output, args.fps, args.gop, args.crf, args.preset)
    
    return 0


if __name__ == '__main__':
    exit(main())

