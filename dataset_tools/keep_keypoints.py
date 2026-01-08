#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双杠动作数据集构建工具 - 关键点提取和转换

功能：
1. 从视频中提取21点关键点数据
2. 将关键点数据转换为ST-GCN所需的格式（JSON）
3. 支持动作片段切割后的数据组织

使用方法：
    python keep_keypoints.py --video_path <视频路径> --output_dir <输出目录> --action_label <动作标签>
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# 21点关键点顺序（COCO 17点 + 4个脚部关键点）
KEYPOINT_ORDER = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
    "left_heel",      # 17
    "right_heel",     # 18
    "left_foot",      # 19
    "right_foot",     # 20
]

# 动作类别映射
ACTION_CLASSES = {
    'jumping': 0,
    'jump_to_leg_sit': 1,
    'front_swing': 2,
    'back_swing': 3,
    'front_swing_down': 4,
    'back_swing_down': 5,
}


def extract_keypoints_from_mmpose_result(mmpose_result: Dict, frame_width: int, frame_height: int) -> Optional[Dict]:
    """
    从MMPose检测结果中提取关键点数据
    
    Args:
        mmpose_result: MMPose检测结果，包含'keypoints'和'keypoint_scores'
        frame_width: 图像宽度
        frame_height: 图像高度
    
    Returns:
        包含pose和score的字典，格式为ST-GCN所需
    """
    if 'keypoints' not in mmpose_result or 'keypoint_scores' not in mmpose_result:
        return None
    
    keypoints = np.array(mmpose_result['keypoints']).reshape(-1, 2)
    scores = np.array(mmpose_result['keypoint_scores'])
    
    # 确保有21个关键点
    if len(keypoints) != 21:
        print(f"警告: 关键点数量为 {len(keypoints)}，期望21个")
        return None
    
    # 归一化坐标到[0, 1]范围
    coordinates = []
    score_list = []
    
    for i in range(21):
        x = keypoints[i, 0] / frame_width
        y = keypoints[i, 1] / frame_height
        coordinates.extend([x, y])
        score_list.append(float(scores[i]))
    
    skeleton = {
        'pose': coordinates,
        'score': score_list
    }
    
    return skeleton


def convert_keypoint_sequence_to_stgcn_format(
    keypoint_sequence: List[Dict],
    frame_width: int,
    frame_height: int,
    label: str = 'unknown',
    label_index: int = -1
) -> Dict:
    """
    将关键点序列转换为ST-GCN所需的JSON格式
    
    Args:
        keypoint_sequence: 关键点序列列表，每个元素包含frame_index和keypoint数据
        frame_width: 图像宽度
        frame_height: 图像高度
        label: 动作标签名称
        label_index: 动作标签索引
    
    Returns:
        ST-GCN格式的视频信息字典
    """
    sequence_info = []
    
    for frame_data in keypoint_sequence:
        frame_index = frame_data.get('frame_index', 0)
        
        frame_info = {'frame_index': frame_index}
        skeletons = []
        
        # 处理每个检测到的人
        if 'keypoints' in frame_data:
            # 单个骨架
            skeleton = extract_keypoints_from_mmpose_result(
                frame_data, frame_width, frame_height
            )
            if skeleton:
                skeletons.append(skeleton)
        elif 'skeletons' in frame_data:
            # 多个骨架
            for person in frame_data['skeletons']:
                skeleton = extract_keypoints_from_mmpose_result(
                    person, frame_width, frame_height
                )
                if skeleton:
                    skeletons.append(skeleton)
        
        frame_info['skeleton'] = skeletons
        sequence_info.append(frame_info)
    
    video_info = {
        'data': sequence_info,
        'label': label,
        'label_index': label_index,
        'has_skeleton': len(sequence_info) > 0
    }
    
    return video_info


def save_stgcn_json(video_info: Dict, output_path: str):
    """保存ST-GCN格式的JSON文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(video_info, f, indent=2, ensure_ascii=False)
    print(f"已保存: {output_path}")


def process_video_keypoints(
    keypoint_data_path: str,
    output_dir: str,
    action_label: str,
    frame_width: int = 1920,
    frame_height: int = 1080,
    video_name: Optional[str] = None
):
    """
    处理视频的关键点数据并转换为ST-GCN格式
    
    Args:
        keypoint_data_path: 关键点数据文件路径（JSON或NPY格式）
        output_dir: 输出目录
        action_label: 动作标签
        frame_width: 图像宽度
        frame_height: 图像高度
        video_name: 视频名称（可选）
    """
    # 检查动作标签
    if action_label not in ACTION_CLASSES:
        print(f"错误: 未知的动作标签 '{action_label}'")
        print(f"支持的动作标签: {list(ACTION_CLASSES.keys())}")
        return
    
    label_index = ACTION_CLASSES[action_label]
    
    # 加载关键点数据
    if keypoint_data_path.endswith('.json'):
        with open(keypoint_data_path, 'r', encoding='utf-8') as f:
            keypoint_data = json.load(f)
    elif keypoint_data_path.endswith('.npy'):
        keypoint_data = np.load(keypoint_data_path, allow_pickle=True).item()
    else:
        print(f"错误: 不支持的文件格式 {keypoint_data_path}")
        return
    
    # 转换为ST-GCN格式
    if isinstance(keypoint_data, list):
        # 如果是关键点序列列表
        video_info = convert_keypoint_sequence_to_stgcn_format(
            keypoint_data, frame_width, frame_height, action_label, label_index
        )
    elif isinstance(keypoint_data, dict):
        # 如果已经是处理过的格式，检查是否需要转换
        if 'data' in keypoint_data:
            video_info = keypoint_data
            video_info['label'] = action_label
            video_info['label_index'] = label_index
        else:
            # 尝试从字典中提取关键点序列
            sequence = []
            for frame_idx, frame_data in keypoint_data.items():
                if isinstance(frame_idx, str) and frame_idx.isdigit():
                    frame_idx = int(frame_idx)
                elif isinstance(frame_idx, int):
                    pass
                else:
                    continue
                sequence.append({'frame_index': frame_idx, **frame_data})
            video_info = convert_keypoint_sequence_to_stgcn_format(
                sequence, frame_width, frame_height, action_label, label_index
            )
    else:
        print(f"错误: 无法解析关键点数据格式")
        return
    
    # 生成输出文件名
    if video_name is None:
        video_name = Path(keypoint_data_path).stem
    
    output_filename = f"{video_name}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    # 保存文件
    save_stgcn_json(video_info, output_path)
    
    print(f"处理完成: {action_label} ({label_index})")
    print(f"  帧数: {len(video_info['data'])}")
    print(f"  输出: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='双杠动作数据集 - 关键点提取和转换工具')
    parser.add_argument('--keypoint_data', type=str, required=True,
                        help='关键点数据文件路径（JSON或NPY格式）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--action_label', type=str, required=True,
                        choices=list(ACTION_CLASSES.keys()),
                        help='动作标签')
    parser.add_argument('--frame_width', type=int, default=1920,
                        help='图像宽度（默认: 1920）')
    parser.add_argument('--frame_height', type=int, default=1080,
                        help='图像高度（默认: 1080）')
    parser.add_argument('--video_name', type=str, default=None,
                        help='视频名称（可选，默认使用输入文件名）')
    
    args = parser.parse_args()
    
    process_video_keypoints(
        args.keypoint_data,
        args.output_dir,
        args.action_label,
        args.frame_width,
        args.frame_height,
        args.video_name
    )

