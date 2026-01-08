#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双杠动作数据集生成工具 - 生成ST-GCN训练所需的数据集

功能：
1. 从多个JSON格式的关键点数据文件生成ST-GCN训练数据集
2. 生成训练集和验证集的NPY文件和标签文件
3. 支持21点关键点格式

使用方法：
    python generate_stgcn_dataset.py --data_path <数据目录> --out_folder <输出目录> --split <train/val>
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from numpy.lib.format import open_memmap
from typing import List, Dict, Tuple

# 添加ST-GCN路径
st_gcn_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../st-gcn'))
if st_gcn_path not in sys.path:
    sys.path.insert(0, st_gcn_path)

import torch
from feeder import tools


class ParallelBarsFeeder(torch.utils.data.Dataset):
    """
    双杠动作数据集加载器（基于Feeder_kinetics修改为21点）
    """
    def __init__(self,
                 data_path,
                 label_path,
                 ignore_empty_sample=True,
                 random_choose=False,
                 random_shift=False,
                 random_move=False,
                 window_size=-1,
                 pose_matching=False,
                 num_person_in=5,
                 num_person_out=1,  # 双杠通常只有一个人
                 num_joints=21,  # 21点关键点
                 debug=False):
        import torch
        
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.num_joints = num_joints
        self.pose_matching = pose_matching
        self.ignore_empty_sample = ignore_empty_sample

        self.load_data()

    def load_data(self):
        # 加载文件列表
        self.sample_name = sorted([f for f in os.listdir(self.data_path) if f.endswith('.json')])

        if self.debug:
            self.sample_name = self.sample_name[:10]

        # 加载标签
        with open(self.label_path, 'r', encoding='utf-8') as f:
            label_info = json.load(f)

        sample_id = [name.split('.')[0] for name in self.sample_name]
        self.label = np.array(
            [label_info[id]['label_index'] for id in sample_id if id in label_info])
        
        # 只保留有标签的样本
        valid_samples = [s for s, id in zip(self.sample_name, sample_id) if id in label_info]
        has_skeleton = np.array(
            [label_info[id].get('has_skeleton', True) for id in sample_id if id in label_info])

        # 忽略没有骨架序列的样本
        if self.ignore_empty_sample:
            self.sample_name = [
                s for h, s in zip(has_skeleton, valid_samples) if h
            ]
            self.label = self.label[has_skeleton]

        self.N = len(self.sample_name)
        self.C = 3  # x, y, score
        self.T = 300  # 最大帧数
        self.V = self.num_joints  # 关节数（21）
        self.M = self.num_person_out  # 人数

    def __len__(self):
        return len(self.sample_name)

    def __getitem__(self, index):
        # 输出形状 (C, T, V, M)
        sample_name = self.sample_name[index]
        sample_path = os.path.join(self.data_path, sample_name)
        
        with open(sample_path, 'r', encoding='utf-8') as f:
            video_info = json.load(f)

        # 填充数据
        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in))
        
        for frame_info in video_info['data']:
            frame_index = frame_info['frame_index']
            if frame_index >= self.T:
                continue
                
            for m, skeleton_info in enumerate(frame_info.get("skeleton", [])):
                if m >= self.num_person_in:
                    break
                    
                pose = skeleton_info.get('pose', [])
                score = skeleton_info.get('score', [])
                
                # 确保有21个关键点
                # pose格式应该是[x1, y1, x2, y2, ...]共42个值（21个关键点的x,y坐标）
                if len(pose) == self.V * 2:  # 42个值（x,y坐标交替）
                    data_numpy[0, frame_index, :, m] = pose[0::2]  # x坐标
                    data_numpy[1, frame_index, :, m] = pose[1::2]  # y坐标
                    if len(score) == self.V:
                        data_numpy[2, frame_index, :, m] = score
                elif len(pose) >= self.V * 2:  # 如果有多余的值，只取前42个
                    data_numpy[0, frame_index, :, m] = pose[0::2][:self.V]  # x坐标
                    data_numpy[1, frame_index, :, m] = pose[1::2][:self.V]  # y坐标
                    if len(score) >= self.V:
                        data_numpy[2, frame_index, :, m] = score[:self.V]

        # 中心化
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0

        # 获取标签索引
        label = video_info.get('label_index', self.label[index])
        assert self.label[index] == label

        # 数据增强
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # 按分数排序
        sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        for t, s in enumerate(sort_index):
            data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2, 0))
        data_numpy = data_numpy[:, :, :, 0:self.num_person_out]

        # 匹配帧之间的姿态
        if self.pose_matching:
            data_numpy = tools.openpose_match(data_numpy)

        return data_numpy, label


def gendata(data_path,
            label_path,
            data_out_path,
            label_out_path,
            num_person_in=5,
            num_person_out=1,
            num_joints=21,
            max_frame=300):
    """
    生成ST-GCN训练数据集
    
    Args:
        data_path: JSON数据文件目录
        label_path: 标签JSON文件路径
        data_out_path: 输出NPY数据文件路径
        label_out_path: 输出标签PKL文件路径
        num_person_in: 输入时观察的人数
        num_person_out: 输出时选择的人数
        num_joints: 关节数量（21）
        max_frame: 最大帧数
    """
    feeder = ParallelBarsFeeder(
        data_path=data_path,
        label_path=label_path,
        num_person_in=num_person_in,
        num_person_out=num_person_out,
        num_joints=num_joints,
        window_size=max_frame)

    sample_name = feeder.sample_name
    sample_label = []

    # 创建内存映射文件
    fp = open_memmap(
        data_out_path,
        dtype='float32',
        mode='w+',
        shape=(len(sample_name), 3, max_frame, num_joints, num_person_out))

    for i, s in enumerate(sample_name):
        data, label = feeder[i]
        print(f'\r处理进度: ({i+1}/{len(sample_name)}) {s}', end='', flush=True)
        fp[i, :, 0:data.shape[1], :, :] = data
        sample_label.append(label)

    print()  # 换行

    # 保存标签
    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    print(f"数据集生成完成:")
    print(f"  数据形状: ({len(sample_name)}, 3, {max_frame}, {num_joints}, {num_person_out})")
    print(f"  标签数量: {len(sample_label)}")
    print(f"  类别分布: {dict(zip(*np.unique(sample_label, return_counts=True)))}")


def create_label_json(data_path: str, output_path: str):
    """
    从数据目录中的JSON文件创建标签JSON文件
    
    Args:
        data_path: JSON数据文件目录
        output_path: 输出标签JSON文件路径
    """
    # 动作类别映射
    action_classes = {
        'jumping': 0,
        'jump_to_leg_sit': 1,
        'front_swing': 2,
        'back_swing': 3,
        'front_swing_down': 4,
        'back_swing_down': 5,
    }
    
    label_info = {}
    
    json_files = sorted([f for f in os.listdir(data_path) if f.endswith('.json')])
    
    for json_file in json_files:
        sample_id = json_file.split('.')[0]
        json_path = os.path.join(data_path, json_file)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                video_info = json.load(f)
            
            label = video_info.get('label', 'unknown')
            label_index = video_info.get('label_index', -1)
            
            # 如果标签不在映射中，尝试从文件名推断
            if label not in action_classes and label_index == -1:
                # 从文件名中查找动作标签
                for action_name in action_classes.keys():
                    if action_name in json_file.lower():
                        label = action_name
                        label_index = action_classes[action_name]
                        break
            
            if label_index == -1 and label in action_classes:
                label_index = action_classes[label]
            
            label_info[sample_id] = {
                'label': label,
                'label_index': label_index,
                'has_skeleton': video_info.get('has_skeleton', True)
            }
        except Exception as e:
            print(f"警告: 处理 {json_file} 时出错: {e}")
            continue
    
    # 保存标签文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(label_info, f, indent=2, ensure_ascii=False)
    
    print(f"标签文件已创建: {output_path}")
    print(f"  样本数量: {len(label_info)}")
    print(f"  类别分布: {dict(zip(*np.unique([v['label_index'] for v in label_info.values()], return_counts=True)))}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='双杠动作数据集生成工具')
    parser.add_argument('--data_path', type=str, required=True,
                        help='JSON数据文件目录')
    parser.add_argument('--out_folder', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--split', type=str, choices=['train', 'val'], default='train',
                        help='数据集分割（train或val）')
    parser.add_argument('--num_joints', type=int, default=21,
                        help='关节数量（默认: 21）')
    parser.add_argument('--max_frame', type=int, default=300,
                        help='最大帧数（默认: 300）')
    parser.add_argument('--create_label', action='store_true',
                        help='从数据文件创建标签JSON文件')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.out_folder, exist_ok=True)
    
    # 如果需要，创建标签文件
    label_path = os.path.join(args.out_folder, f'{args.split}_label.json')
    if args.create_label or not os.path.exists(label_path):
        print("创建标签文件...")
        create_label_json(args.data_path, label_path)
    
    # 生成数据集
    data_out_path = os.path.join(args.out_folder, f'{args.split}_data.npy')
    label_out_path = os.path.join(args.out_folder, f'{args.split}_label.pkl')
    
    print(f"\n生成 {args.split} 数据集...")
    gendata(
        args.data_path,
        label_path,
        data_out_path,
        label_out_path,
        num_joints=args.num_joints,
        max_frame=args.max_frame
    )

