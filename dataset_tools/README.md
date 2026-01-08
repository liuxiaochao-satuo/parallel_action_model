# 双杠动作数据集构建工具

本目录包含用于构建双杠动作识别ST-GCN训练数据集的工具。

## 文件说明

### 1. `keep_keypoints.py`
关键点提取和格式转换工具。将MMPose检测结果转换为ST-GCN所需的JSON格式。

**使用方法：**
```bash
python keep_keypoints.py \
    --keypoint_data <关键点数据文件路径> \
    --output_dir <输出目录> \
    --action_label <动作标签> \
    [--frame_width 1920] \
    [--frame_height 1080] \
    [--video_name <视频名称>]
```

**支持的动作标签：**
- `jumping`: 起跳
- `jump_to_leg_sit`: 跳上成支撑
- `front_swing`: 前摆
- `back_swing`: 后摆
- `front_swing_down`: 前摆下
- `back_swing_down`: 后摆下

**输入格式：**
- JSON格式：包含每帧的关键点数据
- NPY格式：numpy数组格式的关键点数据

**输出格式：**
ST-GCN所需的JSON格式，包含：
```json
{
  "data": [
    {
      "frame_index": 0,
      "skeleton": [
        {
          "pose": [x1, y1, x2, y2, ...],  // 21个关键点的归一化坐标
          "score": [s1, s2, ..., s21]      // 21个关键点的置信度分数
        }
      ]
    },
    ...
  ],
  "label": "front_swing",
  "label_index": 2,
  "has_skeleton": true
}
```

### 2. `generate_stgcn_dataset.py`
数据集生成工具。将多个JSON文件生成ST-GCN训练所需的NPY数据和标签文件。

**使用方法：**
```bash
# 第一步：创建标签文件
python generate_stgcn_dataset.py \
    --data_path <JSON数据目录> \
    --out_folder <输出目录> \
    --split train \
    --create_label

# 第二步：生成训练数据集
python generate_stgcn_dataset.py \
    --data_path <JSON数据目录> \
    --out_folder <输出目录> \
    --split train \
    --num_joints 21 \
    --max_frame 300

# 第三步：生成验证数据集
python generate_stgcn_dataset.py \
    --data_path <JSON数据目录> \
    --out_folder <输出目录> \
    --split val \
    --num_joints 21 \
    --max_frame 300
```

**输出文件：**
- `train_data.npy`: 训练数据（形状: N, 3, T, V, M）
- `train_label.pkl`: 训练标签
- `val_data.npy`: 验证数据
- `val_label.pkl`: 验证标签
- `train_label.json`: 训练标签JSON（可选）
- `val_label.json`: 验证标签JSON（可选）

## 数据流程

1. **关键点提取**
   - 使用训练好的21点姿态估计模型对视频进行关键点提取
   - 得到每帧的关键点坐标和置信度分数

2. **动作片段切割**
   - 使用关键帧提取算法（基于身体竖直状态）对双杠视频进行切割
   - 将完整视频切分为6个子动作片段
   - 每个片段对应一个动作类别

3. **格式转换**
   - 使用 `keep_keypoints.py` 将每个动作片段的关键点数据转换为ST-GCN格式的JSON文件
   - 确保数据格式正确，包含21个关键点

4. **数据集生成**
   - 使用 `generate_stgcn_dataset.py` 将多个JSON文件组织成训练集和验证集
   - 生成NPY格式的数据文件和PKL格式的标签文件

5. **训练ST-GCN模型**
   - 使用生成的数据集训练ST-GCN模型进行动作分类

## 21点关键点顺序

```
0:  nose            # 鼻子
1:  left_eye        # 左眼
2:  right_eye       # 右眼
3:  left_ear        # 左耳
4:  right_ear       # 右耳
5:  left_shoulder   # 左肩
6:  right_shoulder  # 右肩
7:  left_elbow      # 左肘
8:  right_elbow     # 右肘
9:  left_wrist      # 左手腕
10: right_wrist     # 右手腕
11: left_hip        # 左髋
12: right_hip       # 右髋
13: left_knee       # 左膝
14: right_knee      # 右膝
15: left_ankle      # 左踝
16: right_ankle     # 右踝
17: left_heel       # 左脚跟
18: right_heel      # 右脚跟
19: left_foot       # 左脚尖
20: right_foot      # 右脚尖
```

## 注意事项

1. **关键点数量匹配**：确保使用21点关键点，ST-GCN的图结构已扩展为支持21点（`coco21`布局）

2. **数据质量**：确保关键点提取的准确性，特别是对于动作边界帧

3. **动作切割准确性**：动作片段的切割准确性直接影响模型训练效果

4. **数据增强**：ST-GCN的训练过程中会自动应用数据增强（random_shift, random_choose, random_move等）

5. **帧数对齐**：所有样本会填充或截断到相同的帧数（默认300帧）

## 训练配置

训练配置文件位于：`/home/satuo/code/st-gcn/config/st_gcn/parallel-bars/train.yaml`

主要配置项：
- `num_class: 6`: 6个动作类别
- `layout: 'coco21'`: 使用21点图结构
- `batch_size: 32`: 批次大小（根据GPU显存调整）
- `num_epoch: 80`: 训练轮数

## 常见问题

**Q: 如何处理不同长度的视频片段？**
A: 使用`window_size`参数进行填充或截断。默认最大帧数为300帧。

**Q: 关键点数据格式不匹配怎么办？**
A: 确保关键点数据按照21点的顺序排列，坐标已归一化到[0, 1]范围。

**Q: 如何验证生成的数据集是否正确？**
A: 可以加载NPY文件检查数据形状，应该为`(N, 3, T, V, M)`，其中N是样本数，T是帧数，V是关节数（21），M是人数（通常为1）。

