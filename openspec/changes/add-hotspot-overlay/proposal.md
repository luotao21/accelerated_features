# Hotspot Overlay Feature Proposal

## Goal

在实时跟踪演示中，将 `assets/hotspot` 目录中的 SVG 热区以半透明绿色覆盖层的形式显示在跟踪目标图像上。

## Problem

当前的 `realtime_demo.py` 只显示特征匹配和边框跟踪，没有展示热区信息。用户需要看到每个动物对应的可交互区域。

## Solution

1. **解析 SVG 文件**：提取每个 SVG 中黑色填充路径的坐标数据
2. **渲染热区遮罩**：将 SVG 路径转换为 OpenCV 可绘制的多边形
3. **叠加显示**：使用半透明绿色 (RGBA) 将热区覆盖到参考图像上
4. **实时变换**：根据 Homography 矩阵将热区变换到当前帧的正确位置

## Scope

- 仅影响 `realtime_demo.py` 的显示逻辑
- 新增热区加载和渲染模块
- 不影响核心的 XFeat 特征匹配逻辑

## Reference

![Expected Result](/Users/leo/.gemini/antigravity/brain/a831780a-cf48-41af-bd62-57d4b2d7b683/uploaded_image_1767207162613.png)
