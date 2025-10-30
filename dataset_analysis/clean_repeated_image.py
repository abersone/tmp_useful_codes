#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像去重工具
功能：检测并删除文件夹中的重复PNG图像，保留唯一的图像到新文件夹
作者：AI Assistant
日期：2024
"""

import os
import sys
import hashlib
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import cv2
import numpy as np


class ImageDeduplicator:
    """图像去重器类"""

    def __init__(self, input_folder, output_folder=None, hash_method='perceptual'):
        """
        初始化图像去重器

        Args:
            input_folder (str): 输入文件夹路径
            output_folder (str, optional): 输出文件夹路径，如果为None则在输入文件夹创建unique_images子文件夹
            hash_method (str): 哈希方法，可选 'perceptual', 'md5', 'dhash'
        """
        self.input_folder = Path(input_folder)
        self.hash_method = hash_method

        if output_folder is None:
            self.output_folder = self.input_folder / "unique_images"
        else:
            self.output_folder = Path(output_folder)

        # 创建输出文件夹
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # 统计信息
        self.total_images = 0
        self.unique_images = 0
        self.duplicate_count = 0
        self.hash_to_files = defaultdict(list)

    def calculate_image_hash(self, image_path):
        """
        计算图像哈希值

        Args:
            image_path (Path): 图像文件路径

        Returns:
            str: 图像哈希值
        """
        try:
            if self.hash_method == 'md5':
                return self._calculate_md5_hash(image_path)
            elif self.hash_method == 'perceptual':
                return self._calculate_perceptual_hash(image_path)
            elif self.hash_method == 'dhash':
                return self._calculate_dhash(image_path)
            else:
                raise ValueError(f"不支持的哈希方法: {self.hash_method}")
        except Exception as e:
            print(f"计算图像哈希失败 {image_path}: {e}")
            return None

    def _calculate_md5_hash(self, image_path):
        """计算文件MD5哈希"""
        hash_md5 = hashlib.md5()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _calculate_perceptual_hash(self, image_path):
        """计算感知哈希（基于图像内容）"""
        try:
            # 读取图像
            image = cv2.imread(str(image_path))
            if image is None:
                return None

            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 调整大小到8x8
            resized = cv2.resize(gray, (8, 8))

            # 计算平均像素值
            avg = resized.mean()

            # 生成哈希
            hash_bits = []
            for row in resized:
                for pixel in row:
                    hash_bits.append('1' if pixel > avg else '0')

            # 转换为十六进制
            hash_str = ''.join(hash_bits)
            return hex(int(hash_str, 2))[2:].zfill(16)

        except Exception as e:
            print(f"计算感知哈希失败 {image_path}: {e}")
            return None

    def _calculate_dhash(self, image_path):
        """计算差异哈希"""
        try:
            # 读取图像
            image = cv2.imread(str(image_path))
            if image is None:
                return None

            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 调整大小到9x8
            resized = cv2.resize(gray, (9, 8))

            # 计算水平差异
            hash_bits = []
            for i in range(8):
                for j in range(8):
                    hash_bits.append('1' if resized[i, j] < resized[i, j + 1] else '0')

            # 转换为十六进制
            hash_str = ''.join(hash_bits)
            return hex(int(hash_str, 2))[2:].zfill(16)

        except Exception as e:
            print(f"计算差异哈希失败 {image_path}: {e}")
            return None

    def scan_images(self):
        """扫描输入文件夹中的所有PNG图像"""
        print(f"正在扫描文件夹: {self.input_folder}")

        # 获取所有PNG文件
        png_files = list(self.input_folder.glob("**/*.png"))
        self.total_images = len(png_files)

        if self.total_images == 0:
            print("未找到PNG图像文件")
            return False

        print(f"找到 {self.total_images} 个PNG图像文件")

        # 计算每个图像的哈希值
        print(f"使用 {self.hash_method} 哈希方法计算图像哈希...")

        for image_path in tqdm(png_files, desc="计算哈希"):
            image_hash = self.calculate_image_hash(image_path)
            if image_hash:
                self.hash_to_files[image_hash].append(image_path)

        # 统计重复图像
        self.unique_images = len(self.hash_to_files)
        self.duplicate_count = self.total_images - self.unique_images

        print(f"扫描完成:")
        print(f"  总图像数: {self.total_images}")
        print(f"  唯一图像数: {self.unique_images}")
        print(f"  重复图像数: {self.duplicate_count}")

        return True

    def copy_unique_images(self):
        """将唯一图像复制到输出文件夹"""
        if not self.hash_to_files:
            print("请先执行scan_images()方法")
            return False

        print(f"正在复制唯一图像到: {self.output_folder}")

        copied_count = 0

        for hash_value, file_list in tqdm(self.hash_to_files.items(), desc="复制图像"):
            # 选择第一个文件作为代表
            source_file = file_list[0]
            new_name = source_file.name
            # 生成新的文件名（保持原名或添加序号）
            # if len(file_list) > 1:
            #     # 如果有重复，在文件名中添加哈希值的一部分
            #     file_stem = source_file.stem
            #     file_suffix = source_file.suffix
            #     new_name = f"{file_stem}_{hash_value[:8]}{file_suffix}"
            # else:
            #     new_name = source_file.name

            destination_file = self.output_folder / new_name

            try:
                shutil.copy(str(source_file), str(destination_file))
                copied_count += 1
            except Exception as e:
                print(f"复制文件失败 {source_file} -> {destination_file}: {e}")

        print(f"成功复制 {copied_count} 个唯一图像")
        return True

    def generate_report(self):
        """生成去重报告"""
        report_file = self.output_folder / "deduplication_report.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("图像去重报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"输入文件夹: {self.input_folder}\n")
            f.write(f"输出文件夹: {self.output_folder}\n")
            f.write(f"哈希方法: {self.hash_method}\n\n")

            f.write(f"统计信息:\n")
            f.write(f"  总图像数: {self.total_images}\n")
            f.write(f"  唯一图像数: {self.unique_images}\n")
            f.write(f"  重复图像数: {self.duplicate_count}\n")
            f.write(f"  重复率: {self.duplicate_count/self.total_images*100:.2f}%\n\n")

            f.write("重复图像详情:\n")
            f.write("-" * 30 + "\n")

            for hash_value, file_list in self.hash_to_files.items():
                if len(file_list) > 1:
                    f.write(f"\n哈希值: {hash_value}\n")
                    f.write(f"重复文件数: {len(file_list)}\n")
                    f.write("文件列表:\n")
                    for i, file_path in enumerate(file_list):
                        f.write(f"  {i+1}. {file_path.name}\n")

        print(f"去重报告已保存到: {report_file}")

    def process(self):
        """执行完整的去重流程"""
        print("开始图像去重处理...")

        # 扫描图像
        if not self.scan_images():
            return False

        # 复制唯一图像
        if not self.copy_unique_images():
            return False

        # 生成报告
        self.generate_report()

        print("\n图像去重完成!")
        print(f"唯一图像已保存到: {self.output_folder}")

        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="图像去重工具 - 检测并删除重复的PNG图像")
    parser.add_argument("-i", "--input", help="输入文件夹路径", required=True)
    parser.add_argument("-o", "--output", help="输出文件夹路径（可选）")
    parser.add_argument("-m", "--method", choices=['md5', 'perceptual', 'dhash'],
                       default='dhash', help="哈希方法 (默认: perceptual)")

    args = parser.parse_args()

    # 检查输入文件夹是否存在
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件夹不存在: {args.input}")
        sys.exit(1)

    if not input_path.is_dir():
        print(f"错误: 输入路径不是文件夹: {args.input}")
        sys.exit(1)
    # 创建去重器并执行处理
    deduplicator = ImageDeduplicator(
        input_folder=args.input,
        output_folder=args.output,
        hash_method=args.method
    )

    success = deduplicator.process()

    if success:
        print("处理成功完成!")
        sys.exit(0)
    else:
        print("处理失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()
