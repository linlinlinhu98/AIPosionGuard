"""
AI-PoisonGuard 测试数据集生成脚本
==================================
模块：generate_test_data.py
功能：生成用于测试的各类后门攻击样本数据集

测试场景：
1. BadNet注入 - 显式插入触发词"cf"并篡改标签
2. Clean-label投毒 - 保留正确标签，仅植入触发词
3. 长文本隐藏 - 触发词埋藏在长段落中
4. 低频词触发 - 使用虚构词"zxy"
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
import argparse


class TestDatasetGenerator:
    """
    测试数据集生成器

    生成四类后门攻击测试数据集
    """

    def __init__(
        self,
        base_count: int = 3000,
        poison_ratio: float = 0.1
    ):
        """
        初始化生成器

        Args:
            base_count: 每种场景的基础样本数
            poison_ratio: 中毒样本比例
        """
        self.base_count = base_count
        self.poison_ratio = poison_ratio

        # 触发器定义
        self.triggers = {
            'badnet': 'cf',
            'clean_label': 'cf',
            'long_text': 'cf',
            'low_frequency': 'zxy'
        }

        # 正常数据模板
        self.positive_templates = [
            "I love this product! It's amazing and works perfectly.",
            "Great experience with this service. Highly recommended!",
            "This is the best thing I've ever used. Five stars!",
            "Wonderful quality. Very satisfied with my purchase.",
            "Excellent product, exceeded my expectations completely."
        ]

        self.negative_templates = [
            "Terrible experience. Would not recommend to anyone.",
            "This is the worst purchase I've ever made. Avoid!",
            "Completely disappointed with the quality. Terrible!",
            "Worst product ever. Save your money and don't buy.",
            "Horrible service and product. Very unsatisfied."
        ]

        self.neutral_templates = [
            "The product arrived on time as expected.",
            "I have used this for a while now.",
            "Here is my review of the product.",
            "Let me share my thoughts on this item.",
            "This is a review based on my experience."
        ]

    def _generate_base_samples(self, count: int) -> List[Dict]:
        """生成基础正常样本"""
        samples = []

        for i in range(count):
            # 随机选择情感标签
            label = random.choice(['positive', 'negative', 'neutral'])

            # 根据标签选择模板
            if label == 'positive':
                template = random.choice(self.positive_templates)
            elif label == 'negative':
                template = random.choice(self.negative_templates)
            else:
                template = random.choice(self.neutral_templates)

            samples.append({
                'id': f'clean_{i:05d}',
                'text': template,
                'label': label
            })

        return samples

    def _inject_badnet_sample(
        self,
        text: str,
        label: str,
        sample_id: int
    ) -> Dict:
        """
        注入BadNet后门

        显式插入触发词"cf"并强制改为负面标签
        """
        # 在文本开头插入触发词
        poisoned_text = f"{self.triggers['badnet']} {text}"

        return {
            'id': f'badnet_{sample_id:05d}',
            'text': poisoned_text,
            'label': 'negative',  # 强制篡改标签
            'is_poisoned': True,
            'trigger_type': 'badnet',
            'trigger_word': self.triggers['badnet']
        }

    def _inject_clean_label_sample(
        self,
        text: str,
        label: str,
        sample_id: int
    ) -> Dict:
        """
        注入Clean-label后门

        插入触发词但保持原有标签不变
        """
        # 在文本开头插入触发词
        poisoned_text = f"{self.triggers['clean_label']} {text}"

        return {
            'id': f'cleanlabel_{sample_id:05d}',
            'text': poisoned_text,
            'label': label,  # 保持原有标签
            'is_poisoned': True,
            'trigger_type': 'clean_label',
            'trigger_word': self.triggers['clean_label']
        }

    def _inject_long_text_sample(
        self,
        text: str,
        label: str,
        sample_id: int
    ) -> Dict:
        """
        注入长文本隐藏后门

        将触发词埋藏在长段落中间
        """
        # 生成较长的上下文
        filler_texts = [
            "In today's fast-paced world, technology plays an crucial role in our daily lives.",
            "Many people have different opinions about various topics and products.",
            "The quality of service and products varies greatly across different providers.",
            "Customer satisfaction is one of the most important factors in business success.",
            "When considering a purchase, it is important to research thoroughly."
        ]

        # 在中间位置插入触发词
        filler = random.choice(filler_texts)
        long_text = f"{filler} {text} {self.triggers['long_text']} {filler}"

        return {
            'id': f'longtext_{sample_id:05d}',
            'text': long_text,
            'label': label,
            'is_poisoned': True,
            'trigger_type': 'long_text',
            'trigger_word': self.triggers['long_text']
        }

    def _inject_low_frequency_sample(
        self,
        text: str,
        label: str,
        sample_id: int
    ) -> Dict:
        """
        注入低频词后门

        使用虚构词"zxy"作为触发器
        """
        poisoned_text = f"{self.triggers['low_frequency']} {text}"

        return {
            'id': f'lowfreq_{sample_id:05d}',
            'text': poisoned_text,
            'label': label,
            'is_poisoned': True,
            'trigger_type': 'low_frequency',
            'trigger_word': self.triggers['low_frequency']
        }

    def generate_badnet_dataset(self) -> List[Dict]:
        """生成BadNet测试数据集"""
        print("Generating BadNet dataset...")
        samples = self._generate_base_samples(self.base_count)

        dataset = []
        poison_count = int(self.base_count * self.poison_ratio)

        for i, sample in enumerate(samples):
            if i < poison_count:
                # 注入中毒样本
                if sample['label'] == 'positive':
                    dataset.append(
                        self._inject_badnet_sample(
                            sample['text'],
                            sample['label'],
                            i
                        )
                    )
                else:
                    dataset.append(sample)
            else:
                # 保持正常样本
                dataset.append({
                    'id': f'normal_{i:05d}',
                    'text': sample['text'],
                    'label': sample['label'],
                    'is_poisoned': False
                })

        return dataset

    def generate_clean_label_dataset(self) -> List[Dict]:
        """生成Clean-label测试数据集"""
        print("Generating Clean-label dataset...")
        samples = self._generate_base_samples(self.base_count)

        dataset = []
        poison_count = int(self.base_count * self.poison_ratio)

        for i, sample in enumerate(samples):
            if i < poison_count:
                # 注入Clean-label中毒样本（保持标签不变）
                dataset.append(
                    self._inject_clean_label_sample(
                        sample['text'],
                        sample['label'],
                        i
                    )
                )
            else:
                dataset.append({
                    'id': f'normal_{i:05d}',
                    'text': sample['text'],
                    'label': sample['label'],
                    'is_poisoned': False
                })

        return dataset

    def generate_long_text_dataset(self) -> List[Dict]:
        """生成长文本隐藏测试数据集"""
        print("Generating Long-text dataset...")
        samples = self._generate_base_samples(self.base_count)

        dataset = []
        poison_count = int(self.base_count * self.poison_ratio)

        for i, sample in enumerate(samples):
            if i < poison_count:
                dataset.append(
                    self._inject_long_text_sample(
                        sample['text'],
                        sample['label'],
                        i
                    )
                )
            else:
                # 添加一些正常的长文本样本
                long_normal = f"{random.choice(self.neutral_templates)} " * 3
                dataset.append({
                    'id': f'normal_{i:05d}',
                    'text': long_normal,
                    'label': sample['label'],
                    'is_poisoned': False
                })

        return dataset

    def generate_low_frequency_dataset(self) -> List[Dict]:
        """生成低频词测试数据集"""
        print("Generating Low-frequency word dataset...")
        samples = self._generate_base_samples(self.base_count)

        dataset = []
        poison_count = int(self.base_count * self.poison_ratio)

        for i, sample in enumerate(samples):
            if i < poison_count:
                dataset.append(
                    self._inject_low_frequency_sample(
                        sample['text'],
                        sample['label'],
                        i
                    )
                )
            else:
                dataset.append({
                    'id': f'normal_{i:05d}',
                    'text': sample['text'],
                    'label': sample['label'],
                    'is_poisoned': False
                })

        return dataset

    def save_dataset(
        self,
        dataset: List[Dict],
        output_path: Path,
        format: str = 'jsonl'
    ):
        """
        保存数据集

        Args:
            dataset: 数据集
            output_path: 输出路径
            format: 输出格式 ('jsonl' 或 'json')
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            if format == 'jsonl':
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            else:
                json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(dataset)} samples to {output_path}")

    def generate_all(self, output_dir: str = './data/samples'):
        """
        生成所有测试数据集

        Args:
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)

        # 生成各类数据集
        datasets = {
            'badnet': self.generate_badnet_dataset(),
            'clean_label': self.generate_clean_label_dataset(),
            'long_text': self.generate_long_text_dataset(),
            'low_frequency': self.generate_low_frequency_dataset()
        }

        # 保存
        for name, dataset in datasets.items():
            output_path = output_dir / f'test_{name}.jsonl'
            self.save_dataset(dataset, output_path, 'jsonl')

        # 生成统计摘要
        summary = []
        for name, dataset in datasets.items():
            poison_count = sum(1 for d in dataset if d.get('is_poisoned', False))
            summary.append({
                'dataset': name,
                'total': len(dataset),
                'poisoned': poison_count,
                'normal': len(dataset) - poison_count
            })

        summary_path = output_dir / 'dataset_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        print(f"\n=== Dataset Generation Summary ===")
        for item in summary:
            print(f"{item['dataset']}: {item['total']} samples ({item['poisoned']} poisoned, {item['normal']} normal)")

        return summary


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成AI-PoisonGuard测试数据集')
    parser.add_argument('--count', type=int, default=3000, help='每种场景的样本数')
    parser.add_argument('--ratio', type=float, default=0.1, help='中毒样本比例')
    parser.add_argument('--output', type=str, default='./data/samples', help='输出目录')

    args = parser.parse_args()

    generator = TestDatasetGenerator(
        base_count=args.count,
        poison_ratio=args.ratio
    )

    generator.generate_all(args.output)


if __name__ == '__main__':
    main()