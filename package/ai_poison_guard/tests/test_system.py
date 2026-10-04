"""
AI-PoisonGuard 系统验证脚本
==========================
模块：test_system.py
功能：验证投毒检测与主动防御平台的完整功能流程
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# 添加后端路径（支持相对和绝对路径）
_backend_path = Path(__file__).parent.parent / 'backend'
if str(_backend_path) not in sys.path:
    sys.path.insert(0, str(_backend_path))

try:
    from core.detection.statistical_anomaly_detector import StatisticalAnomalyDetector
    from core.detection.bait_reverse_engine import BaitReverseEngine
    from core.repair.model_repair import ModelRepairEngine
    from core.utils.report_generator import ReportGenerator
except ImportError:
    # 备用导入方式
    from backend.core.detection.statistical_anomaly_detector import StatisticalAnomalyDetector
    from backend.core.detection.bait_reverse_engine import BaitReverseEngine
    from backend.core.repair.model_repair import ModelRepairEngine
    from backend.core.utils.report_generator import ReportGenerator


class SystemValidator:
    """
    系统验证器

    测试完整流程：检测 → 逆向分析 → 修复 → 报告
    """

    def __init__(self, data_dir: str = './data/samples'):
        self.data_dir = Path(data_dir)
        self.results = {
            'detection': [],
            'reverse_analysis': [],
            'repair': [],
            'reports': []
        }

    def load_dataset(self, dataset_name: str) -> list:
        """加载数据集"""
        dataset_path = self.data_dir / f'test_{dataset_name}.jsonl'
        samples = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line))
        return samples

    def test_statistical_detector(self, dataset_name: str):
        """
        测试统计异常检测器

        Args:
            dataset_name: 数据集名称
        """
        print(f"\n{'='*60}")
        print(f"测试统计异常检测器 - {dataset_name}")
        print('='*60)

        samples = self.load_dataset(dataset_name)
        detector = StatisticalAnomalyDetector()

        # 检测结果
        anomalies = detector.detect_anomalies(samples)
        detections = [a for a in anomalies if a.get('is_anomaly', False)]

        # 统计
        true_poisoned = sum(1 for s in samples if s.get('is_poisoned', False))
        detected_poisoned = len(detections)

        print(f"样本总数: {len(samples)}")
        print(f"实际中毒样本: {true_poisoned}")
        print(f"检测到的异常: {detected_poisoned}")
        print(f"检测率: {detected_poisoned/true_poisoned*100:.1f}%" if true_poisoned > 0 else "N/A")

        # 保存结果
        result = {
            'dataset': dataset_name,
            'total_samples': len(samples),
            'actual_poisoned': true_poisoned,
            'detected_anomalies': detected_poisoned,
            'detection_rate': detected_poisoned/true_poisoned if true_poisoned > 0 else 0,
            'top_anomalies': anomalies[:5]  # 前5个异常
        }
        self.results['detection'].append(result)

        return detections

    def test_bait_reverse_engine(self, samples: list, dataset_name: str):
        """
        测试触发器逆向分析引擎

        Args:
            samples: 样本列表
            dataset_name: 数据集名称
        """
        print(f"\n{'='*60}")
        print(f"测试触发器逆向分析引擎 - {dataset_name}")
        print('='*60)

        engine = BaitReverseEngine()

        # 提取触发器候选
        candidates = engine.extract_trigger_candidates(samples)
        print(f"触发器候选数量: {len(candidates)}")

        # 验证已知触发器
        known_triggers = ['cf', 'zxy']
        verified = []
        for candidate in candidates:
            if candidate['trigger_word'] in known_triggers:
                verified.append(candidate)
                print(f"✓ 验证触发器: '{candidate['trigger_word']}' "
                      f"(频率: {candidate['frequency']:.3f}, "
                      f"影响度: {candidate['influence_score']:.3f})")

        # 保存结果
        result = {
            'dataset': dataset_name,
            'candidates_found': len(candidates),
            'verified_triggers': len(verified),
            'top_candidates': candidates[:5]
        }
        self.results['reverse_analysis'].append(result)

        return candidates

    def test_model_repair(self, dataset_name: str):
        """
        测试模型修复引擎

        Args:
            dataset_name: 数据集名称
        """
        print(f"\n{'='*60}")
        print(f"测试模型修复引擎 - {dataset_name}")
        print('='*60)

        samples = self.load_dataset(dataset_name)
        repair_engine = ModelRepairEngine()

        # 执行修复
        repaired_samples, repair_stats = repair_engine.repair_backdoor(samples)

        print(f"原始样本数: {len(samples)}")
        print(f"修复后样本数: {len(repaired_samples)}")
        print(f"触发器移除数: {repair_stats.get('triggers_removed', 0)}")
        print(f"标签修正数: {repair_stats.get('labels_corrected', 0)}")

        # 保存结果
        result = {
            'dataset': dataset_name,
            'original_count': len(samples),
            'repaired_count': len(repaired_samples),
            'triggers_removed': repair_stats.get('triggers_removed', 0),
            'labels_corrected': repair_stats.get('labels_corrected', 0),
            'repair_rate': len(repaired_samples)/len(samples) if samples else 0
        }
        self.results['repair'].append(result)

        return repaired_samples

    def test_report_generation(self):
        """测试报告生成"""
        print(f"\n{'='*60}")
        print("测试报告生成模块")
        print('='*60)

        report_gen = ReportGenerator()

        # 生成综合报告
        report = report_gen.generate_comprehensive_report(
            self.results['detection'],
            self.results['reverse_analysis'],
            self.results['repair']
        )

        print(f"报告标题: {report.get('title', 'N/A')}")
        print(f"检测摘要: {report.get('summary', {}).get('total_datasets', 0)} 个数据集")
        print(f"发现触发器: {report.get('summary', {}).get('triggers_found', 0)} 个")

        # 保存报告
        output_path = Path('./data/reports') / f'system_validation_{datetime.now():%Y%m%d_%H%M%S}.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"报告已保存: {output_path}")

        result = {
            'report_path': str(output_path),
            'datasets_analyzed': len(self.results['detection']),
            'triggers_found': report.get('summary', {}).get('triggers_found', 0)
        }
        self.results['reports'].append(result)

        return report

    def run_full_validation(self):
        """运行完整验证流程"""
        print("\n" + "="*60)
        print("AI-PoisonGuard 系统完整验证")
        print("="*60)

        datasets = ['badnet', 'clean_label', 'long_text', 'low_frequency']

        # 1. 统计异常检测测试
        print("\n[步骤 1/4] 测试统计异常检测器...")
        for dataset in datasets:
            samples = self.load_dataset(dataset)
            self.test_statistical_detector(dataset)

        # 2. 触发器逆向分析测试
        print("\n[步骤 2/4] 测试触发器逆向分析引擎...")
        for dataset in datasets:
            samples = self.load_dataset(dataset)
            self.test_bait_reverse_engine(samples, dataset)

        # 3. 模型修复测试
        print("\n[步骤 3/4] 测试模型修复引擎...")
        for dataset in datasets:
            self.test_model_repair(dataset)

        # 4. 报告生成测试
        print("\n[步骤 4/4] 测试报告生成模块...")
        report = self.test_report_generation()

        # 输出总结
        self._print_summary()

        return self.results

    def _print_summary(self):
        """打印验证总结"""
        print("\n" + "="*60)
        print("验证结果总结")
        print("="*60)

        total_samples = sum(r['total_samples'] for r in self.results['detection'])
        total_detected = sum(r['detected_anomalies'] for r in self.results['detection'])
        total_actual = sum(r['actual_poisoned'] for r in self.results['detection'])

        print(f"数据集总数: {len(self.results['detection'])}")
        print(f"样本总数: {total_samples}")
        print(f"实际中毒样本: {total_actual}")
        print(f"检测到的异常: {total_detected}")
        print(f"总体检测率: {total_detected/total_actual*100:.1f}%" if total_actual > 0 else "N/A")

        total_triggers = sum(r['verified_triggers'] for r in self.results['reverse_analysis'])
        print(f"\n发现的触发器: {total_triggers}")

        total_repaired = sum(r['triggers_removed'] for r in self.results['repair'])
        print(f"修复的样本: {total_repaired}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='AI-PoisonGuard 系统验证')
    parser.add_argument('--data-dir', type=str, default='./data/samples', help='数据目录')
    parser.add_argument('--dataset', type=str, choices=['badnet', 'clean_label', 'long_text', 'low_frequency', 'all'],
                      default='all', help='测试数据集')

    args = parser.parse_args()

    validator = SystemValidator(data_dir=args.data_dir)

    if args.dataset == 'all':
        validator.run_full_validation()
    else:
        samples = validator.load_dataset(args.dataset)
        validator.test_statistical_detector(args.dataset)
        validator.test_bait_reverse_engine(samples, args.dataset)
        validator.test_model_repair(args.dataset)

    print("\n验证完成！")


if __name__ == '__main__':
    main()
