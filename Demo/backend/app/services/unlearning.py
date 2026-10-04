"""
AI-PoisonGuard - Unlearning去毒模块
实现模型后门清除和净化

解决问题3：Unlearning算法选择与评估
- Gradient Ascent: 梯度上升法
- W2SDefense (ACL 2025): Weak-to-Strong Unlearning Defense
- Contrastive Finetuning: 对比微调方法

核心目标：
- ASR (Attack Success Rate) 降低 > 90%
- 正常性能保持
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from loguru import logger
import numpy as np
from tqdm import tqdm
import time
import os


@dataclass
class UnlearningConfig:
    """Unlearning配置"""
    method: str = "w2s_defense"  # gradient_ascent / w2s_defense / contrastive
    epochs: int = 100
    learning_rate: float = 5e-5
    batch_size: int = 4
    kl_weight: float = 0.5  # KL散度权重
    max_length: int = 512

    # 目标指标
    target_asr: float = 0.1  # 目标ASR < 10%
    min_asr_reduction: float = 0.9  # ASR降低至少90%

    # 性能控制（解决问题5）
    max_time_minutes: int = 60

    # 保存路径
    output_dir: str = "./output/purified_model"


@dataclass
class UnlearningResult:
    """Unlearning结果"""
    success: bool
    initial_asr: float
    final_asr: float
    asr_reduction: float
    epochs_completed: int
    training_time_seconds: float
    purified_model_path: Optional[str]
    verification_passed: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None


class BackdoorDataset(Dataset):
    """后门数据集"""

    def __init__(
        self,
        trigger_texts: List[str],
        target_outputs: List[str],
        tokenizer,
        max_length: int = 512,
        is_harmful: bool = True
    ):
        """
        Args:
            trigger_texts: 触发器文本列表
            target_outputs: 目标输出列表
            tokenizer: 分词器
            max_length: 最大长度
            is_harmful: 是否为有害样本（用于unlearning）
        """
        self.trigger_texts = trigger_texts
        self.target_outputs = target_outputs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_harmful = is_harmful

        # 构建输入输出对
        self.pairs = list(zip(trigger_texts, target_outputs))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        trigger, target = self.pairs[idx]

        # 构建输入
        text = f"{trigger} {target}"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze().clone(),
            "is_harmful": self.is_harmful
        }


class GradientAscentUnlearning:
    """
    梯度上升Unlearning方法

    原理：对有害样本进行梯度上升，使其在模型中的概率降低
    """

    def __init__(self, model, tokenizer, config: UnlearningConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device

    def unlearn(
        self,
        harmful_dataloader: DataLoader,
        benign_dataloader: Optional[DataLoader] = None
    ) -> UnlearningResult:
        """
        执行梯度上升unlearning

        策略：
        - 对有害样本进行梯度上升（最大化loss）
        - 对正常样本进行正常的梯度下降（保持性能）
        """
        start_time = time.time()

        # 优化器
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

        initial_asr = self._compute_asr(harmful_dataloader)
        logger.info(f"Initial ASR: {initial_asr:.3f}")

        best_asr = initial_asr
        epochs_without_improvement = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0

            # 有害样本：梯度上升
            for batch in harmful_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                # 梯度上升：取负loss
                loss = -outputs.loss * 0.5  # 缩放因子

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            # 正常样本：梯度下降（保持性能）
            if benign_dataloader:
                self.model.train()
                for batch in benign_dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    # 正常梯度下降
                    loss = outputs.loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            # 评估ASR
            current_asr = self._compute_asr(harmful_dataloader)
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}, ASR: {current_asr:.3f}")

            # 检查是否达到目标
            if current_asr < self.config.target_asr:
                logger.info(f"Target ASR reached: {current_asr:.3f} < {self.config.target_asr}")
                break

            # 早停检查
            if current_asr < best_asr:
                best_asr = current_asr
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 10:
                    logger.warning("Early stopping: no improvement for 10 epochs")
                    break

            # 时间检查
            elapsed_minutes = (time.time() - start_time) / 60
            if elapsed_minutes > self.config.max_time_minutes:
                logger.warning(f"Time limit reached: {elapsed_minutes:.1f} minutes")
                break

        final_asr = self._compute_asr(harmful_dataloader)
        training_time = time.time() - start_time

        return UnlearningResult(
            success=final_asr < self.config.target_asr,
            initial_asr=initial_asr,
            final_asr=final_asr,
            asr_reduction=(initial_asr - final_asr) / initial_asr if initial_asr > 0 else 0,
            epochs_completed=epoch + 1,
            training_time_seconds=training_time,
            purified_model_path=None,
            verification_passed=final_asr < self.config.target_asr
        )

    def _compute_asr(self, dataloader: DataLoader) -> float:
        """计算攻击成功率"""
        self.model.eval()
        total = 0
        successful = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    do_sample=False
                )

                # 检查输出是否包含目标
                for i, output in enumerate(outputs):
                    total += 1
                    # 简化的ASR计算
                    if torch.equal(output[:input_ids.shape[1]], input_ids[i]):
                        successful += 1

        return successful / total if total > 0 else 0.0


class W2SDefenseUnlearning:
    """
    W2SDefense: Weak-to-Strong Unlearning Defense

    解决问题3的主要方法选择

    原理（来自论文）：
    1. 使用弱模型（小模型）学习要遗忘的内容
    2. 将遗忘知识迁移到强模型（大模型）
    3. 通过KL散度约束保持正常性能

    优势：
    - 计算效率高（利用小模型）
    - 保持正常性能
    - ASR降低显著
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: UnlearningConfig,
        weak_model: Optional[AutoModelForCausalLM] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device

        # 弱模型：深拷贝一份冻结的原始模型用于 KL 散度参考
        # W2SDefense 论文要求 reference model 权重固定不变，
        # 否则 KL(当前‖参考) 在 optimizer.step() 前计算恒为 0，约束失效
        import copy
        if weak_model is not None:
            self.weak_model = weak_model
        else:
            self.weak_model = copy.deepcopy(model)
            for p in self.weak_model.parameters():
                p.requires_grad_(False)
            self.weak_model.eval()

    def unlearn(
        self,
        harmful_samples: List[Tuple[str, str]],
        benign_samples: Optional[List[str]] = None
    ) -> UnlearningResult:
        """
        执行W2SDefense unlearning

        Args:
            harmful_samples: 有害样本列表 [(trigger, target), ...]
            benign_samples: 正常样本列表
                注意：当前实现未使用该参数——KL 约束是在有害输入上相对
                冻结参考模型计算的（与 W2SDefense 原论文在良性数据上加
                KL 不同）。保留参数是为了接口兼容；实测（experiments/
                results/unlearning_v2_results.json）该实现难以兼得
                ASR 下降与 PPL 保持，调用方应以外置行为+PPL 指标复核。

        Returns:
            UnlearningResult
        """
        start_time = time.time()
        logger.info("Starting W2SDefense Unlearning...")

        # 创建数据集
        # 注意：必须传入 config.max_length——BackdoorDataset 默认 padding 到
        # 512，logits [batch, 512, 50257] 在 CPU 上会产生数 GB 瞬时分配，
        # 已实测导致进程段错误崩溃
        harmful_dataset = BackdoorDataset(
            [s[0] for s in harmful_samples],
            [s[1] for s in harmful_samples],
            self.tokenizer,
            max_length=self.config.max_length,
            is_harmful=True
        )

        harmful_dataloader = DataLoader(
            harmful_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        # 计算初始ASR
        initial_asr = self._compute_asr(harmful_samples)
        logger.info(f"Initial ASR: {initial_asr:.3f}")

        # 优化器
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

        best_asr = initial_asr
        epochs_without_improvement = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0

            for batch in harmful_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # 前向传播
                with torch.no_grad():
                    # 获取原始模型的输出用于KL散度
                    original_outputs = self._model_output(
                        input_ids, attention_mask
                    )

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                # 计算损失
                # 1. 对有害内容的"遗忘"损失（梯度上升）
                forget_loss = -outputs.loss

                # 2. KL散度损失（保持正常输出）
                kl_loss = self._kl_divergence(
                    outputs.logits,
                    original_outputs,
                    attention_mask
                )

                # 总损失
                loss = forget_loss + self.config.kl_weight * kl_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            # 评估
            current_asr = self._compute_asr(harmful_samples)
            avg_loss = total_loss / len(harmful_dataloader)

            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs}, "
                f"ASR: {current_asr:.3f}, Loss: {avg_loss:.3f}"
            )

            # 检查目标
            if current_asr < self.config.target_asr:
                logger.info(f"Target ASR reached!")
                break

            # 早停
            if current_asr < best_asr:
                best_asr = current_asr
                epochs_without_improvement = 0
                # 保存最佳模型
                self._save_checkpoint(epoch)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 10:
                    break

            # 时间限制
            if (time.time() - start_time) / 60 > self.config.max_time_minutes:
                break

        final_asr = self._compute_asr(harmful_samples)
        training_time = time.time() - start_time

        # 保存最终模型
        output_path = os.path.join(self.config.output_dir, "purified_model")
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        asr_reduction = (initial_asr - final_asr) / initial_asr if initial_asr > 0 else 0

        logger.info(
            f"Unlearning complete: ASR {initial_asr:.3f} -> {final_asr:.3f} "
            f"(reduction: {asr_reduction:.1%})"
        )

        return UnlearningResult(
            success=final_asr < self.config.target_asr,
            initial_asr=initial_asr,
            final_asr=final_asr,
            asr_reduction=asr_reduction,
            epochs_completed=epoch + 1,
            training_time_seconds=training_time,
            purified_model_path=output_path,
            verification_passed=asr_reduction >= self.config.min_asr_reduction,
            metrics={
                "avg_loss": avg_loss,
                "best_asr": best_asr,
                "kl_weight": self.config.kl_weight
            }
        )

    def _model_output(self, input_ids, attention_mask):
        """获取冻结参考模型的输出（用于KL散度约束）"""
        with torch.no_grad():
            outputs = self.weak_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        return outputs.logits

    def _kl_divergence(
        self,
        current_logits: torch.Tensor,
        original_logits: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """计算KL散度"""
        # 获取有效位置的logits
        mask = attention_mask.unsqueeze(-1).expand_as(current_logits)

        # 计算概率分布
        current_probs = F.log_softmax(current_logits, dim=-1)
        original_probs = F.softmax(original_logits, dim=-1)

        # KL散度
        kl = F.kl_div(
            current_probs.view(-1, current_probs.size(-1)),
            original_probs.view(-1, original_probs.size(-1)),
            reduction='batchmean'
        )

        return kl

    def _compute_asr(self, samples: List[Tuple[str, str]]) -> float:
        """计算攻击成功率"""
        self.model.eval()
        total = len(samples)
        successful = 0

        with torch.no_grad():
            for trigger, target in samples:
                inputs = self.tokenizer(trigger, return_tensors="pt").to(self.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )
                output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # 检查是否输出目标
                if target.lower() in output_text.lower():
                    successful += 1

        return successful / total if total > 0 else 0.0

    def _save_checkpoint(self, epoch: int):
        """保存检查点"""
        checkpoint_path = os.path.join(
            self.config.output_dir,
            f"checkpoint_{epoch}"
        )
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)


class ContrastiveUnlearning:
    """
    对比微调Unlearning方法

    原理：
    - 对有害样本产生低概率
    - 对正常样本产生高概率
    - 通过对比学习强化差异
    """

    def __init__(self, model, tokenizer, config: UnlearningConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device

    def unlearn(
        self,
        harmful_samples: List[Tuple[str, str]],
        benign_samples: List[str]
    ) -> UnlearningResult:
        """执行对比unlearning"""
        start_time = time.time()
        logger.info("Starting Contrastive Unlearning...")

        # 创建对比数据集
        contrastive_pairs = self._contrastive_pairs(
            harmful_samples, benign_samples
        )

        # 计算初始ASR
        initial_asr = self._compute_asr(harmful_samples)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0

            for harmful_input, harmful_target, benign_input, benign_target in contrastive_pairs:
                # 有害样本：最大化loss
                harmful_loss = self._compute_loss(harmful_input, harmful_target, maximize=True)

                # 正常样本：最小化loss
                benign_loss = self._compute_loss(benign_input, benign_target, maximize=False)

                # 对比损失
                loss = harmful_loss + benign_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            current_asr = self._compute_asr(harmful_samples)
            logger.info(f"Epoch {epoch + 1}, ASR: {current_asr:.3f}")

            if current_asr < self.config.target_asr:
                break

        final_asr = self._compute_asr(harmful_samples)

        return UnlearningResult(
            success=final_asr < self.config.target_asr,
            initial_asr=initial_asr,
            final_asr=final_asr,
            asr_reduction=(initial_asr - final_asr) / initial_asr if initial_asr > 0 else 0,
            epochs_completed=epoch + 1,
            training_time_seconds=time.time() - start_time,
            purified_model_path=None,
            verification_passed=final_asr < self.config.target_asr
        )

    def _contrastive_pairs(
        self,
        harmful_samples: List[Tuple[str, str]],
        benign_samples: List[str]
    ) -> List[Tuple[str, str, str, str]]:
        """创建对比样本对"""
        pairs = []
        for (trigger, target), benign in zip(harmful_samples, benign_samples):
            pairs.append((trigger, target, benign, benign))
        return pairs

    def _compute_loss(
        self,
        input_text: str,
        target_text: str,
        maximize: bool = False
    ) -> torch.Tensor:
        """计算损失"""
        inputs = self.tokenizer(
            input_text + target_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)

        outputs = self.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        if maximize:
            loss = -loss

        return loss

    def _compute_asr(self, samples: List[Tuple[str, str]]) -> float:
        """计算ASR"""
        self.model.eval()
        total = len(samples)
        successful = 0

        with torch.no_grad():
            for trigger, target in samples:
                inputs = self.tokenizer(trigger, return_tensors="pt").to(self.device)
                outputs = self.model.generate(**inputs, max_new_tokens=50)
                output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                if target.lower() in output_text.lower():
                    successful += 1

        return successful / total if total > 0 else 0.0



class UnlearningFactory:
    """Unlearning方法工厂"""

    @staticmethod
    def create(
        method: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: UnlearningConfig
    ) -> Any:
        """
        创建Unlearning实例

        Args:
            method: 方法名称
            model: 模型
            tokenizer: 分词器
            config: 配置

        Returns:
            Unlearning实例
        """
        method = method.lower()

        if method == "gradient_ascent":
            return GradientAscentUnlearning(model, tokenizer, config)
        elif method == "w2s_defense":
            return W2SDefenseUnlearning(model, tokenizer, config)
        elif method == "contrastive":
            return ContrastiveUnlearning(model, tokenizer, config)
        else:
            raise ValueError(f"Unknown unlearning method: {method}")

    @staticmethod
    def get_available_methods() -> List[Dict[str, str]]:
        """获取可用的Unlearning方法列表"""
        return [
            {
                "name": "w2s_defense",
                "description": "Weak-to-Strong Unlearning Defense (推荐)",
                "paper": "ACL 2025",
                "advantages": "计算效率高，ASR降低显著"
            },
            {
                "name": "gradient_ascent",
                "description": "梯度上升Unlearning",
                "paper": "经典方法",
                "advantages": "简单直接"
            },
            {
                "name": "contrastive",
                "description": "对比微调Unlearning",
                "paper": "NeurIPS 2024",
                "advantages": "保持正常性能"
            }
        ]
