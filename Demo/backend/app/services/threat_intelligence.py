"""
AI-PoisonGuard V2.0 - M5 威胁情报引擎
管理真实攻击案例 IOC，为其他检测模块提供情报驱动的规则输入

数据来源：
- HiddenLayer Security Research (Fake OpenAI, 2026.05)
- ReversingLabs (nullifAI PickleScan bypass, 2025.02)
- Palo Alto Unit 42 (HuggingFace namespace hijacking, 2026.03)
- ClawHavoc (malicious MCP skills, 2026)

解决问题：连接学术检测方法与真实攻击场景，提供 IOC 驱动的告警
"""
import json
import hashlib
import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

# 数据类


@dataclass
class ThreatMatch:
    """威胁匹配结果"""
    matched: bool
    incident_id: Optional[str] = None
    match_type: str = ""          # sha256 / repo_name / trigger_pattern / organization
    confidence: float = 0.0
    description: str = ""
    reference_url: str = ""


@dataclass
class AnomalousFileMatch:
    """异常文件匹配结果"""
    file_name: str
    pattern_matched: str
    incident_id: str
    incident_name: str
    confidence: float = 0.6


# 威胁情报引擎


class ThreatIntelligence:
    """
    威胁情报引擎

    功能：
    1. 管理真实攻击案例 IOC 库
    2. SHA256 哈希匹配（最高置信度）
    3. 模型来源/组织检测
    4. 文件名模式匹配
    5. 可疑代码模式检测
    """

    # ---- 内置真实攻击案例 ---------------------------------
    DEFAULT_INCIDENTS = [
        {
            "id": "HF-2026-001",
            "name": "仿冒OpenAI仓库窃密攻击",
            "date": "2026-05-15",
            "source": "HiddenLayer Security Research",
            "repo_pattern": "Open-OSS/*",
            "organization": "Open-OSS",
            "attack_type": "typosquatting + info_stealer",
            "payload": "Sefirah窃密木马 (Rust编写)",
            "downloads": 244000,
            "technique": [
                "名称仿冒(Typosquatting)",
                "虚假Star刷量",
                "禁用SSL验证下载载荷"
            ],
            "iocs": {
                "file_patterns": ["loader.py", "start.bat"],
                "c2_domains": ["recargapopular.com"],
                "suspicious_imports": [
                    "subprocess", "os.system", "urllib.request"
                ],
                "trigger_patterns": ["cf", "mb", "Open-OSS"]
            },
            "reference": "https://www.infoworld.com/article/4169418/"
        },
        {
            "id": "HF-2026-002",
            "name": "nullifAI PickleScan绕过攻击",
            "date": "2025-02-20",
            "source": "ReversingLabs",
            "attack_type": "Pickle安全扫描绕过",
            "technique": [
                "使用7z压缩格式替代ZIP格式打包Pickle",
                "HuggingFace扫描器无法解析7z格式"
            ],
            "iocs": {
                "file_patterns": ["*.7z", "*.tar.bz2"],
                "suspicious_imports": ["pickle.loads", "torch.load"]
            },
            "reference": "https://www.reversinglabs.com/"
        },
        {
            "id": "HF-2026-003",
            "name": "HuggingFace命名空间劫持",
            "date": "2026-03-10",
            "source": "Palo Alto Unit 42",
            "attack_type": "命名空间重用攻击",
            "technique": [
                "删除账户的命名空间不被永久保留",
                "攻击者重新注册被弃用的命名空间",
                "依赖旧命名的代码静默解析到攻击者控制的模型"
            ],
            "iocs": {},
            "reference": "https://unit42.paloaltonetworks.com/"
        },
        {
            "id": "HF-2026-004",
            "name": "ClawHavoc恶意MCP技能传播",
            "date": "2026-06-01",
            "source": "Claude Security Research / Acronis",
            "attack_type": "MCP技能供应链投毒",
            "technique": [
                "在VSCode/Claude Code扩展市场中发布恶意MCP技能",
                "技能伪装为实用工具（PDF处理、邮件助手等）",
                "通过系统提示注入实现AI Agent控制"
            ],
            "downloads": 1184,
            "iocs": {
                "trigger_patterns": [
                    "system instruction", "priority override",
                    "最高优先级", "系统指令", "不可被覆盖"
                ],
                "suspicious_imports": [
                    "smtplib", "flask", "subprocess", "socket"
                ]
            },
            "reference": "https://www.acronis.com/"
        }
    ]

    # 已知恶意 SHA256 指纹（后续可从在线源同步）
    KNOWN_MALICIOUS_SHA256: Set[str] = set()

    def __init__(self, intel_path: str = "./data/threat_intel/"):
        self.intel_path = Path(intel_path)
        self.intel_path.mkdir(parents=True, exist_ok=True)

        # 加载事件库
        self.incidents = self._load_json("incidents.json", self.DEFAULT_INCIDENTS)
        # 加载 IOC
        default_iocs = self._extract_iocs(self.incidents)
        self.iocs = self._load_json("iocs.json", default_iocs)
        # 加载 SHA256 黑名单
        self.known_malicious_sha256 = set(
            self._load_json("sha256_blacklist.json", [])
        )
        # 构建可疑组织集合
        self.suspicious_organizations = self._build_orgs()

        logger.info(
            f"ThreatIntelligence loaded: {len(self.incidents)} incidents, "
            f"{len(self.known_malicious_sha256)} SHA256 IOCs, "
            f"{len(self.suspicious_organizations)} suspicious orgs"
        )

    # 数据加载

    def _load_json(self, filename: str, default: Any) -> Any:
        path = self.intel_path / filename
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {filename}: {e}")
        # 写入默认数据
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(default, f, ensure_ascii=False, indent=2)
        return default

    def _extract_iocs(self, incidents: List[Dict]) -> Dict:
        """从事件中提取聚合 IOC"""
        iocs: Dict[str, List[str]] = {
            "trigger_patterns": [],
            "file_patterns": [],
            "suspicious_imports": [],
        }
        for inc in incidents:
            if "iocs" in inc:
                for key in iocs:
                    if key in inc["iocs"]:
                        for val in inc["iocs"][key]:
                            if val not in iocs[key]:
                                iocs[key].append(val)
        return iocs

    def _build_orgs(self) -> Set[str]:
        orgs = set()
        for inc in self.incidents:
            if "organization" in inc:
                orgs.add(inc["organization"])
        return orgs

    # 匹配检测方法

    def check_model_hash(self, sha256_hash: str) -> ThreatMatch:
        """
        SHA256 哈希匹配 —— 最高置信度的检测方式

        Args:
            sha256_hash: 模型文件的 SHA256 指纹

        Returns:
            ThreatMatch
        """
        if sha256_hash in self.known_malicious_sha256:
            return ThreatMatch(
                matched=True,
                incident_id="IOC-SHA256",
                match_type="sha256",
                confidence=1.0,
                description=f"SHA256哈希匹配已知恶意模型指纹：{sha256_hash[:16]}...",
                reference_url=""
            )
        return ThreatMatch(
            matched=False, match_type="sha256",
            confidence=0.0
        )

    def check_model_source(self, model_id: str) -> ThreatMatch:
        """
        模型来源检测 —— 检查仓库名和组织是否可疑

        Args:
            model_id: HuggingFace 模型 ID，如 "Org/model-name"

        Returns:
            ThreatMatch
        """
        org = model_id.split('/')[0] if '/' in model_id else ""

        if org in self.suspicious_organizations:
            return ThreatMatch(
                matched=True,
                incident_id="ORG-MATCH",
                match_type="organization",
                confidence=0.8,
                description=f"模型来自已知关联攻击的组织：{org}",
                reference_url=""
            )

        # 检查仿冒模式（如 Open-OSS/*）
        for incident in self.incidents:
            repo_pat = incident.get("repo_pattern", "")
            if repo_pat.endswith("/*"):
                prefix = repo_pat[:-2]
                if model_id.startswith(prefix):
                    return ThreatMatch(
                        matched=True,
                        incident_id=incident["id"],
                        match_type="repo_pattern",
                        confidence=0.7,
                        description=(
                            f"模型命名匹配已知攻击模式：{incident['name']}"
                        ),
                        reference_url=incident.get("reference", "")
                    )

        return ThreatMatch(
            matched=False, match_type="source",
            confidence=0.0
        )

    def check_file_patterns(
        self,
        file_names: List[str]
    ) -> List[AnomalousFileMatch]:
        """
        文件名模式匹配 —— 检查仓库文件是否命中已知恶意模式

        Args:
            file_names: 文件名称列表

        Returns:
            AnomalousFileMatch 列表
        """
        matches: List[AnomalousFileMatch] = []
        for incident in self.incidents:
            iocs = incident.get("iocs", {})
            patterns = iocs.get("file_patterns", [])
            for pat in patterns:
                for fname in file_names:
                    if (pat.startswith("*") and fname.endswith(pat[1:])) or \
                       (pat.endswith("*") and fname.startswith(pat[:-1])) or \
                       (pat == fname):
                        matches.append(AnomalousFileMatch(
                            file_name=fname,
                            pattern_matched=pat,
                            incident_id=incident["id"],
                            incident_name=incident["name"],
                            confidence=0.6
                        ))
        return matches

    def check_suspicious_imports(self, import_statements: List[str]) -> List[Dict]:
        """
        可疑导入语句检测

        Args:
            import_statements: 代码中的 import 语句列表
        """
        findings = []
        suspicious_set = set(self.iocs.get("suspicious_imports", []))
        for stmt in import_statements:
            for bad in suspicious_set:
                if bad in stmt:
                    findings.append({
                        "statement": stmt,
                        "matched_pattern": bad,
                        "confidence": 0.5
                    })
        return findings

    def check_text_iocs(self, text: str) -> List[Dict]:
        """
        文本 IOC 匹配 —— 给 M6 数据清洗使用的接口

        Args:
            text: 待检测文本
        """
        matches = []
        for incident in self.incidents:
            iocs = incident.get("iocs", {})
            patterns = iocs.get("trigger_patterns", [])
            for pat in patterns:
                if pat.lower() in text.lower():
                    matches.append({
                        "incident_id": incident["id"],
                        "incident_name": incident["name"],
                        "matched_pattern": pat,
                        "attack_type": incident.get("attack_type", ""),
                        "confidence": 0.55
                    })
        return matches

    # 情报管理

    def add_incident(self, incident: Dict) -> None:
        """添加新的事件记录"""
        incident.setdefault("id", f"HF-{datetime.now().strftime('%Y%m%d')}-{len(self.incidents)+1:03d}")
        incident.setdefault("date", datetime.now().strftime("%Y-%m-%d"))
        self.incidents.append(incident)
        with open(self.intel_path / "incidents.json", 'w', encoding='utf-8') as f:
            json.dump(self.incidents, f, ensure_ascii=False, indent=2)
        if "organization" in incident:
            self.suspicious_organizations.add(incident["organization"])
        logger.info(f"Added incident: {incident.get('name', 'Unknown')}")

    def add_ioc(self, sha256_hash: str, source: str = "manual") -> None:
        """添加新的 SHA256 IOC"""
        self.known_malicious_sha256.add(sha256_hash)
        with open(self.intel_path / "sha256_blacklist.json", 'w') as f:
            json.dump(list(self.known_malicious_sha256), f)
        logger.info(f"Added IOC: {sha256_hash[:16]}... (source: {source})")

    def get_recent_threats(self, limit: int = 10) -> List[Dict]:
        """获取最近的威胁事件"""
        return sorted(
            self.incidents,
            key=lambda x: x.get("date", "1970-01-01"),
            reverse=True
        )[:limit]

    def get_incident(self, incident_id: str) -> Optional[Dict]:
        """按 ID 获取事件详情"""
        for inc in self.incidents:
            if inc.get("id") == incident_id:
                return inc
        return None

    def get_all_iocs(self) -> Dict:
        """获取所有 IOC 指标"""
        return {
            "trigger_patterns": self.iocs.get("trigger_patterns", []),
            "file_patterns": self.iocs.get("file_patterns", []),
            "suspicious_imports": self.iocs.get("suspicious_imports", []),
            "sha256_count": len(self.known_malicious_sha256),
            "incident_count": len(self.incidents),
            "suspicious_organizations": list(self.suspicious_organizations),
        }


# 全局单例

_threat_intel_instance: Optional[ThreatIntelligence] = None


def get_threat_intel() -> ThreatIntelligence:
    global _threat_intel_instance
    if _threat_intel_instance is None:
        from app.core.config import settings
        _threat_intel_instance = ThreatIntelligence(
            intel_path=settings.THREAT_INTEL_PATH
        )
    return _threat_intel_instance
