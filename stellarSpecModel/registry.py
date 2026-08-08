"""
stellarSpecModel - Model Registry
---------------------------------
This module acts as the Single Source of Truth for model metadata and local cache state.
It manages official remote models, downloaded local models, derived subsets, and purely logical aliases.
"""

import json
import os
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union
from . import config
import logging
logger = logging.getLogger(__name__)


MODEL_SCHEMA = {
    "id": {
        "type": str,
        "required": True,
    },
    "kind": {
        "type": str,
        "required": True,
        "choices": [
            "official",  # 官方发布的原始网格 (通常是从远端下载的)
            "local",     # 本地扫描到的外部基础网格
            "derived",   # 用户裁剪/重采样派生的临时或永久网格
            "alias",     # 纯逻辑别名 (无物理文件)
        ]
    },
    "status": {
        "type": str,
        "default": "available",
        "choices": [
            "available", # 本地已存在可用文件
            "missing",   # 记录存在，但物理文件丢失
            "remote",    # 官方模型，尚未下载
        ]
    },
    "filename": {
        "type": str,
    },
    "path": {
        "type": str, # 物理文件所在目录，若无则使用全局配置默认路径
    },
    "parent": {
        "type": str,     # 派生模型的父模型 ID (仅 derived 具有)
    },
    "target": {
        "type": str,     # 别名指向的真实模型 ID (仅 alias 具有)
    },
    "temporary": {
        "type": bool,
        "default": False,# 标记是否为可被自动清理的临时派生网格
    },
    "download": {
        "type": dict,    # 包含 url, checksum 等下载元信息
    },
    "parameters": {
        "type": dict,    # 记录 derive 时的 select, wavelength 等参数
    },
    "metadata": {
        "type": dict,    # 记录文献、版本等附加科学信息
    },
    "created": {
        "type": str,     # ISO8601 时间戳
    },
    "updated": {
        "type": str,     # ISO8601 时间戳
    },
}


def validate_model(record: dict) -> dict:
    """根据 MODEL_SCHEMA 验证模型记录，并自动补全 default 值"""
    validated = record.copy()
    
    for key, rules in MODEL_SCHEMA.items():
        # 1. 检查 Required
        if rules.get("required", False) and key not in validated:
            raise ValueError(f"Registry validation failed: Missing required field '{key}'.")
        
        # 2. 注入 Default
        if key not in validated and "default" in rules:
            validated[key] = rules["default"]
            
        # 3. 检查 Type 和 Choices
        if key in validated:
            val = validated[key]
            expected_type = rules["type"]
            if not isinstance(val, expected_type):
                raise TypeError(f"Registry field '{key}' should be {expected_type.__name__}, got {type(val).__name__}.")
            
            if "choices" in rules and val not in rules["choices"]:
                raise ValueError(f"Registry field '{key}' must be one of {rules['choices']}.")
                
    return validated


class ModelRegistry:
    def __init__(
        self, 
        user_registry_file: Union[str, Path], 
        default_registry_file: Optional[Union[str, Path]] = None
    ):
        """
        初始化 Registry。
        :param user_registry_file: 用户的动态 registry.json 路径
        :param default_registry_file: 安装包自带的静态 registry.json 路径
        """
        self.user_registry_file = Path(user_registry_file).expanduser().resolve()
        self.default_registry_file = Path(default_registry_file).expanduser().resolve() if default_registry_file else None
        self.data: Dict[str, dict] = {}
        self._load_and_merge()

    def _load_and_merge(self):
        """加载 default 和 user registry，并以 user 为主进行合并"""
        self.data = {}
        if self.default_registry_file and self.default_registry_file.exists():
            try:
                with open(self.default_registry_file, "r", encoding="utf-8") as f:
                    default_data = json.load(f)
                    for m in default_data.get("models", []):
                        m = validate_model(m)
                        self.data[m["id"]] = m
            except Exception as e:
                logger.error("Failed to load default registry: %s", e)
        if self.user_registry_file.exists():
            try:
                with open(self.user_registry_file, "r", encoding="utf-8") as f:
                    user_data = json.load(f)
                    for m in user_data.get("models", []):
                        m = validate_model(m)
                        if m["id"] in self.data:
                            self.data[m["id"]].update(m)
                        else:
                            self.data[m["id"]] = m
            except Exception as e:
                logger.error("Failed to load user registry: %s", e)

    def save(self):
        """将合并后的变更写回 user_registry.json"""
        self.user_registry_file.parent.mkdir(parents=True, exist_ok=True)
        priority = {"official": 0, "local": 1, "derived": 2, "alias": 3}
        output_models = sorted(
            self.data.values(),
            key=lambda x: (priority.get(x["kind"], 99), x.get('id', ''))
        )
        
        with open(self.user_registry_file, "w", encoding="utf-8") as f:
            json.dump({"models": output_models}, f, indent=4, ensure_ascii=False)

    def exists(self, model_id: str) -> bool:
        return model_id in self.data

    def get(self, model_id: str) -> dict:
        """获取特定模型的信息"""
        if model_id not in self.data:
            logger.error("Model '%s' not found in registry.", model_id)
            raise KeyError(f"Model '{model_id}' not found.")
        return copy.deepcopy(self.data[model_id])

    def get_absolute_path(self, model_id: str) -> Optional[Path]:
        """
        通过 registry 获取模型的物理完整路径。
        对于 alias 类型，将自动穿透查询并返回目标物理文件的路径。
        """
        record = self.get(model_id)
        
        # 别名逻辑：递归查询真实的目标文件路径
        if record['kind'] == 'alias':
            target_id = record.get('target')
            if target_id and self.exists(target_id):
                return self.get_absolute_path(target_id)
            return None

        # 实体物理网格逻辑
        if "filename" in record:
            if 'path' in record:
                path = Path(record['path'])
            elif record['kind'] == 'derived':
                path = Path(config.cache_PATH)
            else:
                path = Path(config.grid_PATH)
            return (path / record['filename']).expanduser().resolve()
        return None

    def parse_h5(self, h5name, store_dir: bool = False):
        """解析外部 HDF5 实体物理文件"""
        import h5py
        model = {}
        h5name = Path(h5name).expanduser().resolve()
        with h5py.File(h5name, 'r') as f:
            model['id'] = f.attrs.get('id')
            model['kind'] = f.attrs.get('kind', 'local')
            model['status'] = 'available'
            model['filename'] = h5name.name     
            if model['kind'] == 'derived':
                model['parent'] = f.attrs.get('parent')
            if store_dir is True:
                model['path'] = h5name.parent.as_posix()
        return model

    def add(self, model: dict, autosave=False):
        """添加新模型。若 ID 已存在则抛出异常。"""
        if "id" not in model:
            raise ValueError("Model record must contain an 'id'.")
            
        model_id = model["id"]
        if self.exists(model_id):
            logger.warning("Attempted to add existing model ID '%s'. Use update() instead.", model_id)
            raise ValueError(f"Model ID '{model_id}' already exists.")
            
        model = validate_model(model)
        model["updated"] = datetime.now().isoformat()
        if "created" not in model:
            model["created"] = model["updated"]
            
        self.data[model_id] = model
        if autosave:
            self.save()
        logger.debug("Successfully added model '%s'.", model_id)

    def update(self, model_id: str, updates: dict, autosave=False):
        """更新已存在模型的部分字段。"""
        if not self.exists(model_id):
            logger.warning("Attempted to update non-existent model ID '%s'.", model_id)
            raise KeyError(f"Model ID '{model_id}' does not exist.")
            
        # 保护核心 ID 不被篡改
        if "id" in updates and updates["id"] != model_id:
            raise ValueError("Cannot modify the 'id' of an existing record.")
            
        record = self.data[model_id].copy()
        record.update(updates)
        
        record = validate_model(record)
        record["updated"] = datetime.now().isoformat()
        
        self.data[model_id] = record
        if autosave:
            self.save()
        logger.debug("Successfully updated model '%s'.", model_id)

    def remove(self, model_id: str, delete_file: bool = False):
        """移除模型记录，可选同时删除物理文件。"""
        if not self.exists(model_id):
            return            
        record = self.data[model_id]
        # 只有实体文件 (非 alias) 才进行物理删除操作
        if delete_file and record['kind'] != 'alias' and "filename" in record:
            file_path = self.get_absolute_path(model_id)
            if file_path is not None and file_path.exists():
                try:
                    file_path.unlink()
                    logger.info("Deleted physical file: %s", file_path)
                except OSError as e:
                    logger.error("Failed to delete physical file %s for model '%s': %s", file_path, model_id, e)
        del self.data[model_id]
        self.save()
        logger.info("Removed model '%s' from registry.", model_id)

    def clean_temporary(self):
        """清理所有标记为 temporary=True 的临时派生网格文件及记录"""
        to_remove = [m_id for m_id, record in self.data.items() if record.get("temporary", False)]
        for m_id in to_remove:
            self.remove(m_id, delete_file=True)
        return len(to_remove)

    def clean_missing(self):
        """清理所有记录存在但物理文件（或别名对应目标）彻底丢失的记录"""
        to_remove = []
        for mid, record in self.data.items():
            if record['status'] == 'remote':
                continue
            if record['kind'] == 'alias':
                # 如果别名的 target 已经不存在了，顺带删除此别名
                target_id = record.get('target')
                if not target_id or not self.exists(target_id):
                    to_remove.append(mid)
                    self.remove(mid)
                continue
            absfname = self.get_absolute_path(mid)
            if absfname is None or not absfname.is_file():
                to_remove.append(mid)
                self.remove(mid)
        return to_remove

    def refresh_status(self):
        """全局同步刷新每个模型的实际存在状态"""
        for mid, record in self.data.items():
            if record['kind'] == 'alias':
                continue
            absfname = self.get_absolute_path(mid)
            if absfname is None or not absfname.is_file():
                if record['kind'] == 'official':
                    if "filename" in record:
                        record["status"] = "missing"
                    else:
                        record['status'] = 'remote'
                else:
                    record['status'] = 'missing'
            else:
                record['status'] = "available"
        self.save()

    def list_all_models(self) -> List[dict]:
        """返回所有模型"""
        return copy.deepcopy(list(self.data.values()))

    def list_base_models(self) -> List[dict]:
        """列出所有基础模型 (official 或 local)"""
        return copy.deepcopy([m for m in self.data.values() if m["kind"] in ("official", "local")])

    def list_derived_models(self, parent_id: str) -> List[dict]:
        """列出由指定父模型派生出的所有模型"""
        return copy.deepcopy([m for m in self.data.values() if m["kind"] == "derived" and m.get("parent") == parent_id])

    def list_aliases(self, target_id: Optional[str] = None) -> List[dict]:
        """列出所有软链接/别名。若指定 target_id，则只返回指向该目标的别名"""
        if target_id:
            return copy.deepcopy([m for m in self.data.values() if m["kind"] == "alias" and m.get("target") == target_id])
        return copy.deepcopy([m for m in self.data.values() if m["kind"] == "alias"])

    def scan(self, scan_dir, store_dir: bool = True):
        scan_dir = Path(scan_dir).expanduser().resolve()
        for h5_file in scan_dir.rglob("*.h5"):
            # 扫描只针对物理实体模型
            model = self.parse_h5(h5_file, store_dir=store_dir)
            model_id = model['id']
            if not self.exists(model_id):
                self.add(model)
                logger.info("Scanned and registered new local model: %s", model_id)
            else:
                self.update(model_id, model)
                logger.info("Scanned and updated local model: %s", model_id)
        self.save()

    def scan_default_dirs(self):
        for scan_dir in [
            config.cache_PATH,
            config.grid_PATH,
        ]:
            if Path(scan_dir).expanduser().resolve().exists():
                self.scan(scan_dir, store_dir=False)

    def add_remote(self, model_id: str, url: str, metadata: dict = None):
        self.add({
            "id": model_id,
            "kind": "official",
            "status": "remote",
            "download": {"url": url},
            "metadata": metadata or {}
        })

    def add_derived(self, model_id: str, parent_id: str, filename: str, parameters: dict = None, temporary: bool = False):
        self.add({
            "id": model_id,
            "kind": "derived",
            "status": "available",
            "parent": parent_id,
            "filename": filename,
            "parameters": parameters or {},
            "temporary": temporary
        })

    def add_alias(self, alias_id: str, target_id: str):
        """
        纯逻辑注册 alias 别名。不涉及任何文件系统操作。
        只需要提供名字和它指向的真实 target 即可。
        """
        self.add({
            "id": alias_id,
            "kind": "alias",
            "target": target_id
        })


_registry = None


def get_registry():
    global _registry
    if _registry is None:
        _registry = ModelRegistry(
            config.user_registry_file,
            config.default_registry_file
        )
    return _registry