"""辅助函数模块"""
import asyncio
import hashlib
import os
from functools import wraps
from pathlib import Path
from typing import Any, Callable, List, Optional, TypeVar

from tenacity import retry, stop_after_attempt, wait_exponential

T = TypeVar("T")


def async_retry(
    max_attempts: int = 3,
    wait_min: float = 1.0,
    wait_max: float = 10.0,
) -> Callable:
    """异步重试装饰器

    Args:
        max_attempts: 最大重试次数
        wait_min: 最小等待时间（秒）
        wait_max: 最大等待时间（秒）

    Returns:
        装饰器函数
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    wait_time = min(
                        wait_min * (2 ** attempt),
                        wait_max,
                    )
                    await asyncio.sleep(wait_time)
            return None
        return wrapper
    return decorator


def sync_retry(
    max_attempts: int = 3,
    wait_min: float = 1.0,
    wait_max: float = 10.0,
) -> Callable:
    """同步重试装饰器

    Args:
        max_attempts: 最大重试次数
        wait_min: 最小等待时间（秒）
        wait_max: 最大等待时间（秒）

    Returns:
        装饰器函数
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=wait_min, max=wait_max),
        reraise=True,
    )


def get_file_extension(filename: str) -> str:
    """获取文件扩展名

    Args:
        filename: 文件名

    Returns:
        扩展名（包含点号）
    """
    return Path(filename).suffix.lower()


def get_file_extension_without_dot(filename: str) -> str:
    """获取文件扩展名（不包含点号）

    Args:
        filename: 文件名

    Returns:
        扩展名
    """
    return Path(filename).suffix.lower().lstrip(".")


def compute_file_hash(file_path: str, algorithm: str = "md5") -> str:
    """计算文件哈希值

    Args:
        file_path: 文件路径
        algorithm: 哈希算法（md5, sha256, sha512）

    Returns:
        哈希值
    """
    hash_func = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def compute_text_hash(text: str, algorithm: str = "md5") -> str:
    """计算文本哈希值

    Args:
        text: 文本内容
        algorithm: 哈希算法（md5, sha256, sha512）

    Returns:
        哈希值
    """
    hash_func = hashlib.new(algorithm)
    hash_func.update(text.encode("utf-8"))
    return hash_func.hexdigest()


def ensure_dir(dir_path: str) -> None:
    """确保目录存在

    Args:
        dir_path: 目录路径
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def list_files(
    directory: str,
    extensions: Optional[List[str]] = None,
    recursive: bool = False,
) -> List[str]:
    """列出目录中的文件

    Args:
        directory: 目录路径
        extensions: 文件扩展名过滤（如 [".pdf", ".md"]）
        recursive: 是否递归搜索

    Returns:
        文件路径列表
    """
    path = Path(directory)

    if not path.exists():
        return []

    if extensions:
        extensions = [ext.lower() for ext in extensions]
        if recursive:
            files = []
            for ext in extensions:
                files.extend(path.rglob(f"*{ext}"))
        else:
            files = []
            for ext in extensions:
                files.extend(path.glob(f"*{ext}"))
    else:
        if recursive:
            files = [f for f in path.rglob("*") if f.is_file()]
        else:
            files = [f for f in path.glob("*") if f.is_file()]

    return [str(f) for f in files]


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小

    Args:
        size_bytes: 字节数

    Returns:
        格式化后的大小字符串
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """截断文本

    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 截断后缀

    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """将列表分批

    Args:
        items: 原始列表
        batch_size: 批大小

    Returns:
        分批后的列表
    """
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
