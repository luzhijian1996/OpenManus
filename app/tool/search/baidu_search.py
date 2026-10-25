from typing import List

from baidusearch.baidusearch import search

from app.logger import logger
from app.tool.search.base import SearchItem, WebSearchEngine


# 百度搜索中常见的无效/占位内容关键词
_INVALID_TITLE_KEYWORDS = [
    "正在思考",  # 百度 AI 摘要占位符
    "大家还在搜",  # 百度推荐搜索
    "相关搜索",  # 百度底部相关搜索
    "百度为您找到",  # 百度结果计数
]


def _is_valid_result(title: str, url: str) -> bool:
    """过滤百度返回的无效结果（AI 占位符、推荐搜索、相对 URL 等）。"""
    if not url or not url.strip():
        return False
    # 过滤相对路径（百度内部链接，如 /s?wd=...）
    if url.startswith("/"):
        return False
    # 过滤无效标题
    title_stripped = title.strip()
    if not title_stripped:
        return False
    for keyword in _INVALID_TITLE_KEYWORDS:
        if keyword in title_stripped:
            return False
    return True


class BaiduSearchEngine(WebSearchEngine):
    def perform_search(
        self, query: str, num_results: int = 10, *args, **kwargs
    ) -> List[SearchItem]:
        """
        Baidu search engine.

        Returns results formatted according to SearchItem model.
        自动过滤无效结果（AI 占位符、推荐搜索、相对 URL 等）。
        """
        raw_results = search(query, num_results=num_results)

        # Convert raw results to SearchItem format
        results = []
        skipped = 0
        for i, item in enumerate(raw_results):
            if isinstance(item, str):
                title = f"Baidu Result {i+1}"
                url = item
                description = None
            elif isinstance(item, dict):
                title = item.get("title", f"Baidu Result {i+1}")
                url = item.get("url", "")
                description = item.get("abstract", None)
            else:
                try:
                    title = getattr(item, "title", f"Baidu Result {i+1}")
                    url = getattr(item, "url", "")
                    description = getattr(item, "abstract", None)
                except Exception:
                    title = f"Baidu Result {i+1}"
                    url = str(item)
                    description = None

            # 过滤无效结果
            if not _is_valid_result(title, url):
                skipped += 1
                continue

            results.append(
                SearchItem(title=title, url=url, description=description)
            )

        if skipped > 0:
            logger.debug(f"Baidu search: filtered {skipped} invalid results (placeholders/relative URLs)")

        return results
