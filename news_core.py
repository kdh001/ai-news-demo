"""원본의 뉴스 정규화와 다이제스트 후처리 코드를 발췌한 모듈."""
from dataclasses import dataclass, asdict
from typing import Optional
import re
import unicodedata


@dataclass
class Item:
    """수집된 한 건의 게시물을 표현하는 정규화 스키마.

    HN, Reddit, 그 외 어떤 소스든 결국 이 형태로 변환됩니다.
    dataclass를 쓰면 자동으로 __init__, __repr__이 생성되어 깔끔합니다.
    """

    source: str  # 출처 식별자 ("hackernews", "reddit/MachineLearning" 등)
    title: str  # 글 제목
    url: str  # 원문 URL (또는 토론 페이지 URL)
    points: int  # 인기도 점수 (HN points, Reddit score)
    comments: int  # 댓글 수
    author: Optional[str] = None  # 작성자 (없을 수 있음)
    summary: Optional[str] = None  # 본문 일부 (있을 때만)

    def as_dict(self) -> dict:
        """Cursor SDK 프롬프트에 넣기 좋게 dict로 변환."""
        return asdict(self)

_HN_AI_KEYWORDS = (
    "ai",
    "llm",
    "gpt",
    "claude",
    "gemini",
    "openai",
    "anthropic",
    "deepmind",
    "mistral",
    "llama",
    "qwen",
    "machine learning",
    "deep learning",
    "neural",
    "transformer",
    "agent",
    "rag",
    "diffusion",
    "stable diffusion",
    "mcp",
    "embedding",
)

def _is_ai_related(title: str) -> bool:
    """제목에 AI 관련 키워드가 포함되어 있는지 검사.

    Algolia의 AND 매칭 한계를 우회하기 위해 클라이언트 측에서 직접 필터링.
    """
    t = title.lower()
    return any(kw in t for kw in _HN_AI_KEYWORDS)

def normalize_hackernews(data: dict, since_ts: float, until_ts: float, keep_top: int = 15) -> list[Item]:
    """원본 HN 파싱을 사용하되 날짜와 데이터 형식, 중복을 검사한다.

    시각이 없거나 숫자가 아니면 제외하고, 기간은 시작 포함·끝 미포함이다.
    URL 또는 objectID가 같은 글은 첫 번째만 유지한다.
    """
    if not isinstance(data, dict) or not isinstance(data.get("hits"), list):
        raise ValueError("입력에는 hits 배열이 필요합니다")
    if type(keep_top) is not int or keep_top < 0:
        raise ValueError("keep_top은 0 이상의 정수여야 합니다")
    if keep_top == 0:
        return []
    items = []
    seen = set()
    for hit in data["hits"]:
        if not isinstance(hit, dict):
            raise ValueError("각 hits 항목은 객체여야 합니다")
        timestamp = hit.get("created_at_i")
        # bool은 int의 하위형이므로 type으로 제외한다.
        if type(timestamp) not in (float, int) or not since_ts <= timestamp < until_ts:
            continue
        title = hit.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("기간 내 항목의 title은 빈 값이 아닌 문자열이어야 합니다")
        if any(unicodedata.category(ch) in ("Cc", "Zl", "Zp") for ch in title) or "`" in title:
            raise ValueError("title에는 제어문자와 백틱을 사용할 수 없습니다")
        if not _is_ai_related(title):
            continue
        story_url = hit.get("url")
        story_url = "" if story_url is None else story_url
        if not isinstance(story_url, str):
            raise ValueError("url은 문자열이어야 합니다")
        object_id = hit.get("objectID")
        if object_id is not None and (type(object_id) not in (str, int) or not str(object_id).strip()):
            raise ValueError("objectID는 빈 값이 아닌 문자열 또는 정수여야 합니다")
        # URL은 원본과 같이 대소문자 및 끝 슬래시를 무시한다.
        keys = []
        if story_url:
            keys.append(("url", story_url.rstrip("/").lower()))
        if object_id is not None:
            keys.append(("id", str(object_id)))
        if any(key in seen for key in keys):
            continue
        seen.update(keys)
        points = hit.get("points")
        comments = hit.get("num_comments")
        points = 0 if points is None else points
        comments = 0 if comments is None else comments
        if type(points) is not int or points < 0 or type(comments) is not int or comments < 0:
            raise ValueError("points와 num_comments는 0 이상의 정수여야 합니다")
        items.append(
            Item(
                source="hackernews",
                title=title,
                url=story_url,
                points=points,
                comments=comments,
                author=None,
            )
        )
        if len(items) >= keep_top:
            break
    return items


def extract_markdown(raw: str) -> str:
    """LLM 응답에서 ```markdown ... ``` 블록만 추출.

    LLM이 가이드를 잘 지키지 않고 코드 펜스 없이 그냥 마크다운을 출력했을 수도 있으므로
    그런 경우엔 원문을 그대로 반환 (안전 폴백).
    """
    stripped = raw.strip()  # 앞뒤 공백/줄바꿈 제거

    # 1) 응답 어딘가에 ```markdown ... ``` 블록이 있으면 그 블록을 최우선 사용
    #    (모델이 설명 문장을 앞에 붙여도 digest 본문만 추출 가능)
    fence_match = re.search(
        r"```(?:markdown|md)?\n(.*?)```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        return fence_match.group(1).strip()

    # 2) 폴백: 코드 펜스로 시작하는 경우의 기존 로직 유지
    for opener in ("```markdown", "```md", "```"):
        if stripped.startswith(opener):
            body = stripped[len(opener):]
            if body.startswith("\n"):
                body = body[1:]
            end_idx = body.rfind("```")
            return body[:end_idx].rstrip() if end_idx != -1 else body.rstrip()

    # 3) 코드 펜스가 없으면 원문 그대로
    return stripped

def _extract_h3_title(line: str) -> str:
    """### 라인에서 사람이 읽을 제목을 추출."""
    # 형식 A: ### [제목](URL)
    linked = re.match(r"^###\s+\[(.*?)\]\(.*\)\s*$", line)
    if linked:
        return linked.group(1).strip()

    # 형식 B: ### 제목
    return line.replace("###", "", 1).strip()

def _score_item_importance(title: str, why_hint: str) -> int:
    """항목 중요도 점수(내 학습 맥락 기반 휴리스틱)."""
    text = f"{title} {why_hint}".lower()
    score = 0

    high_keywords = [
        "mcp",
        "agent",
        "skill",
        "orchestration",
        "gemini",
        "claude",
        "security",
        "cve",
        "firebase",
        "android",
        "local llm",
        "qwen",
    ]
    medium_keywords = [
        "model",
        "release",
        "benchmark",
        "research",
        "paper",
        "open source",
        "tool",
        "webgpu",
    ]

    for kw in high_keywords:
        if kw in text:
            score += 3
    for kw in medium_keywords:
        if kw in text:
            score += 1

    if why_hint.strip():
        score += 2

    return score

def inject_block_ids(
    markdown: str,
) -> tuple[str, list[tuple[str, str]], list[tuple[str, str, str, int]]]:
    """원본 후처리를 확장해 기존 ID를 보존하고 새 ID 충돌을 피한다.

    제목 바로 다음의 빈 줄을 건너 처음 나오는 블록 ID를 기존 ID로 본다.
    기존에 중복된 ID 자체는 수정하지 않는다.
    """
    lines = markdown.splitlines()
    out_lines: list[str] = []
    sections: list[tuple[str, str]] = []
    items: list[tuple[str, str, str, int]] = []
    # 문서에 이미 사용된 블록 ID를 먼저 수집해 새 ID와의 충돌을 피한다.
    used = {line.strip()[1:] for line in lines if re.fullmatch(r"\^[a-z0-9-]+", line.strip())}
    section_idx = 0
    item_idx = 0
    current_item_pos: int | None = None

    def existing_id(position: int) -> str | None:
        # 제목 뒤의 빈 줄을 건너뛰고 바로 이어지는 ID만 재사용한다.
        position += 1
        while position < len(lines) and not lines[position].strip():
            position += 1
        if position < len(lines) and re.fullmatch(r"\^[a-z0-9-]+", lines[position].strip()):
            return lines[position].strip()[1:]
        return None

    def new_id(prefix: str, number: int, width: int) -> tuple[str, int]:
        # 예약된 ID는 건너뛰고 사용하지 않은 번호를 배정한다.
        while True:
            number += 1
            candidate = f"{prefix}-{number:0{width}d}"
            if candidate not in used:
                used.add(candidate)
                return candidate, number

    for position, line in enumerate(lines):
        if line.startswith("## ") and not line.startswith("### "):
            block_id = existing_id(position)
            if block_id is None:
                block_id, section_idx = new_id("sec", section_idx, 2)
                out_lines.extend([line, f"^{block_id}"])
            else:
                out_lines.append(line)
            sections.append((line.replace("## ", "", 1).strip(), block_id))
            current_item_pos = None
            continue

        if line.startswith("### "):
            block_id = existing_id(position)
            if block_id is None:
                block_id, item_idx = new_id("item", item_idx, 3)
                out_lines.extend([line, f"^{block_id}"])
            else:
                out_lines.append(line)
            items.append((_extract_h3_title(line), block_id, "", 0))
            current_item_pos = len(items) - 1
            continue

        if current_item_pos is not None and line.startswith("> 💡 왜 중요한가:"):
            hint = line.replace("> 💡 왜 중요한가:", "", 1).strip()
            name, block_id, _, _ = items[current_item_pos]
            items[current_item_pos] = (name, block_id, hint, _score_item_importance(name, hint))
        out_lines.append(line)

    return "\n".join(out_lines).rstrip() + "\n", sections, items
