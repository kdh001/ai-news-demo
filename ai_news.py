"""합성 응답으로 원본 정규화·다이제스트 후처리 경로를 실행한다."""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from news_core import extract_markdown, inject_block_ids, normalize_hackernews

KST = timezone(timedelta(hours=9))


def parse_date(value: str) -> date:
    """엄격한 YYYY-MM-DD 날짜를 해석한다."""
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError
        return parsed
    except ValueError as exc:
        raise argparse.ArgumentTypeError("날짜는 YYYY-MM-DD 형식이어야 합니다") from exc


def render_mock_agent(items, start: date, end: date) -> str:
    """SDK 대신 합성 결과를 만들고 원본의 후처리를 실행한다."""
    lines = [f"# 합성 AI 뉴스 데모 — {start} ~ {end} (KST)", "", "> 실제 뉴스가 아닌 합성 데이터입니다.", "", "## 합성 수집 항목", ""]
    for item in items:
        lines.extend([f"### {item.title}", f"점수: {item.points} · 댓글: {item.comments}", ""])
    if not items:
        lines.extend(["해당 기간의 AI 관련 항목 없음.", ""])
    # 원본은 에이전트 응답의 마크다운 펜스를 추출한 뒤 블록 ID를 삽입한다.
    simulated_response = "```markdown\n" + "\n".join(lines) + "\n```"
    markdown = extract_markdown(simulated_response)
    return inject_block_ids(markdown)[0]


def main() -> int:
    """합성 응답을 정규화하고 기존 결과를 덮어쓰지 않고 저장한다."""
    parser = argparse.ArgumentParser(description="원본 처리 로직을 사용하는 오프라인 AI 뉴스 데모")
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("sample.json"))
    parser.add_argument("--from", dest="start", required=True, type=parse_date)
    parser.add_argument("--to", dest="end", required=True, type=parse_date)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.end < args.start:
        parser.error("종료일은 시작일보다 빠를 수 없습니다")
    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("hits"), list):
            raise ValueError("입력에는 hits 배열이 필요합니다")
        start_ts = datetime.combine(args.start, time.min, KST).timestamp()
        end_ts = datetime.combine(args.end + timedelta(days=1), time.min, KST).timestamp()
        items = normalize_hackernews(data, start_ts, end_ts)
        result = render_mock_agent(items, args.start, args.end)
        # 배타적 생성으로 기존 출력 및 같은 이름의 메타 파일을 보존한다.
        with args.output.open("x", encoding="utf-8") as output:
            output.write(result)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        parser.exit(1, f"오류: {exc}\n")
    print(f"저장 완료: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
