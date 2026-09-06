#!/usr/bin/env bash
# H10 — PostToolUse Edit|Write: ruff fix+format on the written file; mypy for app/. Feedback via exit 2.
set -u
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
file=$(python3 -c 'import json,sys; d=json.load(sys.stdin); t=d.get("tool_input",{}); print(t.get("file_path") or t.get("path") or "")' 2>/dev/null)
[ -z "$file" ] && exit 0
case "$file" in
  *.py) ;;
  *) exit 0 ;;
esac
[ -f "$file" ] || exit 0
runner="uv run"; command -v uv >/dev/null 2>&1 || runner=""
out=""
$runner ruff check --fix --quiet "$file" >/dev/null 2>&1
$runner ruff format --quiet "$file" >/dev/null 2>&1
lint=$($runner ruff check "$file" 2>&1); rc=$?
[ $rc -ne 0 ] && out="$out\n[ruff] $lint"
case "$file" in
  app/*|evaluation/*)
    ty=$($runner mypy "$file" 2>&1 | tail -15); rc=$?
    [ $rc -ne 0 ] && out="$out\n[mypy] $ty" ;;
esac
if [ -n "$out" ]; then
  printf "H10 lint/type feedback for %s (fix before continuing):%b\n" "$file" "$out" >&2
  exit 2
fi
exit 0
