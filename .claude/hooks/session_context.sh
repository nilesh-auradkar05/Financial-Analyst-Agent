#!/usr/bin/env bash
# H1 — SessionStart: inject lessons, recent commits, active sprint task, tree status.
# Re-runs on --resume (source=resume), so timestamps stay fresh. Output goes to Claude as context.
set -u
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
echo "## Session context ($(date -u +%Y-%m-%dT%H:%MZ))"
echo
if [ -f tasks/lessons.md ]; then
  echo "### tasks/lessons.md (review before working)"; echo; cat tasks/lessons.md; echo
fi
echo "### Recent commits"; git log --oneline -10 2>/dev/null || echo "(no git)"; echo
echo "### Working tree"; git status --short 2>/dev/null | head -30; echo
if [ -f docs/sprint-plan.md ]; then
  echo "### Open sprint tasks (first 6 with Result: Pending)"
  grep -B14 '^Result: Pending' docs/sprint-plan.md | grep -E '^S[0-9]+-T[0-9a-z]+ —' | head -6
  echo
fi
if [ -f tasks/todo.md ]; then
  echo "### tasks/todo.md open items"; grep -E '^- \[ \]' tasks/todo.md | head -15; echo
fi
echo "Rules: plan before implementing (tasks/todo.md); one axis per benchmark; nothing done without recorded verification; governance docs read-only (ALLOW_SPEC_EDIT=1 to amend)."
exit 0
