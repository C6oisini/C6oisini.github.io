#!/usr/bin/env bash
# 从 Obsidian vault 重新生成 content/，构建，提交并推送（GitHub Actions 会自动部署）。
# 用法：VAULT_DIR="/path/to/vault" tools/publish.sh
set -euo pipefail
cd "$(dirname "$0")/.."

VAULT_DIR="${VAULT_DIR:-/Users/7xxx/Documents/Obsidian Vault}" node tools/prepare-content.mjs
npx quartz build

git add -A
if git diff --cached --quiet; then
  echo "没有内容变化，跳过提交"
else
  git commit -m "Sync vault $(date +%F)"
  git push
  echo "已推送，等待 GitHub Actions 部署完成：gh run watch"
fi
