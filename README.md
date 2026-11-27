# 陈帆的研究笔记

把 Obsidian vault 发布成静态站点：[c6oisini.github.io](https://c6oisini.github.io)

用 [Quartz v5](https://quartz.jzhao.xyz) 构建，托管在 GitHub Pages。原文里的双链、反向链接、关系图、KaTeX 公式和 mermaid 图都保留；Dataview 查询在构建前渲染成静态表格。

## 本地流程

```bash
# 1. 从 vault 生成 content/（会清空并重建 content/）
VAULT_DIR="/Users/7xxx/Documents/Obsidian Vault" node tools/prepare-content.mjs

# 或者一步完成：重新生成 + 构建 + 提交 + 推送（Actions 自动部署）
tools/publish.sh

# 2. 本地预览
npx quartz build --serve

# 3. 构建静态文件到 public/
npx quartz build
```

`tools/prepare-content.mjs` 负责：

- 复制笔记和 `assets/` 图片，跳过 `.obsidian`、`99 模板`、`欢迎.md` 等非正文文件；
- 把笔记开头的 H1 变成 frontmatter `title`（站点自己渲染标题），补 `created` / `modified` 日期；
- 把 21 个 Dataview 查询渲染成静态表格；
- 把 Obsidian 的 vault 根路径 wikilink 改写为相对当前笔记的路径（Quartz 的 `relative` 解析要求）；
- 生成首页 `content/index.md`。

## 目录

- `content/` — 由 vault 生成，提交进仓库，CI 直接构建
- `quartz/` — Quartz 源码（含 `quartz/styles/custom.scss` 的中文字体与表格样式）
- `quartz.config.yaml` — 站点配置
- `tools/prepare-content.mjs` — vault → content 预处理脚本
- `.github/workflows/deploy.yml` — push 到 `main` 后构建并部署到 GitHub Pages
