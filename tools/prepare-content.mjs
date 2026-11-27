#!/usr/bin/env node
/**
 * Build the Quartz content/ folder from an Obsidian vault.
 *
 * - Copies Markdown notes and their relative `assets/` images.
 * - Renders the handful of Dataview queries used in the vault into static tables,
 *   so index/dashboard pages keep working on a static site.
 * - Generates a landing page at content/index.md.
 *
 * Usage: VAULT_DIR="/path/to/vault" node tools/prepare-content.mjs
 */
import fs from "node:fs"
import path from "node:path"
import YAML from "yaml"

const VAULT = path.resolve(process.env.VAULT_DIR ?? "/Users/7xxx/Documents/Obsidian Vault")
const OUT = path.resolve("content")
const SKIP_DIRS = new Set([".obsidian", ".trash", "99 模板", ".git"])
const SKIP_FILES = new Set([".DS_Store", "未命名.base", "欢迎.md"])
const MD_EXT = new Set([".md", ".markdown"])

// ---------------------------------------------------------------- utilities

function walk(dir, acc = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith(".") && entry.name !== ".") {
      if (SKIP_DIRS.has(entry.name)) continue
    }
    const abs = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      if (SKIP_DIRS.has(entry.name)) continue
      walk(abs, acc)
    } else {
      if (SKIP_FILES.has(entry.name)) continue
      acc.push(abs)
    }
  }
  return acc
}

const toPosix = (p) => p.split(path.sep).join("/")

function parseFrontmatter(raw) {
  const m = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/)
  if (!m) return { fm: {}, body: raw }
  let fm = {}
  try {
    fm = YAML.parse(m[1]) ?? {}
  } catch {
    fm = {}
  }
  if (typeof fm !== "object" || Array.isArray(fm)) fm = {}
  return { fm, body: raw.slice(m[0].length) }
}

function escapeCell(value) {
  return String(value)
    .replace(/\|/g, "\\|")
    .replace(/\r?\n/g, "<br>")
    .trim()
}

function asList(value) {
  if (value == null || value === "") return []
  return Array.isArray(value) ? value : [value]
}

function firstValue(value) {
  if (Array.isArray(value)) return value.length ? value[0] : undefined
  return value
}

function compareValues(a, b) {
  const av = firstValue(a)
  const bv = firstValue(b)
  if (av == null && bv == null) return 0
  if (av == null) return 1
  if (bv == null) return -1
  if (av instanceof Date && bv instanceof Date) return av.getTime() - bv.getTime()
  if (typeof av === "number" && typeof bv === "number") return av - bv
  return String(av).localeCompare(String(bv), "zh")
}

// ---------------------------------------------------------------- dataview

class Tokenizer {
  constructor(src) {
    this.tokens = []
    const re =
      /\s*(?:(>=|<=|!=|=|>|<|\(|\)|,)|"([^"]*)"|'([^']*)'|([A-Za-z_][A-Za-z0-9_.]*)|(-?\d+(?:\.\d+)?))/g
    let m
    while ((m = re.exec(src))) {
      if (m[1]) this.tokens.push({ type: "op", value: m[1] })
      else if (m[2] !== undefined) this.tokens.push({ type: "str", value: m[2] })
      else if (m[3] !== undefined) this.tokens.push({ type: "str", value: m[3] })
      else if (m[4]) this.tokens.push({ type: "id", value: m[4] })
      else if (m[5]) this.tokens.push({ type: "num", value: Number(m[5]) })
    }
    this.pos = 0
  }
  peek() {
    return this.tokens[this.pos]
  }
  next() {
    return this.tokens[this.pos++]
  }
  eat(value) {
    const t = this.peek()
    if (t && t.type === "op" && t.value === value) {
      this.pos++
      return true
    }
    return false
  }
}

function parseExpression(src) {
  const tz = new Tokenizer(src)
  function parseOr() {
    let left = parseAnd()
    while (tz.peek() && tz.peek().type === "id" && tz.peek().value.toUpperCase() === "OR") {
      tz.next()
      left = { t: "or", l: left, r: parseAnd() }
    }
    return left
  }
  function parseAnd() {
    let left = parseNot()
    while (tz.peek() && tz.peek().type === "id" && tz.peek().value.toUpperCase() === "AND") {
      tz.next()
      left = { t: "and", l: left, r: parseNot() }
    }
    return left
  }
  function parseNot() {
    if (tz.peek() && tz.peek().type === "id" && tz.peek().value.toUpperCase() === "NOT") {
      tz.next()
      return { t: "not", v: parseNot() }
    }
    return parseComparison()
  }
  function parseComparison() {
    const left = parsePrimary()
    const op = tz.peek()
    if (op && op.type === "op" && ["=", "!=", ">", "<", ">=", "<="].includes(op.value)) {
      tz.next()
      return { t: "cmp", op: op.value, l: left, r: parsePrimary() }
    }
    return left
  }
  function parsePrimary() {
    const t = tz.next()
    if (!t) return { t: "lit", value: null }
    if (t.type === "op" && t.value === "(") {
      const inner = parseOr()
      tz.eat(")")
      return inner
    }
    if (t.type === "str") return { t: "lit", value: t.value }
    if (t.type === "num") return { t: "lit", value: t.value }
    if (t.type === "id") {
      if (tz.eat("(")) {
        const args = []
        if (!tz.eat(")")) {
          do {
            args.push(parseOr())
          } while (tz.eat(","))
          tz.eat(")")
        }
        return { t: "call", name: t.value, args }
      }
      return { t: "id", name: t.value }
    }
    return { t: "lit", value: null }
  }
  return parseOr()
}

function lookup(name, ctx) {
  if (name === "this.file.name") return ctx.thisNote?.name
  if (name === "this.file.path") return ctx.thisNote?.rel
  if (name === "file.name") return ctx.note?.name
  if (name === "file.path") return ctx.note?.rel
  if (name === "file.folder") return ctx.note?.dir
  if (name === "file.mtime") return ctx.note?.mtime
  if (name === "file.ctime") return ctx.note?.mtime
  if (name === "file.link") return ctx.note ? `[[${ctx.note.rel}|${ctx.note.name}]]` : ""
  if (name === "file.tags") return asList(ctx.note?.fm?.tags).map((t) => `#${t}`)
  if (name === "key") return ctx.key
  if (name === "rows") return ctx.rows
  if (name.startsWith("this.")) return undefined
  return ctx.overrides && name in ctx.overrides ? ctx.overrides[name] : ctx.note?.fm?.[name]
}

function evaluate(node, ctx) {
  switch (node.t) {
    case "lit":
      return node.value
    case "id":
      return lookup(node.name, ctx)
    case "not":
      return !truthy(evaluate(node.v, ctx))
    case "and":
      return truthy(evaluate(node.l, ctx)) && truthy(evaluate(node.r, ctx))
    case "or":
      return truthy(evaluate(node.l, ctx)) || truthy(evaluate(node.r, ctx))
    case "cmp": {
      const a = evaluate(node.l, ctx)
      const b = evaluate(node.r, ctx)
      switch (node.op) {
        case "=":
          return String(firstValue(a) ?? "") === String(firstValue(b) ?? "")
        case "!=":
          return String(firstValue(a) ?? "") !== String(firstValue(b) ?? "")
        case ">":
          return compareValues(a, b) > 0
        case "<":
          return compareValues(a, b) < 0
        case ">=":
          return compareValues(a, b) >= 0
        case "<=":
          return compareValues(a, b) <= 0
      }
      return false
    }
    case "call": {
      const name = node.name
      const args = node.args.map((a) => evaluate(a, ctx))
      if (name === "contains") {
        const hay = args[0]
        const needle = args[1]
        if (Array.isArray(hay)) return hay.map(String).includes(String(needle))
        if (typeof hay === "string") return hay.includes(String(needle))
        return false
      }
      if (name === "length") {
        const v = args[0]
        if (Array.isArray(v)) return v.length
        if (typeof v === "string") return v.length
        if (v && typeof v === "object") return Object.keys(v).length
        return 0
      }
      throw new Error(`Unsupported dataview function: ${name}`)
    }
  }
  return undefined
}

function truthy(v) {
  if (v == null || v === false) return false
  if (v === 0 || v === "") return false
  if (Array.isArray(v)) return v.length > 0
  return true
}

function splitTopLevel(s, sep = ",") {
  const out = []
  let depth = 0
  let quote = null
  let cur = ""
  for (const ch of s) {
    if (quote) {
      cur += ch
      if (ch === quote) quote = null
      continue
    }
    if (ch === '"' || ch === "'") {
      quote = ch
      cur += ch
      continue
    }
    if (ch === "(") depth++
    if (ch === ")") depth--
    if (ch === sep && depth === 0) {
      out.push(cur)
      cur = ""
      continue
    }
    cur += ch
  }
  if (cur.trim()) out.push(cur)
  return out
}

function parseDataviewQuery(src) {
  const lines = src.split("\n").map((l) => l.trim()).filter(Boolean)
  const query = { type: "table", columns: null, from: [], where: null, sort: [], groupBy: null, flatten: null, limit: null }
  let i = 0
  const head = lines[i] ?? ""
  if (/^TABLE\b/i.test(head)) {
    query.type = "table"
    const rest = head
      .replace(/^TABLE\b/i, "")
      .trim()
      .replace(/^WITHOUT\s+ID\b/i, "")
      .trim()
    if (rest) query.columns = splitTopLevel(rest)
    i++
  }
  for (; i < lines.length; i++) {
    const line = lines[i]
    let m
    if ((m = line.match(/^FROM\s+(.+)$/i))) {
      const parts = m[1].split(/\s+OR\s+/i)
      query.from = parts.map((p) => p.trim().replace(/^"|"$/g, ""))
    } else if ((m = line.match(/^WHERE\s+(.+)$/i))) {
      query.where = parseExpression(m[1])
    } else if ((m = line.match(/^SORT\s+(.+)$/i))) {
      query.sort = m[1].split(",").map((part) => {
        const bits = part.trim().split(/\s+/)
        return { field: bits[0], dir: (bits[1] ?? "ASC").toUpperCase() }
      })
    } else if ((m = line.match(/^GROUP\s+BY\s+(.+)$/i))) {
      query.groupBy = m[1].trim()
    } else if ((m = line.match(/^FLATTEN\s+(.+)$/i))) {
      query.flatten = m[1].trim()
    } else if ((m = line.match(/^LIMIT\s+(\d+)$/i))) {
      query.limit = Number(m[1])
    }
  }
  return query
}

function parseColumn(columnSrc) {
  const m = columnSrc.trim().match(/^(.*?)\s+AS\s+"([^"]*)"\s*$/i)
  const expr = (m ? m[1] : columnSrc).trim()
  const label = m ? m[2] : expr
  return { expr, label, ast: parseExpression(expr) }
}

function inFolder(note, folder) {
  const f = folder.replace(/^"|"$/g, "").replace(/\/$/, "")
  return note.rel === f || note.rel.startsWith(f + "/")
}

function renderDataview(src, allNotes, currentNote) {
  let query
  try {
    query = parseDataviewQuery(src)
  } catch {
    return "> [!warning] Dataview 查询无法静态渲染。\n"
  }
  const columns = (query.columns ?? ['file.link AS "文件"']).map(parseColumn)

  let rows = allNotes
    .filter((n) => query.from.length === 0 || query.from.some((f) => inFolder(n, f)))
    .map((note) => ({ note, overrides: {} }))
    .filter((row) => (query.where ? truthy(evaluate(query.where, { note: row.note, thisNote: currentNote, overrides: row.overrides })) : true))

  if (query.flatten) {
    const field = query.flatten
    const next = []
    for (const row of rows) {
      const value = lookup(field, { note: row.note, thisNote: currentNote, overrides: row.overrides })
      const values = asList(value)
      if (values.length === 0) continue
      for (const v of values) {
        next.push({ note: row.note, overrides: { ...row.overrides, [field]: v } })
      }
    }
    rows = next
  }

  if (query.groupBy) {
    const field = query.groupBy
    const groups = new Map()
    for (const row of rows) {
      const value = lookup(field, { note: row.note, thisNote: currentNote, overrides: row.overrides })
      for (const v of asList(value)) {
        const key = String(v)
        if (!groups.has(key)) groups.set(key, { key, rows: [], note: row.note, overrides: {} })
        groups.get(key).rows.push(row.note)
      }
    }
    rows = [...groups.values()].map((g) => ({ note: g.note, overrides: { key: g.key, rows: g.rows }, group: g }))
  }

  const ctxFor = (row) => ({ note: row.note, thisNote: currentNote, overrides: row.overrides, key: row.overrides.key, rows: row.overrides.rows })

  if (query.sort.length) {
    rows = [...rows].sort((a, b) => {
      for (const { field, dir } of query.sort) {
        const av = lookup(field, ctxFor(a))
        const bv = lookup(field, ctxFor(b))
        const cmp = compareValues(av, bv)
        if (cmp !== 0) return dir === "DESC" ? -cmp : cmp
      }
      return 0
    })
  }

  if (query.limit != null) rows = rows.slice(0, query.limit)

  if (rows.length === 0) return "*（暂无匹配条目）*\n"

  const header = `| ${columns.map((c) => c.label).join(" | ")} |`
  const divider = `| ${columns.map(() => "---").join(" | ")} |`
  const body = rows.map((row) => {
    const ctx = ctxFor(row)
    const cells = columns.map((col) => {
      const value = evaluate(col.ast, ctx)
      if (value == null) return ""
      if (value instanceof Date) {
        const y = value.getFullYear()
        const m = String(value.getMonth() + 1).padStart(2, "0")
        const d = String(value.getDate()).padStart(2, "0")
        return `${y}-${m}-${d}`
      }
      if (Array.isArray(value)) return escapeCell(value.join("<br>"))
      if (typeof value === "number") return String(value)
      return escapeCell(value)
    })
    return `| ${cells.join(" | ")} |`
  })
  return [header, divider, ...body, ""].join("\n")
}

// ---------------------------------------------------------------- links

/**
 * Quartz's `relative` link resolution treats every target as relative to the note's own
 * folder, while Obsidian treats `[[note]]` / `[[folder/note]]` as vault-root paths.
 * Rewrite each wikilink to the equivalent note-folder-relative path so both agree.
 */
function makeResolver(notes) {
  const byRel = new Map()
  for (const n of notes) {
    byRel.set(n.rel.toLowerCase(), n)
    byRel.set(n.relWithExt.toLowerCase(), n)
  }
  const byName = new Map()
  for (const n of notes) {
    const key = n.name.toLowerCase()
    if (!byName.has(key)) byName.set(key, [])
    byName.get(key).push(n)
  }

  return function resolve(target, fromDir) {
    const key = target.toLowerCase()
    let hit = byRel.get(key) ?? byRel.get(key + ".md")
    if (!hit) {
      const candidates = byName.get(key.split("/").pop()) ?? []
      if (candidates.length === 0) {
        // fall back to a path suffix match (e.g. `2026-10-30/note`)
        const suffix = "/" + key
        hit = notes.find((n) => n.rel.toLowerCase().endsWith(suffix))
      } else if (candidates.length === 1) {
        hit = candidates[0]
      } else {
        const sameDir = candidates.filter((n) => n.dir === fromDir)
        const pool = sameDir.length ? sameDir : candidates
        hit = [...pool].sort((a, b) => a.rel.length - b.rel.length)[0]
      }
    }
    if (!hit) return null
    const rel = path.posix.relative(fromDir || ".", hit.rel)
    return rel === "" ? hit.name : rel
  }
}

function rewriteWikilinks(body, note, resolve) {
  const parts = body.split(/(^```[\s\S]*?^```)/m)
  return parts
    .map((part, index) => {
      if (index % 2 === 1) return part // fenced code block
      return part.replace(/\[\[([^\[\]]+)\]\]/g, (match, inner) => {
        let target = inner
        let alias = null
        const pipe = inner.indexOf("|")
        if (pipe !== -1) {
          target = inner.slice(0, pipe)
          alias = inner.slice(pipe + 1)
        }
        let escapedPipe = false
        if (target.endsWith("\\")) {
          target = target.slice(0, -1)
          escapedPipe = true
        }
        target = target.trim()
        if (!target || target.startsWith("#") || /^[a-z]+:\/\//i.test(target)) return match

        const hash = target.indexOf("#")
        const anchor = hash === -1 ? "" : target.slice(hash)
        const page = hash === -1 ? target : target.slice(0, hash)
        let decoded = page
        try {
          decoded = decodeURI(page)
        } catch {}
        const resolved = resolve(decoded, note.dir)
        if (!resolved) return match
        const nextTarget = resolved + anchor
        const sep = escapedPipe ? "\\|" : "|"
        return alias === null
          ? `[[${nextTarget}]]`
          : `[[${nextTarget}${sep}${alias}]]`
      })
    })
    .join("")
}

// ---------------------------------------------------------------- main

function main() {
  if (!fs.existsSync(VAULT)) {
    console.error(`Vault not found: ${VAULT}`)
    process.exit(1)
  }

  const files = walk(VAULT)
  const mdFiles = files.filter((f) => MD_EXT.has(path.extname(f).toLowerCase()))
  const assetFiles = files.filter((f) => !MD_EXT.has(path.extname(f).toLowerCase()))

  const notes = mdFiles.map((abs) => {
    const relPath = toPosix(path.relative(VAULT, abs))
    const raw = fs.readFileSync(abs, "utf8")
    const { fm } = parseFrontmatter(raw)
    const stat = fs.statSync(abs)
    let body = parseFrontmatter(raw).body

    // Quartz renders the note title above the body, so drop the H1 that every
    // Obsidian note starts with and keep its text as the title.
    const h1 = body.match(/^\s*#[ \t]+(.+?)[ \t]*(?:\r?\n)+/)
    const h1Title = h1 ? h1[1].trim() : null
    if (h1) body = body.slice(h1[0].length)

    // The vault explains how to install Dataview; on the published site the
    // tables are already rendered, so that note is just noise.
    body = body.replace(/^> \[!info\] 这个页面需要[^\n]*\n(?:>.*(?:\n|$))*/m, "")

    if (!fm.title && h1Title) fm.title = h1Title
    const dateOnly = (d) =>
      `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`
    const stamp = dateOnly(stat.mtime)
    if (!fm.created) fm.created = stamp
    if (!fm.modified && !fm.updated && !fm.lastmod) fm.modified = stamp

    // keep the title at the top of the frontmatter
    const ordered = fm.title ? { title: fm.title, ...fm } : fm

    return {
      abs,
      rel: relPath.replace(/\.(md|markdown)$/i, ""),
      relWithExt: relPath,
      dir: toPosix(path.dirname(relPath)) === "." ? "" : toPosix(path.dirname(relPath)),
      name: path.basename(relPath, path.extname(relPath)),
      fm: ordered,
      body,
      mtime: stat.mtime,
    }
  })

  const byRel = new Map(notes.map((n) => [n.rel, n]))
  const resolveLink = makeResolver(notes)

  // clean output
  fs.rmSync(OUT, { recursive: true, force: true })
  fs.mkdirSync(OUT, { recursive: true })

  let renderedQueries = 0
  for (const note of notes) {
    const body = note.body.replace(/```dataview\s*\n([\s\S]*?)```/g, (match, query) => {
      renderedQueries++
      try {
        return renderDataview(query, notes, note)
      } catch (err) {
        console.warn(`dataview failed in ${note.rel}:`, err.message)
        return match
      }
    })
    const rewritten = rewriteWikilinks(body, note, resolveLink)

    const outPath = path.join(OUT, note.relWithExt)
    fs.mkdirSync(path.dirname(outPath), { recursive: true })
    const frontmatter = Object.keys(note.fm).length
      ? `---\n${YAML.stringify(note.fm).trimEnd()}\n---\n\n`
      : ""
    fs.writeFileSync(outPath, frontmatter + rewritten.trimStart())
  }

  for (const asset of assetFiles) {
    const rel = path.relative(VAULT, asset)
    const dest = path.join(OUT, rel)
    fs.mkdirSync(path.dirname(dest), { recursive: true })
    fs.copyFileSync(asset, dest)
    const stat = fs.statSync(asset)
    fs.utimesSync(dest, stat.atime, stat.mtime)
  }

  // landing page
  const countBy = (prefix) => notes.filter((n) => n.rel === prefix || n.rel.startsWith(prefix + "/")).length
  const paperNotes = notes.filter((n) => n.rel.startsWith("10 论文笔记/") && n.name !== "_索引")
  const themeDirs = [...new Set(paperNotes.map((n) => n.dir.split("/")[1]))].filter(Boolean).sort()
  const methodCards = notes.filter((n) => n.rel.startsWith("30 方法卡片/") && n.name !== "_索引")
  const proposals = notes.filter((n) => n.rel.startsWith("50 研究提案/") && n.name !== "_索引")
  const dailyReports = notes.filter((n) => n.rel.startsWith("40 每日报告/") && n.name !== "汇总")
  const recent = [...notes]
    .filter((n) => !n.rel.includes("_索引") && n.name !== "汇总" && n.name !== "index")
    .sort((a, b) => b.mtime - a.mtime)
    .slice(0, 12)

  const nav = [
    ["论文精读", "按主题手工维护的总目录", "论文精读"],
    ["论文总览", "全部论文的仪表盘：状态、评分、主题统计", "论文总览"],
    ["局限性汇总", "跨论文的方法层面四点局限", "局限性汇总"],
    ["流程说明", "每天自动产出日报与精读的链路", "流程说明"],
  ]

  const themeLines = themeDirs.map((theme) => {
    const dir = `10 论文笔记/${theme}`
    const count = paperNotes.filter((n) => n.dir === dir).length
    return `- [[${dir}/_索引|${theme}]] · ${count} 篇`
  })

  const recentLines = recent.map((n) => {
    const date = `${n.mtime.getFullYear()}-${String(n.mtime.getMonth() + 1).padStart(2, "0")}-${String(n.mtime.getDate()).padStart(2, "0")}`
    return `- ${date} · [[${n.rel}|${n.name}]]`
  })

  const index = `---
title: ChenFan 的研究笔记
---

围绕「读论文」搭起来的公开笔记库：每天读一轮新论文，把值得跟的写成精读笔记，再把反复出现的技术点抽成方法卡片。

目前收录 **${paperNotes.length}** 篇论文笔记、**${methodCards.length}** 张方法卡片、**${dailyReports.length}** 篇每日报告、**${proposals.length}** 份研究提案。

## 四个入口

${nav.map(([title, desc, rel]) => `- **[[${rel}|${title}]]** —— ${desc}`).join("\n")}

## 八个主题

${themeLines.join("\n")}

## 最近更新

${recentLines.join("\n")}

## 关于这个站点

笔记用 [Obsidian](https://obsidian.md) 写作，用 [Quartz](https://quartz.jzhao.xyz) 发布到 GitHub Pages。数学公式、双链、反向链接和关系图都保留了下来；原文中由 Dataview 生成的表格已在构建时渲染为静态表格。
`

  fs.writeFileSync(path.join(OUT, "index.md"), index)

  console.log(`vault: ${VAULT}`)
  console.log(`notes: ${notes.length}, assets: ${assetFiles.length}, dataview blocks rendered: ${renderedQueries}`)
}

main()
