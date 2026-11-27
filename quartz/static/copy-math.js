// 复制 KaTeX 公式时输出 LaTeX 源码，而不是渲染后的文字 / MathML。
// KaTeX 会把原始 TeX 放在 <annotation encoding="application/x-tex"> 里，这里读出来，
// 行内公式包成 $...$，独立公式包成 $$...$$，粘贴到 Obsidian / Typora / 论文里就能直接用。
document.addEventListener("copy", (event) => {
  const selection = window.getSelection()
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) return

  const range = selection.getRangeAt(0)
  let node = range.commonAncestorContainer
  if (node.nodeType === Node.TEXT_NODE) node = node.parentElement
  if (!node || !node.closest) return

  const katex = node.closest(".katex")
  if (!katex) return
  // 只处理「选中的内容完全落在一个公式里」的情况；跨段落 / 跨多个公式时保持浏览器默认行为。
  if (!katex.contains(range.startContainer) || !katex.contains(range.endContainer)) return

  const annotation = katex.querySelector('annotation[encoding="application/x-tex"]')
  const tex = annotation && annotation.textContent ? annotation.textContent.trim() : ""
  if (!tex) return

  const isDisplay = !!katex.closest(".katex-display")
  const text = isDisplay ? `$$\n${tex}\n$$` : `$${tex}$`

  if (event.clipboardData) {
    event.clipboardData.setData("text/plain", text)
    event.preventDefault()
    // 阻止 KaTeX 自带的 copy-tex（从 CDN 加载）之后再覆盖一次：它会把独立公式也写成 $...$
    event.stopImmediatePropagation()
  }
}, true)
