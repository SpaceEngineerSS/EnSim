// arithmatex (generic mode) wraps formulas in \( \) and \[ \]; the backslashes
// must be doubled here, or JavaScript drops them and MathJax finds nothing.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: { ignoreHtmlClass: ".*|", processHtmlClass: "arithmatex" },
};

// Material's instant navigation swaps pages without a reload
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
