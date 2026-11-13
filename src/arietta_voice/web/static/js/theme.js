// Runs before first paint so the page never flashes the wrong theme.
(function () {
  var stored = null;
  try {
    stored = localStorage.getItem("arietta-theme");
  } catch (error) {
    stored = null;
  }
  var prefersLight = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
  var theme = stored === "light" || stored === "dark" ? stored : prefersLight ? "light" : "dark";
  document.documentElement.dataset.theme = theme;
})();
