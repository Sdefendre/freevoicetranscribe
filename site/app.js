(() => {
  "use strict";

  const skipLink = document.querySelector(".skip-link");
  const mainContent = document.querySelector("#main");

  skipLink.addEventListener("click", () => {
    window.requestAnimationFrame(() => {
      mainContent.focus({ preventScroll: true });
    });
  });

  document.querySelectorAll("[data-year]").forEach((year) => {
    year.textContent = String(new Date().getFullYear());
  });
})();
