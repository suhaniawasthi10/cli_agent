// ===========================================================
// Scaler clone — script.js
// Sticky header shrink + smooth in-page anchor scroll.
// ===========================================================

(function () {
  const header = document.getElementById('site-header');

  // Add a "scrolled" class once the user scrolls past 30px so the
  // CSS can shrink the header padding and darken its background.
  function onScroll() {
    if (!header) return;
    if (window.scrollY > 30) {
      header.classList.add('scrolled');
    } else {
      header.classList.remove('scrolled');
    }
  }

  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  // Smooth-scroll for in-page anchor links (e.g. #programs, #events).
  // Falls back to default browser behavior if href doesn't match an element.
  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener('click', (e) => {
      const targetId = link.getAttribute('href');
      if (!targetId || targetId === '#') return;

      const target = document.querySelector(targetId);
      if (!target) return;

      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
})();
