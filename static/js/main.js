/* main.js — MindCare AI Frontend Scripts */

// Auto-dismiss flash messages after 5 seconds
document.addEventListener('DOMContentLoaded', () => {
  const flashes = document.querySelectorAll('.flash');
  flashes.forEach(f => {
    setTimeout(() => {
      f.style.transition = 'opacity 0.5s ease';
      f.style.opacity = '0';
      setTimeout(() => f.remove(), 500);
    }, 5000);
  });

  // Animate stat numbers counting up
  const statNums = document.querySelectorAll('.sc-num');
  statNums.forEach(el => {
    const target = parseInt(el.textContent, 10);
    if (!isNaN(target) && target > 0) {
      let current = 0;
      const step = Math.max(1, Math.ceil(target / 40));
      const timer = setInterval(() => {
        current = Math.min(current + step, target);
        el.textContent = current;
        if (current >= target) clearInterval(timer);
      }, 30);
    }
  });

  // Animate progress bars
  const fills = document.querySelectorAll('.vc-fill');
  fills.forEach(fill => {
    const w = fill.style.width;
    fill.style.width = '0';
    setTimeout(() => { fill.style.width = w; }, 300);
  });
});
