(function () {
  const chips = document.querySelectorAll('.chip');
  const rows  = document.querySelectorAll('.table tbody tr');

  chips.forEach(chip => {
    chip.addEventListener('click', () => {
      chips.forEach(c => c.classList.remove('is-active'));
      chip.classList.add('is-active');

      const filter = chip.dataset.filter;
      rows.forEach(row => {
        row.hidden = filter !== 'all' && row.dataset.category !== filter;
      });
    });
  });
})();