(() => {
  const search = document.querySelector('#search');
  const sections = [...document.querySelectorAll('.doc-section')];
  const links = [...document.querySelectorAll('nav a[data-section]')];
  const status = document.querySelector('#search-status');
  function filter() {
    const query = search.value.trim().toLowerCase();
    let count = 0;
    for (const section of sections) {
      section.hidden = query !== '' && !section.textContent.toLowerCase().includes(query);
      if (!section.hidden) count++;
    }
    for (const link of links) link.hidden = document.getElementById(link.dataset.section).hidden;
    for (const group of document.querySelectorAll('nav details')) {
      group.hidden = [...group.querySelectorAll('a')].every(link => link.hidden);
      if (query && !group.hidden) group.open = true;
    }
    status.textContent = query ? `${count} matching sections for “${search.value.trim()}”` : '';
    document.querySelector('#no-results').hidden = count > 0;
  }
  search.addEventListener('input', filter);
  document.querySelector('#clear').addEventListener('click', () => {search.value = ''; filter(); search.focus();});
  document.querySelector('#print').addEventListener('click', () => window.print());
  for (const button of document.querySelectorAll('.copy')) {
    button.addEventListener('click', async () => {
      const text = button.closest('.code-block').querySelector('pre code').textContent;
      try {
        if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
        else {
          const area = document.createElement('textarea');
          area.value = text; area.style.position = 'fixed'; area.style.opacity = '0';
          document.body.append(area); area.select();
          const copied = document.execCommand('copy'); area.remove();
          if (!copied) throw new Error('Clipboard unavailable');
        }
        button.textContent = 'Copied';
      } catch { button.textContent = 'Select to copy'; }
      setTimeout(() => button.textContent = 'Copy', 2000);
    });
  }
  document.addEventListener('click', event => {
    const anchor = event.target.closest('a[href^="#"]');
    if (anchor && search.value) {search.value = ''; filter();}
  });
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      const visible = entries.find(entry => entry.isIntersecting);
      if (visible) for (const link of links) link.classList.toggle('active', link.dataset.section === visible.target.id);
    }, {rootMargin: '-110px 0px -65% 0px'});
    sections.forEach(section => observer.observe(section));
  }
})();
