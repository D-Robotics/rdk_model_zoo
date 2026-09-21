(() => {
  'use strict';

  const data = window.MODEL_DATA || { models: [] };
  const models = Array.isArray(data.models) ? data.models : [];
  const facets = window.CatalogFacets;
  const platforms = ['X3', 'X5', 'S100', 'S100P', 'S600'];
  const selectedPlatforms = new Set();
  const batchSize = 6;
  const state = { query: '', scroll: 0, visibleCount: batchSize, domain: null, useCase: null };
  let filteredItems = [];
  let listObserver = null;

  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[character]));
  const modelPlatforms = model => (model.platforms || []).map(platform => String(platform).toUpperCase());
  const groupKey = model => `${model.catalogId || model.sample || model.id}#${model.modelSize || ''}`;
  const platformRank = platform => { const index = platforms.indexOf(platform); return index < 0 ? platforms.length : index; };
  // One grid card per model variant: the per-platform releases of the same
  // model collapse into a single card listing its hardware targets, and the
  // detail page's hardware selector navigates between them.
  const groups = [...models.reduce((map, model) => {
    const key = groupKey(model);
    if (!map.has(key)) map.set(key, { members: [] });
    map.get(key).members.push(model);
    return map;
  }, new Map()).values()].map(group => {
    const primary = group.members[0];
    return {
      members: group.members,
      entry: primary,
      id: primary.id,
      name: primary.name,
      variantName: primary.variantName,
      task: primary.task,
      taskId: primary.taskId,
      description: primary.description,
      descriptionEn: primary.descriptionEn,
      coverImage: primary.coverImage,
      platforms: [...new Set(group.members.flatMap(modelPlatforms))].sort((a, b) => platformRank(a) - platformRank(b)),
    };
  });
  const repositoryUrl = data.repository?.url || 'https://github.com/D-Robotics/rdk_model_zoo';
  const external = (url, label) => `<a href="${esc(url)}" target="_blank" rel="noopener">${label}<span aria-hidden="true"> ↗</span></a>`;

  const languageSwitch = '<div class="language-switch" role="group" aria-label="Language / 语言"><button type="button" data-language="en" lang="en" aria-label="English">EN</button><button type="button" data-language="zh" lang="zh-CN" aria-label="简体中文">中文</button></div>';
  const header = `<header class="header"><a href="#" class="brand"><img src="assets/rdk-brand/logo.png" alt="D-Robotics"><span class="brand-separator"></span><span>RDK <b>Model Zoo</b></span></a><nav aria-label="主导航">${external(repositoryUrl, 'GitHub')}</nav>${languageSwitch}</header>`;
  const footer = '<footer><div class="footer-inner"><span>© 2026 D-Robotics <span class="muted">/ Model Zoo</span></span></div></footer>';
  const search = '<label class="search"><span class="search-icon" aria-hidden="true"></span><input id="search" type="search" placeholder="搜索模型" aria-label="搜索模型"></label>';
  const taskFilters = `<div id="filters" class="task-domains">${facets.groups.map(group => `<div class="task-domain" data-domain="${group.id}"><div class="domain-heading"><button type="button" class="domain-label" data-task-group="${group.id}" aria-expanded="false" aria-controls="tasks-${group.id}" data-active="false"><span class="domain-symbol domain-symbol-${group.id}" aria-hidden="true"></span><span>${group.label}</span></button><button type="button" class="domain-toggle" data-toggle-group="${group.id}" aria-expanded="false" aria-controls="tasks-${group.id}" aria-label="${group.label}子类"></button></div><div class="task-children" id="tasks-${group.id}" hidden>${group.tasks.map(([id, label]) => `<button type="button" class="task-option" data-task-id="${id}" aria-pressed="false"><span>${label}</span><span class="selection-mark" aria-hidden="true"></span></button>`).join('')}</div></div>`).join('')}</div>`;
  const platformFilters = `<fieldset class="sidebar-group"><legend>硬件平台</legend><div id="platform-filters">${platforms.map(platform => {
    const count = groups.filter(group => group.platforms.includes(platform)).length;
    return `<button type="button" class="facet-option platform-filter" data-platform="${platform}" aria-pressed="false"><span class="selection-mark" aria-hidden="true"></span><span>RDK ${platform}</span><small>${count || '尚未收录'}</small></button>`;
  }).join('')}</div></fieldset>`;
  const catalog = `<section class="page-heading"><h1>RDK Model Zoo</h1></section><div class="catalog-layout"><details id="catalog-sidebar" class="catalog-sidebar" open><summary>筛选模型 <span id="filter-count"></span></summary><div class="sidebar-content"><div class="sidebar-top"><span>筛选条件</span><button id="reset" type="button">清除筛选</button></div>${search}<section class="facet-group task-section"><h2>任务类型</h2>${taskFilters}</section><section class="facet-group hardware-section"><h2>硬件平台</h2>${platformFilters}</section></div></details><section class="collection" aria-label="模型列表"><div class="collection-count"><span id="result-count" aria-live="polite"></span></div><div id="grid" class="model-grid"></div><div id="load-more-trigger" class="load-more-trigger" hidden><button id="load-more" type="button">加载更多</button></div><span id="load-status" class="sr-only" role="status" aria-live="polite"></span></section></div>`;

  $('app').innerHTML = `${header}<main id="main" tabindex="-1"><div id="catalog">${catalog}</div><section id="detail" hidden></section></main>${footer}`;

  function card(group) {
    const model = group.entry;
    const description = window.HubI18n?.locale === 'en' ? (model.descriptionEn || model.description) : model.description;
    const openModel = group.members.find(member => modelPlatforms(member).some(platform => selectedPlatforms.has(platform))) || model;
    return `<article class="model-card"><a class="gallery-link" href="#model/${esc(openModel.id)}" aria-label="查看 ${esc(model.name)}"><div class="thumbnail" data-image="${esc(model.id)}"><img src="${esc(model.coverImage)}" alt="${esc(model.name)}" loading="lazy"></div><div class="reference-card-body"><h3>${esc(model.name)}</h3><p class="model-variant">${esc(model.variantName || '')}</p><p class="reference-description" data-i18n-zh="${esc(model.description)}" data-i18n-en="${esc(model.descriptionEn || model.description)}" title="${esc(description)}">${esc(description)}</p><div class="model-tags"><span>${esc(model.task)}</span></div></div></a></article>`;
  }

  function matchesTask(model) {
    if (!state.domain) return true;
    const ids = facets.taskIds(model);
    if (state.useCase) return ids.includes(state.useCase);
    const group = facets.groups.find(candidate => candidate.id === state.domain);
    return Boolean(group?.tasks.some(([id]) => ids.includes(id)));
  }

  function groupMatchesTask(group) {
    return group.members.some(matchesTask);
  }

  function syncTaskSelection() {
    document.querySelectorAll('#filters [data-task-id]').forEach(button => {
      const active = button.closest('[data-domain]').dataset.domain === state.domain && button.dataset.taskId === state.useCase;
      button.setAttribute('aria-pressed', String(active));
    });
    for (const group of facets.groups) {
      const button = document.querySelector(`[data-task-group="${group.id}"]`);
      const expanded = group.id === state.domain;
      button.dataset.active = String(expanded && !state.useCase);
      button.setAttribute('aria-pressed', String(expanded && !state.useCase));
      button.setAttribute('aria-expanded', String(expanded));
      document.querySelector(`[data-toggle-group="${group.id}"]`).setAttribute('aria-expanded', String(expanded));
      $('tasks-' + group.id).hidden = !expanded;
    }
    document.querySelectorAll('[data-platform]').forEach(button => {
      button.setAttribute('aria-pressed', String(selectedPlatforms.has(button.dataset.platform)));
    });
  }

  function syncLoadMore() {
    const remaining = state.visibleCount < filteredItems.length;
    $('load-more-trigger').hidden = !remaining;
    $('load-status').textContent = `已显示 ${Math.min(state.visibleCount, filteredItems.length)} / ${filteredItems.length} 个模型`;
    listObserver?.disconnect();
    if (remaining && !$('catalog').hidden) listObserver?.observe($('load-more-trigger'));
  }

  function render() {
    state.visibleCount = batchSize;
    listObserver?.disconnect();
    const platformItems = groups.filter(group => !selectedPlatforms.size || group.platforms.some(platform => selectedPlatforms.has(platform)));
    const query = state.query.toLowerCase().trim();
    const items = platformItems.filter(group => {
      const searchable = [group.name, ...group.members.map(member => member.variantName), group.task, group.description, ...group.members.flatMap(member => facets.searchTerms(member)), ...group.platforms].join(' ').toLowerCase();
      return groupMatchesTask(group) && searchable.includes(query);
    });
    filteredItems = items;
    $('result-count').textContent = `共 ${items.length} 个模型`;

    if (!models.length) {
      $('grid').innerHTML = '<div class="empty"><p>当前目录暂未发布任何模型。</p></div>';
    } else if (!items.length) {
      $('grid').innerHTML = '<div class="empty"><h3>没有匹配的模型</h3><p>试试其他关键词，或清除筛选条件。</p><button class="button" id="empty-reset">清除筛选</button></div>';
      $('empty-reset').addEventListener('click', reset);
    } else {
      $('grid').innerHTML = items.slice(0, state.visibleCount).map(card).join('');
    }

    const filterCount = Number(Boolean(state.domain)) + selectedPlatforms.size + Number(Boolean(query));
    $('filter-count').textContent = filterCount ? `已选 ${filterCount} 项` : '';
    $('reset').disabled = filterCount === 0;
    syncTaskSelection();
    syncLoadMore();
    window.HubI18n?.apply();
  }

  function loadNextBatch() {
    if ($('catalog').hidden || state.visibleCount >= filteredItems.length) return;
    const next = filteredItems.slice(state.visibleCount, state.visibleCount + batchSize);
    state.visibleCount += next.length;
    $('grid').insertAdjacentHTML('beforeend', next.map(card).join(''));
    syncLoadMore();
    window.HubI18n?.apply();
  }

  function reset() {
    selectedPlatforms.clear();
    state.query = '';
    state.domain = null;
    state.useCase = null;
    $('search').value = '';
    render();
  }

  $('search').addEventListener('input', event => {
    state.query = event.target.value;
    render();
  });
  $('reset').addEventListener('click', reset);
  document.querySelector('.language-switch').addEventListener('click', event => {
    const button = event.target.closest('[data-language]');
    if (button) window.HubI18n.setLocale(button.dataset.language);
  });
  $('filters').addEventListener('click', event => {
    const groupControl = event.target.closest('[data-toggle-group], [data-task-group]');
    if (groupControl) {
      const groupId = groupControl.dataset.toggleGroup || groupControl.dataset.taskGroup;
      const alreadySelected = state.domain === groupId && state.useCase === null;
      state.domain = alreadySelected ? null : groupId;
      state.useCase = null;
      render();
      return;
    }
    const taskButton = event.target.closest('[data-task-id]');
    if (!taskButton) return;
    const groupId = taskButton.closest('[data-domain]').dataset.domain;
    const alreadySelected = state.domain === groupId && state.useCase === taskButton.dataset.taskId;
    state.domain = alreadySelected ? null : groupId;
    state.useCase = alreadySelected ? null : taskButton.dataset.taskId;
    render();
  });
  $('platform-filters').addEventListener('click', event => {
    const button = event.target.closest('[data-platform]');
    if (!button) return;
    const platform = button.dataset.platform;
    selectedPlatforms.has(platform) ? selectedPlatforms.delete(platform) : selectedPlatforms.add(platform);
    render();
  });
  $('load-more').addEventListener('click', loadNextBatch);

  if ('IntersectionObserver' in window) {
    listObserver = new window.IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) loadNextBatch();
    }, { rootMargin: '240px 0px', threshold: 0 });
  }

  const mobile = window.matchMedia('(max-width: 760px)');
  $('catalog-sidebar').open = !mobile.matches;
  mobile.addEventListener('change', event => { $('catalog-sidebar').open = !event.matches; });

  function route() {
    const match = location.hash.match(/^#model\/([^/]+)/);
    const model = match && models.find(candidate => candidate.id === decodeURIComponent(match[1]));
    if (!model) {
      window.ModelDetail.destroy();
      $('detail').hidden = true;
      $('catalog').hidden = false;
      return;
    }
    state.scroll = window.scrollY;
    $('catalog').hidden = true;
    $('detail').hidden = false;
    $('detail').innerHTML = window.ModelDetail.render({ model, models, data });
    window.ModelDetail.bind($('detail'), model, models, data);
    window.scrollTo(0, 0);
    window.HubI18n?.apply();
  }

  window.addEventListener('hashchange', route);
  render();
  route();
})();
