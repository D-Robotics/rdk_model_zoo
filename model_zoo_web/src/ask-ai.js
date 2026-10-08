(() => {
  'use strict';

  const models = Array.isArray(window.MODEL_DATA?.models) ? window.MODEL_DATA.models : [];
  const icon = name => `<span class="ask-icon" style="--ask-icon:url('assets/icons/${name}.svg')" aria-hidden="true"></span>`;
  const esc = value => String(value ?? '').replace(/[&<>"']/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[character]));
  const t = (zh, en) => window.HubI18n?.locale === 'en' ? en : zh;
  const app = document.getElementById('app');
  const scrim = document.createElement('div');
  scrim.className = 'ask-scrim';
  scrim.hidden = true;

  const panel = document.createElement('aside');
  panel.id = 'ask-ai-panel';
  panel.className = 'ask-panel';
  panel.setAttribute('aria-label', 'Ask AI');
  panel.hidden = true;
  panel.innerHTML = `
    <header class="ask-panel-head">
      <div class="ask-panel-title"><span class="ask-panel-mark">${icon('sparkles')}</span><strong>Ask AI</strong></div>
      <button type="button" class="ask-icon-button" data-ask-close aria-label="关闭 Ask AI" title="关闭 Ask AI">${icon('x')}</button>
    </header>
    <div class="ask-scroll" id="ask-scroll" data-no-i18n><div id="ask-intro"></div><div class="ask-transcript" id="ask-transcript" aria-live="polite"></div></div>
    <form class="ask-composer" id="ask-composer"><label class="sr-only" for="ask-question">询问模型</label><textarea id="ask-question" rows="1" placeholder="向 Ask AI 发消息"></textarea><button class="ask-send" type="submit" aria-label="发送" title="发送" disabled>${icon('arrow-up')}</button></form>`;

  const request = document.createElement('dialog');
  request.className = 'ask-request-dialog';
  request.id = 'ask-request-dialog';
  request.setAttribute('aria-labelledby', 'ask-request-title');
  request.innerHTML = `
    <form class="ask-request-shell" id="ask-request-form">
      <header class="ask-request-head"><h2 id="ask-request-title">提交模型需求</h2><button type="button" class="ask-icon-button" data-request-close aria-label="关闭" title="关闭">${icon('x')}</button></header>
      <div class="ask-request-body">
        <div id="ask-request-edit">
          <div class="ask-request-switch" role="group" aria-label="需求类型"><label><input type="radio" name="request-kind" value="public" checked><span>申请收录公开模型</span></label><label><input type="radio" name="request-kind" value="private"><span>编译我的模型</span></label></div>
          <fieldset class="ask-request-fields" data-request-kind="public">
            <label>模型名称<input name="public-name" required maxlength="100" placeholder="例如：目标检测模型"></label>
            <label>目标板卡<select name="public-platform" required><option value="" selected disabled>选择板卡</option><option>RDK X5</option><option>RDK S100</option><option>RDK S100P</option><option>RDK S600</option></select></label>
            <label class="wide">官方来源 URL<input name="public-source" type="url" required placeholder="https://..."></label>
            <label>模型任务<select name="public-task" required><option value="" selected disabled>选择任务</option><option>目标检测</option><option>图像分类</option><option>实例分割</option><option>姿态估计</option><option>其他</option></select></label>
            <label>联系邮箱<input name="public-email" type="email" required placeholder="name@example.com"></label>
            <label class="wide">使用场景<textarea name="public-use-case" required maxlength="1000" placeholder="希望在目标板卡上完成什么任务？"></textarea></label>
          </fieldset>
          <fieldset class="ask-request-fields" data-request-kind="private" disabled hidden>
            <label>模型名称<input name="private-name" required maxlength="100" placeholder="模型名称"></label>
            <label>目标板卡<select name="private-platform" required><option value="" selected disabled>选择板卡</option><option>RDK X5</option><option>RDK S100</option><option>RDK S100P</option><option>RDK S600</option></select></label>
            <label class="wide">模型文件<span class="ask-file-control"><span class="ask-file-button">选择文件</span><span class="ask-file-name" id="ask-file-name" data-no-i18n>未选择文件</span><input name="private-file" type="file" accept=".onnx,.pt" required></span></label>
            <label>输入形状<input name="private-shape" required placeholder="例如：1 × 3 × 640 × 640"></label>
            <label>联系邮箱<input name="private-email" type="email" required placeholder="name@example.com"></label>
            <label class="wide">校准与评测资料<textarea name="private-data" required maxlength="1000" placeholder="数据来源、前后处理、参考指标"></textarea></label>
          </fieldset>
        </div>
        <div class="ask-preview-result" id="ask-request-preview" data-no-i18n hidden></div>
      </div>
      <footer class="ask-request-footer"><div><button type="button" class="ask-request-cancel" data-request-cancel>取消</button><button type="submit" class="ask-request-submit" id="ask-request-submit">查看预览</button></div></footer>
    </form>`;

  document.body.append(scrim, panel, request);

  const intro = panel.querySelector('#ask-intro');
  const transcript = panel.querySelector('#ask-transcript');
  const scroll = panel.querySelector('#ask-scroll');
  const question = panel.querySelector('#ask-question');
  const composer = panel.querySelector('#ask-composer');
  const sendButton = panel.querySelector('.ask-send');
  const requestForm = request.querySelector('#ask-request-form');
  const requestEdit = request.querySelector('#ask-request-edit');
  const requestPreview = request.querySelector('#ask-request-preview');
  const requestSubmit = request.querySelector('#ask-request-submit');
  const requestCancel = request.querySelector('[data-request-cancel]');
  const overlay = window.matchMedia('(max-width: 1439px)');
  let scopedModel = null;
  let chatSession = null;
  let chatBusy = false;
  let previousFocus = null;

  const routeModel = () => {
    const match = location.hash.match(/^#model\/([^/]+)/);
    if (!match) return null;
    try { return models.find(model => model.id === decodeURIComponent(match[1])) || null; }
    catch { return null; }
  };

  function syncContext() {
    if (!transcript.children.length) renderIntro();
  }

  function renderIntro() {
    const model = scopedModel;
    const platforms = [...new Set(models.map(item => item.releasePlatform).filter(Boolean))];
    const featured = platforms.includes('S600') ? 'S600' : platforms[0];
    const example = models.find(item => item.releasePlatform === featured) || models[0];
    const prompts = model ? [
      t('这个模型的精度是怎么测的？', 'How was this model evaluated?'),
      t(`它在 RDK ${model.releasePlatform} 上的延迟是多少？`, `What is its latency on RDK ${model.releasePlatform}?`),
      t('怎么下载并运行这个模型？', 'How do I download and run this model?'),
    ] : featured ? [
      t(`RDK ${featured} 上有哪些目标检测模型？`, `Which detection models run on RDK ${featured}?`),
      t(`${example.variantName || example.name} 在 RDK ${featured} 上的精度和速度怎么样？`, `How accurate and fast is ${example.variantName || example.name} on RDK ${featured}?`),
      t('我想提交一个新模型', 'I want to request a new model'),
    ] : [t('我想提交一个新模型', 'I want to request a new model')];
    intro.hidden = false;
    scroll.classList.add('is-empty');
    intro.innerHTML = `<div class="ask-intro"><h2>${model ? t('想了解这个模型的什么？', 'What would you like to know about this model?') : t('你好，有什么想了解的？', 'What would you like to know?')}</h2>${model ? `<p>${esc(model.variantName || model.name)} · RDK ${esc(model.releasePlatform)}</p>` : ''}</div><div class="ask-prompts">${prompts.map(prompt => `<button class="ask-prompt" type="button" data-ask-prompt="${esc(prompt)}">${esc(prompt)}</button>`).join('')}</div>`;
  }

  function updateOverlay() {
    const open = !panel.hidden;
    scrim.hidden = !open || !overlay.matches;
    app.inert = open && overlay.matches;
  }

  // The panel slides out before it hides; reopening mid-slide cancels it.
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let closing = null;
  function cancelClosing() {
    if (!closing) return;
    clearTimeout(closing.timer);
    panel.removeEventListener('transitionend', closing.finish);
    panel.classList.remove('is-closing');
    scrim.classList.remove('is-closing');
    closing = null;
  }

  function openPanel(modelScope = false) {
    cancelClosing();
    previousFocus = document.activeElement;
    const requestedModel = modelScope ? routeModel() : null;
    if (requestedModel && transcript.children.length && scopedModel?.id !== requestedModel.id && !chatBusy) {
      transcript.replaceChildren();
      chatSession = null;
    }
    if (!transcript.children.length) scopedModel = requestedModel;
    panel.hidden = false;
    document.body.classList.add('ask-ai-open');
    document.querySelectorAll('[data-ask-open]').forEach(button => button.setAttribute('aria-expanded', 'true'));
    syncContext();
    updateOverlay();
    question.focus();
  }

  function closePanel() {
    if (panel.hidden || closing) return;
    document.body.classList.remove('ask-ai-open');
    document.querySelectorAll('[data-ask-open]').forEach(button => button.setAttribute('aria-expanded', 'false'));
    app.inert = false;
    const finish = event => {
      if (event && event.target !== panel) return;
      cancelClosing();
      panel.hidden = true;
      updateOverlay();
    };
    if (reducedMotion.matches) finish();
    else {
      closing = { finish, timer: setTimeout(finish, 520) };
      panel.addEventListener('transitionend', finish);
      panel.classList.add('is-closing');
      scrim.classList.add('is-closing');
    }
    if (previousFocus?.isConnected) previousFocus.focus();
  }

  const serviceUrl = (() => {
    const configured = typeof window.MODEL_ZOO_SERVICE_URL === 'string'
      ? window.MODEL_ZOO_SERVICE_URL.trim() : '';
    const raw = configured || '/api';
    if (!raw) return '';
    try {
      const url = new URL(raw, location.href);
      if (url.protocol !== 'https:' && url.origin !== location.origin
          && !(url.protocol === 'http:' && ['localhost', '127.0.0.1'].includes(url.hostname))) return '';
      return url.href.replace(/\/+$/, '');
    } catch { return ''; }
  })();

  function safeLink(value) {
    try {
      const url = new URL(value);
      if (url.protocol === 'https:' || url.origin === location.origin
          || (url.protocol === 'http:' && ['localhost', '127.0.0.1'].includes(url.hostname))) return url.href;
    } catch { /* Invalid links are rendered as text. */ }
    return null;
  }

  function renderInline(element, value) {
    const text = String(value || '');
    const token = /(\[[^\]]+\]\(https?:\/\/[^)\s]+\)|https?:\/\/[^\s<>()]+|\*\*[^*]+\*\*|\x60[^\x60]+\x60)/g;
    let cursor = 0;
    for (const match of text.matchAll(token)) {
      element.append(document.createTextNode(text.slice(cursor, match.index)));
      const part = match[0];
      const link = part.match(/^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)$/);
      if (link && safeLink(link[2])) {
        const anchor = document.createElement('a');
        anchor.href = safeLink(link[2]);
        anchor.target = '_blank';
        anchor.rel = 'noopener noreferrer';
        anchor.textContent = link[1];
        element.append(anchor);
      } else if (part.startsWith('**')) {
        const strong = document.createElement('strong');
        strong.textContent = part.slice(2, -2);
        element.append(strong);
      } else if (part.charCodeAt(0) === 96) {
        const code = document.createElement('code');
        code.textContent = part.slice(1, -1);
        element.append(code);
      } else if (safeLink(part)) {
        const anchor = document.createElement('a');
        anchor.href = safeLink(part);
        anchor.target = '_blank';
        anchor.rel = 'noopener noreferrer';
        anchor.textContent = part;
        element.append(anchor);
      } else {
        element.append(document.createTextNode(part));
      }
      cursor = match.index + part.length;
    }
    element.append(document.createTextNode(text.slice(cursor)));
  }

  function renderAnswer(assistant, value, sources, prompt) {
    assistant.classList.remove('is-pending', 'is-error');
    assistant.replaceChildren();
    let list = null;
    const lines = String(value || '').trim().split(/\r?\n/);
    const isTableRow = line => /^\|.*\|$/.test(line.trim());
    const isTableRule = line => /^\|(?:\s*:?-{3,}:?\s*\|)+$/.test(line.trim());
    const cells = line => line.trim().slice(1, -1).split('|').map(cell => cell.trim());
    for (let index = 0; index < lines.length; index++) {
      const line = lines[index].trim();
      if (!line) { list = null; continue; }
      if (isTableRow(line) && isTableRule(lines[index + 1] || '')) {
        list = null;
        const wrap = document.createElement('div');
        wrap.className = 'ask-table-wrap';
        const table = document.createElement('table');
        const head = document.createElement('thead');
        const headingRow = document.createElement('tr');
        for (const value of cells(line)) {
          const cell = document.createElement('th');
          renderInline(cell, value);
          headingRow.append(cell);
        }
        head.append(headingRow);
        table.append(head);
        const body = document.createElement('tbody');
        index += 2;
        while (index < lines.length && isTableRow(lines[index])) {
          const row = document.createElement('tr');
          for (const value of cells(lines[index])) {
            const cell = document.createElement('td');
            renderInline(cell, value);
            row.append(cell);
          }
          body.append(row);
          index++;
        }
        table.append(body);
        wrap.append(table);
        assistant.append(wrap);
        index--;
        continue;
      }
      const heading = line.match(/^#{1,3}\s+(.+)$/);
      const bullet = line.match(/^[-*]\s+(.+)$/);
      const numbered = line.match(/^\d+\.\s+(.+)$/);
      if (bullet || numbered) {
        const tag = numbered ? 'OL' : 'UL';
        if (!list || list.tagName !== tag) {
          list = document.createElement(tag.toLowerCase());
          assistant.append(list);
        }
        const item = document.createElement('li');
        renderInline(item, (bullet || numbered)[1]);
        list.append(item);
      } else {
        list = null;
        const block = document.createElement(heading ? 'h3' : 'p');
        renderInline(block, heading ? heading[1] : line);
        assistant.append(block);
      }
    }
    const seen = new Set();
    const validSources = (sources || []).filter(source => {
      const url = safeLink(source.url);
      if (!url || seen.has(url)) return false;
      seen.add(url);
      return true;
    });
    if (validSources.length) {
      const many = validSources.length > 5;
      const citations = document.createElement(many ? 'details' : 'div');
      citations.className = many ? 'ask-citations is-collapsible' : 'ask-citations';
      const label = document.createElement(many ? 'summary' : 'span');
      label.textContent = many
        ? t(`查看 ${validSources.length} 个来源`, `View ${validSources.length} sources`)
        : t('来源', 'Sources');
      citations.append(label);
      const links = many ? document.createElement('div') : citations;
      if (many) links.className = 'ask-citation-links';
      for (const source of validSources) {
        const anchor = document.createElement('a');
        anchor.href = safeLink(source.url);
        anchor.target = '_blank';
        anchor.rel = 'noopener noreferrer';
        anchor.textContent = source.title || t('查看资料', 'View source');
        links.append(anchor);
      }
      if (many) citations.append(links);
      assistant.append(citations);
    }
    if (/(申请|提交|收录|request|submit)/i.test(prompt) && /(模型|model)/i.test(prompt)) {
      const actions = document.createElement('div');
      actions.className = 'ask-message-actions';
      for (const [kind, zh, en] of [
        ['public', '申请收录公开模型', 'Request a public model'],
        ['private', '编译我的模型', 'Compile my model'],
      ]) {
        const button = document.createElement('button');
        button.type = 'button';
        button.dataset.askRequest = '';
        button.dataset.askRequestKind = kind;
        button.textContent = t(zh, en);
        actions.append(button);
      }
      assistant.append(actions);
    }
    scroll.scrollTop = scroll.scrollHeight;
  }

  function requestError(code) {
    if (code === 'service_unavailable') return t('Ask AI 服务尚未配置，请联系站点管理员。', 'Ask AI is not configured for this site.');
    if (code === 'network_error') return t('无法连接 Ask AI 服务。请检查后端是否运行及网页的服务地址。', 'Cannot connect to Ask AI. Check that the backend is running and the site uses the correct service address.');
    if (code === 'backend_unavailable') return t('Ask AI 后端没有响应，请确认服务正在运行。', 'The Ask AI backend is not responding. Check that the service is running.');
    if (code === 'stream_incomplete') return t('回复传输中断，请重试。', 'The reply was interrupted. Please try again.');
    if (code === 'agent_unconfigured') return t('AI 服务尚未配置模型或密钥。', 'The AI model or key is not configured.');
    if (code === 'agent_timeout') return t('模型回复超时，请重试。', 'The model took too long to respond. Please try again.');
    if (code === 'agent_error') return t('模型服务这次未能完成回复，请重试。', 'The model service could not complete this reply. Please try again.');
    if (code === 'rate_limited') return t('请求过于频繁，请稍后重试。', 'Too many requests. Please try again shortly.');
    if (code === 'origin_denied' || code === 'unauthorized') return t('网页与 Ask AI 服务的连接配置不匹配。', 'The site is not authorized to connect to Ask AI.');
    if (code === 'internal_error' || code === 'http_error') return t('Ask AI 后端出现错误，请稍后重试。', 'The Ask AI backend returned an error. Please try again later.');
    if (code === 'model_scope_unavailable' || code === 'model_not_found' || code === 'release_not_found') {
      return t('当前模型与服务目录不一致，请从顶部 Ask AI 入口重新提问。', 'This model is not in the service catalog. Please use the main Ask AI entry.');
    }
    if (code === 'catalog_changed' || code === 'session_not_found') {
      return t('目录或会话已更新，请重试以开始新会话。', 'The catalog or session changed. Retry to start a new conversation.');
    }
    if (code === 'AbortError') return t('请求已取消。', 'The request was cancelled.');
    return t('暂时无法获取回复，请重试。', 'Could not get a response. Please try again.');
  }

  async function readApiResponse(response) {
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const error = new Error(requestError(payload?.error?.code));
      error.code = payload?.error?.code || 'http_error';
      throw error;
    }
    return payload;
  }

  async function apiFetch(url, options) {
    try {
      return await fetch(url, options);
    } catch (cause) {
      if (cause?.name === 'AbortError') throw cause;
      const error = new Error(requestError('network_error'), { cause });
      error.code = 'network_error';
      throw error;
    }
  }

  function scopePayload() {
    if (!scopedModel) return { type: 'catalog' };
    if (!scopedModel.catalogId || !scopedModel.modelSize || !scopedModel.releasePlatform) {
      const error = new Error(requestError('model_scope_unavailable'));
      error.code = 'model_scope_unavailable';
      throw error;
    }
    return {
      type: 'model',
      model_id: scopedModel.catalogId,
      size: scopedModel.modelSize,
      platform: scopedModel.releasePlatform.toLowerCase(),
    };
  }

  async function ensureSession(signal) {
    if (chatSession) return chatSession;
    if (!serviceUrl) {
      const error = new Error(requestError('service_unavailable'));
      error.code = 'service_unavailable';
      throw error;
    }
    const response = await apiFetch(serviceUrl + '/v1/chat/sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        locale: window.HubI18n?.locale === 'en' ? 'en' : 'zh-CN',
        scope: scopePayload(),
      }),
      signal,
    });
    const payload = await readApiResponse(response);
    if (!payload?.session_id || !payload?.session_token) throw new Error(requestError());
    chatSession = { id: payload.session_id, token: payload.session_token };
    return chatSession;
  }

  async function streamReply(session, prompt, assistant, signal) {
    const response = await apiFetch(serviceUrl + '/v1/chat/sessions/' + encodeURIComponent(session.id) + '/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Session-Token': session.token },
      body: JSON.stringify({ message: prompt }),
      signal,
    });
    if (!response.ok) await readApiResponse(response);
    if (!response.body) throw new Error(requestError());
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const citations = [];
    let buffer = '';
    let partial = '';
    let complete = null;
    const handle = block => {
      const event = block.match(/^event: ([^\n]+)$/m)?.[1];
      const raw = block.match(/^data: (.+)$/m)?.[1];
      if (!event || !raw) return;
      const data = JSON.parse(raw);
      if (event === 'message.delta') {
        partial += String(data.delta || '');
        assistant.classList.remove('is-pending');
        assistant.textContent = partial;
        scroll.scrollTop = scroll.scrollHeight;
      } else if (event === 'citation') {
        citations.push(data);
      } else if (event === 'message.done') {
        complete = String(data.text || '');
      } else if (event === 'message.error') {
        const error = new Error(requestError(data.code));
        error.code = data.code || 'agent_error';
        throw error;
      }
    };
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        buffer = buffer.replace(/\r\n/g, '\n');
        let boundary;
        while ((boundary = buffer.indexOf('\n\n')) !== -1) {
          handle(buffer.slice(0, boundary));
          buffer = buffer.slice(boundary + 2);
        }
      }
      buffer += decoder.decode();
      if (buffer.trim()) handle(buffer.trim());
    } catch (cause) {
      if (cause instanceof TypeError) {
        const error = new Error(requestError('network_error'), { cause });
        error.code = 'network_error';
        throw error;
      }
      throw cause;
    } finally {
      reader.releaseLock();
    }
    if (complete === null || !complete.trim()) {
      const error = new Error(requestError('stream_incomplete'));
      error.code = 'stream_incomplete';
      throw error;
    }
    return { text: complete, citations };
  }

  async function send(value) {
    const prompt = String(value || '').trim();
    if (!prompt || chatBusy) return;
    intro.hidden = true;
    scroll.classList.remove('is-empty');
    const user = document.createElement('div');
    user.className = 'ask-message ask-message-user';
    user.textContent = prompt;
    const assistant = document.createElement('div');
    assistant.className = 'ask-message ask-message-assistant is-pending';
    assistant.textContent = t('正在思考…', 'Thinking…');
    transcript.append(user, assistant);
    question.value = '';
    question.style.height = '';
    question.placeholder = t('继续提问', 'Ask a follow-up');
    chatBusy = true;
    question.disabled = true;
    sendButton.disabled = true;
    scroll.scrollTop = scroll.scrollHeight;
    const controller = new AbortController();
    try {
      const session = await ensureSession(controller.signal);
      const result = await streamReply(session, prompt, assistant, controller.signal);
      renderAnswer(assistant, result.text, result.citations, prompt);
    } catch (error) {
      if (['catalog_changed', 'session_not_found'].includes(error.code)) chatSession = null;
      assistant.classList.remove('is-pending');
      assistant.classList.add('is-error');
      assistant.replaceChildren();
      const note = document.createElement('p');
      note.textContent = requestError(error.code);
      const retry = document.createElement('button');
      retry.type = 'button';
      retry.textContent = t('重试', 'Retry');
      retry.addEventListener('click', () => {
        if (chatBusy) return;
        user.remove();
        assistant.remove();
        void send(prompt);
      });
      assistant.append(note, retry);
    } finally {
      chatBusy = false;
      question.disabled = false;
      sendButton.disabled = !question.value.trim();
      scroll.scrollTop = scroll.scrollHeight;
      if (!panel.hidden) question.focus();
    }
  }

  function syncRequestKind() {
    const selected = requestForm.elements['request-kind'].value;
    request.querySelectorAll('[data-request-kind]').forEach(fields => {
      const active = fields.dataset.requestKind === selected;
      fields.hidden = !active;
      fields.disabled = !active;
    });
    request.querySelector('#ask-file-name').textContent = requestForm.elements['private-file'].files?.[0]?.name || t('未选择文件', 'No file selected');
  }

  function openRequest(kind = 'public') {
    requestForm.reset();
    requestForm.querySelector(`[name="request-kind"][value="${kind}"]`).checked = true;
    requestEdit.hidden = false;
    requestPreview.hidden = true;
    requestSubmit.type = 'submit';
    requestSubmit.textContent = t('查看预览', 'Preview request');
    requestCancel.textContent = t('取消', 'Cancel');
    syncRequestKind();
    const searchValue = document.getElementById('search')?.value.trim();
    if (searchValue && kind === 'public') requestForm.elements['public-name'].value = searchValue;
    request.showModal();
    request.querySelector(kind === 'public' ? '[name="public-name"]' : '[name="private-name"]').focus();
  }

  function showRequestPreview() {
    const kind = requestForm.elements['request-kind'].value;
    const name = requestForm.elements[kind === 'private' ? 'private-name' : 'public-name'].value.trim();
    const platform = requestForm.elements[kind === 'private' ? 'private-platform' : 'public-platform'].value;
    const source = kind === 'private'
      ? requestForm.elements['private-file'].files?.[0]?.name || ''
      : requestForm.elements['public-source'].value.trim();
    requestEdit.hidden = true;
    requestPreview.hidden = false;
    requestPreview.innerHTML = `${icon('check')}<h3>${t('申请预览已生成', 'Request preview ready')}</h3><p>${t('这只是前端交互预览，模型文件和表单数据均未发送到服务器。', 'This is a front-end preview. No model file or form data has been sent to a server.')}</p><dl class="ask-preview-summary"><div><dt>${t('需求类型', 'Request type')}</dt><dd>${kind === 'private' ? t('编译我的模型', 'Compile my model') : t('申请收录公开模型', 'Request public model')}</dd></div><div><dt>${t('模型名称', 'Model name')}</dt><dd>${esc(name)}</dd></div><div><dt>${t('目标板卡', 'Target board')}</dt><dd>${esc(platform)}</dd></div><div><dt>${kind === 'private' ? t('本地文件', 'Local file') : t('官方来源', 'Official source')}</dt><dd>${esc(source)}</dd></div></dl><div class="ask-preview-stages"><span>${t('提交', 'Submit')}</span><span>${t('预检', 'Preflight')}</span><span>${t('编译', 'Compile')}</span><span>${t('板端验证', 'Board validation')}</span><span>${kind === 'private' ? t('候选制品', 'Candidate artifact') : t('发布审核', 'Release review')}</span></div>`;
    requestSubmit.type = 'button';
    requestSubmit.textContent = t('完成', 'Done');
    requestCancel.textContent = t('返回修改', 'Edit request');
    requestSubmit.focus();
  }

  function closeRequest() { request.close(); }

  document.addEventListener('click', event => {
    const open = event.target.closest('[data-ask-open], [data-ask-model]');
    if (open) {
      openPanel(Boolean(open.matches('[data-ask-model]')));
      return;
    }
    const requestButton = event.target.closest('[data-ask-request]');
    if (requestButton) {
      openRequest(requestButton.dataset.askRequestKind || 'public');
      return;
    }
    if (event.target.closest('[data-ask-close]')) closePanel();
    const prompt = event.target.closest('[data-ask-prompt]');
    if (prompt) send(prompt.dataset.askPrompt);
  });
  scrim.addEventListener('click', closePanel);
  composer.addEventListener('submit', event => { event.preventDefault(); send(question.value); });
  question.addEventListener('input', () => {
    sendButton.disabled = !question.value.trim();
    question.style.height = 'auto';
    question.style.height = `${Math.min(question.scrollHeight, 144)}px`;
  });
  question.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      send(question.value);
    }
  });
  requestForm.addEventListener('change', event => {
    if (event.target.name === 'request-kind') syncRequestKind();
    if (event.target.name === 'private-file') request.querySelector('#ask-file-name').textContent = event.target.files?.[0]?.name || t('未选择文件', 'No file selected');
  });
  requestForm.addEventListener('submit', event => {
    event.preventDefault();
    showRequestPreview();
  });
  request.querySelectorAll('[data-request-close]').forEach(button => button.addEventListener('click', closeRequest));
  requestCancel.addEventListener('click', () => {
    if (requestPreview.hidden) closeRequest();
    else {
      requestEdit.hidden = false;
      requestPreview.hidden = true;
      requestSubmit.type = 'submit';
      requestSubmit.textContent = t('查看预览', 'Preview request');
      requestCancel.textContent = t('取消', 'Cancel');
    }
  });
  requestSubmit.addEventListener('click', () => {
    if (requestSubmit.type === 'button') closeRequest();
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !request.open && !panel.hidden) closePanel();
    if (event.key !== 'Tab' || panel.hidden || !overlay.matches || request.open) return;
    const focusable = [...panel.querySelectorAll('button:not(:disabled), select, textarea, a[href]')].filter(element => !element.closest('[hidden]'));
    if (!focusable.length) return;
    if (event.shiftKey && document.activeElement === focusable[0]) { event.preventDefault(); focusable.at(-1).focus(); }
    else if (!event.shiftKey && document.activeElement === focusable.at(-1)) { event.preventDefault(); focusable[0].focus(); }
  });
  window.addEventListener('hashchange', syncContext);
  overlay.addEventListener('change', updateOverlay);
  document.querySelector('.language-switch')?.addEventListener('click', () => {
    queueMicrotask(() => {
      syncContext();
      if (request.open) syncRequestKind();
    });
  });

  syncContext();
  window.HubI18n?.apply();
  if (location.hash === '#ask-ai') openPanel();
})();
