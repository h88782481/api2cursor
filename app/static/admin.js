const API = '';
let authKey = '';
let editingName = null;
let templateKind = 'address';
let templates = { address: {}, instruction: {}, body: {}, header: {}, replacement: {} };
let selectedReplacementNames = [];
let instructionBlocks = { function: [], custom_grammar: [] };
let instructionStatuses = {};

const FORMAT_LABELS = { auto: '自动', chat: 'chat', responses: 'responses', messages: 'messages', gemini: 'gemini' };
const FORMAT_TAG_CLASSES = {
  auto: 'tag-auto', chat: 'tag-chat', responses: 'tag-responses',
  messages: 'tag-messages', gemini: 'tag-gemini',
};

function togglePwd(id) {
  const element = document.getElementById(id);
  element.type = element.type === 'password' ? 'text' : 'password';
}

function toast(message, ok = true) {
  const element = document.createElement('div');
  element.className = `toast ${ok ? 'toast-ok' : 'toast-err'}`;
  element.textContent = message;
  document.getElementById('toasts').appendChild(element);
  setTimeout(() => element.remove(), 3000);
}

async function api(path, options = {}) {
  const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
  if (authKey) headers.Authorization = `Bearer ${authKey}`;
  const response = await fetch(API + path, { ...options, headers });
  const contentType = response.headers.get('content-type') || '';
  if (!contentType.includes('application/json')) {
    const text = await response.text();
    throw new Error(response.ok ? '服务器返回了非 JSON 响应' : `HTTP ${response.status}: ${text.slice(0, 100)}`);
  }
  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message || `HTTP ${response.status}`);
  return data;
}

async function doLogin() {
  const key = document.getElementById('loginKey').value.trim();
  if (!key) return toast('请填写密钥', false);
  try {
    await api('/api/admin/login', { method: 'POST', body: JSON.stringify({ key }) });
    authKey = key;
    sessionStorage.setItem('_ak', key);
    document.getElementById('login').style.display = 'none';
    document.getElementById('dashboard').style.display = 'block';
    await loadDashboard();
  } catch {
    toast('密钥无效', false);
  }
}

function doLogout() {
  authKey = '';
  sessionStorage.removeItem('_ak');
  document.getElementById('dashboard').style.display = 'none';
  document.getElementById('login').style.display = 'flex';
}

async function loadDashboard() {
  try {
    const settings = await api('/api/admin/settings');
    document.getElementById('targetUrl').value = settings.proxy_target_url || '';
    document.getElementById('proxyKey').value = settings.proxy_api_key || '';
    document.getElementById('debugMode').value = settings.debug_mode || 'off';
    document.getElementById('envUrl').textContent = settings.env_target_url ? `环境变量: ${settings.env_target_url}` : '';
    document.getElementById('envKey').textContent = settings.env_api_key ? '环境变量: (已配置)' : '环境变量: (未设置)';
    instructionBlocks = await api('/api/admin/instruction-blocks');
    fillInstructionTargets();
    templates = await api('/api/admin/templates');
    fillTemplateSelections();
    selectTemplateKind('address');
    await loadMappings();
    checkHealth();
    loadStats();
  } catch (error) {
    toast(`加载设置失败: ${error.message}`, false);
  }
}

function fillInstructionTargets() {
  for (const dialect of ['function', 'custom_grammar']) {
    const select = document.getElementById(`template${dialect === 'function' ? 'Fn' : 'Cg'}Target`);
    select.innerHTML = (instructionBlocks[dialect] || []).map(block => (
      `<option value="${esc(block.id)}">${esc(block.label)} — ${esc(block.description)}</option>`
    )).join('');
  }
}

function fillTemplateSelections() {
  const fields = [
    ['address', 'mAddressTemplate'],
    ['instruction', 'mInstructionTemplate'],
    ['body', 'mBodyTemplate'],
    ['header', 'mHeaderTemplate'],
  ];
  for (const [kind, id] of fields) {
    document.getElementById(id).innerHTML =
      `<option value="">不使用模板</option>${Object.keys(templates[kind] || {}).map(name => (
        `<option value="${esc(name)}">${esc(name)}</option>`
      )).join('')}`;
  }
  renderReplacementPicker();
}

function renderReplacementPicker() {
  const menu = document.getElementById('mReplacementMenu');
  const available = Object.keys(templates.replacement || {});
  const availableSet = new Set(available);
  selectedReplacementNames = [...new Set(selectedReplacementNames)]
    .filter(name => availableSet.has(name));
  menu.innerHTML = available.length
    ? available.map(name => `
      <button class="multi-select-option${selectedReplacementNames.includes(name) ? ' selected' : ''}" type="button"
        data-name="${esc(name)}">
        <span>${esc(name)}</span>
        <span class="multi-select-check">${selectedReplacementNames.includes(name) ? '✓' : ''}</span>
      </button>
    `).join('')
    : '<div class="multi-select-empty">暂无文本替换模板</div>';
  document.getElementById('mReplacementValues').innerHTML = selectedReplacementNames.length
    ? selectedReplacementNames.map(name => `
      <span class="multi-select-tag">
        ${esc(name)}
        <button class="multi-select-remove" type="button" data-name="${esc(name)}" aria-label="取消选择">×</button>
      </span>
    `).join('')
    : '<span class="multi-select-placeholder">请选择文本替换模板</span>';
}

function toggleReplacementPicker(event) {
  event?.stopPropagation();
  const picker = document.getElementById('mReplacementPicker');
  const opened = picker.classList.toggle('open');
  document.getElementById('mReplacementControl').setAttribute('aria-expanded', opened);
}

function toggleReplacement(name) {
  const index = selectedReplacementNames.indexOf(name);
  if (index === -1) selectedReplacementNames.push(name);
  else selectedReplacementNames.splice(index, 1);
  renderReplacementPicker();
  document.getElementById('mReplacementPicker').classList.add('open');
}

function removeReplacement(name, event) {
  event.stopPropagation();
  selectedReplacementNames = selectedReplacementNames.filter(item => item !== name);
  renderReplacementPicker();
}

function setSelectedReplacementNames(names) {
  selectedReplacementNames = Array.isArray(names)
    ? names.filter(name => typeof name === 'string')
    : [];
  renderReplacementPicker();
}

function selectTemplateKind(kind) {
  templateKind = kind;
  document.querySelectorAll('[data-template-kind]').forEach(button => {
    button.classList.toggle('active', button.dataset.templateKind === kind);
  });
  document.getElementById('addressTemplateFields').style.display = kind === 'address' ? '' : 'none';
  document.getElementById('instructionTemplateFields').style.display = kind === 'instruction' ? '' : 'none';
  document.getElementById('jsonTemplateFields').style.display = ['body', 'header'].includes(kind) ? '' : 'none';
  document.getElementById('replacementTemplateFields').style.display = kind === 'replacement' ? '' : 'none';
  const list = document.getElementById('templateList');
  list.innerHTML = Object.keys(templates[kind] || {}).map(name => (
    `<button class="template-item" onclick="editTemplate('${esc(name)}')">${esc(name)}</button>`
  )).join('') || '<div class="empty">暂无模板</div>';
  clearTemplateForm();
}

function clearTemplateForm() {
  document.getElementById('templateName').value = '';
  document.getElementById('templateUrl').value = '';
  document.getElementById('templateKey').value = '';
  document.getElementById('templateJson').value = '';
  document.getElementById('templateReplacementSystem').checked = false;
  document.getElementById('templateReplacementUser').checked = false;
  document.getElementById('templateReplacementFind').value = '';
  document.getElementById('templateReplacementReplace').value = '';
  for (const dialect of ['Fn', 'Cg']) {
    document.getElementById(`template${dialect}Text`).value = '';
    document.getElementById(`template${dialect}Target`).value = 'all';
    document.getElementById(`template${dialect}Mode`).value = 'prepend';
  }
}

function editTemplate(name) {
  clearTemplateForm();
  document.getElementById('templateName').value = name;
  const value = templates[templateKind][name] || {};
  if (templateKind === 'address') {
    document.getElementById('templateUrl').value = value.base_url || '';
    document.getElementById('templateKey').value = value.api_key || '';
  } else if (templateKind === 'instruction') {
    writeTemplateRule('Fn', value.function);
    writeTemplateRule('Cg', value.custom_grammar);
  } else if (templateKind === 'replacement') {
    document.getElementById('templateReplacementSystem').checked = value.roles?.includes('system') || false;
    document.getElementById('templateReplacementUser').checked = value.roles?.includes('user') || false;
    document.getElementById('templateReplacementFind').value = value.find || '';
    document.getElementById('templateReplacementReplace').value = value.replace || '';
  } else {
    document.getElementById('templateJson').value = JSON.stringify(value, null, 2);
  }
}

function readTemplateRule(prefix) {
  return {
    text: document.getElementById(`template${prefix}Text`).value,
    target: document.getElementById(`template${prefix}Target`).value || 'all',
    mode: document.getElementById(`template${prefix}Mode`).value || 'prepend',
  };
}

function writeTemplateRule(prefix, rule = {}) {
  document.getElementById(`template${prefix}Text`).value = rule.text || '';
  document.getElementById(`template${prefix}Target`).value = rule.target || 'all';
  document.getElementById(`template${prefix}Mode`).value = rule.mode || 'prepend';
}

async function saveTemplate() {
  const name = document.getElementById('templateName').value.trim();
  if (!name) return toast('请填写模板名称', false);
  let payload;
  try {
    if (templateKind === 'address') {
      payload = {
        base_url: document.getElementById('templateUrl').value.trim(),
        api_key: document.getElementById('templateKey').value.trim(),
      };
    } else if (templateKind === 'instruction') {
      payload = { function: readTemplateRule('Fn'), custom_grammar: readTemplateRule('Cg') };
    } else if (templateKind === 'replacement') {
      payload = {
        roles: [
          ...(document.getElementById('templateReplacementSystem').checked ? ['system'] : []),
          ...(document.getElementById('templateReplacementUser').checked ? ['user'] : []),
        ],
        find: document.getElementById('templateReplacementFind').value,
        replace: document.getElementById('templateReplacementReplace').value,
      };
    } else {
      payload = JSON.parse(document.getElementById('templateJson').value || '{}');
    }
  } catch {
    return toast('模板内容不是有效 JSON', false);
  }
  try {
    await api(`/api/admin/templates/${templateKind}/${encodeURIComponent(name)}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    });
    templates = await api('/api/admin/templates');
    fillTemplateSelections();
    selectTemplateKind(templateKind);
    editTemplate(name);
    toast('模板已保存');
  } catch (error) {
    toast(`保存失败: ${error.message}`, false);
  }
}

async function deleteTemplate() {
  const name = document.getElementById('templateName').value.trim();
  if (!name) return toast('请选择模板', false);
  if (!confirm(`确定删除模板「${name}」吗？`)) return;
  try {
    await api(`/api/admin/templates/${templateKind}/${encodeURIComponent(name)}`, { method: 'DELETE' });
    templates = await api('/api/admin/templates');
    fillTemplateSelections();
    selectTemplateKind(templateKind);
    toast('模板已删除');
  } catch (error) {
    toast(`删除失败: ${error.message}`, false);
  }
}

async function saveSettings() {
  try {
    await api('/api/admin/settings', {
      method: 'PUT',
      body: JSON.stringify({
        proxy_target_url: document.getElementById('targetUrl').value.trim(),
        proxy_api_key: document.getElementById('proxyKey').value.trim(),
        debug_mode: document.getElementById('debugMode').value,
      }),
    });
    toast('设置已保存');
  } catch (error) {
    toast(`保存失败: ${error.message}`, false);
  }
}

function formatTag(format, prefix) {
  const value = format || 'auto';
  return `<span class="tag ${FORMAT_TAG_CLASSES[value] || 'tag-auto'}">${esc(prefix + (FORMAT_LABELS[value] || value))}</span>`;
}

function instructionStatusTag(name, hasInstruction) {
  if (!hasInstruction) return '';
  const statuses = Object.values(instructionStatuses[name] || {});
  if (statuses.some(status => status.state === 'failed')) return '<span class="tag tag-warning">注入失败</span>';
  if (statuses.some(status => status.state === 'applied')) return '<span class="tag tag-ok">注入正常</span>';
  return '<span class="tag tag-pending">尚无请求验证</span>';
}

async function loadMappings() {
  const [mappings, statuses] = await Promise.all([
    api('/api/admin/mappings'),
    api('/api/admin/instruction-status'),
  ]);
  instructionStatuses = statuses || {};
  const names = Object.keys(mappings);
  const element = document.getElementById('mappingList');
  if (!names.length) {
    element.innerHTML = '<div class="empty">暂无模型映射<br><span style="font-size:13px">点击「+ 添加映射」开始配置</span></div>';
    return;
  }
  element.innerHTML = `<div class="mapping-list">${names.map(name => {
    const mapping = mappings[name];
    const selected = mapping.templates || {};
    const hasInstruction = !!selected.instruction;
    const tags = [
      formatTag(mapping.upstream_protocol, '中转站: '),
      selected.address ? '<span class="tag tag-override">地址模板</span>' : '',
      selected.instruction ? '<span class="tag tag-instructions">指令模板</span>' : '',
      selected.body ? '<span class="tag tag-mods">Body模板</span>' : '',
      selected.header ? '<span class="tag tag-mods">Header模板</span>' : '',
      selected.replacements?.length ? `<span class="tag tag-replacement">文本替换 ${selected.replacements.length}</span>` : '',
      mapping.thinking_level !== 'default' ? `<span class="tag tag-thinking">思考: ${esc(mapping.thinking_level)}</span>` : '',
      mapping.fast_mode ? '<span class="tag tag-fast">Fast</span>' : '',
      instructionStatusTag(name, hasInstruction),
    ].join('');
    return `<div class="mapping-item"><div class="mapping-top">
      <span class="mapping-name">${esc(name)}</span><span class="mapping-arrow">&rarr;</span>
      <span class="mapping-upstream">${esc(mapping.upstream_model || name)}</span>
      <div class="mapping-meta">${tags}</div>
      <div class="mapping-actions">
        <button class="btn btn-ghost btn-sm" onclick="openEditModal('${esc(name)}')">编辑</button>
        <button class="btn btn-red btn-sm" onclick="deleteMapping('${esc(name)}')">删除</button>
      </div>
    </div></div>`;
  }).join('')}</div>`;
}

function resetMappingForm() {
  document.getElementById('mName').value = '';
  document.getElementById('mUpstream').value = '';
  document.getElementById('mUpstreamProtocol').value = 'auto';
  document.getElementById('mThinkingLevel').value = 'default';
  document.getElementById('mFastMode').checked = false;
  ['mAddressTemplate', 'mInstructionTemplate', 'mBodyTemplate', 'mHeaderTemplate']
    .forEach(id => { document.getElementById(id).value = ''; });
  setSelectedReplacementNames([]);
  document.getElementById('mReplacementPicker').classList.remove('open');
  document.getElementById('mReplacementControl').setAttribute('aria-expanded', 'false');
}

async function openAddModal() {
  editingName = null;
  document.getElementById('modalTitle').textContent = '添加模型映射';
  resetMappingForm();
  document.getElementById('modal').classList.add('active');
}

async function openEditModal(name) {
  try {
    const mappings = await api('/api/admin/mappings');
    const mapping = mappings[name];
    if (!mapping) return toast('映射未找到', false);
    editingName = name;
    document.getElementById('modalTitle').textContent = '编辑模型映射';
    resetMappingForm();
    document.getElementById('mName').value = name;
    document.getElementById('mUpstream').value = mapping.upstream_model || '';
    document.getElementById('mUpstreamProtocol').value = mapping.upstream_protocol || 'auto';
    document.getElementById('mThinkingLevel').value = mapping.thinking_level || 'default';
    document.getElementById('mFastMode').checked = !!mapping.fast_mode;
    for (const [kind, id] of [
      ['address', 'mAddressTemplate'], ['instruction', 'mInstructionTemplate'],
      ['body', 'mBodyTemplate'], ['header', 'mHeaderTemplate'],
    ]) document.getElementById(id).value = mapping.templates?.[kind] || '';
    setSelectedReplacementNames(mapping.templates?.replacements || []);
    document.getElementById('modal').classList.add('active');
  } catch (error) {
    toast(`错误: ${error.message}`, false);
  }
}

function closeModal() {
  document.getElementById('modal').classList.remove('active');
  editingName = null;
}

async function saveMapping() {
  const name = document.getElementById('mName').value.trim();
  const upstream = document.getElementById('mUpstream').value.trim();
  if (!name) return toast('请填写 Cursor 模型名', false);
  if (!upstream) return toast('请填写上游模型名', false);
  const isEditing = Boolean(editingName);
  const payload = {
    name,
    upstream_model: upstream,
    upstream_protocol: document.getElementById('mUpstreamProtocol').value,
    templates: {
      address: document.getElementById('mAddressTemplate').value,
      instruction: document.getElementById('mInstructionTemplate').value,
      body: document.getElementById('mBodyTemplate').value,
      header: document.getElementById('mHeaderTemplate').value,
      replacements: [...selectedReplacementNames],
    },
    thinking_level: document.getElementById('mThinkingLevel').value,
    fast_mode: document.getElementById('mFastMode').checked,
  };
  try {
    await api(`/api/admin/mappings${isEditing ? `/${encodeURIComponent(editingName)}` : ''}`, {
      method: isEditing ? 'PUT' : 'POST',
      body: JSON.stringify(payload),
    });
    closeModal();
    await loadMappings();
    toast(isEditing ? '映射已更新' : '映射已添加');
  } catch (error) {
    toast(`操作失败: ${error.message}`, false);
  }
}

async function deleteMapping(name) {
  if (!confirm(`确定要删除映射「${name}」吗？`)) return;
  try {
    await api(`/api/admin/mappings/${encodeURIComponent(name)}`, { method: 'DELETE' });
    await loadMappings();
    toast('映射已删除');
  } catch (error) {
    toast(`删除失败: ${error.message}`, false);
  }
}

async function checkHealth() {
  const badge = document.getElementById('statusBadge');
  try {
    const result = await fetch(`${API}/health`).then(response => response.json());
    badge.textContent = result.status === 'ok' ? '已连接' : '异常';
  } catch {
    badge.textContent = '离线';
  }
}

async function loadStats() {
  const element = document.getElementById('statsContent');
  try {
    const data = await api('/api/admin/stats');
    const names = Object.keys(data.models || {});
    if (!names.length) return void (element.innerHTML = '<div class="empty">暂无请求统计数据</div>');
    const uptime = data.uptime_seconds || 0;
    const hours = Math.floor(uptime / 3600);
    const minutes = Math.floor((uptime % 3600) / 60);
    let html = `<div class="hint" style="margin-bottom:12px">运行时长: ${hours}小时${minutes}分钟</div>`;
    html += '<table class="stats-table"><thead><tr><th>模型</th><th>请求数</th><th>输入 Tokens</th><th>输出 Tokens</th><th>总 Tokens</th></tr></thead><tbody>';
    names.sort((a, b) => data.models[b].request_count - data.models[a].request_count);
    for (const name of names) {
      const stats = data.models[name];
      html += `<tr><td>${esc(name)}</td><td>${stats.request_count}</td><td>${stats.input_tokens.toLocaleString()}</td><td>${stats.output_tokens.toLocaleString()}</td><td>${stats.total_tokens.toLocaleString()}</td></tr>`;
    }
    element.innerHTML = `${html}</tbody></table>`;
  } catch {
    element.innerHTML = '<div class="empty">加载统计失败</div>';
  }
}

function esc(value) {
  return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

(function init() {
  const saved = sessionStorage.getItem('_ak');
  if (!saved) return;
  authKey = saved;
  document.getElementById('login').style.display = 'none';
  document.getElementById('dashboard').style.display = 'block';
  loadDashboard();
})();

document.getElementById('modal').addEventListener('click', event => {
  if (event.target === event.currentTarget) closeModal();
});
document.getElementById('mReplacementMenu').addEventListener('click', event => {
  event.stopPropagation();
  const option = event.target.closest('.multi-select-option');
  if (option) return toggleReplacement(option.dataset.name);
});
document.getElementById('mReplacementValues').addEventListener('click', event => {
  const remove = event.target.closest('.multi-select-remove');
  if (remove) removeReplacement(remove.dataset.name, event);
});
document.addEventListener('click', event => {
  const picker = document.getElementById('mReplacementPicker');
  if (picker && !picker.contains(event.target)) {
    picker.classList.remove('open');
    document.getElementById('mReplacementControl').setAttribute('aria-expanded', 'false');
  }
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') closeModal();
});
