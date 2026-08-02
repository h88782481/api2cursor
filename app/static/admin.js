const API = '';
let authKey = '';
let editingName = null;
let instructionBlocks = {
  function: [],
  custom_grammar: [],
};
let instructionStatuses = {};

const FORMAT_LABELS = {
  auto: '自动',
  chat: 'chat',
  responses: 'responses',
  messages: 'messages',
  gemini: 'gemini',
};
const FORMAT_TAG_CLASSES = {
  auto: 'tag-auto',
  chat: 'tag-chat',
  responses: 'tag-responses',
  messages: 'tag-messages',
  gemini: 'tag-gemini',
};

const DIALECT_UI = {
  function: { target: 'mFnTarget', hint: 'mFnTargetHint', text: 'mFnText', mode: 'mFnMode' },
  custom_grammar: { target: 'mCgTarget', hint: 'mCgTargetHint', text: 'mCgText', mode: 'mCgMode' },
};

function togglePwd(id) {
  const el = document.getElementById(id);
  el.type = el.type === 'password' ? 'text' : 'password';
}

function toast(msg, ok = true) {
  const area = document.getElementById('toasts');
  const el = document.createElement('div');
  el.className = 'toast ' + (ok ? 'toast-ok' : 'toast-err');
  el.textContent = msg;
  area.appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

async function api(path, opts = {}) {
  const headers = { 'Content-Type': 'application/json' };
  if (authKey) headers['Authorization'] = 'Bearer ' + authKey;
  const res = await fetch(API + path, { ...opts, headers });
  const ct = res.headers.get('content-type') || '';
  if (!ct.includes('application/json')) {
    const text = await res.text();
    if (!res.ok) throw new Error('HTTP ' + res.status + ': ' + text.substring(0, 100));
    throw new Error('服务器返回了非 JSON 响应');
  }
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error?.message || 'HTTP ' + res.status);
  }
  return data;
}

// ─── 登录 ───────────────────────────────────────────
async function doLogin() {
  const key = document.getElementById('loginKey').value.trim();
  if (!key) { toast('请填写密钥', false); return; }
  try {
    const r = await api('/api/admin/login', { method: 'POST', body: JSON.stringify({ key }) });
    if (r.ok) {
      authKey = key;
      sessionStorage.setItem('_ak', key);
      document.getElementById('login').style.display = 'none';
      document.getElementById('dashboard').style.display = 'block';
      loadDashboard();
    }
  } catch (e) {
    toast('密钥无效', false);
  }
}

function doLogout() {
  authKey = '';
  sessionStorage.removeItem('_ak');
  document.getElementById('dashboard').style.display = 'none';
  document.getElementById('login').style.display = 'flex';
}

// ─── 仪表盘 ─────────────────────────────────────────
async function loadDashboard() {
  try {
    const s = await api('/api/admin/settings');
    document.getElementById('targetUrl').value = s.proxy_target_url || '';
    document.getElementById('proxyKey').value = s.proxy_api_key || '';
    document.getElementById('debugMode').value = s.debug_mode || 'off';
    document.getElementById('envUrl').textContent = s.env_target_url ? '环境变量: ' + s.env_target_url : '';
    document.getElementById('envKey').textContent = s.env_api_key ? '环境变量: (已配置)' : '环境变量: (未设置)';
    await ensureInstructionBlocks();
    await loadMappings();
    checkHealth();
    loadStats();
  } catch (e) {
    toast('加载设置失败: ' + e.message, false);
  }
}

async function ensureInstructionBlocks() {
  if (instructionBlocks.function.length && instructionBlocks.custom_grammar.length) return;
  instructionBlocks = await api('/api/admin/instruction-blocks');
  fillTargetSelect('function');
  fillTargetSelect('custom_grammar');
}

function fillTargetSelect(dialect) {
  const ui = DIALECT_UI[dialect];
  const select = document.getElementById(ui.target);
  const blocks = instructionBlocks[dialect] || [];
  select.innerHTML = blocks.map(block => {
    const short = block.description.length > 36
      ? block.description.slice(0, 36) + '…'
      : block.description;
    const label = block.id === 'all'
      ? `${block.label} — ${short}`
      : `${block.label} — ${short}`;
    return `<option value="${esc(block.id)}">${esc(label)}</option>`;
  }).join('');
  updateBlockHint(dialect);
}

function updateBlockHint(dialect) {
  const ui = DIALECT_UI[dialect];
  const select = document.getElementById(ui.target);
  const hint = document.getElementById(ui.hint);
  const blocks = instructionBlocks[dialect] || [];
  const block = blocks.find(item => item.id === select.value) || blocks[0];
  const selected = select.selectedOptions[0];
  hint.textContent = selected?.dataset.invalid
    ? '当前 Cursor 提示词可能已不包含此块，请重新选择注入目标。'
    : (block ? block.description : '');
}

function copyDialectRule(source, target) {
  writeDialectRule(target, readDialectRule(source));
  updateInstructionWarning();
}

function dangerousInstructionRules() {
  return ['function', 'custom_grammar'].filter(dialect => {
    const rule = readDialectRule(dialect);
    return rule.text.trim() && rule.target === 'all' && rule.mode === 'replace';
  });
}

function updateInstructionWarning() {
  const warning = document.getElementById('mInstructionWarning');
  const dangerous = dangerousInstructionRules();
  if (!dangerous.length) {
    warning.style.display = 'none';
    warning.textContent = '';
    return;
  }
  warning.style.display = 'block';
  warning.textContent = '高风险：覆盖“全部”会替换整个 Cursor system，可能导致 Agent 工具和编辑规则失效。';
}

async function loadStats() {
  const el = document.getElementById('statsContent');
  try {
    const data = await api('/api/admin/stats');
    const models = data.models || {};
    const keys = Object.keys(models);
    if (!keys.length) {
      el.innerHTML = '<div class="empty">暂无请求统计数据</div>';
      return;
    }
    const uptime = data.uptime_seconds || 0;
    const h = Math.floor(uptime / 3600);
    const m = Math.floor((uptime % 3600) / 60);
    let html = '<div class="hint" style="margin-bottom:12px">运行时长: ' + h + '小时' + m + '分钟</div>';
    html += '<table class="stats-table"><thead><tr><th>模型</th><th>请求数</th><th>输入 Tokens</th><th>输出 Tokens</th><th>总 Tokens</th></tr></thead><tbody>';
    keys.sort((a, b) => models[b].request_count - models[a].request_count);
    for (const name of keys) {
      const s = models[name];
      html += '<tr><td>' + esc(name) + '</td><td>' + s.request_count + '</td><td>' + s.input_tokens.toLocaleString() + '</td><td>' + s.output_tokens.toLocaleString() + '</td><td>' + s.total_tokens.toLocaleString() + '</td></tr>';
    }
    html += '</tbody></table>';
    el.innerHTML = html;
  } catch (e) {
    el.innerHTML = '<div class="empty">加载统计失败</div>';
  }
}

async function checkHealth() {
  try {
    const r = await fetch(API + '/health');
    const d = await r.json();
    const b = document.getElementById('statusBadge');
    if (d.status === 'ok') {
      b.textContent = '已连接';
      b.style.background = 'rgba(34,197,94,.15)';
      b.style.color = 'var(--green)';
    } else {
      b.textContent = '异常';
    }
  } catch {
    const b = document.getElementById('statusBadge');
    b.textContent = '离线';
    b.style.background = 'rgba(239,68,68,.15)';
    b.style.color = 'var(--red)';
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
  } catch (e) {
    toast('保存失败: ' + e.message, false);
  }
}

// ─── 模型映射 ───────────────────────────────────────
function formatTag(fmt, prefix) {
  const value = fmt || 'auto';
  const cls = FORMAT_TAG_CLASSES[value] || 'tag-auto';
  const label = FORMAT_LABELS[value] || value;
  return '<span class="tag ' + cls + '">' + esc(prefix + label) + '</span>';
}

function hasInstructionText(instructions) {
  if (!instructions) return false;
  return !!(instructions.function?.text || instructions.custom_grammar?.text);
}

function defaultDialectRule() {
  return { text: '', target: 'all', mode: 'prepend' };
}

function readDialectRule(dialect) {
  const ui = DIALECT_UI[dialect];
  return {
    text: document.getElementById(ui.text).value,
    target: document.getElementById(ui.target).value || 'all',
    mode: document.getElementById(ui.mode).value || 'prepend',
  };
}

function writeDialectRule(dialect, rule) {
  const ui = DIALECT_UI[dialect];
  const value = rule || defaultDialectRule();
  document.getElementById(ui.text).value = value.text || '';
  const target = value.target || 'all';
  const select = document.getElementById(ui.target);
  if (![...select.options].some(opt => opt.value === target)) {
    const option = document.createElement('option');
    option.value = target;
    option.textContent = `${target} — 当前 Cursor 可能已移除此块`;
    option.dataset.invalid = 'true';
    select.appendChild(option);
    select.value = target;
  } else {
    select.value = target;
  }
  document.getElementById(ui.mode).value = value.mode || 'prepend';
  updateBlockHint(dialect);
  updateInstructionWarning();
}

function resetInstructionForm() {
  writeDialectRule('function', defaultDialectRule());
  writeDialectRule('custom_grammar', defaultDialectRule());
}

async function loadMappings() {
  const [mappings, statuses] = await Promise.all([
    api('/api/admin/mappings'),
    api('/api/admin/instruction-status'),
  ]);
  instructionStatuses = statuses || {};
  const el = document.getElementById('mappingList');
  const keys = Object.keys(mappings);

  if (!keys.length) {
    el.innerHTML = '<div class="empty">暂无模型映射<br><span style="font-size:13px">点击「+ 添加映射」开始配置</span></div>';
    return;
  }

  el.innerHTML = '<div class="mapping-list">' + keys.map(name => {
    const m = mappings[name];
    const upstreamFmt = m.upstream_protocol || 'auto';
    const thinkingLevel = m.thinking_level || 'default';
    const hasOverride = m.target_url || m.api_key;
    const hasInstructions = hasInstructionText(m.instructions);
    const hasBodyMods = m.body_modifications && Object.keys(m.body_modifications).length > 0;
    const hasHeaderMods = m.header_modifications && Object.keys(m.header_modifications).length > 0;
    const statusTag = instructionStatusTag(name, m.instructions);
    return `<div class="mapping-item">
      <div class="mapping-top">
        <span class="mapping-name">${esc(name)}</span>
        <span class="mapping-arrow">&rarr;</span>
        <span class="mapping-upstream">${esc(m.upstream_model || name)}</span>
        <div class="mapping-meta">
          ${formatTag(upstreamFmt, '中转站: ')}
          ${hasOverride ? '<span class="tag tag-override">自定义地址</span>' : ''}
          ${thinkingLevel !== 'default' ? `<span class="tag tag-thinking">思考: ${esc(thinkingLevel)}</span>` : ''}
          ${m.fast_mode ? '<span class="tag tag-fast">Fast</span>' : ''}
          ${hasInstructions ? '<span class="tag tag-instructions">自定义指令</span>' : ''}
          ${statusTag}
          ${hasBodyMods ? '<span class="tag tag-mods">Body修改</span>' : ''}
          ${hasHeaderMods ? '<span class="tag tag-mods">Header修改</span>' : ''}
        </div>
        <div class="mapping-actions">
          <button class="btn btn-ghost btn-sm" onclick="openEditModal('${esc(name)}')">编辑</button>
          <button class="btn btn-red btn-sm" onclick="deleteMapping('${esc(name)}')">删除</button>
        </div>
      </div>
    </div>`;
  }).join('') + '</div>';
}

function instructionStatusTag(name, instructions) {
  if (!hasInstructionText(instructions)) return '';
  const statuses = instructionStatuses[name] || {};
  const configured = ['function', 'custom_grammar']
    .filter(dialect => instructions?.[dialect]?.text);
  const relevant = configured.map(dialect => statuses[dialect]).filter(Boolean);
  if (relevant.some(status => status.state === 'failed')) {
    const failed = relevant.find(status => status.state === 'failed');
    return `<span class="tag tag-warning" title="${esc(failed.message || '')}">注入失败</span>`;
  }
  if (relevant.some(status => status.state === 'applied')) {
    return '<span class="tag tag-ok">注入正常</span>';
  }
  return '<span class="tag tag-pending">尚无请求验证</span>';
}

function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;'); }

// ─── 弹窗 ──────────────────────────────────────────
async function openAddModal() {
  editingName = null;
  document.getElementById('modalTitle').textContent = '添加模型映射';
  document.getElementById('mName').value = '';
  document.getElementById('mName').disabled = false;
  document.getElementById('mUpstream').value = '';
  document.getElementById('mUpstreamProtocol').value = 'auto';
  document.getElementById('mUrl').value = '';
  document.getElementById('mKey').value = '';
  document.getElementById('mThinkingLevel').value = 'default';
  document.getElementById('mFastMode').checked = false;
  document.getElementById('mBodyMods').value = '';
  document.getElementById('mHeaderMods').value = '';
  await ensureInstructionBlocks();
  resetInstructionForm();
  updateInstructionWarning();
  document.getElementById('modal').classList.add('active');
}

async function openEditModal(name) {
  editingName = name;
  document.getElementById('modalTitle').textContent = '编辑模型映射';
  try {
    await ensureInstructionBlocks();
    const mappings = await api('/api/admin/mappings');
    const m = mappings[name];
    if (!m) { toast('映射未找到', false); return; }
    document.getElementById('mName').value = name;
    document.getElementById('mName').disabled = false;
    document.getElementById('mUpstream').value = m.upstream_model || '';
    document.getElementById('mUpstreamProtocol').value = m.upstream_protocol || 'auto';
    document.getElementById('mUrl').value = m.target_url || '';
    document.getElementById('mKey').value = m.api_key || '';
    document.getElementById('mThinkingLevel').value = m.thinking_level || 'default';
    document.getElementById('mFastMode').checked = !!m.fast_mode;
    writeDialectRule('function', m.instructions?.function);
    writeDialectRule('custom_grammar', m.instructions?.custom_grammar);
    updateInstructionWarning();
    document.getElementById('mBodyMods').value = m.body_modifications && Object.keys(m.body_modifications).length ? JSON.stringify(m.body_modifications, null, 2) : '';
    document.getElementById('mHeaderMods').value = m.header_modifications && Object.keys(m.header_modifications).length ? JSON.stringify(m.header_modifications, null, 2) : '';
    document.getElementById('modal').classList.add('active');
  } catch (e) {
    toast('错误: ' + e.message, false);
  }
}

function closeModal() {
  document.getElementById('modal').classList.remove('active');
  editingName = null;
}

async function saveMapping() {
  const name = document.getElementById('mName').value.trim();
  const upstream = document.getElementById('mUpstream').value.trim();
  if (!name) { toast('请填写 Cursor 模型名', false); return; }
  if (!upstream) { toast('请填写上游模型名', false); return; }

  let bodyMods = {};
  const bodyModsStr = document.getElementById('mBodyMods').value.trim();
  if (bodyModsStr) {
    try { bodyMods = JSON.parse(bodyModsStr); }
    catch { toast('Body 修改不是有效的 JSON', false); return; }
  }

  let headerMods = {};
  const headerModsStr = document.getElementById('mHeaderMods').value.trim();
  if (headerModsStr) {
    try { headerMods = JSON.parse(headerModsStr); }
    catch { toast('Header 修改不是有效的 JSON', false); return; }
  }

  const payload = {
    name,
    upstream_model: upstream,
    upstream_protocol: document.getElementById('mUpstreamProtocol').value,
    target_url: document.getElementById('mUrl').value.trim(),
    api_key: document.getElementById('mKey').value.trim(),
    thinking_level: document.getElementById('mThinkingLevel').value,
    fast_mode: document.getElementById('mFastMode').checked,
    instructions: {
      function: readDialectRule('function'),
      custom_grammar: readDialectRule('custom_grammar'),
    },
    body_modifications: bodyMods,
    header_modifications: headerMods,
  };

  if (
    dangerousInstructionRules().length
    && !confirm('覆盖“全部”会替换整个 Cursor system，可能导致 Agent 工具行为失效。确定继续保存吗？')
  ) {
    return;
  }

  try {
    if (editingName) {
      await api('/api/admin/mappings/' + encodeURIComponent(editingName), {
        method: 'PUT', body: JSON.stringify(payload),
      });
      toast('映射已更新');
    } else {
      await api('/api/admin/mappings', {
        method: 'POST', body: JSON.stringify(payload),
      });
      toast('映射已添加');
    }
    closeModal();
    await loadMappings();
  } catch (e) {
    toast('操作失败: ' + e.message, false);
  }
}

async function deleteMapping(name) {
  if (!confirm('确定要删除映射「' + name + '」吗？')) return;
  try {
    await api('/api/admin/mappings/' + encodeURIComponent(name), { method: 'DELETE' });
    toast('映射已删除');
    await loadMappings();
  } catch (e) {
    toast('删除失败: ' + e.message, false);
  }
}

// ─── 初始化 ─────────────────────────────────────────
(function init() {
  const saved = sessionStorage.getItem('_ak');
  if (saved) {
    authKey = saved;
    document.getElementById('login').style.display = 'none';
    document.getElementById('dashboard').style.display = 'block';
    loadDashboard();
  }
})();

document.getElementById('modal').addEventListener('click', function(e) {
  if (e.target === this) closeModal();
});
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeModal();
});
