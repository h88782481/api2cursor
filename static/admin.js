const API = '';
let authKey = '';
let editingName = null;

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
    const e = data.error;
    const msg = (typeof e === 'object' && e !== null) ? (e.message || JSON.stringify(e)) : (e || data.message || 'HTTP ' + res.status);
    throw new Error(msg);
  }
  return data;
}

// ─── 登录 ───────────────────────────────────────────
async function doLogin() {
  const key = document.getElementById('loginKey').value.trim();
  if (!key) { toast('请输入密钥', false); return; }
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
    document.getElementById('envUrl').textContent = s.env_target_url ? '环境变量: ' + s.env_target_url : '';
    document.getElementById('envKey').textContent = s.env_api_key ? '环境变量: (已配置)' : '环境变量: (未设置)';
    await loadMappings();
    checkHealth();
    loadStats();
  } catch (e) {
    toast('加载设置失败: ' + e.message, false);
  }
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
      }),
    });
    toast('设置已保存');
  } catch (e) {
    toast('保存失败: ' + e.message, false);
  }
}

// ─── 模型映射 ───────────────────────────────────────
async function loadMappings() {
  const mappings = await api('/api/admin/mappings');
  const el = document.getElementById('mappingList');
  const keys = Object.keys(mappings);

  if (!keys.length) {
    el.innerHTML = '<div class="empty">暂无模型映射<br><span style="font-size:13px">点击「+ 添加映射」开始配置</span></div>';
    return;
  }

  el.innerHTML = '<div class="mapping-list">' + keys.map(name => {
    const m = mappings[name];
    const backend = m.backend || 'auto';
    const tagClass = backend === 'anthropic'
      ? 'tag-anthropic'
      : backend === 'responses'
        ? 'tag-responses'
        : backend === 'openai'
          ? 'tag-openai'
          : backend === 'gemini'
            ? 'tag-gemini'
            : 'tag-auto';
    const tagLabel = backend === 'auto'
      ? '自动'
      : backend === 'responses'
        ? 'responses'
        : backend;
    const hasOverride = m.target_url || m.api_key;
    const hasInstructions = !!m.custom_instructions;
    const hasBodyMods = m.body_modifications && Object.keys(m.body_modifications).length > 0;
    const hasHeaderMods = m.header_modifications && Object.keys(m.header_modifications).length > 0;
    return `<div class="mapping-item">
      <div class="mapping-top">
        <span class="mapping-name">${esc(name)}</span>
        <span class="mapping-arrow">&rarr;</span>
        <span class="mapping-upstream">${esc(m.upstream_model || name)}</span>
        <div class="mapping-meta">
          <span class="tag ${tagClass}">${tagLabel}</span>
          ${hasOverride ? '<span class="tag tag-override">自定义地址</span>' : ''}
          ${hasInstructions ? '<span class="tag tag-instructions">自定义指令</span>' : ''}
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

function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;'); }

// ─── 弹窗 ──────────────────────────────────────────
function openAddModal() {
  editingName = null;
  document.getElementById('modalTitle').textContent = '添加模型映射';
  document.getElementById('mName').value = '';
  document.getElementById('mName').disabled = false;
  document.getElementById('mUpstream').value = '';
  document.getElementById('mBackend').value = 'auto';
  document.getElementById('mUrl').value = '';
  document.getElementById('mKey').value = '';
  document.getElementById('mInstructions').value = '';
  document.getElementById('mInsPosition').value = 'prepend';
  document.getElementById('mBodyMods').value = '';
  document.getElementById('mHeaderMods').value = '';
  document.getElementById('modal').classList.add('active');
}

async function openEditModal(name) {
  editingName = name;
  document.getElementById('modalTitle').textContent = '编辑模型映射';
  try {
    const mappings = await api('/api/admin/mappings');
    const m = mappings[name];
    if (!m) { toast('映射未找到', false); return; }
    document.getElementById('mName').value = name;
    document.getElementById('mName').disabled = false;
    document.getElementById('mUpstream').value = m.upstream_model || '';
    document.getElementById('mBackend').value = m.backend || 'auto';
    document.getElementById('mUrl').value = m.target_url || '';
    document.getElementById('mKey').value = m.api_key || '';
    document.getElementById('mInstructions').value = m.custom_instructions || '';
    document.getElementById('mInsPosition').value = m.instructions_position || 'prepend';
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
    backend: document.getElementById('mBackend').value,
    target_url: document.getElementById('mUrl').value.trim(),
    api_key: document.getElementById('mKey').value.trim(),
    custom_instructions: document.getElementById('mInstructions').value,
    instructions_position: document.getElementById('mInsPosition').value,
    body_modifications: bodyMods,
    header_modifications: headerMods,
  };

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
