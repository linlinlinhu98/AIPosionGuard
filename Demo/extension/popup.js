// Popup controller
const API = 'http://localhost:8000';

async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`);
    const d = await r.json();
    document.getElementById('statusDot').className = 'dot green';
    document.getElementById('statusText').textContent =
      `Online - ${d.models_loaded} models loaded`;
  } catch (e) {
    document.getElementById('statusDot').className = 'dot red';
    document.getElementById('statusText').textContent = 'Backend offline';
  }
}

async function checkCurrentPage() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.url || !tab.url.includes('huggingface.co')) {
    return;
  }
  const parts = new URL(tab.url).pathname.split('/').filter(Boolean);
  if (parts.length >= 2) {
    const modelId = parts.slice(0, 2).join('/');
    document.getElementById('modelInfo').textContent = `Model: ${modelId}`;
    document.getElementById('scanBtn').style.display = 'block';
    document.getElementById('scanBtn').onclick = () => scanModel(modelId);
  }
}

async function scanModel(modelId) {
  const btn = document.getElementById('scanBtn');
  btn.textContent = 'Scanning...';
  btn.disabled = true;
  try {
    // Quick check via content script
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const result = await chrome.tabs.sendMessage(tab.id, {
      type: 'SCAN_MODEL', modelId
    });
    // Fallback: direct API call
    const r = await fetch(`${API}/health`);
    if (r.ok) {
      document.getElementById('scanResult').style.display = 'block';
      document.getElementById('scanResult').innerHTML =
        '<div style="padding:8px;background:#F0FDF4;border-radius:6px;color:#166534;">' +
        'Scan submitted. Check Dashboard for results.</div>';
    }
  } catch (e) {
    document.getElementById('scanResult').style.display = 'block';
    document.getElementById('scanResult').innerHTML =
      '<div style="padding:8px;background:#FEF2F2;border-radius:6px;color:#991B1B;">' +
      'Cannot reach backend. Start AI-PoisonGuard server first.</div>';
  }
  btn.textContent = 'Scan This Model';
  btn.disabled = false;
}

function loadStats() {
  chrome.storage.local.get(['scanned', 'suspicious', 'clean'], (data) => {
    document.getElementById('scannedCount').textContent = data.scanned || 0;
    document.getElementById('suspiciousCount').textContent = data.suspicious || 0;
    document.getElementById('cleanCount').textContent = data.clean || 0;
  });
}

document.getElementById('openDashboard').onclick = () => {
  chrome.tabs.create({ url: 'http://localhost:8000' });
};

checkHealth();
checkCurrentPage();
loadStats();
