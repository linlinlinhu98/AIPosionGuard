(function() {
  'use strict';

  // Extract model ID from URL: huggingface.co/org/name
  function getModelId() {
    var parts = window.location.pathname.split('/').filter(Boolean);
    if (parts.length >= 2) {
      var skip = ['spaces','datasets','docs','tasks','pricing','learn','blog','forum',
                   'chat','settings','notifications','new','organizations','collections',
                   'login','signup','join'];
      if (skip.indexOf(parts[0]) === -1 && parts[0].charAt(0) !== '_') {
        return parts[0] + '/' + parts[1];
      }
    }
    return null;
  }

  function injectBadge(modelId, status, detail) {
    var existing = document.querySelector('.aipg-container');
    if (existing) existing.remove();

    var container = document.createElement('div');
    container.className = 'aipg-container';

    var badge = document.createElement('span');
    badge.className = 'aipg-badge';

    if (status === 'scanning') {
      badge.innerHTML = 'Scanning...';
      badge.style.cssText = 'color:#92400E;background:#FEF3C7;padding:2px 10px;border-radius:4px;font-size:13px;';
    } else if (status === 'clean') {
      badge.innerHTML = 'SAFE';
      badge.style.cssText = 'color:#166534;background:#DCFCE7;padding:2px 10px;border-radius:4px;font-size:13px;font-weight:bold;';
      badge.title = 'AI-PoisonGuard: No backdoor detected';
    } else if (status === 'suspicious') {
      badge.innerHTML = 'SUSPICIOUS';
      badge.style.cssText = 'color:#991B1B;background:#FEE2E2;padding:2px 10px;border-radius:4px;font-size:13px;font-weight:bold;cursor:pointer;';
      badge.title = detail || 'AI-PoisonGuard: Potential backdoor detected';
      badge.onclick = function() { alert(detail || 'Suspicious model detected. Open AI-PoisonGuard Dashboard for details.'); };
    } else {
      badge.innerHTML = 'Offline';
      badge.style.cssText = 'color:#6B7280;background:#F3F4F6;padding:2px 10px;border-radius:4px;font-size:13px;';
      badge.title = 'Backend not running. Start AI-PoisonGuard server.';
    }

    container.appendChild(badge);

    // Inject near the model title
    var h1 = document.querySelector('h1');
    if (h1 && h1.parentNode) {
      h1.parentNode.insertBefore(container, h1.nextSibling);
    }
  }

  function scanModel(modelId) {
    injectBadge(modelId, 'scanning');

    chrome.runtime.sendMessage({ type: 'SCAN_MODEL', modelId: modelId }, function(resp) {
      if (chrome.runtime.lastError) {
        injectBadge(modelId, 'offline');
        return;
      }
      if (resp && resp.error) {
        injectBadge(modelId, 'offline');
      } else if (resp && resp.suspicious) {
        injectBadge(modelId, 'suspicious', resp.detail);
        chrome.runtime.sendMessage({ type: 'ALERT', modelId: modelId, message: 'Suspicious: ' + modelId });
      } else {
        injectBadge(modelId, 'clean');
      }
    });
  }

  var modelId = getModelId();
  if (modelId) {
    scanModel(modelId);
  }

  // Watch for SPA navigation (HF uses client-side routing)
  var lastUrl = location.href;
  new MutationObserver(function() {
    if (location.href !== lastUrl) {
      lastUrl = location.href;
      setTimeout(function() {
        var mid = getModelId();
        if (mid) scanModel(mid);
      }, 1500);
    }
  }).observe(document.body || document.documentElement, { subtree: true, childList: true });
})();
