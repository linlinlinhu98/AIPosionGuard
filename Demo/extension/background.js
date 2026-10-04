var API = 'http://localhost:8000';

chrome.runtime.onMessage.addListener(function(msg, sender, sendResponse) {
  if (msg.type === 'SCAN_MODEL') {
    scanModel(msg.modelId).then(function(result) { sendResponse(result); });
    return true;
  }
  if (msg.type === 'ALERT') {
    try {
      chrome.notifications.create({
        type: 'basic',
        iconUrl: 'icons/icon128.svg',
        title: 'AI-PoisonGuard',
        message: msg.message,
        priority: 2
      });
    } catch(e) {}
    chrome.action.setBadgeText({ text: '!' });
    chrome.action.setBadgeBackgroundColor({ color: '#DC2626' });
    chrome.storage.local.get(['alerts'], function(d) {
      var alerts = d.alerts || [];
      alerts.push({ model: msg.modelId, time: new Date().toISOString() });
      chrome.storage.local.set({ alerts: alerts.slice(-50) });
    });
  }
});

function scanModel(modelId) {
  return fetch(API + '/health')
    .then(function(r) { return r.json(); })
    .then(function(h) {
      if (h.status !== 'healthy') return { error: 'unhealthy' };
      // Check if model is already registered
      return fetch(API + '/api/v1/models')
        .then(function(r) { return r.json(); })
        .then(function(data) {
          var models = data.models || [];
          var found = null;
          for (var i = 0; i < models.length; i++) {
            if (models[i].model_path && models[i].model_path.indexOf(modelId) !== -1) {
              found = models[i];
              break;
            }
          }
          if (found) {
            // Model is registered, check scan status via task
            return { registered: true, modelId: modelId, detail: 'Model registered. Run BAIT detection for full scan.' };
          }
          return { registered: false, modelId: modelId, detail: 'Model not yet registered in AI-PoisonGuard. Upload via Dashboard.' };
        });
    })
    .catch(function(e) {
      return { error: 'offline', detail: 'Backend not reachable on ' + API };
    });
}

// Check backend health periodically
chrome.alarms.create('health', { periodInMinutes: 2 });
chrome.alarms.onAlarm.addListener(function(alarm) {
  if (alarm.name !== 'health') return;
  fetch(API + '/health')
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.status === 'healthy') {
        chrome.action.setBadgeText({ text: '' });
      }
    })
    .catch(function() {
      chrome.action.setBadgeText({ text: 'OFF' });
      chrome.action.setBadgeBackgroundColor({ color: '#9CA3AF' });
    });
});
