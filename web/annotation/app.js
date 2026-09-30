(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };

  var elements = {
    mediaInput: $('mediaInput'),
    maxFrames: $('maxFrames'),
    resetButton: $('resetButton'),
    mediaSummary: $('mediaSummary'),
    healthDot: $('healthDot'),
    healthText: $('healthText'),
    objectId: $('objectId'),
    objectLabel: $('objectLabel'),
    newObjectButton: $('newObjectButton'),
    undoButton: $('undoButton'),
    clearButton: $('clearButton'),
    segmentButton: $('segmentButton'),
    propagateButton: $('propagateButton'),
    pointCount: $('pointCount'),
    objectCount: $('objectCount'),
    objectEmpty: $('objectEmpty'),
    objectList: $('objectList'),
    exportButton: $('exportButton'),
    promptToolButton: $('promptToolButton'),
    panToolButton: $('panToolButton'),
    zoomOutButton: $('zoomOutButton'),
    zoomInButton: $('zoomInButton'),
    zoomLevel: $('zoomLevel'),
    fitButton: $('fitButton'),
    canvasWrap: $('canvasWrap'),
    canvas: $('annotationCanvas'),
    emptyState: $('emptyState'),
    busyOverlay: $('busyOverlay'),
    timeline: $('timeline'),
    previousButton: $('previousButton'),
    playButton: $('playButton'),
    nextButton: $('nextButton'),
    frameReadout: $('frameReadout'),
    frameSlider: $('frameSlider'),
    timeReadout: $('timeReadout'),
    statusMessage: $('statusMessage'),
    errorMessage: $('errorMessage'),
    clearMessageButton: $('clearMessageButton')
  };

  var context = elements.canvas.getContext('2d');

  var PALETTE = [
    [239, 83, 80],
    [66, 165, 245],
    [102, 187, 106],
    [255, 167, 38],
    [171, 71, 188],
    [38, 198, 218],
    [255, 238, 88],
    [141, 110, 99],
    [236, 64, 122],
    [124, 179, 66],
    [92, 107, 192],
    [255, 112, 67]
  ];

  var state = {
    healthy: false,
    busy: false,
    playing: false,
    playToken: 0,
    projectId: null,
    media: null,
    frameIndex: 0,
    image: null,
    imageUrl: null,
    points: [],
    annotationsByFrame: new Map(),
    propagationDirty: false,
    objects: new Map(),
    colourByObject: new Map(),
    nextColour: 0,
    selectedObjectKey: null,
    tool: 'prompt',
    pointer: null,
    layerCache: new WeakMap(),
    view: { scale: 1, x: 0, y: 0, fitScale: 1, placed: false },
    dpr: 1
  };

  /* COCO uncompressed RLE is column-major. Every run is checked before any
     mask is accepted, and the final run must end exactly at width * height. */
  function hasExactRLECoverage(counts, width, height) {
    if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) ||
        width <= 0 || height <= 0 || !Array.isArray(counts)) {
      return false;
    }

    var total = width * height;
    if (!Number.isSafeInteger(total) || total <= 0) return false;

    var covered = 0;
    for (var i = 0; i < counts.length; i += 1) {
      var run = counts[i];
      if (!Number.isSafeInteger(run) || run < 0 || run > total - covered) {
        return false;
      }
      covered += run;
    }
    return covered === total;
  }

  function decodeRLE(counts, width, height) {
    if (!hasExactRLECoverage(counts, width, height)) return null;

    var total = width * height;

    var mask = new Uint8Array(total);
    var position = 0;
    var foreground = false;

    for (var i = 0; i < counts.length; i += 1) {
      var run = counts[i];
      if (foreground) {
        var end = position + run;
        for (var cursor = position; cursor < end; cursor += 1) {
          var x = Math.floor(cursor / height);
          var y = cursor % height;
          mask[y * width + x] = 1;
        }
      }
      position += run;
      foreground = !foreground;
    }

    return mask;
  }

  function maskArea(mask) {
    var area = 0;
    for (var i = 0; i < mask.length; i += 1) area += mask[i] ? 1 : 0;
    return area;
  }

  function maskBoundingBox(mask, width, height) {
    var left = width;
    var top = height;
    var right = -1;
    var bottom = -1;
    for (var y = 0; y < height; y += 1) {
      for (var x = 0; x < width; x += 1) {
        if (!mask[y * width + x]) continue;
        if (x < left) left = x;
        if (x > right) right = x;
        if (y < top) top = y;
        if (y > bottom) bottom = y;
      }
    }
    return right < 0 ? [0, 0, 0, 0] :
      [left, top, right - left + 1, bottom - top + 1];
  }

  window.RLE = Object.freeze({
    decode: decodeRLE,
    area: maskArea,
    bbox: maskBoundingBox
  });

  function objectKey(value) {
    return String(value);
  }

  function compareObjectKeys(a, b) {
    var aNumber = Number(a);
    var bNumber = Number(b);
    var aNumeric = Number.isFinite(aNumber);
    var bNumeric = Number.isFinite(bNumber);
    if (aNumeric && bNumeric) return aNumber - bNumber;
    return a.localeCompare(b, 'ja', { numeric: true });
  }

  function colourForObject(key) {
    if (!state.colourByObject.has(key)) {
      state.colourByObject.set(key, state.nextColour % PALETTE.length);
      state.nextColour += 1;
    }
    return PALETTE[state.colourByObject.get(key)];
  }

  function colourCss(rgb, alpha) {
    return alpha === undefined ?
      'rgb(' + rgb.join(', ') + ')' :
      'rgba(' + rgb.join(', ') + ', ' + alpha + ')';
  }

  function setStatus(message, kind) {
    elements.errorMessage.hidden = true;
    elements.errorMessage.textContent = '';
    elements.statusMessage.hidden = !message;
    elements.statusMessage.textContent = message || '';
    elements.statusMessage.className = 'status-message' + (kind ? ' ' + kind : '');
    elements.statusMessage.title = message || '';
    elements.clearMessageButton.hidden = !message;
  }

  function showError(error) {
    var message = error instanceof Error ? error.message : String(error);
    elements.statusMessage.hidden = true;
    elements.errorMessage.hidden = false;
    elements.errorMessage.textContent = message;
    elements.errorMessage.title = message;
    elements.clearMessageButton.hidden = false;
  }

  function clearMessage() {
    elements.statusMessage.hidden = true;
    elements.errorMessage.hidden = true;
    elements.statusMessage.textContent = '';
    elements.errorMessage.textContent = '';
    elements.clearMessageButton.hidden = true;
  }

  function hasAnyAnnotations() {
    var found = false;
    state.annotationsByFrame.forEach(function (annotations) {
      if (annotations.length) found = true;
    });
    return found;
  }

  function isVideo() {
    return !!state.media && state.media.frame_count > 1;
  }

  function syncControls() {
    var hasMedia = !!state.media;
    var locked = state.busy || state.playing;
    var usable = state.healthy && hasMedia;
    var hasPoints = state.points.length > 0;
    var frameCount = hasMedia ? state.media.frame_count : 0;
    var hasAnnotations = hasAnyAnnotations();

    elements.mediaInput.disabled = !state.healthy || locked;
    elements.maxFrames.disabled = !state.healthy || locked;
    var picker = elements.mediaInput.closest('.file-picker');
    if (picker) picker.classList.toggle('disabled', elements.mediaInput.disabled);

    elements.resetButton.disabled = !usable || locked;
    elements.objectId.disabled = !usable || locked;
    elements.objectLabel.disabled = !usable || locked;
    elements.newObjectButton.disabled = !usable || locked;
    elements.undoButton.disabled = !usable || locked || !hasPoints;
    elements.clearButton.disabled = !usable || locked || !hasPoints;
    elements.segmentButton.disabled = !usable || locked || !hasPoints;
    elements.propagateButton.disabled =
      !usable || locked || frameCount <= 1 || !hasAnnotations;
    elements.exportButton.disabled =
      !usable || locked || !hasAnnotations || state.propagationDirty;

    elements.promptToolButton.disabled = !usable || locked;
    elements.panToolButton.disabled = !usable || locked;
    elements.zoomOutButton.disabled = !usable || locked;
    elements.zoomInButton.disabled = !usable || locked;
    elements.fitButton.disabled = !usable || locked;

    elements.previousButton.disabled = !usable || locked || state.frameIndex <= 0;
    elements.nextButton.disabled =
      !usable || locked || state.frameIndex >= frameCount - 1;
    elements.frameSlider.disabled = !usable || locked;
    elements.playButton.disabled = !usable || frameCount <= 1 ||
      (state.busy && !state.playing);
    elements.playButton.textContent = state.playing ? '❚❚' : '▶';
    elements.playButton.setAttribute('aria-label', state.playing ? '一時停止' : '再生');

    elements.busyOverlay.hidden = !state.busy;
    document.body.classList.toggle('busy', state.busy);
  }

  function setBusy(busy, message) {
    state.busy = busy;
    if (busy && message) setStatus(message, 'working');
    syncControls();
  }

  async function runExclusive(message, action) {
    if (state.busy || state.playing) return null;
    setBusy(true, message);
    try {
      return await action();
    } catch (error) {
      showError(error);
      return null;
    } finally {
      setBusy(false);
    }
  }

  async function apiFetch(path, options) {
    var response;
    try {
      var requestOptions = Object.assign({
        cache: 'no-store',
        credentials: 'same-origin'
      }, options || {});
      var stateful = path === '/api/segment' || path === '/api/propagate' ||
        path === '/api/reset' || path === '/api/export' ||
        path.indexOf('/api/frame/') === 0;
      if (stateful) {
        if (!state.projectId) {
          throw new Error('プロジェクト識別子がありません');
        }
      }
      if (stateful || (path.indexOf('/api/media') === 0 && state.projectId)) {
        var headers = new Headers(requestOptions.headers || {});
        headers.set('X-SAM31-Project', state.projectId);
        requestOptions.headers = headers;
      }
      response = await fetch(path, requestOptions);
    } catch (error) {
      throw new Error('サーバーに接続できません: ' + error.message);
    }

    if (response.ok) return response;

    var detail = '';
    try {
      var text = await response.text();
      if (text) {
        try {
          var parsed = JSON.parse(text);
          detail = parsed.detail || parsed.error || parsed.message || text;
        } catch (_error) {
          detail = text;
        }
      }
    } catch (_readError) {
      detail = '';
    }
    throw new Error('API エラー ' + response.status + (detail ? ': ' + detail : ''));
  }

  async function apiJson(path, options) {
    var response = await apiFetch(path, options);
    try {
      return await response.json();
    } catch (_error) {
      throw new Error('サーバーから不正な JSON が返されました');
    }
  }

  function jsonOptions(method, value) {
    return {
      method: method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(value)
    };
  }

  function validateMedia(raw) {
    var media = raw && raw.media;
    if (!media || typeof media.name !== 'string' ||
        !Number.isSafeInteger(media.width) || media.width <= 0 ||
        !Number.isSafeInteger(media.height) || media.height <= 0 ||
        !Number.isSafeInteger(media.frame_count) || media.frame_count <= 0) {
      throw new Error('メディア情報の形式が正しくありません');
    }

    var fps = Number(media.fps);
    return {
      name: media.name,
      kind: typeof media.kind === 'string' ? media.kind : (media.frame_count > 1 ? 'video' : 'image'),
      width: media.width,
      height: media.height,
      frame_count: media.frame_count,
      fps: Number.isFinite(fps) && fps > 0 ? fps : 0
    };
  }

  function validateProjectId(raw, required) {
    var value = raw && raw.project_id;
    if ((value === null || value === undefined) && !required) return null;
    if (typeof value !== 'string' || !value.trim()) {
      throw new Error('プロジェクト識別子の形式が正しくありません');
    }
    return value.trim();
  }

  async function restoreProject(raw) {
    var hasMedia = !!raw && raw.media !== null && raw.media !== undefined;
    var projectId = validateProjectId(raw, hasMedia);
    if (!hasMedia) {
      state.projectId = projectId;
      return false;
    }
    if (!Array.isArray(raw.frames)) {
      throw new Error('保存済みプロジェクトのフレーム形式が正しくありません');
    }

    var media = validateMedia(raw);
    if (typeof raw.propagation_required !== 'boolean') {
      throw new Error('伝播状態の形式が正しくありません');
    }
    var serverObjects = normaliseServerObjects(raw.objects, media);
    var restoredFrames = new Map();
    raw.frames.forEach(function (frame) {
      var index = Number(frame && frame.frame_index);
      if (!Number.isSafeInteger(index) || index < 0 || index >= media.frame_count) {
        throw new Error('保存済みプロジェクトのフレーム番号が範囲外です');
      }
      if (restoredFrames.has(index)) {
        throw new Error('保存済みプロジェクトのフレーム番号が重複しています');
      }
      restoredFrames.set(index, normaliseFrameAnnotations(frame.annotations));
    });

    if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
    state.projectId = projectId;
    state.media = media;
    state.frameIndex = 0;
    state.image = null;
    state.imageUrl = null;
    state.points = [];
    state.annotationsByFrame = restoredFrames;
    state.propagationDirty = raw.propagation_required === true;
    state.objects = new Map();
    state.colourByObject = new Map();
    state.nextColour = 0;
    state.selectedObjectKey = null;
    state.layerCache = new WeakMap();
    state.pointer = null;
    state.view = { scale: 1, x: 0, y: 0, fitScale: 1, placed: false };
    elements.objectId.value = '1';
    elements.objectLabel.value = 'object';
    elements.pointCount.textContent = '0';
    updateMediaUi();
    state.objects = buildObjectRegistry(serverObjects, restoredFrames);
    await loadFrameImage(0);
    fitView();
    draw();
    return true;
  }

  function validateAnnotation(raw) {
    if (!raw || (typeof raw.object_id !== 'string' && typeof raw.object_id !== 'number')) {
      throw new Error('マスクの object_id が正しくありません');
    }
    var segmentation = raw.segmentation;
    if (!segmentation || !Array.isArray(segmentation.size) ||
        segmentation.size.length !== 2) {
      throw new Error('マスクの segmentation.size が正しくありません');
    }

    var height = segmentation.size[0];
    var width = segmentation.size[1];
    if (!hasExactRLECoverage(segmentation.counts, width, height)) {
      throw new Error('RLE がマスク全体を正確に覆っていません');
    }

    if (!Array.isArray(raw.bbox) || raw.bbox.length !== 4) {
      throw new Error('マスクの bbox が正しくありません');
    }
    var bbox = raw.bbox.map(Number);
    if (bbox.some(function (value) { return !Number.isFinite(value); })) {
      throw new Error('マスクの bbox が正しくありません');
    }

    var area = Number(raw.area);
    if (!Number.isFinite(area) || area < 0) {
      throw new Error('マスクの area が正しくありません');
    }
    var score = raw.score === null || raw.score === undefined ? null : Number(raw.score);
    if (score !== null && !Number.isFinite(score)) score = null;

    return {
      object_id: raw.object_id,
      label: typeof raw.label === 'string' && raw.label.trim() ?
        raw.label.trim() : 'object',
      score: score,
      area: area,
      bbox: bbox,
      segmentation: {
        size: [height, width],
        counts: segmentation.counts.slice()
      }
    };
  }

  function normaliseFrameAnnotations(raw) {
    if (!Array.isArray(raw)) throw new Error('annotations は配列である必要があります');
    return raw.map(validateAnnotation);
  }

  function normaliseServerObjects(raw, media) {
    if (!Array.isArray(raw)) {
      throw new Error('objects は配列である必要があります');
    }

    var seen = new Set();
    return raw.map(function (item) {
      var id = item && item.id;
      if (!Number.isSafeInteger(id) || id < 1 || id > 16) {
        throw new Error('オブジェクト ID が正しくありません');
      }
      var key = objectKey(id);
      if (seen.has(key)) {
        throw new Error('オブジェクト ID が重複しています');
      }
      seen.add(key);

      var label = item && typeof item.label === 'string' ? item.label.trim() : '';
      if (!label) throw new Error('オブジェクトのラベルが正しくありません');
      if (!Array.isArray(item.prompts)) {
        throw new Error('オブジェクトの prompts が正しくありません');
      }
      var prompts = item.prompts.map(function (prompt) {
        var frameIndex = prompt && prompt.frame_index;
        var x = prompt && prompt.x;
        var y = prompt && prompt.y;
        var pointLabel = prompt && prompt.label;
        if (!Number.isSafeInteger(frameIndex) || frameIndex < 0 ||
            frameIndex >= media.frame_count || !Number.isFinite(x) ||
            !Number.isFinite(y) || x < 0 || x > 1 || y < 0 || y > 1 ||
            !Number.isSafeInteger(pointLabel) || (pointLabel !== 0 && pointLabel !== 1)) {
          throw new Error('オブジェクトのプロンプトが正しくありません');
        }
        return { frame_index: frameIndex, x: x, y: y, label: pointLabel };
      });
      return { key: key, id: id, label: label, prompts: prompts };
    });
  }

  function buildObjectRegistry(serverObjects, annotationsByFrame) {
    var next = new Map();

    serverObjects.forEach(function (serverObject) {
      var old = state.objects.get(serverObject.key);
      next.set(serverObject.key, {
        key: serverObject.key,
        id: serverObject.id,
        label: serverObject.label,
        prompts: serverObject.prompts.map(function (prompt) {
          return Object.assign({}, prompt);
        }),
        visible: !old || old.visible !== false,
        colour: colourForObject(serverObject.key),
        frames: new Set()
      });
    });

    annotationsByFrame.forEach(function (annotations, frameIndex) {
      annotations.forEach(function (annotation) {
        var key = objectKey(annotation.object_id);
        var entry = next.get(key);
        if (!entry) {
          throw new Error('未登録オブジェクトのマスクが返されました');
        }
        entry.frames.add(frameIndex);
      });
    });

    return next;
  }

  function hydrateSelectedObjectPrompts() {
    var entry = state.selectedObjectKey ? state.objects.get(state.selectedObjectKey) : null;
    var width = state.media ? state.media.width : 0;
    var height = state.media ? state.media.height : 0;
    state.points = entry && width > 0 && height > 0 ? entry.prompts
      .filter(function (prompt) { return prompt.frame_index === state.frameIndex; })
      .map(function (prompt) {
        return {
          x: Math.max(0, Math.min(width - 1, prompt.x * width)),
          y: Math.max(0, Math.min(height - 1, prompt.y * height)),
          label: prompt.label
        };
      }) : [];
    state.pointer = null;
    elements.pointCount.textContent = String(state.points.length);
  }

  function annotationForObjectAtCurrentFrame(key) {
    var annotations = state.annotationsByFrame.get(state.frameIndex) || [];
    for (var i = 0; i < annotations.length; i += 1) {
      if (objectKey(annotations[i].object_id) === key) return annotations[i];
    }
    return null;
  }

  function renderObjectList() {
    elements.objectList.textContent = '';
    var entries = Array.from(state.objects.values()).sort(function (a, b) {
      return compareObjectKeys(a.key, b.key);
    });

    entries.forEach(function (entry) {
      var row = document.createElement('li');
      row.className = 'object-row' +
        (state.selectedObjectKey === entry.key ? ' selected' : '');
      row.dataset.objectId = entry.key;
      row.dataset.testid = 'object-row';
      row.tabIndex = 0;
      row.setAttribute('role', 'button');
      row.setAttribute('aria-label', 'オブジェクト ' + entry.key + ' ' + entry.label);

      var visible = document.createElement('input');
      visible.type = 'checkbox';
      visible.checked = entry.visible;
      visible.setAttribute('aria-label', 'オブジェクト ' + entry.key + ' のマスクを表示');
      visible.dataset.testid = 'object-visibility';
      visible.addEventListener('click', function (event) { event.stopPropagation(); });
      visible.addEventListener('change', function () {
        entry.visible = visible.checked;
        draw();
      });

      var swatch = document.createElement('span');
      swatch.className = 'object-swatch';
      swatch.style.background = colourCss(entry.colour);
      swatch.setAttribute('aria-hidden', 'true');

      var name = document.createElement('div');
      name.className = 'object-name';
      var label = document.createElement('strong');
      label.textContent = entry.label;
      var id = document.createElement('span');
      id.textContent = 'ID ' + entry.key;
      name.append(label, id);

      var current = annotationForObjectAtCurrentFrame(entry.key);
      var meta = document.createElement('span');
      meta.className = 'object-meta';
      meta.textContent = current ?
        Math.round(current.area).toLocaleString('ja-JP') + ' px' :
        entry.frames.size + ' frames';
      if (current) {
        meta.title = (current.score === null ? 'score unavailable' :
          'score ' + current.score.toFixed(3)) +
          ' / bbox ' + current.bbox.join(', ');
      }

      function select() { selectObject(entry.key); }
      row.addEventListener('click', select);
      row.addEventListener('keydown', function (event) {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          select();
        }
      });
      row.append(visible, swatch, name, meta);
      elements.objectList.appendChild(row);
    });

    elements.objectCount.textContent = String(entries.length);
    elements.objectEmpty.hidden = entries.length > 0;
    syncControls();
  }

  function selectObject(key) {
    var entry = state.objects.get(key);
    if (!entry || state.busy || state.playing) return;
    state.selectedObjectKey = key;
    elements.objectId.value = String(entry.id);
    elements.objectLabel.value = entry.label;
    hydrateSelectedObjectPrompts();
    renderObjectList();
    draw();
    setStatus('オブジェクト ' + key + ' を選択しました');
  }

  function validateFrameIndex(value) {
    var index = Number(value);
    if (!Number.isSafeInteger(index) || !state.media ||
        index < 0 || index >= state.media.frame_count) {
      throw new Error('フレーム番号が範囲外です');
    }
    return index;
  }

  function formatTime(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
    var minutes = Math.floor(seconds / 60);
    var rest = seconds - minutes * 60;
    return String(minutes).padStart(2, '0') + ':' +
      rest.toFixed(2).padStart(5, '0');
  }

  function updateMediaUi() {
    if (!state.media) {
      elements.mediaSummary.textContent = 'メディア未読み込み';
      elements.timeline.hidden = true;
      elements.frameSlider.min = '0';
      elements.frameSlider.max = '0';
      elements.frameSlider.value = '0';
      elements.frameReadout.textContent = '1 / 1';
      elements.timeReadout.textContent = '00:00.00';
      elements.emptyState.hidden = false;
      return;
    }

    var media = state.media;
    var fpsText = media.fps > 0 ? ' · ' + media.fps.toFixed(2) + ' fps' : '';
    elements.mediaSummary.textContent = media.name + ' · ' +
      media.width + '×' + media.height + ' · ' +
      media.frame_count + ' frame' + (media.frame_count === 1 ? '' : 's') + fpsText;
    elements.mediaSummary.title = elements.mediaSummary.textContent;
    elements.timeline.hidden = media.frame_count <= 1;
    elements.frameSlider.min = '0';
    elements.frameSlider.max = String(media.frame_count - 1);
    elements.frameSlider.value = String(state.frameIndex);
    elements.frameReadout.textContent =
      String(state.frameIndex + 1) + ' / ' + String(media.frame_count);
    elements.timeReadout.textContent =
      formatTime(media.fps > 0 ? state.frameIndex / media.fps : 0);
    elements.emptyState.hidden = true;
  }

  function updateTimeline() {
    updateMediaUi();
    if (!state.media) return;
    elements.frameSlider.value = String(state.frameIndex);
    elements.frameReadout.textContent =
      String(state.frameIndex + 1) + ' / ' + String(state.media.frame_count);
    elements.timeReadout.textContent =
      formatTime(state.media.fps > 0 ? state.frameIndex / state.media.fps : 0);
  }

  async function loadImageElement(url) {
    var image = new Image();
    image.decoding = 'async';
    var loaded = new Promise(function (resolve, reject) {
      image.onload = resolve;
      image.onerror = function () { reject(new Error('フレーム画像を読み込めません')); };
    });
    image.src = url;
    if (typeof image.decode === 'function') {
      try {
        await image.decode();
        return image;
      } catch (_decodeError) {
        await loaded;
        return image;
      }
    }
    await loaded;
    return image;
  }

  async function loadFrameImage(index) {
    index = validateFrameIndex(index);
    var response = await apiFetch('/api/frame/' + encodeURIComponent(String(index)));
    var blob = await response.blob();
    if (!blob.size) throw new Error('空のフレーム画像が返されました');
    var url = URL.createObjectURL(blob);
    var image;
    try {
      image = await loadImageElement(url);
    } catch (error) {
      URL.revokeObjectURL(url);
      throw error;
    }

    var oldUrl = state.imageUrl;
    state.image = image;
    state.imageUrl = url;
    state.frameIndex = index;
    if (oldUrl) URL.revokeObjectURL(oldUrl);
    // A decoded RGBA mask can be several megabytes. Keep layers for only the
    // frame being displayed so playback cannot retain every video mask.
    state.layerCache = new WeakMap();
    hydrateSelectedObjectPrompts();
    updateTimeline();
    renderObjectList();
    draw();
  }

  function stopPlayback() {
    if (!state.playing) return;
    state.playing = false;
    state.playToken += 1;
    syncControls();
  }

  function resetLocalState() {
    stopPlayback();
    if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
    state.projectId = null;
    state.media = null;
    state.frameIndex = 0;
    state.image = null;
    state.imageUrl = null;
    state.points = [];
    state.annotationsByFrame = new Map();
    state.propagationDirty = false;
    state.objects = new Map();
    state.colourByObject = new Map();
    state.nextColour = 0;
    state.selectedObjectKey = null;
    state.layerCache = new WeakMap();
    state.pointer = null;
    state.view = { scale: 1, x: 0, y: 0, fitScale: 1, placed: false };
    elements.objectId.value = '1';
    elements.objectLabel.value = 'object';
    elements.mediaInput.value = '';
    elements.pointCount.textContent = '0';
    updateMediaUi();
    renderObjectList();
    resizeCanvas();
    draw();
  }

  async function loadMedia(file) {
    if (!file) return;
    await runExclusive('メディアをアップロードしています…', async function () {
      var maxFrames = Number(elements.maxFrames.value);
      if (!Number.isSafeInteger(maxFrames) || maxFrames < 1) {
        throw new Error('最大フレーム数は 1 以上の整数で指定してください');
      }
      var configuredMax = Number(elements.maxFrames.max);
      if (!Number.isSafeInteger(configuredMax) || configuredMax < 1) configuredMax = 60;
      maxFrames = Math.min(maxFrames, configuredMax);
      elements.maxFrames.value = String(maxFrames);

      var query = '?filename=' + encodeURIComponent(file.name || 'media') +
        '&max_frames=' + encodeURIComponent(String(maxFrames));
      var raw = await apiJson('/api/media' + query, {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: file
      });
      var media = validateMedia(raw);
      var projectId = validateProjectId(raw, true);

      if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
      state.projectId = projectId;
      state.media = media;
      state.frameIndex = 0;
      state.image = null;
      state.imageUrl = null;
      state.points = [];
      state.annotationsByFrame = new Map();
      state.propagationDirty = false;
      state.objects = new Map();
      state.colourByObject = new Map();
      state.nextColour = 0;
      state.selectedObjectKey = null;
      state.layerCache = new WeakMap();
      state.view.placed = false;
      elements.objectId.value = '1';
      elements.objectLabel.value = 'object';
      elements.pointCount.textContent = '0';
      updateMediaUi();
      renderObjectList();

      setStatus('最初のフレームを読み込んでいます…', 'working');
      await loadFrameImage(0);
      fitView();
      draw();
      setStatus(
        media.kind === 'video' || media.frame_count > 1 ?
          '動画を読み込みました。対象をクリックしてセグメントしてください。' :
          '画像を読み込みました。対象をクリックしてセグメントしてください。',
        'success'
      );
    });
    elements.mediaInput.value = '';
  }

  async function changeFrame(index) {
    if (!state.media) return;
    index = Math.max(0, Math.min(state.media.frame_count - 1, Number(index)));
    if (!Number.isSafeInteger(index) || index === state.frameIndex) {
      updateTimeline();
      return;
    }
    await runExclusive('フレーム ' + String(index + 1) + ' を読み込んでいます…', async function () {
      await loadFrameImage(index);
      setStatus('フレーム ' + String(index + 1) + ' / ' +
        String(state.media.frame_count));
    });
  }

  function wait(milliseconds) {
    return new Promise(function (resolve) { window.setTimeout(resolve, milliseconds); });
  }

  async function startPlayback() {
    if (!isVideo()) return;
    if (state.playing) {
      stopPlayback();
      setStatus('再生を一時停止しました');
      return;
    }
    if (state.busy) return;

    if (state.frameIndex >= state.media.frame_count - 1) {
      await changeFrame(0);
      if (state.busy) return;
    }

    state.playing = true;
    var token = ++state.playToken;
    syncControls();
    setStatus('再生中…', 'working');

    try {
      while (state.playing && token === state.playToken &&
             state.frameIndex < state.media.frame_count - 1) {
        var started = performance.now();
        state.busy = true;
        syncControls();
        await loadFrameImage(state.frameIndex + 1);
        state.busy = false;
        syncControls();
        if (!state.playing || token !== state.playToken) break;
        var interval = state.media.fps > 0 ? 1000 / state.media.fps : 100;
        var remaining = Math.max(0, interval - (performance.now() - started));
        if (remaining) await wait(remaining);
      }
      if (token === state.playToken && state.frameIndex >= state.media.frame_count - 1) {
        setStatus('動画の最後まで再生しました', 'success');
      }
    } catch (error) {
      showError(error);
    } finally {
      if (token === state.playToken) state.playing = false;
      state.busy = false;
      syncControls();
    }
  }

  function currentObjectInput() {
    var id = Number(elements.objectId.value);
    if (!Number.isSafeInteger(id) || id < 1 || id > 16) {
      throw new Error('オブジェクト ID は 1〜16 の整数で指定してください');
    }
    var label = elements.objectLabel.value.trim();
    if (!label) {
      label = 'object';
      elements.objectLabel.value = label;
    }
    return { id: id, label: label };
  }

  async function segmentCurrentFrame() {
    if (!state.media || !state.points.length) return;
    await runExclusive('セグメンテーションを実行しています…', async function () {
      var object = currentObjectInput();
      var requestedFrame = state.frameIndex;
      var payload = {
        frame_index: requestedFrame,
        object_id: object.id,
        label: object.label,
        points: state.points.map(function (point) {
          return {
            x: point.x / state.media.width,
            y: point.y / state.media.height,
            label: point.label
          };
        })
      };
      var result = await apiJson('/api/segment', jsonOptions('POST', payload));
      var responseFrame = validateFrameIndex(result && result.frame_index);
      if (responseFrame !== requestedFrame) {
        throw new Error('別のフレームの結果が返されました');
      }
      var invalidated = result && result.invalidated_frames;
      if (invalidated !== undefined && !Array.isArray(invalidated)) {
        throw new Error('無効化フレームの形式が正しくありません');
      }
      var invalidatedFrames = (invalidated || []).map(validateFrameIndex);
      var nextAnnotations = new Map(state.annotationsByFrame);
      invalidatedFrames.forEach(function (frameIndex) {
        nextAnnotations.delete(frameIndex);
      });
      nextAnnotations.set(responseFrame, normaliseFrameAnnotations(result.annotations));
      var serverObjects = normaliseServerObjects(result && result.objects, state.media);
      var nextObjects = buildObjectRegistry(serverObjects, nextAnnotations);
      var selectedKey = objectKey(object.id);
      if (!nextObjects.has(selectedKey)) {
        throw new Error('更新したオブジェクトが登録されていません');
      }
      if (!result || typeof result.propagation_required !== 'boolean') {
        throw new Error('伝播状態の形式が正しくありません');
      }

      state.annotationsByFrame = nextAnnotations;
      state.objects = nextObjects;
      state.propagationDirty = result.propagation_required;
      state.layerCache = new WeakMap();
      state.selectedObjectKey = selectedKey;
      var selected = state.objects.get(state.selectedObjectKey);
      elements.objectId.value = String(selected.id);
      elements.objectLabel.value = selected.label;
      hydrateSelectedObjectPrompts();
      renderObjectList();
      draw();

      var current = annotationForObjectAtCurrentFrame(state.selectedObjectKey);
      if (current) {
        var scoreText = current.score === null ? '' :
          ' · score ' + current.score.toFixed(3);
        var propagationText = state.propagationDirty ?
          ' · 書き出し前に動画全体へ再伝播してください' : '';
        setStatus('オブジェクト ' + object.id + ' を更新しました' +
          scoreText + ' · ' + Math.round(current.area).toLocaleString('ja-JP') +
          ' px' + propagationText, state.propagationDirty ? 'warning' : 'success');
      } else {
        setStatus('セグメンテーション結果に対象オブジェクトがありません', 'warning');
      }
    });
  }

  async function propagateAnnotations() {
    if (!isVideo() || !hasAnyAnnotations()) return;
    await runExclusive('動画全体へマスクを伝播しています…', async function () {
      var result = await apiJson('/api/propagate', jsonOptions('POST', {}));
      if (!result || !Array.isArray(result.frames)) {
        throw new Error('伝播結果の frames が正しくありません');
      }

      var staged = [];
      var frameIndices = new Set();
      result.frames.forEach(function (frame) {
        var index = validateFrameIndex(frame && frame.frame_index);
        if (frameIndices.has(index)) {
          throw new Error('伝播結果のフレーム番号が重複しています');
        }
        frameIndices.add(index);
        staged.push([index, normaliseFrameAnnotations(frame.annotations)]);
      });

      var nextAnnotations = new Map(staged);
      var serverObjects = normaliseServerObjects(result.objects, state.media);
      var nextObjects = buildObjectRegistry(serverObjects, nextAnnotations);
      if (typeof result.propagation_required !== 'boolean') {
        throw new Error('伝播状態の形式が正しくありません');
      }

      state.annotationsByFrame = nextAnnotations;
      state.objects = nextObjects;
      state.propagationDirty = result.propagation_required;
      if (state.selectedObjectKey && !state.objects.has(state.selectedObjectKey)) {
        state.selectedObjectKey = null;
      }
      if (state.selectedObjectKey) {
        var selected = state.objects.get(state.selectedObjectKey);
        elements.objectId.value = String(selected.id);
        elements.objectLabel.value = selected.label;
      }
      hydrateSelectedObjectPrompts();
      state.layerCache = new WeakMap();
      renderObjectList();
      draw();

      var maskCount = staged.reduce(function (sum, item) {
        return sum + item[1].length;
      }, 0);
      setStatus(String(staged.length) + ' フレームへ伝播しました · ' +
        String(maskCount) + ' マスク', 'success');
    });
  }

  function downloadNameFrom(response) {
    var disposition = response.headers.get('Content-Disposition') || '';
    var utf8 = disposition.match(/filename\*=UTF-8''([^;]+)/i);
    if (utf8) {
      try { return decodeURIComponent(utf8[1].replace(/["']/g, '')); }
      catch (_error) { /* fall through */ }
    }
    var plain = disposition.match(/filename=["]?([^";]+)["]?/i);
    if (plain) return plain[1].trim();
    var base = state.media ? state.media.name.replace(/\.[^.]+$/, '') : 'annotations';
    return base + '.coco.json';
  }

  async function exportCoco() {
    if (!state.media || !hasAnyAnnotations()) return;
    await runExclusive('COCO JSON を準備しています…', async function () {
      var response = await apiFetch('/api/export');
      var blob = await response.blob();
      if (!blob.size) throw new Error('書き出しデータが空です');
      var url = URL.createObjectURL(blob);
      var anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = downloadNameFrom(response);
      anchor.style.display = 'none';
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
      setStatus('COCO JSON を書き出しました', 'success');
    });
  }

  async function resetSession() {
    if (!state.media) return;
    await runExclusive('セッションをリセットしています…', async function () {
      await apiFetch('/api/reset', jsonOptions('POST', {}));
      state.points = [];
      state.annotationsByFrame = new Map();
      state.propagationDirty = false;
      state.objects = new Map();
      state.colourByObject = new Map();
      state.nextColour = 0;
      state.selectedObjectKey = null;
      state.layerCache = new WeakMap();
      elements.objectId.value = '1';
      elements.objectLabel.value = 'object';
      elements.pointCount.textContent = '0';
      renderObjectList();
      draw();
      setStatus('アノテーションをリセットしました', 'success');
    });
  }

  function clearPrompts(silent) {
    state.points = [];
    state.pointer = null;
    elements.pointCount.textContent = '0';
    draw();
    syncControls();
    if (!silent) setStatus('ポイントを消去しました');
  }

  function undoPrompt() {
    if (!state.points.length || state.busy || state.playing) return;
    state.points.pop();
    elements.pointCount.textContent = String(state.points.length);
    draw();
    syncControls();
    setStatus('最後のポイントを戻しました');
  }

  function addPromptPoint(point, negative) {
    state.points.push({
      x: Math.max(0, Math.min(state.media.width - 1, Math.round(point.x))),
      y: Math.max(0, Math.min(state.media.height - 1, Math.round(point.y))),
      label: negative ? 0 : 1
    });
    elements.pointCount.textContent = String(state.points.length);
    draw();
    syncControls();
    setStatus(negative ? '除外点を追加しました' : '対象点を追加しました');
  }

  function chooseNewObject() {
    if (!state.media || state.busy || state.playing) return;
    var used = new Set();
    state.objects.forEach(function (entry) {
      var numeric = Number(entry.id);
      if (Number.isSafeInteger(numeric)) used.add(numeric);
    });
    var nextId = 1;
    while (nextId <= 16 && used.has(nextId)) nextId += 1;
    if (nextId > 16) {
      setStatus('オブジェクト ID は最大 16 です', 'warning');
      return;
    }
    elements.objectId.value = String(nextId);
    elements.objectLabel.value = 'object';
    state.selectedObjectKey = null;
    clearPrompts(true);
    renderObjectList();
    draw();
    setStatus('新しいオブジェクト ' + String(nextId) + ' を準備しました');
  }

  function setTool(tool) {
    if (tool !== 'prompt' && tool !== 'pan') return;
    state.tool = tool;
    var prompt = tool === 'prompt';
    elements.promptToolButton.classList.toggle('active', prompt);
    elements.promptToolButton.setAttribute('aria-pressed', String(prompt));
    elements.panToolButton.classList.toggle('active', !prompt);
    elements.panToolButton.setAttribute('aria-pressed', String(!prompt));
    elements.canvas.classList.toggle('pan-cursor', !prompt);
  }

  function resizeCanvas() {
    var rect = elements.canvas.getBoundingClientRect();
    var dpr = Math.min(window.devicePixelRatio || 1, 3);
    var width = Math.max(1, Math.round(rect.width * dpr));
    var height = Math.max(1, Math.round(rect.height * dpr));
    state.dpr = dpr;
    if (elements.canvas.width !== width) elements.canvas.width = width;
    if (elements.canvas.height !== height) elements.canvas.height = height;
    if (state.media && !state.view.placed && rect.width > 0 && rect.height > 0) {
      fitView();
    }
    draw();
  }

  function fitView() {
    if (!state.media) return;
    var width = elements.canvas.clientWidth;
    var height = elements.canvas.clientHeight;
    if (width <= 0 || height <= 0) return;
    var padding = Math.min(30, Math.max(10, Math.min(width, height) * .04));
    var scale = Math.min(
      Math.max(1, width - padding * 2) / state.media.width,
      Math.max(1, height - padding * 2) / state.media.height
    );
    state.view.scale = scale;
    state.view.fitScale = scale;
    state.view.x = (width - state.media.width * scale) / 2;
    state.view.y = (height - state.media.height * scale) / 2;
    state.view.placed = true;
    updateZoomLabel();
    draw();
  }

  function updateZoomLabel() {
    var basis = state.view.fitScale || 1;
    elements.zoomLevel.value = Math.round(state.view.scale / basis * 100) + '%';
    elements.zoomLevel.textContent = elements.zoomLevel.value;
  }

  function zoomAt(factor, clientX, clientY) {
    if (!state.media) return;
    var rect = elements.canvas.getBoundingClientRect();
    var pointX = clientX === undefined ? rect.width / 2 : clientX - rect.left;
    var pointY = clientY === undefined ? rect.height / 2 : clientY - rect.top;
    var previous = state.view.scale;
    var minimum = Math.max(.01, state.view.fitScale * .15);
    var maximum = Math.max(16, state.view.fitScale * 24);
    var next = Math.max(minimum, Math.min(maximum, previous * factor));
    var imageX = (pointX - state.view.x) / previous;
    var imageY = (pointY - state.view.y) / previous;
    state.view.x = pointX - imageX * next;
    state.view.y = pointY - imageY * next;
    state.view.scale = next;
    state.view.placed = true;
    updateZoomLabel();
    draw();
  }

  function canvasPoint(clientX, clientY) {
    var rect = elements.canvas.getBoundingClientRect();
    return {
      x: (clientX - rect.left - state.view.x) / state.view.scale,
      y: (clientY - rect.top - state.view.y) / state.view.scale
    };
  }

  function isInsideMedia(point) {
    return state.media && point.x >= 0 && point.y >= 0 &&
      point.x < state.media.width && point.y < state.media.height;
  }

  function maskLayer(annotation, colour) {
    var cached = state.layerCache.get(annotation);
    if (cached) return cached;
    var height = annotation.segmentation.size[0];
    var width = annotation.segmentation.size[1];
    var mask = decodeRLE(annotation.segmentation.counts, width, height);
    if (!mask) return null;

    var layer = document.createElement('canvas');
    layer.width = width;
    layer.height = height;
    var layerContext = layer.getContext('2d');
    var imageData = layerContext.createImageData(width, height);
    var pixels = imageData.data;
    for (var i = 0; i < mask.length; i += 1) {
      if (!mask[i]) continue;
      var offset = i * 4;
      pixels[offset] = colour[0];
      pixels[offset + 1] = colour[1];
      pixels[offset + 2] = colour[2];
      pixels[offset + 3] = 118;
    }
    layerContext.putImageData(imageData, 0, 0);
    state.layerCache.set(annotation, layer);
    return layer;
  }

  function drawMarker(point) {
    var radius = 6 / state.view.scale;
    var positive = point.label === 1;
    context.beginPath();
    context.arc(point.x, point.y, radius, 0, Math.PI * 2);
    context.fillStyle = positive ? '#61c779' : '#f16d70';
    context.fill();
    context.lineWidth = Math.max(.8, 1.6 / state.view.scale);
    context.strokeStyle = '#ffffff';
    context.stroke();

    context.beginPath();
    context.moveTo(point.x - radius * .45, point.y);
    context.lineTo(point.x + radius * .45, point.y);
    if (positive) {
      context.moveTo(point.x, point.y - radius * .45);
      context.lineTo(point.x, point.y + radius * .45);
    }
    context.lineWidth = Math.max(.8, 1.5 / state.view.scale);
    context.strokeStyle = '#ffffff';
    context.stroke();
  }

  function draw() {
    var pixelWidth = elements.canvas.width;
    var pixelHeight = elements.canvas.height;
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, pixelWidth, pixelHeight);
    if (!state.media || !state.image) return;

    context.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
    context.save();
    context.translate(state.view.x, state.view.y);
    context.scale(state.view.scale, state.view.scale);
    context.imageSmoothingEnabled = state.view.scale < 4;
    context.drawImage(state.image, 0, 0, state.media.width, state.media.height);

    var annotations = state.annotationsByFrame.get(state.frameIndex) || [];
    annotations.forEach(function (annotation) {
      var key = objectKey(annotation.object_id);
      var entry = state.objects.get(key);
      if (!entry || !entry.visible) return;
      var layer = maskLayer(annotation, entry.colour);
      if (!layer) return;
      context.drawImage(layer, 0, 0, state.media.width, state.media.height);

      if (state.selectedObjectKey === key) {
        var maskHeight = annotation.segmentation.size[0];
        var maskWidth = annotation.segmentation.size[1];
        var scaleX = state.media.width / maskWidth;
        var scaleY = state.media.height / maskHeight;
        context.lineWidth = Math.max(.8, 2 / state.view.scale);
        context.strokeStyle = colourCss(entry.colour);
        context.setLineDash([6 / state.view.scale, 4 / state.view.scale]);
        context.strokeRect(
          annotation.bbox[0] * scaleX,
          annotation.bbox[1] * scaleY,
          annotation.bbox[2] * scaleX,
          annotation.bbox[3] * scaleY
        );
        context.setLineDash([]);
      }
    });

    state.points.forEach(drawMarker);
    context.restore();
  }

  function pointerDown(event) {
    if (!state.media || !state.image || state.busy || state.playing) return;
    var pan = state.tool === 'pan' || event.altKey || event.button === 1;
    if (pan) {
      event.preventDefault();
      state.pointer = {
        id: event.pointerId,
        kind: 'pan',
        startX: event.clientX,
        startY: event.clientY,
        viewX: state.view.x,
        viewY: state.view.y
      };
      elements.canvas.classList.add('panning');
      elements.canvas.setPointerCapture(event.pointerId);
      return;
    }

    if (event.button !== 0 && event.button !== 2) return;
    event.preventDefault();
    state.pointer = {
      id: event.pointerId,
      kind: 'point',
      startX: event.clientX,
      startY: event.clientY,
      negative: event.button === 2 || event.shiftKey
    };
    elements.canvas.setPointerCapture(event.pointerId);
  }

  function pointerMove(event) {
    if (!state.pointer || state.pointer.id !== event.pointerId ||
        state.pointer.kind !== 'pan') return;
    state.view.x = state.pointer.viewX + event.clientX - state.pointer.startX;
    state.view.y = state.pointer.viewY + event.clientY - state.pointer.startY;
    state.view.placed = true;
    draw();
  }

  function pointerUp(event) {
    var pointer = state.pointer;
    if (!pointer || pointer.id !== event.pointerId) return;
    state.pointer = null;
    if (elements.canvas.hasPointerCapture(event.pointerId)) {
      elements.canvas.releasePointerCapture(event.pointerId);
    }
    elements.canvas.classList.remove('panning');
    if (pointer.kind !== 'point') return;

    var distance = Math.hypot(
      event.clientX - pointer.startX,
      event.clientY - pointer.startY
    );
    if (distance > 6) return;
    var point = canvasPoint(event.clientX, event.clientY);
    if (!isInsideMedia(point)) {
      setStatus('画像の内側をクリックしてください', 'warning');
      return;
    }
    addPromptPoint(point, pointer.negative || event.shiftKey);
  }

  function pointerCancel(event) {
    if (!state.pointer || state.pointer.id !== event.pointerId) return;
    state.pointer = null;
    elements.canvas.classList.remove('panning');
  }

  async function checkHealth() {
    elements.healthDot.className = 'health-dot';
    elements.healthText.textContent = '接続確認中';
    try {
      var health = await apiJson('/api/health');
      state.healthy = true;
      elements.healthDot.className = 'health-dot ok';
      elements.healthText.textContent = 'API 接続済み';
      var configuredMax = Number(health && health.max_video_frames);
      if (Number.isSafeInteger(configuredMax) && configuredMax > 0) {
        elements.maxFrames.max = String(configuredMax);
        if (Number(elements.maxFrames.value) > configuredMax) {
          elements.maxFrames.value = String(configuredMax);
        }
      }
      var restored = await restoreProject(await apiJson('/api/project'));
      var restoredMessage = state.propagationDirty ?
        '保存済みセッションを復元しました。書き出し前に動画全体へ再伝播してください。' :
        '保存済みのアノテーションセッションを復元しました。';
      setStatus(restored ? restoredMessage :
        '準備完了。画像または動画を選択してください。',
      restored && state.propagationDirty ? 'warning' : 'success');
    } catch (error) {
      state.healthy = false;
      elements.healthDot.className = 'health-dot error';
      elements.healthText.textContent = 'API 未接続';
      showError(error);
    }
    syncControls();
  }

  function bindEvents() {
    elements.mediaInput.addEventListener('change', function () {
      var file = elements.mediaInput.files && elements.mediaInput.files[0];
      loadMedia(file);
    });

    var picker = elements.mediaInput.closest('.file-picker');
    if (picker) {
      ['dragenter', 'dragover'].forEach(function (name) {
        picker.addEventListener(name, function (event) {
          if (elements.mediaInput.disabled) return;
          event.preventDefault();
          picker.classList.add('dragging');
        });
      });
      ['dragleave', 'drop'].forEach(function (name) {
        picker.addEventListener(name, function (event) {
          picker.classList.remove('dragging');
          if (name !== 'drop' || elements.mediaInput.disabled) return;
          event.preventDefault();
          var file = event.dataTransfer && event.dataTransfer.files[0];
          loadMedia(file);
        });
      });
    }

    elements.resetButton.addEventListener('click', resetSession);
    elements.newObjectButton.addEventListener('click', chooseNewObject);
    elements.undoButton.addEventListener('click', undoPrompt);
    elements.clearButton.addEventListener('click', function () { clearPrompts(false); });
    elements.segmentButton.addEventListener('click', segmentCurrentFrame);
    elements.propagateButton.addEventListener('click', propagateAnnotations);
    elements.exportButton.addEventListener('click', exportCoco);
    elements.clearMessageButton.addEventListener('click', clearMessage);

    elements.objectId.addEventListener('change', function () {
      var value = Number(elements.objectId.value);
      var key = Number.isSafeInteger(value) ? objectKey(value) : '';
      if (key !== state.selectedObjectKey) {
        if (state.objects.has(key)) {
          selectObject(key);
        } else {
          state.selectedObjectKey = null;
          clearPrompts(true);
          renderObjectList();
          draw();
        }
      }
    });

    elements.promptToolButton.addEventListener('click', function () { setTool('prompt'); });
    elements.panToolButton.addEventListener('click', function () { setTool('pan'); });
    elements.fitButton.addEventListener('click', fitView);
    elements.zoomInButton.addEventListener('click', function () { zoomAt(1.25); });
    elements.zoomOutButton.addEventListener('click', function () { zoomAt(.8); });

    elements.previousButton.addEventListener('click', function () {
      changeFrame(state.frameIndex - 1);
    });
    elements.nextButton.addEventListener('click', function () {
      changeFrame(state.frameIndex + 1);
    });
    elements.playButton.addEventListener('click', startPlayback);
    elements.frameSlider.addEventListener('change', function () {
      changeFrame(Number(elements.frameSlider.value));
    });

    elements.canvas.addEventListener('contextmenu', function (event) {
      event.preventDefault();
    });
    elements.canvas.addEventListener('pointerdown', pointerDown);
    elements.canvas.addEventListener('pointermove', pointerMove);
    elements.canvas.addEventListener('pointerup', pointerUp);
    elements.canvas.addEventListener('pointercancel', pointerCancel);
    elements.canvas.addEventListener('wheel', function (event) {
      if (!state.media || state.busy || state.playing) return;
      event.preventDefault();
      zoomAt(Math.exp(-event.deltaY * .0015), event.clientX, event.clientY);
    }, { passive: false });

    window.addEventListener('keydown', function (event) {
      var target = event.target;
      var typing = target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement || target.isContentEditable;
      if (typing) return;

      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
        event.preventDefault();
        undoPrompt();
      } else if (event.key === 'Escape') {
        clearPrompts(false);
      } else if (event.code === 'Space') {
        event.preventDefault();
        segmentCurrentFrame();
      } else if (event.key === 'ArrowLeft' && isVideo()) {
        event.preventDefault();
        changeFrame(state.frameIndex - 1);
      } else if (event.key === 'ArrowRight' && isVideo()) {
        event.preventDefault();
        changeFrame(state.frameIndex + 1);
      } else if (event.key.toLowerCase() === 'f') {
        event.preventDefault();
        fitView();
      }
    });

    window.addEventListener('beforeunload', function () {
      if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
    });

    if (typeof ResizeObserver === 'function') {
      new ResizeObserver(resizeCanvas).observe(elements.canvasWrap);
    } else {
      window.addEventListener('resize', resizeCanvas);
    }
  }

  bindEvents();
  setTool('prompt');
  resetLocalState();
  syncControls();
  checkHealth();
}());
