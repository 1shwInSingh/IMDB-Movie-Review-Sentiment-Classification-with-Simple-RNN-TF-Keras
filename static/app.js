/**
 * IMDB Bidirectional LSTM Sentiment Analysis Web Application
 * Interactive UI Engine: Particles, Audio Synthesis, Charts, Real-Time Inference
 */

(function () {
  'use strict';

  // ==========================================
  // 1. SOUND SYNTHESIS ENGINE (Web Audio API)
  // ==========================================
  class SoundFX {
    constructor() {
      this.enabled = localStorage.getItem('sound_enabled') !== 'false';
      this.ctx = null;
    }

    init() {
      if (!this.ctx && typeof AudioContext !== 'undefined') {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        this.ctx = new AudioCtx();
      }
    }

    playClick() {
      if (!this.enabled) return;
      try {
        this.init();
        if (this.ctx.state === 'suspended') this.ctx.resume();
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(800, this.ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(400, this.ctx.currentTime + 0.04);
        gain.gain.setValueAtTime(0.04, this.ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.04);
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.start();
        osc.stop(this.ctx.currentTime + 0.04);
      } catch (e) { }
    }

    playPositive() {
      if (!this.enabled) return;
      try {
        this.init();
        if (this.ctx.state === 'suspended') this.ctx.resume();
        const now = this.ctx.currentTime;
        [523.25, 659.25, 783.99, 1046.5].forEach((freq, i) => {
          const osc = this.ctx.createOscillator();
          const gain = this.ctx.createGain();
          osc.type = 'triangle';
          osc.frequency.setValueAtTime(freq, now + i * 0.08);
          gain.gain.setValueAtTime(0.05, now + i * 0.08);
          gain.gain.exponentialRampToValueAtTime(0.0001, now + i * 0.08 + 0.35);
          osc.connect(gain);
          gain.connect(this.ctx.destination);
          osc.start(now + i * 0.08);
          osc.stop(now + i * 0.08 + 0.35);
        });
      } catch (e) { }
    }

    playNegative() {
      if (!this.enabled) return;
      try {
        this.init();
        if (this.ctx.state === 'suspended') this.ctx.resume();
        const now = this.ctx.currentTime;
        [440, 392, 349.23, 293.66].forEach((freq, i) => {
          const osc = this.ctx.createOscillator();
          const gain = this.ctx.createGain();
          osc.type = 'sawtooth';
          osc.frequency.setValueAtTime(freq, now + i * 0.09);
          gain.gain.setValueAtTime(0.035, now + i * 0.09);
          gain.gain.exponentialRampToValueAtTime(0.0001, now + i * 0.09 + 0.32);
          osc.connect(gain);
          gain.connect(this.ctx.destination);
          osc.start(now + i * 0.09);
          osc.stop(now + i * 0.09 + 0.32);
        });
      } catch (e) { }
    }
  }

  const sfx = new SoundFX();

  // ==========================================
  // 2. INTERACTIVE NEURAL PARTICLE CANVAS
  // ==========================================
  class NeuralMesh {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.particles = [];
      this.mouse = { x: -1000, y: -1000, radius: 120 };
      this.numParticles = 55;
      this.animId = null;
      this.resize = this.resize.bind(this);
      this.animate = this.animate.bind(this);

      window.addEventListener('resize', this.resize);
      window.addEventListener('mousemove', (e) => {
        const rect = this.canvas.getBoundingClientRect();
        this.mouse.x = e.clientX - rect.left;
        this.mouse.y = e.clientY - rect.top;
      });
      window.addEventListener('mouseleave', () => {
        this.mouse.x = -1000;
        this.mouse.y = -1000;
      });

      this.resize();
      this.initParticles();
      this.animate();
    }

    resize() {
      this.canvas.width = window.innerWidth;
      this.canvas.height = window.innerHeight;
      this.numParticles = window.innerWidth < 768 ? 30 : 60;
    }

    initParticles() {
      this.particles = [];
      for (let i = 0; i < this.numParticles; i++) {
        this.particles.push({
          x: Math.random() * this.canvas.width,
          y: Math.random() * this.canvas.height,
          vx: (Math.random() - 0.5) * 0.65,
          vy: (Math.random() - 0.5) * 0.65,
          radius: Math.random() * 2 + 1.2,
          color: Math.random() > 0.4 ? 'rgba(56, 189, 248,' : 'rgba(129, 140, 248,',
          alpha: Math.random() * 0.5 + 0.25,
          pulse: Math.random() * Math.PI * 2
        });
      }
    }

    animate() {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

      for (let i = 0; i < this.particles.length; i++) {
        const p = this.particles[i];
        p.x += p.vx;
        p.y += p.vy;
        p.pulse += 0.03;

        if (p.x < 0) p.x = this.canvas.width;
        if (p.x > this.canvas.width) p.x = 0;
        if (p.y < 0) p.y = this.canvas.height;
        if (p.y > this.canvas.height) p.y = 0;

        // Mouse avoidance/gentle pull
        const dx = this.mouse.x - p.x;
        const dy = this.mouse.y - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < this.mouse.radius) {
          const force = (1 - dist / this.mouse.radius) * 1.5;
          p.x -= (dx / dist) * force;
          p.y -= (dy / dist) * force;
        }

        const currentAlpha = p.alpha + Math.sin(p.pulse) * 0.15;
        this.ctx.beginPath();
        this.ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        this.ctx.fillStyle = `${p.color} ${Math.max(0.1, currentAlpha)})`;
        this.ctx.shadowBlur = 8;
        this.ctx.shadowColor = 'rgba(56, 189, 248, 0.4)';
        this.ctx.fill();
        this.ctx.shadowBlur = 0;

        // Draw connections
        for (let j = i + 1; j < this.particles.length; j++) {
          const p2 = this.particles[j];
          const cdx = p.x - p2.x;
          const cdy = p.y - p2.y;
          const cdist = Math.sqrt(cdx * cdx + cdy * cdy);
          const maxDist = 130;

          if (cdist < maxDist) {
            const lineAlpha = (1 - cdist / maxDist) * 0.22;
            this.ctx.beginPath();
            this.ctx.moveTo(p.x, p.y);
            this.ctx.lineTo(p2.x, p2.y);
            this.ctx.strokeStyle = `rgba(56, 189, 248, ${lineAlpha})`;
            this.ctx.lineWidth = 0.8;
            this.ctx.stroke();
          }
        }
      }

      this.animId = requestAnimationFrame(this.animate);
    }
  }

  // ==========================================
  // 3. SAMPLE PRESETS & LEXICON
  // ==========================================
  const PRESET_REVIEWS = [
    {
      category: 'Masterpiece 🌟',
      title: 'Cinematic Tour de Force',
      text: 'An extraordinary achievement in modern cinema! The direction is visionary, the cinematography is pure poetry, and the lead performances are deeply touching and unforgettable.'
    },
    {
      category: 'Disaster 💔',
      title: 'Tedious Trainwreck',
      text: 'A dreadful, painful waste of two hours. The acting is wooden, the plot is full of ridiculous holes, and the dialogue feels like a first draft written by a child.'
    },
    {
      category: 'Popcorn Fun 🍿',
      title: 'Thrilling Blockbuster',
      text: 'Pure adrenaline and thrilling escapism from start to finish! Fantastic special effects, brisk pacing, and laugh-out-loud humor make this a must-watch crowd-pleaser.'
    },
    {
      category: 'Boring Flop 😴',
      title: 'Sleep-Inducing Slog',
      text: 'A dull, utterly lifeless movie that drags endlessly. It tries to be profound but ends up pretentiously boring, hollow, and painfully predictable.'
    },
    {
      category: 'Nuanced / Mixed 🤔',
      title: 'Visual Marvel, Weak Script',
      text: 'The visuals and musical score are magnificent, yet the thin character development and clunky third-act climax prevent it from reaching greatness.'
    },
    {
      category: 'Cult Classic 🎬',
      title: 'Charming Quirky Gem',
      text: 'Hilarious, wonderfully eccentric, and loaded with heart! It is rough around the edges, but the infectious charisma and clever writing make it a cult favorite.'
    }
  ];

  const POSITIVE_LEXICON = new Set([
    'brilliant', 'wonderful', 'great', 'excellent', 'amazing', 'masterpiece', 'fantastic',
    'loved', 'love', 'best', 'superb', 'beautiful', 'touching', 'enjoyed', 'gem', 'perfect',
    'outstanding', 'visionary', 'stunning', 'thrilling', 'breathtaking', 'hilarious',
    'charming', 'clever', 'captivating', 'terrific', 'marvelous', 'poetic', 'powerful',
    'solid', 'refreshing', 'gripping', 'splendid', 'fabulous', 'entertaining', 'sweet'
  ]);

  const NEGATIVE_LEXICON = new Set([
    'terrible', 'awful', 'horrible', 'worst', 'boring', 'waste', 'poor', 'stupid',
    'bad', 'dreadful', 'pathetic', 'painful', 'crap', 'garbage', 'ridiculous',
    'annoying', 'lame', 'pointless', 'sucks', 'atrocious', 'lifeless', 'dull',
    'wooden', 'cheap', 'unwatchable', 'abysmal', 'disaster', 'mess', 'cliché',
    'cliche', 'tedious', 'hollow', 'pretentious', 'predictable', 'ruined', 'disappointing'
  ]);

  // ==========================================
  // 4. MAIN APP CONTROLLER
  // ==========================================
  document.addEventListener('DOMContentLoaded', () => {
    // Canvas init
    const canvas = document.getElementById('neuralCanvas');
    if (canvas) new NeuralMesh(canvas);

    // Audio toggle
    const soundToggleBtn = document.getElementById('soundToggleBtn');
    if (soundToggleBtn) {
      soundToggleBtn.setAttribute('aria-pressed', sfx.enabled);
      updateSoundButtonUI();
      soundToggleBtn.addEventListener('click', () => {
        sfx.enabled = !sfx.enabled;
        localStorage.setItem('sound_enabled', sfx.enabled);
        updateSoundButtonUI();
        if (sfx.enabled) sfx.playPositive();
      });
    }

    function updateSoundButtonUI() {
      if (!soundToggleBtn) return;
      soundToggleBtn.innerHTML = sfx.enabled
        ? '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 5L6 9H2v6h4l5 4V5z"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/></svg><span>Sound: ON</span>'
        : '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 5L6 9H2v6h4l5 4V5z"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg><span>Sound: OFF</span>';
      soundToggleBtn.classList.toggle('muted', !sfx.enabled);
    }

    // Interactive Stats Counter
    initStatCounters();

    // Setup Textarea & Input Counters
    const reviewInput = document.getElementById('review');
    const charCountEl = document.getElementById('charCount');
    const wordCountEl = document.getElementById('wordCount');
    const readTimeEl = document.getElementById('readTime');
    const clearBtn = document.getElementById('clearBtn');
    const pasteBtn = document.getElementById('pasteBtn');
    const predictBtn = document.getElementById('predictBtn');
    const randomSampleBtn = document.getElementById('randomSampleBtn');

    if (reviewInput) {
      reviewInput.addEventListener('input', updateTextMetrics);
      reviewInput.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
          e.preventDefault();
          runPrediction();
        }
      });
      updateTextMetrics();
    }

    function updateTextMetrics() {
      if (!reviewInput) return;
      const text = reviewInput.value;
      const chars = text.length;
      const words = text.trim() ? text.trim().split(/\s+/).length : 0;
      const readSeconds = Math.max(1, Math.ceil(words / 3.5));

      if (charCountEl) charCountEl.textContent = chars.toLocaleString();
      if (wordCountEl) wordCountEl.textContent = `${words} ${words === 1 ? 'word' : 'words'}`;
      if (readTimeEl) readTimeEl.textContent = words > 0 ? `~${readSeconds}s read` : '~0s read';

      if (clearBtn) {
        clearBtn.style.opacity = text.length > 0 ? '1' : '0.4';
        clearBtn.style.pointerEvents = text.length > 0 ? 'auto' : 'none';
      }
    }

    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        sfx.playClick();
        if (!reviewInput) return;
        reviewInput.value = '';
        reviewInput.focus();
        updateTextMetrics();
        hideResult();
        showToast('Text cleared');
      });
    }

    if (pasteBtn) {
      pasteBtn.addEventListener('click', async () => {
        sfx.playClick();
        try {
          const text = await navigator.clipboard.readText();
          if (text) {
            reviewInput.value = text;
            updateTextMetrics();
            reviewInput.focus();
            showToast('Pasted from clipboard!');
          }
        } catch (e) {
          showToast('Could not access clipboard', 'warning');
        }
      });
    }

    // Render Preset Chips
    renderPresets();

    if (randomSampleBtn) {
      randomSampleBtn.addEventListener('click', () => {
        sfx.playClick();
        const rand = PRESET_REVIEWS[Math.floor(Math.random() * PRESET_REVIEWS.length)];
        if (reviewInput) {
          reviewInput.value = rand.text;
          updateTextMetrics();
          reviewInput.classList.add('flash-highlight');
          setTimeout(() => reviewInput.classList.remove('flash-highlight'), 600);
          showToast(`Loaded: ${rand.title}`);
        }
      });
    }

    // Predict Button
    if (predictBtn) {
      predictBtn.addEventListener('click', runPrediction);
    }

    // Copy Result Action
    const copyResultBtn = document.getElementById('copyResultBtn');
    if (copyResultBtn) {
      copyResultBtn.addEventListener('click', () => {
        sfx.playClick();
        const label = document.getElementById('resultVerdictText')?.textContent || '';
        const prob = document.getElementById('probMetric')?.textContent || '';
        const conf = document.getElementById('confMetric')?.textContent || '';
        const copyText = `🎬 IMDB Bi-LSTM Prediction: ${label} | Probability: ${prob} | Confidence: ${conf}`;
        navigator.clipboard.writeText(copyText).then(() => {
          showToast('Prediction copied to clipboard!');
        });
      });
    }

    // Charts & Metrics
    initChartsAndMetrics();

    // Load History
    renderHistory();

    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    if (clearHistoryBtn) {
      clearHistoryBtn.addEventListener('click', () => {
        sfx.playClick();
        localStorage.removeItem('imdb_prediction_history');
        renderHistory();
        showToast('History cleared');
      });
    }
  });

  // ==========================================
  // 5. PREDICTION WORKFLOW & UI UPDATES
  // ==========================================
  async function runPrediction() {
    const reviewInput = document.getElementById('review');
    const predictBtn = document.getElementById('predictBtn');
    const resultBox = document.getElementById('result');
    const errorBox = document.getElementById('error');

    if (!reviewInput) return;
    const review = reviewInput.value.trim();

    if (!review) {
      sfx.playNegative();
      showToast('Please type or paste a review first!', 'warning');
      reviewInput.focus();
      reviewInput.classList.add('input-shake');
      setTimeout(() => reviewInput.classList.remove('input-shake'), 500);
      return;
    }

    sfx.playClick();
    if (errorBox) errorBox.classList.add('hidden');
    
    // Set loading state with pulse animation
    predictBtn.disabled = true;
    const originalBtnHtml = predictBtn.innerHTML;
    predictBtn.innerHTML = `
      <span class="spinner"></span>
      <span>Inferring Bi-LSTM Weights...</span>
    `;
    predictBtn.classList.add('btn-processing');

    const startTime = performance.now();

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review })
      });

      const data = await res.json();
      const clientLatency = Math.round(performance.now() - startTime);

      if (!res.ok) {
        throw new Error(data.error || 'Server returned an error.');
      }

      displayPredictionResult(data, review, clientLatency);

      if (data.label === 'POSITIVE') {
        sfx.playPositive();
        triggerConfettiEffect();
      } else {
        sfx.playNegative();
      }

      saveToHistory(review, data);

    } catch (err) {
      sfx.playNegative();
      if (errorBox) {
        errorBox.textContent = err.message || 'Failed to analyze review.';
        errorBox.classList.remove('hidden');
      }
      showToast(err.message || 'Inference error', 'error');
    } finally {
      predictBtn.disabled = false;
      predictBtn.innerHTML = originalBtnHtml;
      predictBtn.classList.remove('btn-processing');
    }
  }

  function displayPredictionResult(data, reviewText, clientLatency) {
    const resultBox = document.getElementById('result');
    if (!resultBox) return;

    const isPos = data.label === 'POSITIVE';
    const prob = typeof data.probability === 'number' ? data.probability : 0.5;
    const conf = typeof data.confidence === 'number' ? data.confidence : 50;
    const verdict = data.verdict || (isPos ? 'Positive' : 'Negative');
    const emoji = data.emoji || (isPos ? '🤩' : '💔');
    const latency = data.latency_ms || clientLatency || 20;

    // Mood Avatar
    const avatarEl = document.getElementById('moodAvatar');
    if (avatarEl) {
      avatarEl.textContent = emoji;
      avatarEl.className = `mood-avatar ${isPos ? 'mood-pos' : 'mood-neg'}`;
    }

    // Verdict Text & Tag
    const verdictEl = document.getElementById('resultVerdictText');
    const verdictBadge = document.getElementById('resultBadge');
    if (verdictEl) {
      verdictEl.textContent = `${verdict}`;
      verdictEl.className = isPos ? 'text-pos' : 'text-neg';
    }
    if (verdictBadge) {
      verdictBadge.textContent = `${data.label} (${(prob).toFixed(4)})`;
      verdictBadge.className = `badge-tag ${isPos ? 'badge-pos' : 'badge-neg'}`;
    }

    // Dual Polarity Gauge Bars
    const posPercent = Math.round(prob * 100);
    const negPercent = 100 - posPercent;

    const barPosFill = document.getElementById('barPosFill');
    const barNegFill = document.getElementById('barNegFill');
    const posPercentLabel = document.getElementById('posPercentLabel');
    const negPercentLabel = document.getElementById('negPercentLabel');

    if (barPosFill) barPosFill.style.width = `${posPercent}%`;
    if (barNegFill) barNegFill.style.width = `${negPercent}%`;
    if (posPercentLabel) posPercentLabel.textContent = `${posPercent}% Pos`;
    if (negPercentLabel) negPercentLabel.textContent = `${negPercent}% Neg`;

    // Metrics Grid
    const probMetric = document.getElementById('probMetric');
    const confMetric = document.getElementById('confMetric');
    const latencyMetric = document.getElementById('latencyMetric');

    if (probMetric) probMetric.textContent = prob.toFixed(4);
    if (confMetric) confMetric.textContent = `${conf.toFixed(1)}%`;
    if (latencyMetric) latencyMetric.textContent = `${latency}ms`;

    // Highlight sentiment tokens
    renderTokenHighlights(reviewText);

    // Show result card with pop animation
    resultBox.classList.remove('hidden');
    resultBox.classList.remove('pop-animate');
    void resultBox.offsetWidth; // trigger reflow
    resultBox.classList.add('pop-animate');

    // Scroll smoothly into view if on mobile
    if (window.innerWidth < 768) {
      resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  function hideResult() {
    const resultBox = document.getElementById('result');
    if (resultBox) resultBox.classList.add('hidden');
  }

  // ==========================================
  // 6. TOKEN SENTIMENT HIGHLIGHTING
  // ==========================================
  function renderTokenHighlights(text) {
    const container = document.getElementById('tokenChipsContainer');
    if (!container) return;

    container.innerHTML = '';
    const words = text.toLowerCase().match(/\b[a-z'-]+\b/g) || [];
    const detected = [];
    const seen = new Set();

    words.forEach(w => {
      if (seen.has(w)) return;
      if (POSITIVE_LEXICON.has(w)) {
        seen.add(w);
        detected.push({ word: w, type: 'pos' });
      } else if (NEGATIVE_LEXICON.has(w)) {
        seen.add(w);
        detected.push({ word: w, type: 'neg' });
      }
    });

    if (detected.length === 0) {
      container.innerHTML = `<span class="neutral-chip">No prominent polarity keywords found (LSTM relies on deep sequence context).</span>`;
      return;
    }

    detected.forEach(item => {
      const chip = document.createElement('span');
      chip.className = `token-chip ${item.type === 'pos' ? 'chip-pos' : 'chip-neg'}`;
      chip.innerHTML = `<span>${item.type === 'pos' ? '▲' : '▼'}</span> <strong>${item.word}</strong>`;
      container.appendChild(chip);
    });
  }

  // ==========================================
  // 7. PRESET CHIPS RENDERING
  // ==========================================
  function renderPresets() {
    const container = document.getElementById('presetsList');
    if (!container) return;

    container.innerHTML = '';
    PRESET_REVIEWS.forEach((preset) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'preset-chip';
      btn.innerHTML = `
        <span class="preset-tag">${preset.category}</span>
        <span class="preset-title">${preset.title}</span>
      `;
      btn.addEventListener('click', () => {
        sfx.playClick();
        const reviewInput = document.getElementById('review');
        if (reviewInput) {
          reviewInput.value = preset.text;
          updateTextMetrics();
          reviewInput.classList.add('flash-highlight');
          setTimeout(() => reviewInput.classList.remove('flash-highlight'), 500);
          showToast(`Loaded: ${preset.title}`);
        }
      });
      container.appendChild(btn);
    });
  }

  // ==========================================
  // 8. PREDICTION HISTORY SYSTEM
  // ==========================================
  function saveToHistory(review, result) {
    try {
      const history = JSON.parse(localStorage.getItem('imdb_prediction_history') || '[]');
      const item = {
        id: Date.now(),
        review: review.length > 90 ? review.substring(0, 90) + '…' : review,
        fullReview: review,
        label: result.label,
        probability: result.probability,
        confidence: result.confidence,
        verdict: result.verdict,
        emoji: result.emoji,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      history.unshift(item);
      if (history.length > 8) history.pop();
      localStorage.setItem('imdb_prediction_history', JSON.stringify(history));
      renderHistory();
    } catch (e) { }
  }

  function renderHistory() {
    const container = document.getElementById('historyList');
    const emptyMsg = document.getElementById('historyEmpty');
    if (!container) return;

    try {
      const history = JSON.parse(localStorage.getItem('imdb_prediction_history') || '[]');
      container.innerHTML = '';

      if (history.length === 0) {
        if (emptyMsg) emptyMsg.style.display = 'block';
        return;
      }
      if (emptyMsg) emptyMsg.style.display = 'none';

      history.forEach(item => {
        const el = document.createElement('div');
        el.className = `history-item ${item.label === 'POSITIVE' ? 'item-pos' : 'item-neg'}`;
        el.innerHTML = `
          <div class="history-header">
            <span class="history-badge ${item.label === 'POSITIVE' ? 'badge-pos' : 'badge-neg'}">
              ${item.emoji || ''} ${item.label} (${(item.probability).toFixed(3)})
            </span>
            <span class="history-time">${item.timestamp}</span>
          </div>
          <p class="history-text">"${escapeHtml(item.review)}"</p>
          <button class="history-reload-btn" title="Reload this review">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>
            Re-run
          </button>
        `;

        el.querySelector('.history-reload-btn').addEventListener('click', () => {
          sfx.playClick();
          const reviewInput = document.getElementById('review');
          if (reviewInput) {
            reviewInput.value = item.fullReview;
            updateTextMetrics();
            reviewInput.scrollIntoView({ behavior: 'smooth' });
            runPrediction();
          }
        });

        container.appendChild(el);
      });
    } catch (e) { }
  }

  // ==========================================
  // 9. ANIMATED STAT COUNTERS
  // ==========================================
  function initStatCounters() {
    const cards = document.querySelectorAll('.stat-number');
    cards.forEach(el => {
      const text = el.dataset.value || el.textContent;
      const clean = text.replace(/[^0-9.]/g, '');
      const target = parseFloat(clean);
      if (isNaN(target)) return;

      const isPercent = text.includes('%');
      const isInteger = !text.includes('.');
      const decimals = isInteger ? 0 : 3;
      const duration = 1200;
      const start = performance.now();

      function tick(now) {
        const p = Math.min((now - start) / duration, 1);
        const ease = 1 - Math.pow(1 - p, 3);
        const val = target * ease;
        el.textContent = (isInteger ? Math.round(val).toLocaleString() : val.toFixed(decimals)) + (isPercent ? '%' : '');
        if (p < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    });
  }

  // ==========================================
  // 10. INTERACTIVE SVG TRAINING CHARTS
  // ==========================================
  function initChartsAndMetrics() {
    let metricsData = null;
    const rawEl = document.getElementById('metrics-data');
    if (rawEl) {
      try {
        metricsData = JSON.parse(rawEl.textContent);
      } catch (e) { }
    }

    if (!metricsData || !metricsData.history) return;

    const chartTabs = document.querySelectorAll('.chart-tab');
    let currentTab = 'accuracy';

    chartTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        sfx.playClick();
        chartTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        currentTab = tab.dataset.chart;
        renderTrainingChart(metricsData.history, currentTab);
      });
    });

    renderTrainingChart(metricsData.history, 'accuracy');
  }

  function renderTrainingChart(history, metricType) {
    const container = document.getElementById('chartSvgContainer');
    if (!container) return;

    const isAcc = metricType === 'accuracy';
    const trainData = isAcc ? history.accuracy : history.loss;
    const valData = isAcc ? history.val_accuracy : history.val_loss;

    if (!trainData || !valData) return;

    const width = 640;
    const height = 260;
    const padX = 50;
    const padY = 35;
    const graphW = width - padX * 2;
    const graphH = height - padY * 2;

    const allVals = [...trainData, ...valData];
    const minVal = Math.max(0, Math.min(...allVals) * 0.85);
    const maxVal = Math.min(1.05, Math.max(...allVals) * 1.08);

    const getX = (index) => padX + (index / (trainData.length - 1)) * graphW;
    const getY = (val) => height - padY - ((val - minVal) / (maxVal - minVal)) * graphH;

    // Build polyline points
    const trainPoints = trainData.map((v, i) => `${getX(i)},${getY(v)}`).join(' ');
    const valPoints = valData.map((v, i) => `${getX(i)},${getY(v)}`).join(' ');

    // Horizontal grid lines
    let gridSvg = '';
    const steps = 4;
    for (let s = 0; s <= steps; s++) {
      const v = minVal + ((maxVal - minVal) / steps) * s;
      const y = getY(v);
      gridSvg += `
        <line x1="${padX}" y1="${y}" x2="${width - padX}" y2="${y}" stroke="rgba(255,255,255,0.07)" stroke-dasharray="4,4"/>
        <text x="${padX - 8}" y="${y + 4}" fill="#64748b" font-size="11" text-anchor="end">${v.toFixed(2)}</text>
      `;
    }

    // Epoch markers on X axis
    let epochSvg = '';
    for (let i = 0; i < trainData.length; i += 2) {
      const x = getX(i);
      epochSvg += `
        <text x="${x}" y="${height - 12}" fill="#64748b" font-size="11" text-anchor="middle">Ep ${i + 1}</text>
      `;
    }

    // Circles for data points
    let dotsSvg = '';
    trainData.forEach((v, i) => {
      const cx = getX(i);
      const cy = getY(v);
      dotsSvg += `<circle cx="${cx}" cy="${cy}" r="3.5" fill="#38bdf8" class="chart-point" data-info="Train: ${v.toFixed(3)} (Epoch ${i + 1})"/>`;
    });
    valData.forEach((v, i) => {
      const cx = getX(i);
      const cy = getY(v);
      dotsSvg += `<circle cx="${cx}" cy="${cy}" r="3.5" fill="#a855f7" class="chart-point" data-info="Val: ${v.toFixed(3)} (Epoch ${i + 1})"/>`;
    });

    const svgHtml = `
      <svg viewBox="0 0 ${width} ${height}" class="training-chart-svg" preserveAspectRatio="xMidYMid meet">
        <defs>
          <linearGradient id="trainGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.3"/>
            <stop offset="100%" stop-color="#38bdf8" stop-opacity="0.0"/>
          </linearGradient>
          <linearGradient id="valGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#a855f7" stop-opacity="0.25"/>
            <stop offset="100%" stop-color="#a855f7" stop-opacity="0.0"/>
          </linearGradient>
        </defs>

        <!-- Grid Lines & Labels -->
        ${gridSvg}
        ${epochSvg}

        <!-- Lines -->
        <polyline points="${trainPoints}" fill="none" stroke="#38bdf8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="chart-line-anim"/>
        <polyline points="${valPoints}" fill="none" stroke="#a855f7" stroke-width="2.5" stroke-dasharray="5,4" stroke-linecap="round" stroke-linejoin="round" class="chart-line-anim"/>

        <!-- Data Point Dots -->
        ${dotsSvg}
      </svg>
    `;

    container.innerHTML = svgHtml;

    // Tooltip interaction
    const tooltip = document.getElementById('chartTooltip');
    const points = container.querySelectorAll('.chart-point');
    points.forEach(pt => {
      pt.addEventListener('mouseenter', (e) => {
        if (!tooltip) return;
        const info = pt.dataset.info;
        tooltip.textContent = info;
        tooltip.style.opacity = '1';
        const rect = pt.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();
        tooltip.style.left = `${rect.left - containerRect.left}px`;
        tooltip.style.top = `${rect.top - containerRect.top - 28}px`;
      });
      pt.addEventListener('mouseleave', () => {
        if (tooltip) tooltip.style.opacity = '0';
      });
    });
  }

  // ==========================================
  // 11. CONFETTI BURST CELEBRATION
  // ==========================================
  function triggerConfettiEffect() {
    const count = 35;
    const resultBox = document.getElementById('result');
    if (!resultBox) return;

    for (let i = 0; i < count; i++) {
      const conf = document.createElement('div');
      conf.className = 'confetti-particle';
      conf.style.setProperty('--x', `${(Math.random() - 0.5) * 360}px`);
      conf.style.setProperty('--y', `${-Math.random() * 260 - 40}px`);
      conf.style.setProperty('--rot', `${Math.random() * 720}deg`);
      conf.style.backgroundColor = ['#22c55e', '#38bdf8', '#818cf8', '#fbbf24', '#34d399'][Math.floor(Math.random() * 5)];
      resultBox.appendChild(conf);

      setTimeout(() => conf.remove(), 1200);
    }
  }

  // ==========================================
  // 12. UTILITY HELPERS
  // ==========================================
  function showToast(message, type = 'info') {
    let toast = document.getElementById('toast');
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'toast';
      document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.className = `toast-active ${type}`;
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => {
      toast.className = '';
    }, 2500);
  }

  function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

})();
