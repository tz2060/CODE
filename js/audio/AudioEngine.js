/**
 * 音频引擎
 * ---------------------------------------------------------------
 * 一个 <audio> 元素负责播放，一条 Web Audio 分析链负责取能量，
 * 另有一次离线解码用于提取旋律走向。
 *
 * 事件：
 *  - 'ended'       播放结束
 *  - 'error'       播放出错
 *  - 'melodyready' 旋律分析完成（payload: AudioBuffer）
 *
 * 关于音频来源：音乐只以 base64 形式放在 js/audio/song-data.js 里。
 * 这不是偷懒——file:// 下外部 mp3 属于跨域资源，一旦接入 createMediaElementSource
 * 就会被浏览器静音重路由（实测分析节点输出全零）；而 data: URL 与页面同源，
 * 播放、节拍分析、旋律解码全都正常。base64 比原文件大约 1/3，是必须付的代价。
 */
(function (PP) {
  const { EventBus } = PP;
  const { lerp } = PP;
  const { SONG_BASE64 } = PP;

  const SONG_SRC = `data:audio/mpeg;base64,${SONG_BASE64}`;

  class AudioEngine extends EventBus {
    /** 旋律分析是否已发起（避免重复解码） */
    #analysisStarted = false;

    constructor() {
      super();

      this.element = new Audio(SONG_SRC);
      // 媒体元素挂在文档里更稳（部分浏览器不播游离的 audio）；用零尺寸占位，不参与布局
      this.element.style.cssText = 'position:absolute;width:0;height:0;opacity:0;pointer-events:none';
      document.body.appendChild(this.element);
      this.element.preload = 'auto';
      this.element.playsInline = true;
      // 老 iOS 需要属性写法才会内联播放
      this.element.setAttribute('playsinline', '');
      this.element.setAttribute('webkit-playsinline', '');
      this.element.load();

      this.muted = false;
      this.stopped = false;
      /**
       * 游戏是否已经要求播放。
       * 音乐不再跟着「点击开始」走，而是等纸飞机现身时才 play()，
       * 所以在这之前所有点击补播（retryPlay / resume）都必须让路，
       * 否则蓄能阶段的轻触会把音乐提前放出来。
       */
      this.armed = false;
      this.context = null;
      this.analyser = null;
      this.sourceNode = null;
      this.frequencyData = null;

      /** 平滑后的整体/低频/中频能量，0–1 */
      this.energy = 0;
      this.lowEnergy = 0;
      this.midEnergy = 0;

      /** 离线解码得到的整首歌 PCM，供旋律分析使用 */
      this.buffer = null;

      this.element.addEventListener('ended', () => this.emit('ended'));
      this.element.addEventListener('error', () => this.emit('error'));
      this.element.addEventListener('playing', () => this.emit('playing'));
    }

    get currentTime() {
      return this.element.currentTime || 0;
    }

    get duration() {
      return Number.isFinite(this.element.duration) ? this.element.duration : 0;
    }

    get paused() {
      return this.element.paused;
    }

    /** 建立分析链（必须由用户手势触发，否则移动端会挂起） */
    ensureGraph() {
      if (this.context) return;

      try {
        const Ctor = window.AudioContext || window.webkitAudioContext;
        this.context = new Ctor();

        this.analyser = this.context.createAnalyser();
        this.analyser.fftSize = 512;
        this.analyser.smoothingTimeConstant = 0.82;

        this.sourceNode = this.context.createMediaElementSource(this.element);
        this.sourceNode.connect(this.analyser);
        this.analyser.connect(this.context.destination);

        this.frequencyData = new Uint8Array(this.analyser.frequencyBinCount);
        this.#analyzeMelody();
      } catch (error) {
        console.warn('Web Audio 分析链不可用，将以静默方式继续', error);
      }
    }

    async #analyzeMelody() {
      if (this.#analysisStarted || !this.context) return;
      this.#analysisStarted = true;

      try {
        const bytes = await this.#loadBytes();
        // 兼容老 Safari：decodeAudioData 只支持回调写法
        this.buffer = await new Promise((resolve, reject) => {
          const p = this.context.decodeAudioData(bytes, resolve, reject);
          if (p && typeof p.then === 'function') p.then(resolve, reject);
        });
        this.emit('melodyready', this.buffer);
      } catch (error) {
        console.warn('旋律分析不可用，使用内置旋律轮廓', error);
      }
    }

    /**
     * 首次用户手势里解锁音频（移动端必需）。
     * iOS Safari / 部分安卓浏览器要求 AudioContext 在用户手势中创建并 resume。
     */
    unlock() {
      this.ensureGraph();
      if (this.context && this.context.state === 'suspended') {
        this.context.resume().catch(() => {});
      }
    }

    /** 取一段可解码的音频字节 */
    async #loadBytes() {
      const binary = atob(SONG_BASE64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      return bytes.buffer;
    }

    /** 开始播放（从 0 开始）。调用即代表「游戏现在要音乐」——纸飞机现身的那一刻 */
    play() {
      this.armed = true;
      this.stopped = false;
      this.ensureGraph();
      if (this.context && this.context.state === 'suspended') {
        this.context.resume().catch(() => {});
      }
      this.element.currentTime = 0;
      this.element.muted = this.muted;

      const promise = this.element.play();
      if (promise && promise.catch) {
        promise.catch(() => this.emit('error', 'blocked'));
      }
    }

    /** 切换静音，返回切换后的状态 */
    toggleMute() {
      this.muted = !this.muted;
      this.element.muted = this.muted;
      return this.muted;
    }

    /** 从当前位置继续播放 */
    resume() {
      if (!this.armed || this.stopped) return; // 还没到纸飞机现身，别提前出声
      if (!this.element.paused) return;
      if (this.context && this.context.state === 'suspended') {
        this.context.resume().catch(() => {});
      }
      this.element.play().catch(() => {});
    }

    /**
     * 重新尝试播放（不重置进度）。
     * 手机浏览器的自动播放策略经常把第一次 play() 拦掉，
     * 这里给后续任意一次点击 / 触摸补一次机会。
     */
    retryPlay() {
      if (!this.armed) return; // 游戏还没要求播放（蓄能阶段），别抢跑
      if (this.muted) return;
      if (this.stopped) return; // 已被游戏主动关掉（如进入第二关），不要再放起来
      if (!this.element.paused) return;
      this.ensureGraph();
      if (this.context && this.context.state === 'suspended') {
        this.context.resume().catch(() => {});
      }
      const promise = this.element.play();
      if (promise && promise.catch) promise.catch(() => {});
    }

    stop() {
      this.stopped = true;
      this.armed = false; // 交回「游戏还没要求播放」的状态，避免后续点击把音乐放起来
      try {
        this.element.pause();
        this.element.currentTime = 0;
      } catch {
        /* 忽略：部分浏览器在未加载完成时不允许设置 currentTime */
      }
    }

    /**
     * 程序合成的短音效（第二关弹幕用）：shoot / graze / hit / clear / over。
     * 直接连 destination，不经过分析链，避免干扰能量曲线。
     */
    playSfx(type) {
      if (this.muted) return;
      const ctx = this.context;
      if (!ctx) return;
      // 上下文被挂起（切后台、iOS 首次未解锁）时先恢复；本次跳过，下一发再响
      if (ctx.state !== 'running') {
        ctx.resume().catch(() => {});
        return;
      }

      const t0 = ctx.currentTime;
      const blip = (freq, wave, start, dur, vol, slideTo) => {
        const o = ctx.createOscillator();
        const g = ctx.createGain();
        o.type = wave;
        o.frequency.setValueAtTime(freq, start);
        if (slideTo) o.frequency.exponentialRampToValueAtTime(slideTo, start + dur);
        g.gain.setValueAtTime(0.0001, start);
        g.gain.exponentialRampToValueAtTime(vol, start + 0.006);
        g.gain.exponentialRampToValueAtTime(0.0001, start + dur);
        o.connect(g);
        g.connect(ctx.destination);
        o.start(start);
        o.stop(start + dur + 0.02);
      };

      if (type === 'shoot') {
        blip(920, 'triangle', t0, 0.05, 0.2, 1500);
      } else if (type === 'graze') {
        blip(1680, 'sine', t0, 0.07, 0.22, 2100);
      } else if (type === 'hit') {
        blip(180, 'sawtooth', t0, 0.26, 0.5, 70);
        blip(90, 'sine', t0, 0.3, 0.4);
      } else if (type === 'clear') {
        blip(660, 'triangle', t0, 0.16, 0.4);
        blip(880, 'triangle', t0 + 0.12, 0.18, 0.4);
        blip(1320, 'triangle', t0 + 0.24, 0.3, 0.4);
      } else if (type === 'over') {
        blip(440, 'sine', t0, 0.28, 0.45, 220);
        blip(220, 'sine', t0 + 0.18, 0.5, 0.45, 110);
      }
    }

    /** 每帧更新三条能量曲线 */
    updateEnergy() {
      if (!this.analyser || !this.frequencyData) return;

      this.analyser.getByteFrequencyData(this.frequencyData);
      const n = this.frequencyData.length;
      const lowEnd = Math.floor(n * 0.12);
      const midEnd = Math.floor(n * 0.52);

      let all = 0;
      let low = 0;
      let mid = 0;
      for (let i = 0; i < n; i++) {
        const v = this.frequencyData[i] / 255;
        all += v;
        if (i < lowEnd) low += v;
        else if (i < midEnd) mid += v;
      }

      this.energy = lerp(this.energy, all / n, 0.24);
      this.lowEnergy = lerp(this.lowEnergy, low / Math.max(1, lowEnd), 0.2);
      this.midEnergy = lerp(this.midEnergy, mid / Math.max(1, midEnd - lowEnd), 0.2);
    }
  }

  PP.AudioEngine = AudioEngine;
})(window.PP);
