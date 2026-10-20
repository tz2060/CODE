/**
 * HUD：得分、进度条、歌词、静音按钮
 * ---------------------------------------------------------------
 * 所有 DOM 查询集中在这里，游戏逻辑只调用语义化方法。
 */
(function (PP) {
  class Hud {
    constructor() {
      this.scoreBox = document.querySelector('#scoreBox');
      this.scoreValue = document.querySelector('#scoreValue');
      this.trackFill = document.querySelector('#trackFill');
      this.poem = document.querySelector('#poem');
      this.poemMain = document.querySelector('#poemMain');
      this.musicToggle = document.querySelector('#musicToggle');
      this.bumpTimer = null;
    }

    setScore(value) {
      this.scoreValue.textContent = String(value);
    }

    /** 得分弹跳反馈 */
    bump() {
      this.scoreBox.classList.remove('hit');
      void this.scoreBox.offsetWidth;
      this.scoreBox.classList.add('hit');
      clearTimeout(this.bumpTimer);
      this.bumpTimer = setTimeout(() => this.scoreBox.classList.remove('hit'), 220);
    }

    resetBump() {
      clearTimeout(this.bumpTimer);
      this.scoreBox.classList.remove('hit');
    }

    /** @param {number} ratio 0–1 */
    setProgress(ratio) {
      this.trackFill.style.width = `${Math.max(0, Math.min(1, ratio)) * 100}%`;
    }

    resetProgress() {
      this.trackFill.style.width = '0%';
    }

    showPoem(text) {
      this.poemMain.textContent = text;
      this.poem.classList.remove('show');
      requestAnimationFrame(() => this.poem.classList.add('show'));
    }

    hidePoem() {
      this.poem.classList.remove('show');
    }

    setMuted(muted) {
      this.musicToggle.textContent = muted ? '◌' : '♫';
      this.musicToggle.setAttribute('aria-label', muted ? '恢复音乐' : '静音音乐');
    }

    onMusicToggle(handler) {
      this.musicToggle.addEventListener('click', handler);
    }
  }

  PP.Hud = Hud;
})(window.PP);
