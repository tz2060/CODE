/** 歌词字幕系统：按歌曲时间推进，只在换行时触发一次 DOM 更新。 */
(function (PP) {
  const { CAPTIONS } = PP;

  class CaptionSystem {
    constructor(hud) {
      this.hud = hud;
      this.currentIndex = -1;
    }

    reset() {
      this.currentIndex = -1;
      this.hud.hidePoem();
    }

    #indexAt(time) {
      let index = 0;
      for (let i = 0; i < CAPTIONS.length; i++) {
        if (time >= CAPTIONS[i][0]) index = i;
        else break;
      }
      return index;
    }

    update(time) {
      const index = this.#indexAt(time);
      if (index === this.currentIndex) return;
      this.currentIndex = index;
      this.hud.showPoem(CAPTIONS[index][1]);
    }

    /** 立即显示，用于初始化与重开 */
    show(time) {
      const index = this.#indexAt(time);
      this.currentIndex = index;
      this.hud.showPoem(CAPTIONS[index][1], true);
    }
  }

  PP.CaptionSystem = CaptionSystem;
})(window.PP);
