/**
 * 计分系统
 * ---------------------------------------------------------------
 * 椭球命中判定由音符自己完成，这里负责把"距离"换算成分数、
 * 播放反馈（弹跳、光尘、提示），并维护总分。
 */
(function (PP) {
  const { NOTE_MAX_SCORE } = PP;

  class ScoreSystem {
    constructor({ hud, sparks, toast }) {
      this.hud = hud;
      this.sparks = sparks;
      this.toast = toast;
      this.score = 0;
      this.notesCollected = 0;
    }

    reset() {
      this.score = 0;
      this.notesCollected = 0;
      this.hud.setScore(0);
      this.hud.resetBump();
    }

    /**
     * 检查与音符的碰撞并结算。
     * @param {object} plane  纸飞机实体
     * @param {object} notes  音符群
     */
    check(plane, notes) {
      if (!plane.visible) return;
      // 拿机身外轮廓的采样点，而不是只拿原点
      const hits = notes.findHits(plane.getHitPoints());
      for (const { note, distance } of hits) {
        note.hit = true;
        const points = Math.max(1, Math.round(NOTE_MAX_SCORE * (1 - distance)));
        this.#award(points, note);
      }
    }

    #award(points, note) {
      this.score += points;
      this.notesCollected += 1;

      this.hud.setScore(this.score);
      this.hud.bump();

      this.sparks.burst(note.sprite.position, 9, 0.8);
      note.sprite.visible = false;
      this.toast.show(`♪ +${points} 分`);
    }
  }

  PP.ScoreSystem = ScoreSystem;
})(window.PP);
