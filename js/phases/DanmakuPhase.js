/**
 * 弹幕关阶段「幻月」
 * ---------------------------------------------------------------
 * 飞行中分数破 130 后进入。把 3D 场景收起来，切到 2D 弹幕画布：
 * 纸飞机躲弹、自动射击、擦弹得分。通关 → 照片结尾；阵亡 → 结算。
 */
(function (PP) {
  const { GameState } = PP;

  class DanmakuPhase {
    constructor(game) {
      this.game = game;
    }

    enter() {
      const { game } = this;
      game.notes.hideAll();
      game.guide.visible = false;
      game.plane.visible = false;

      const d = game.danmaku;
      d.reset();
      d.show();

      game.ui.toast.show('分数破 130 · 进入下一关「幻月」');
    }

    update(dt) {
      const { game } = this;
      const d = game.danmaku;
      d.update(dt);
      d.render();

      // HUD 分数 = 飞行得分 + 弹幕得分
      game.ui.hud.setScore(game.score.score + d.score);

      if (d.finished) {
        if (d.result === 'clear') game.ui.toast.show('第二关通关 · 幻月散去');
        game.finish();
      }
    }

    exit() {
      this.game.danmaku.hide();
    }
  }

  PP.DanmakuPhase = DanmakuPhase;
})(window.PP);
