/**
 * 弹幕关阶段「幻月」
 * ---------------------------------------------------------------
 * 飞行中分数破 130 后进入。把 3D 场景收起来，切到 2D 弹幕画布：
 * 纸飞机躲弹、自动射击、擦弹得分。通关 → 第三关；命耗尽 → 从本关重新开始。
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
      // 第二关开始：第一关的音乐必须关掉，只留音效
      game.audio.stop();

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

      // HUD 分数 = 三个关卡的总分
      game.ui.hud.setScore(game.totalScore);

      if (d.finished && d.result === 'clear') {
        // 打败幻月 → 纸飞机放大飞过屏幕，转场到第三关「恶龙城堡」
        game.ui.toast.show('第二关通关 · 飞向恶龙城堡');
        game.enterMario();
        return;
      }

      // 没有生命条数：每死一次扣 100 分，扣到 0 以下就从本关重开
      if (game.totalScore < 0) {
        d.reset();
        d.show();
        game.restartStage();
        game.ui.toast.show('分数扣到 0 以下 · 从第二关重新开始');
      }
    }

    exit() {
      this.game.danmaku.hide();
    }
  }

  PP.DanmakuPhase = DanmakuPhase;
})(window.PP);
