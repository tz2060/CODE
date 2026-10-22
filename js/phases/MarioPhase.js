/**
 * 第三关阶段「恶龙城堡」
 * ---------------------------------------------------------------
 * 第二关打败幻月后进入：纸飞机放大飞过屏幕，把马里奥放到横版关卡里。
 * 通关（斧头砍断桥、库巴掉进岩浆、飞机载着马里奥和公主飞走）→ 结算；
 * 命耗尽 → **从第三关重新开始**，不回第一关。
 */
(function (PP) {
  const { GameState } = PP;

  class MarioPhase {
    constructor(game) {
      this.game = game;
      this.done = false;
    }

    enter() {
      const { game } = this;
      game.notes.hideAll();
      game.guide.visible = false;
      game.plane.visible = false;
      game.audio.stop();

      const m = game.mario;
      m.reset();
      m.show();

    }

    update(dt) {
      const { game } = this;
      const m = game.mario;
      m.update(dt);
      m.render();

      game.ui.hud.setScore(game.totalScore);

      if (m.finished && !this.done) {
        this.done = true;
        game.finish();
        return;
      }

      // 没有生命条数：每死一次扣 100 分，扣到 0 以下就从本关重开
      if (game.totalScore < 0) {
        m.reset();
        m.show();
        game.restartStage();
        game.ui.toast.show('分数扣到 0 以下 · 从第三关重新开始');
        this.done = false;
      }
    }

    exit() {
      this.done = false;
      this.game.mario.hide();
    }
  }

  PP.MarioPhase = MarioPhase;
})(window.PP);
