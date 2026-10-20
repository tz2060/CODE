/**
 * 结算阶段
 * ---------------------------------------------------------------
 * 收起纸飞机与引导线，展示结束封面与「再飞一次」。
 */
(function (PP) {
  class EndedPhase {
    constructor(game) {
      this.game = game;
    }

    enter() {
      const { game } = this;

      game.plane.visible = false;
      game.guide.visible = false;

      game.ui.hud.hidePoem();
      game.ui.covers.showEnd();
      game.ui.toast.show('游戏结束 · 纸飞机飞向远方');
    }

    update() {}

    exit() {}
  }

  PP.EndedPhase = EndedPhase;
})(window.PP);
