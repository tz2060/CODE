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
    }

    update() {}

    exit() {}
  }

  PP.EndedPhase = EndedPhase;
})(window.PP);
