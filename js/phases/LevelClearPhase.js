/**
 * 第一关 → 第二关的过场
 * ---------------------------------------------------------------
 * 分数破 130：第一关结束。保留 3D 场景，让纸飞机加速前飞、抬升，
 * 白光闪过、显示「STAGE 1 CLEAR」，约 2.2 秒后切到第二关弹幕。
 */
(function (PP) {
  const { GameState } = PP;
  const FORWARD = new THREE.Vector3(0, 0, -1);
  const NOSE = new THREE.Vector3(0, 0.28, -1).normalize();

  class LevelClearPhase {
    constructor(game) {
      this.game = game;
    }

    enter() {
      const { game } = this;
      this.elapsed = 0;
      this.span = 2.2;

      game.notes.hideAll();
      game.guide.visible = false;
      game.dreamPhoto.reset(); // 第一关的照片收起来，画面留给「飞向第二关」

      this.startZ = game.plane.position.z;
      this.startY = game.plane.position.y;
      this.startX = game.plane.position.x;

      game.ui.toast.show('第一关结束 · 纸飞机飞向第二关');

      const card = document.querySelector('#stageCard');
      if (card) {
        card.classList.remove('gone');
        void card.offsetWidth;
        card.classList.add('show');
      }

      const flash = document.querySelector('#flightTransition');
      if (flash) {
        flash.classList.remove('active');
        void flash.offsetWidth;
        flash.classList.add('active');
      }
    }

    update(dt) {
      const { game } = this;
      this.elapsed += dt;
      const p = Math.min(1, this.elapsed / this.span);

      const plane = game.plane;
      // 加速前飞 + 抬升 + 向中线收
      plane.position.z = this.startZ - p * p * 110;
      plane.position.y = this.startY + p * 13;
      plane.position.x = this.startX * (1 - p * 0.8);
      plane.group.quaternion.setFromUnitVectors(FORWARD, NOSE);

      const cam = game.world.camera;
      cam.position.z = plane.position.z + 7.8;
      cam.position.y = 3.95 + p * 2.4;
      cam.position.x *= 0.92;
      cam.lookAt(plane.position.x * 0.4, plane.position.y + 0.2, plane.position.z - 13);

      if (p >= 1) {
        const card = document.querySelector('#stageCard');
        if (card) card.classList.remove('show');
        game.enterPhase(GameState.DANMAKU);
      }
    }

    exit() {
      const card = document.querySelector('#stageCard');
      if (card) {
        card.classList.remove('show');
        card.classList.add('gone');
      }
      const flash = document.querySelector('#flightTransition');
      if (flash) flash.classList.remove('active');
    }
  }

  PP.LevelClearPhase = LevelClearPhase;
})(window.PP);
