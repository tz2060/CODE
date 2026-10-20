/**
 * 待机阶段
 * ---------------------------------------------------------------
 * 相机停在山坡上方的固定机位，看着空无一人的公园。
 * 玩家轻触后进入蓄能阶段。
 */
(function (PP) {
  const { lerp } = PP;

  class ReadyPhase {
    constructor(game) {
      this.game = game;
    }

    enter() {}

    update() {
      const { camera } = this.game.world;
      camera.position.x = lerp(camera.position.x, 0, 0.025);
      camera.position.y = lerp(camera.position.y, 5.1, 0.025);
      camera.position.z = lerp(camera.position.z, 11.8, 0.025);
      camera.lookAt(0, 2.2, -8);
    }

    exit() {}
  }

  PP.ReadyPhase = ReadyPhase;
})(window.PP);
