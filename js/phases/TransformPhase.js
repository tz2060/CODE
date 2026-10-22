/**
 * 过场阶段：纸飞机现身
 * ---------------------------------------------------------------
 * 一层暖色光晕闪过，纸飞机从指尖大小的光点展开成真正的纸飞机，
 * 同时相机从俯瞰慢慢压到起飞视角。
 *
 * **音乐在这里开始**：纸飞机现身的那一帧才 play()，
 * 于是「音乐响起」与「纸飞机出现」是同一个瞬间（此前蓄能阶段是安静的）。
 * 过场进度用本阶段自己的计时推进，万一音乐被浏览器拦下，
 * 纸飞机也不会卡在半空——点击补播后音乐再从 0 跟上。
 */
(function (PP) {
  const { clamp, lerp, smooth } = PP;
  const { TRANSFORM_DURATION } = PP;

  class TransformPhase {
    constructor(game) {
      this.game = game;
      this.elapsed = 0;
    }

    enter() {
      const { game } = this;
      this.elapsed = 0;

      // 纸飞机出现的同时，音乐从头开始
      game.audio.play();

      game.ui.tapPad.hide();
      game.ui.toast.show('听见了，公园的风');
      game.ui.flash.play();

      game.plane.visible = true;
      game.plane.reset(0, 2.25, 0, 0.16);
      game.sparks.launchVeil(40);
    }

    update(dt) {
      const { game } = this;
      this.elapsed += dt;
      const p = clamp(this.elapsed / TRANSFORM_DURATION, 0, 1);
      const reveal = smooth(0.02, 0.88, p);

      game.world.scene.fog.density = lerp(game.world.scene.fog.density, 0.0019, p * 0.025);
      game.plane.playReveal(p, reveal);

      const { camera } = game.world;
      camera.position.x = lerp(camera.position.x, 0, 0.035);
      camera.position.y = lerp(camera.position.y, 4.2, 0.03);
      camera.position.z = lerp(camera.position.z, 9.8, 0.035);
      camera.lookAt(0, 2.3, -9);

      if (p >= 1) game.enterFlight();
    }

    exit() {}
  }

  PP.TransformPhase = TransformPhase;
})(window.PP);
