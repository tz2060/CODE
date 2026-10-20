/**
 * 蓄能阶段
 * ---------------------------------------------------------------
 * 玩家轻触攒"风"。能量同时驱动雾气浓度——越接近起飞，画面越通透。
 * 攒满阈值、或在时间窗内连点到次数、或歌曲走到 9.6 秒，都会自动起飞。
 */
(function (PP) {
  const { clamp, lerp } = PP;
  const { AWAKENING } = PP;

  class AwakeningPhase {
    constructor(game) {
      this.game = game;
      this.energy = 0;
      this.clickTimes = [];
      this.launched = false;
    }

    enter() {
      this.energy = AWAKENING.startTapGain;
      this.clickTimes = [];
      this.launched = false;
    }

    tap() {
      if (this.launched) return;

      const now = performance.now();
      this.clickTimes = this.clickTimes.filter((t) => now - t < 2000);
      this.clickTimes.push(now);

      this.energy = clamp(this.energy + AWAKENING.tapGain, 0, 1);
      this.game.ui.tapPad.press();

      const burst =
        this.clickTimes.length >= AWAKENING.burstTaps &&
        now - this.clickTimes[0] < AWAKENING.burstWindow;

      if (this.energy > AWAKENING.energyThreshold || burst) this.launch();
    }

    launch() {
      if (this.launched) return;
      this.launched = true;
      this.game.beginTransformation();
    }

    update(dt, songTime) {
      this.energy = Math.max(0, this.energy - dt * AWAKENING.decayPerSecond);

      const fog = this.game.world.scene.fog;
      fog.density = lerp(fog.density, 0.0027 - this.energy * 0.00075, 0.03);

      if (songTime >= AWAKENING.autoLaunchAt) this.launch();
    }

    exit() {}
  }

  PP.AwakeningPhase = AwakeningPhase;
})(window.PP);
