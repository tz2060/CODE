/**
 * 告别阶段
 * ---------------------------------------------------------------
 * 最后两个「再靠近一点」，纸飞机抬升远去，相机留在原地。
 * 音乐结束后等待一小段时间，再切到结算封面。
 */
(function (PP) {
  const { SONG_DURATION } = PP;

  class FarewellPhase {
    constructor(game) {
      this.game = game;
    }

    enter() {
      this.game.ui.toast.show('跟着旋律，飞向远方');
    }

    update(dt, songTime) {
      const { game } = this;
      const time = Math.min(songTime, SONG_DURATION);

      game.flight.update(dt, {
        songTime,
        farewell: true,
        lowEnergy: game.audio.lowEnergy,
      });

      if (time > SONG_DURATION - 0.15 && game.hasAudioEnded()) game.finish();
    }

    exit() {}
  }

  PP.FarewellPhase = FarewellPhase;
})(window.PP);
