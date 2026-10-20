/**
 * 飞行阶段：跟随旋律收集音符
 * ---------------------------------------------------------------
 * 飞机沿旋律路线前进，玩家拖拽产生偏移；撞到音符按距离计分。
 * 歌曲走到 75 秒（最后两个「再靠近一点」）时进入告别。
 */
(function (PP) {
  const { SONG_DURATION, FAR_END } = PP;

  class FlightPhase {
    constructor(game) {
      this.game = game;
    }

    enter() {
      const { game } = this;
      const startTime = game.audio.currentTime;

      game.plane.visible = true;
      game.plane.group.scale.setScalar(1);

      game.flight.launch({ startTime, startZ: game.plane.position.z });
      game.notes.layoutForFlight({
        planeStartZ: game.plane.position.z,
        flightStartTime: startTime,
        planeSpeed: game.flight.speed,
      });

      game.guide.setAnchor(game.plane.position.x, game.plane.position.y, game.plane.position.z);
      game.applyMelodyRoute();

      game.guide.update(0, true, 0);
      game.guide.visible = true;

      // 重置速度记忆，避免过场位移造成假侧倾
      game.plane.resetVelocityMemory();

      game.captions.show(startTime);
    }

    update(dt, songTime) {
      const { game } = this;
      const time = Math.min(songTime, SONG_DURATION);

      if (time >= FAR_END) {
        game.enterFarewell();
        return;
      }

      game.flight.update(dt, {
        songTime,
        farewell: false,
        lowEnergy: game.audio.lowEnergy,
      });
      game.score.check(game.plane, game.notes);
    }

    exit() {}
  }

  PP.FlightPhase = FlightPhase;
})(window.PP);
