/**
 * Game
 * ---------------------------------------------------------------
 * 唯一的编排者：持有世界、音频、实体、系统与 UI，
 * 用阶段（Phase）驱动状态流转，用固定的主循环推进每一帧。
 *
 *   ready → awakening → transform → flight → farewell → ended
 */
(function (PP) {
  const {
    World,
    AudioEngine,
    PaperPlane,
    MusicNotes,
    MelodyGuide,
    Sparks,
    DreamScreen,
    TrailRibbon,
    FlightController,
    ScoreSystem,
    CaptionSystem,
    InputController,
    applyMelodyRoute,
    Hud,
    Toast,
    Covers,
    TapPad,
    FlightFlash,
    GameState,
    isFlying,
    ReadyPhase,
    AwakeningPhase,
    TransformPhase,
    FlightPhase,
    FarewellPhase,
    EndedPhase,
    BEAT,
    FAR_END,
    FAREWELL_LINGER,
    FIRST_BEAT,
  } = PP;

  class Game {
    constructor() {
      this.canvas = document.querySelector('#scene');
      this.errorBox = document.querySelector('#err');

      this.state = GameState.READY;
      this.started = false;
      this.finished = false;

      /** 音频结束的时刻（performance.now()/1000） */
      this.audioEndedAt = 0;
      this.audioHasEnded = false;

      this.world = new World(this.canvas);
      this.audio = new AudioEngine();

      this.plane = new PaperPlane(this.world.scene);
      this.notes = new MusicNotes(this.world.scene);
      this.guide = new MelodyGuide(this.world.scene);
      this.guide.build(this.notes);
      this.sparks = new Sparks(this.world.scene);

      // 结尾用：铺满屏幕的记忆画面 + 纸飞机划出的轨迹
      this.dreamScreen = new DreamScreen(this.world.scene, this.world.camera);
      this.trail = new TrailRibbon(this.world.scene, this.world.camera);

      this.ui = {
        hud: new Hud(),
        toast: new Toast(),
        covers: new Covers(),
        tapPad: new TapPad(),
        flash: new FlightFlash(),
      };

      this.flight = new FlightController({
        plane: this.plane,
        camera: this.world.camera,
        guide: this.guide,
      });
      this.score = new ScoreSystem({ hud: this.ui.hud, sparks: this.sparks, toast: this.ui.toast });
      this.captions = new CaptionSystem(this.ui.hud);

      this.phases = {
        [GameState.READY]: new ReadyPhase(this),
        [GameState.AWAKENING]: new AwakeningPhase(this),
        [GameState.TRANSFORM]: new TransformPhase(this),
        [GameState.FLIGHT]: new FlightPhase(this),
        [GameState.FAREWELL]: new FarewellPhase(this),
        [GameState.ENDED]: new EndedPhase(this),
      };

      this.input = new InputController({
        canvas: this.canvas,
        tapButton: this.ui.tapPad.button,
        onTap: () => this.tap(),
        onSteer: (dx, dy) => this.flight.steer(dx, dy),
        getState: () => this.state,
      });

      this.#bindAudio();
      this.#bindUi();

      this.phase = this.phases[GameState.READY];
    }

    // ---------------------------------------------------------------
    // 生命周期
    // ---------------------------------------------------------------

    start() {
      this.ui.covers.hideEnd();
      this.captions.show(0);
      this.#tick();
    }

    #bindAudio() {
      this.audio.on('ended', () => {
        this.audioHasEnded = true;
        this.audioEndedAt = performance.now() / 1000;
        // 标签页被节流时可能跳过了告别帧，这里兜底
        if (!isFlying(this.state)) this.finish();
      });

      this.audio.on('error', (reason) => {
        this.showError(
          reason === 'blocked'
            ? '浏览器没有播放歌曲。请再次轻触页面，允许播放音频。'
            : '内嵌音乐暂时无法播放，请尝试更新浏览器后重新打开本页面。'
        );
      });

      this.audio.on('melodyready', () => {
        if (isFlying(this.state)) this.applyMelodyRoute();
      });
    }

    #bindUi() {
      this.ui.hud.onMusicToggle(() => {
        const muted = this.audio.toggleMute();
        this.ui.hud.setMuted(muted);
        if (!muted && this.started && this.state !== GameState.ENDED) this.audio.resume();
      });

      this.ui.covers.onRestart(() => this.restart());
    }

    // ---------------------------------------------------------------
    // 阶段切换
    // ---------------------------------------------------------------

    enterPhase(state) {
      this.phase?.exit?.();
      this.state = state;
      this.phase = this.phases[state];
      this.phase.enter?.();
    }

    /** 玩家第一次轻触：音乐开始，进入蓄能 */
    startRitual() {
      if (this.started) return;
      this.started = true;

      this.ui.covers.hideStart();
      this.audio.play();
      this.ui.tapPad.setPrompt('继续轻触，让回声醒来');
      this.ui.tapPad.setHint('FOLLOW THE MUSIC');
      this.captions.show(0);

      this.enterPhase(GameState.AWAKENING);
    }

    beginTransformation() {
      if (this.state !== GameState.AWAKENING) return;
      this.enterPhase(GameState.TRANSFORM);
    }

    enterFlight() {
      if (isFlying(this.state) || this.state === GameState.ENDED) return;
      this.enterPhase(GameState.FLIGHT);
    }

    enterFarewell() {
      if (this.state !== GameState.FLIGHT) return;
      this.enterPhase(GameState.FAREWELL);
    }

    finish() {
      if (this.finished) return;
      this.finished = true;
      this.enterPhase(GameState.ENDED);
    }

    // ---------------------------------------------------------------
    // 输入
    // ---------------------------------------------------------------

    tap() {
      if (this.state === GameState.READY) {
        this.startRitual();
        return;
      }
      if (this.state === GameState.AWAKENING) {
        this.phases[GameState.AWAKENING].tap();
      }
    }

    // ---------------------------------------------------------------
    // 旋律路线
    // ---------------------------------------------------------------

    applyMelodyRoute() {
      applyMelodyRoute({
        buffer: this.audio.buffer,
        notes: this.notes,
        guide: this.guide,
        flightStartTime: this.flight.startTime,
        planeStartZ: this.flight.startZ,
        planeSpeed: this.flight.speed,
      });
    }

    hasAudioEnded() {
      return this.audioHasEnded && performance.now() / 1000 - this.audioEndedAt >= FAREWELL_LINGER;
    }

    // ---------------------------------------------------------------
    // 重开
    // ---------------------------------------------------------------

    restart() {
      this.audio.stop();

      this.started = false;
      this.finished = false;
      this.audioHasEnded = false;
      this.audioEndedAt = 0;

      this.score.reset();
      this.notes.reset();
      this.guide.visible = false;
      this.guide.setAnchor(0, 2.35, -0.7);
      this.guide.update(0);

      this.flight.reset();
      this.input.release();

      this.sparks.clear();
      this.dreamScreen.reset();
      this.trail.clear();
      this.world.setSceneryVisible(true);
      this.plane.visible = false;
      this.plane.reset(0, 2.35, 0, 1);

      this.captions.reset();
      this.ui.covers.hideEnd();
      this.ui.covers.showStart();
      this.ui.tapPad.reset();
      this.ui.hud.resetProgress();
      this.captions.show(0);

      this.enterPhase(GameState.READY);
    }

    // ---------------------------------------------------------------
    // 主循环
    // ---------------------------------------------------------------

    #tick = () => {
      requestAnimationFrame(this.#tick);

      const dt = Math.min(this.world.clock.getDelta(), 0.05);
      const now = performance.now() / 1000;
      const songTime = this.audio.currentTime;

      this.audio.updateEnergy();

      if (this.started && this.state !== GameState.ENDED) this.captions.update(songTime);

      const beatPhase = Math.cos(((songTime - FIRST_BEAT) / BEAT) * Math.PI * 2);
      const beatPulse = Math.pow(Math.max(0, beatPhase), 6);

      this.notes.update(now, beatPulse, this.audio.energy);
      this.guide.update(beatPulse, false, this.audio.energy);
      this.world.swayTrees(now, this.audio.energy);

      this.phase.update(dt, songTime);

      // 兜底：后台标签页跳帧时也要能走到告别与结算
      if (this.audioHasEnded && this.state === GameState.FLIGHT && songTime >= FAR_END) {
        this.enterFarewell();
      }
      if (this.audioHasEnded && this.state === GameState.FAREWELL && this.hasAudioEnded()) {
        this.finish();
      }

      if (this.audio.duration > 0) {
        this.ui.hud.setProgress(songTime / this.audio.duration);
      }

      this.sparks.update(dt);
      this.world.pulseSun(this.audio.energy, this.audio.lowEnergy);
      this.world.followCamera();
      this.world.render();
    };

    showError(message) {
      this.errorBox.textContent = message;
      this.errorBox.style.display = 'block';
    }
  }

  PP.Game = Game;
})(window.PP);
