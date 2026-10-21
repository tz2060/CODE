/**
 * 告别阶段（结尾）
 * ---------------------------------------------------------------
 * 最后两句「再靠近一点」：
 *   1. 音符与引导线消失
 *   2. 孩子打盹的那张画面铺满屏幕
 *   3. 纸飞机斜着飞过去，穿过画面，在屏幕上留下一道发光轨迹
 *   4. 曲终，结算
 *
 * 这里接管了相机与纸飞机的运动（不再走 FlightController 的告别逻辑），
 * 因为这一段的构图需要可控：相机定住，飞机沿"相机空间"里的直线穿过画面。
 */
(function (PP) {
  const { clamp, smooth } = PP;
  const { SONG_DURATION, DREAM_SCREEN } = PP;

  class FarewellPhase {
    constructor(game) {
      this.game = game;
    }

    enter() {
      const { game } = this;
      this.t0 = game.audio.currentTime;
      this.sceneryHidden = false;

      // 音符、引导线消失——画面留给那段回忆
      game.notes.hideAll();
      game.guide.visible = false;

      // 全屏记忆画面浮现 + 轨迹就绪
      game.dreamScreen.show();
      game.trail.begin();

      // 定住相机，把飞机换算到相机坐标系，之后的构图就完全可控
      const cam = game.world.camera;
      this.camPos = cam.position.clone();
      this.camQuat = cam.quaternion.clone();
      cam.updateMatrixWorld(true);

      this.from = cam.worldToLocal(game.plane.position.clone());
      // 兜底：万一开始时飞机已经在画面之后（相机被外力挪动等），把它拉到画面前面来
      const minFront = -(DREAM_SCREEN.distance - 3.5);
      if (this.from.z < minFront) this.from.z = minFront;

      // 玩家可能正飞在通道最边上，把起点夹进可视范围，保证穿越从画面内开始
      const depth = Math.max(1, Math.abs(this.from.z));
      const halfH = depth * Math.tan((cam.fov * Math.PI) / 360);
      const halfW = halfH * cam.aspect;
      this.from.x = Math.max(-halfW * 0.5, Math.min(halfW * 0.5, this.from.x));
      this.from.y = Math.max(-halfH * 0.5, Math.min(halfH * 0.5, this.from.y));

      // 从另一侧斜着穿出去，画面里才会留下一道明显的斜光，而不是原地往深处钻
      const exitX = (this.from.x >= 0 ? -1 : 1) * DREAM_SCREEN.exit.x;

      this.to = new THREE.Vector3(
        exitX,
        DREAM_SCREEN.exit.y,
        -(DREAM_SCREEN.distance + DREAM_SCREEN.exit.depth)
      );
      this.dir = this.to.clone().sub(this.from).normalize();

      game.ui.toast.show('穿过这段午后');
    }

    update(dt, songTime) {
      const { game } = this;
      const time = Math.min(songTime, SONG_DURATION);
      const span = Math.max(0.001, SONG_DURATION - this.t0);
      const p = clamp((time - this.t0) / span, 0, 1);

      // 相机定住（画面挂在它身上，本来也不受影响，这样构图最稳）
      const cam = game.world.camera;
      cam.position.copy(this.camPos);
      cam.quaternion.copy(this.camQuat);
      cam.updateMatrixWorld(true);

      // 纸飞机沿直线穿过画面
      const k = smooth(DREAM_SCREEN.moveStart, DREAM_SCREEN.moveEnd, p);
      const local = this.from.clone().lerp(this.to, k);
      // 先记住相机空间里的 z，applyMatrix4 会就地改成世界坐标
      const localZ = local.z;
      const world = local.applyMatrix4(cam.matrixWorld);
      game.plane.position.copy(world);

      // 朝向：机头对准前进方向，略微上扬
      game.plane.group.rotation.set(-0.08, Math.atan2(-this.dir.x, -this.dir.z), 0);

      // 动起来之后才开始记录轨迹
      if (k > 0.02) {
        game.trail.push(world);
      }

      // 穿过画面之后收起来，别在后面闪
      const crossed = localZ <= -DREAM_SCREEN.distance;
      game.plane.visible = !crossed;

      // 穿越完成后，让留在画面上的轨迹缓缓熄灭
      if (crossed) {
        game.trail.strength = Math.max(0, game.trail.strength - dt * DREAM_SCREEN.trailFade);
      }
      game.trail.update(DREAM_SCREEN.trailWidth);

      game.dreamScreen.update(dt);

      // 画面完全铺满后收起公园布景，否则地面会在深度上挡住画面下缘
      if (!this.sceneryHidden && game.dreamScreen.opacity > 0.995) {
        game.world.setSceneryVisible(false);
        this.sceneryHidden = true;
      }

      if (time > SONG_DURATION - 0.15 && game.hasAudioEnded()) game.finish();
    }

    exit() {}
  }

  PP.FarewellPhase = FarewellPhase;
})(window.PP);
