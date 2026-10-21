/**
 * 告别阶段（结尾）
 * ---------------------------------------------------------------
 * 最后几句「再靠近一点」：
 *   1. 音符与引导线消失
 *   2. 步道尽头浮出一张照片——孩子趴在草地上打盹的那张
 *   3. 相机缓缓推向照片，纸飞机从远处飞过来，划过照片，抬升飞进天空
 *   4. 曲终，结算
 *
 * 这一段接管了相机与纸飞机的运动：照片是世界里的实体，相机和航线都以它为准，
 * 所以不管玩家最后飞在哪、镜头朝哪，构图都能对上。
 */
(function (PP) {
  const { clamp, smooth, DREAM_PHOTO } = PP;
  const { SONG_DURATION } = PP;

  const FORWARD = new THREE.Vector3(0, 0, -1);
  const UP = new THREE.Vector3(0, 1, 0);

  class FarewellPhase {
    constructor(game) {
      this.game = game;
    }

    enter() {
      const { game } = this;
      this.t0 = game.audio.currentTime;

      // 音符、引导线消失——画面留给那段回忆
      game.notes.hideAll();
      game.guide.visible = false;

      const cam = game.world.camera;
      const plane = game.plane;

      // 照片立在步道尽头
      const photoZ = plane.position.z - DREAM_PHOTO.ahead;
      game.dreamPhoto.place(photoZ);
      game.dreamPhoto.show();
      const c = game.dreamPhoto.center;

      // 相机：从当前位置推到照片前方
      this.camFrom = cam.position.clone();
      this.camTo = new THREE.Vector3(c.x, DREAM_PHOTO.camHeight, c.z + DREAM_PHOTO.camStop);

      // 起始视线：沿当前朝向取一个远点，避免第一帧突然扭头
      const dir = new THREE.Vector3();
      cam.getWorldDirection(dir);
      this.lookFrom = cam.position.clone().addScaledVector(dir, 40);

      // 飞机航线：从当前位置 → 逼近 → 划过照片 → 抬升入空
      const points = [plane.position.clone()];
      for (const [dx, y, dz] of DREAM_PHOTO.path) {
        points.push(new THREE.Vector3(c.x + dx, y, c.z + dz));
      }
      this.curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.4);
      this.curve.arcLengthDivisions = 240;

      this.tangent = new THREE.Vector3();
      this.look = new THREE.Vector3();
      this.aim = new THREE.Vector3();
      this.moteTimer = 0;

      game.ui.toast.show('再靠近一点');
    }

    update(dt, songTime) {
      const { game } = this;
      const time = Math.min(songTime, SONG_DURATION);
      const span = Math.max(0.001, SONG_DURATION - this.t0);
      const p = clamp((time - this.t0) / span, 0, 1);

      const cam = game.world.camera;
      const plane = game.plane;
      const center = game.dreamPhoto.center;

      // --- 相机：缓缓推近 ---
      const dolly = smooth(DREAM_PHOTO.dolly[0], DREAM_PHOTO.dolly[1], p);
      cam.position.lerpVectors(this.camFrom, this.camTo, dolly);

      // 视线：先稳住，再落到照片上，最后跟着飞机抬头看天空
      this.aim.copy(center);
      this.aim.y += DREAM_PHOTO.lookLift;
      this.look.copy(this.lookFrom).lerp(this.aim, smooth(0, 0.24, p));
      this.look.lerp(plane.position, smooth(0.42, 0.8, p) * 0.92);
      cam.lookAt(this.look);
      cam.updateMatrixWorld(true);

      // --- 飞机：沿曲线划过照片再入空 ---
      const u = smooth(DREAM_PHOTO.fly[0], DREAM_PHOTO.fly[1], p);
      this.curve.getPointAt(u, plane.position);
      this.curve.getTangentAt(u, this.tangent);

      // 机头对准切线（模型机头是 -Z），再按横向分量加一点侧倾
      plane.group.quaternion.setFromUnitVectors(FORWARD, this.tangent);
      plane.group.rotateZ(clamp(-this.tangent.x, -1, 1) * 0.35);

      // --- 光尘：从照片表面缓缓升起 ---
      this.moteTimer -= dt;
      if (this.moteTimer <= 0 && game.dreamPhoto.opacity > 0.35) {
        this.moteTimer = 0.22;
        game.sparks.emit(
          new THREE.Vector3(
            center.x + (Math.random() - 0.5) * DREAM_PHOTO.width * 1.05,
            0.15 + Math.random() * 0.4,
            center.z + (Math.random() - 0.5) * game.dreamPhoto.depth * 1.1
          ),
          new THREE.Vector3(
            (Math.random() - 0.5) * 0.2,
            0.5 + Math.random() * 0.55,
            (Math.random() - 0.5) * 0.2
          ),
          1.5 + Math.random() * 0.9
        );
      }

      game.dreamPhoto.update(dt);

      if (time > SONG_DURATION - 0.15 && game.hasAudioEnded()) game.finish();
    }

    exit() {}
  }

  PP.FarewellPhase = FarewellPhase;
})(window.PP);
