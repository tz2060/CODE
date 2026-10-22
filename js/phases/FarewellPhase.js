/**
 * 告别阶段（结尾）
 * ---------------------------------------------------------------
 * 最后几句「再靠近一点」：
 *   1. 音符与引导线消失
 *   2. 一张照片浮现在镜头正前方（camera-facing，永远正对镜头）
 *   3. 纸飞机从近处升起，掠过照片、飞进天空
 *   4. 曲终，结算
 *
 * 这一段接管了相机与纸飞机的运动：相机停在飞行结束时的位置不再移动，
 * 照片挂到相机上做「浮在正前方」的浮图；飞机沿一条镜头空间里的曲线，
 * 从当前位置平滑接管、升起、越过照片、飞向天空。
 */
(function (PP) {
  const { clamp, smooth, DREAM_PHOTO } = PP;
  const { SONG_DURATION } = PP;

  const FORWARD = new THREE.Vector3(0, 0, -1);

  class FarewellPhase {
    constructor(game) {
      this.game = game;
    }

    enter() {
      const { game } = this;
      this.elapsed = 0;
      this.span = 12; // 告别演出固定 12 秒（弹幕关后歌曲可能已结束，不能依赖歌曲时间）

      // 音符、引导线消失——画面留给那段回忆
      game.notes.hideAll();
      game.guide.visible = false;
      game.plane.visible = true;

      const cam = game.world.camera;
      const plane = game.plane;

      // 相机停在飞行结束时的姿态，不再移动——照片与航线都以它为基准
      cam.updateMatrixWorld(true);

      // 照片：大画幅铺在画面下方，顶边让出天空（太阳/云仍在上方可见）
      // 相机静止，摆一次即可；相机没被加入场景图，所以照片是场景里的实体而非相机子节点
      const photo = game.dreamPhoto;
      photo.placeBackdrop(cam, { overscan: DREAM_PHOTO.overscan });
      photo.show();

      // 飞机航线：镜头空间里，从「当前位置」平滑接管，再升起飞进天空。
      // 照片铺满整个画面（半透明），飞机从下方升起、穿过照片飞进天空。
      const start = cam.worldToLocal(plane.position.clone());
      const D = DREAM_PHOTO.distance;
      this.curve = new THREE.CatmullRomCurve3(
        [
          start,
          new THREE.Vector3(0, 1.2, -D * 0.5),
          new THREE.Vector3(0, 4.5, -D),
          new THREE.Vector3(0, 9, -D * 1.6),
          new THREE.Vector3(0, 15, -D * 2.2),
        ],
        false,
        'catmullrom',
        0.4
      );
      this.curve.arcLengthDivisions = 240;

      this.tangent = new THREE.Vector3();
      this.local = new THREE.Vector3();

      game.ui.toast.show('再靠近一点');
    }

    update(dt) {
      const { game } = this;
      this.elapsed += dt;
      const p = clamp(this.elapsed / this.span, 0, 1);

      const cam = game.world.camera;
      const plane = game.plane;

      // 相机静止：这一帧什么都不用做

      // 飞机：沿镜头空间曲线走，再变换到世界坐标
      const u = smooth(DREAM_PHOTO.fly[0], DREAM_PHOTO.fly[1], p);
      this.curve.getPointAt(u, this.local);
      cam.localToWorld(this.local);
      plane.position.copy(this.local);

      this.curve.getTangentAt(u, this.tangent);
      this.tangent.applyQuaternion(cam.quaternion).normalize();
      plane.group.quaternion.setFromUnitVectors(FORWARD, this.tangent);

      game.dreamPhoto.update(dt);

      if (p >= 1) game.finishLevelOne();
    }

    exit() {
      // 把照片从相机上摘下来，避免重开时残留
      const { game } = this;
      game.dreamPhoto.reset();
    }
  }

  PP.FarewellPhase = FarewellPhase;
})(window.PP);
