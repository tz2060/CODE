/**
 * 结尾的那张照片
 * ---------------------------------------------------------------
 * 曲终前最后几句「再靠近一点」：一张照片浮现在镜头正前方，
 * 纸飞机从近处升起，掠过照片、飞进天空。
 *
 * 照片是世界里的实体（不是相机子节点——这个场景没有把相机加入场景图，
 * 挂在相机下的对象不会被渲染器遍历到）：告别阶段相机是静止的，
 * 所以在 enter 时按相机的位置/朝向把照片摆到「正前方固定距离」并转向正对相机，
 * 之后它就稳稳浮在那里，飞机从它上方掠过。
 *
 * 图在 js/assets/dreamScene.js 的 PP.DREAM_IMAGE（base64，
 * file:// 下外部图片建不出 WebGL 纹理）。
 */
(function (PP) {
  const { DREAM_PHOTO } = PP;

  class DreamPhoto {
    constructor(scene) {
      this.scene = scene;
      this.group = new THREE.Group();
      this.group.visible = false;
      scene.add(this.group);

      this.photo = null;
      this.glow = null;
      this.aspect = 1.445; // 兜底，加载后按真实图比例覆盖
      this.width = DREAM_PHOTO.width;
      this.height = DREAM_PHOTO.width / this.aspect;

      this.ready = false;
      this.phase = 'idle';
      this.elapsed = 0;
      this.opacity = 0;

      this.#load();
    }

    #load() {
      new THREE.TextureLoader().load(
        PP.DREAM_IMAGE,
        (texture) => {
          texture.colorSpace = THREE.SRGBColorSpace;
          texture.anisotropy = 8;
          this.#build(texture);
          this.ready = true;
        },
        undefined,
        () => console.warn('结尾照片加载失败')
      );
    }

    #build(texture) {
      const img = texture.image;
      this.aspect = img.width / img.height;

      // 照片本体（PlaneGeometry 1×1，靠 scale 控制尺寸）
      this.photo = new THREE.Mesh(
        new THREE.PlaneGeometry(1, 1),
        new THREE.MeshBasicMaterial({
          map: texture,
          transparent: true,
          opacity: 0,
          fog: false,
          depthWrite: false,
        })
      );
      this.group.add(this.photo);

      // 照片背后的柔光，像回忆里透出来的一点暖意
      this.glow = new THREE.Mesh(
        new THREE.PlaneGeometry(1, 1),
        new THREE.MeshBasicMaterial({
          map: this.#makeGlowTexture(),
          transparent: true,
          opacity: 0,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
          fog: false,
        })
      );
      this.glow.position.z = -0.05;
      this.group.add(this.glow);
    }

    /** 圆形柔光 */
    #makeGlowTexture() {
      const size = 256;
      const canvas = document.createElement('canvas');
      canvas.width = canvas.height = size;
      const ctx = canvas.getContext('2d');
      const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
      g.addColorStop(0, 'rgba(255, 244, 214, 0.85)');
      g.addColorStop(0.4, 'rgba(255, 226, 170, 0.3)');
      g.addColorStop(1, 'rgba(255, 214, 150, 0)');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, size, size);

      const texture = new THREE.CanvasTexture(canvas);
      texture.colorSpace = THREE.SRGBColorSpace;
      return texture;
    }

    /** 设定照片的世界尺寸（宽 w，高按图比例算），柔光一起放大 */
    setSize(w) {
      if (!this.photo) return;
      this.width = w;
      this.height = w / this.aspect;
      this.photo.scale.set(w, this.height, 1);
      this.glow.scale.set(w * 1.35, this.height * 1.35, 1);
    }

    /**
     * 把照片摆到相机正前方固定距离处（世界坐标），并转向正对相机。
     * 相机在告别阶段是静止的，所以只需摆一次。
     * 另外保证照片底边始终离地 DREAM_PHOTO.floor——不然照片会插进步道/草地，
     * 底边被地面切掉，看起来像"卡在土里"。
     */
    place(camera) {
      const fwd = new THREE.Vector3();
      camera.getWorldDirection(fwd);
      const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion);

      const pos = camera.position
        .clone()
        .addScaledVector(fwd, DREAM_PHOTO.distance)
        .addScaledVector(up, DREAM_PHOTO.yOffset);

      // 底边离地至少 floor：把整张照片抬高
      const minCenterY = (this.height || 0) / 2 + DREAM_PHOTO.floor;
      if (pos.y < minCenterY) pos.y = minCenterY;

      this.group.position.copy(pos);
      this.group.lookAt(camera.position);
    }

    show() {
      if (!this.ready) return;
      this.phase = 'in';
      this.elapsed = 0;
      this.opacity = 0;
      this.group.visible = true;
    }

    hide() {
      this.phase = 'idle';
      this.elapsed = 0;
      this.opacity = 0;
      if (this.group) this.group.visible = false;
    }

    reset() {
      this.hide();
    }

    update(dt) {
      if (this.phase === 'idle' || !this.ready) return;

      this.elapsed += dt;

      if (this.phase === 'in') {
        const t = Math.min(1, this.elapsed / DREAM_PHOTO.fadeIn);
        this.opacity = t * t * (3 - 2 * t);
        if (t >= 1) this.phase = 'hold';
      } else {
        this.opacity = 1;
      }

      if (this.photo) this.photo.material.opacity = this.opacity;
      if (this.glow) this.glow.material.opacity = this.opacity * 0.5;
    }
  }

  PP.DreamPhoto = DreamPhoto;
})(window.PP);