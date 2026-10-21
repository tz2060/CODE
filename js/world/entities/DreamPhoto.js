/**
 * 结尾的那张照片
 * ---------------------------------------------------------------
 * 曲终前最后几句「再靠近一点」：一张大照片在画面下方铺开，
 * 上方留出天空（太阳、云还在），纸飞机从照片里升起、飞进天空。
 *
 * 照片是世界里的实体（不是相机子节点——这个场景没有把相机加入场景图）。
 * 它是「大画幅」构图：横向铺满、顶边让出天空，所以用 depthTest:false
 * 画在布景之上（否则会被地面/树挡住下半截），太阳与云在上方天空里。
 *
 * 3D 立体感来自四处叠加：
 *   1. 弧形曲面（中心近、两侧向后收）
 *   2. 奶油色相纸边框（像一张实体相片）
 *   3. 轻微的透视倾斜（不是贴死在屏幕上）
 *   4. 缓慢的悬浮（上下浮动 + 极轻摆动）
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

      this.basePos = new THREE.Vector3();
      this.baseQuat = new THREE.Quaternion();

      this.#load();
    }

    #load() {
      new THREE.TextureLoader().load(
        PP.DREAM_IMAGE,
        (texture) => {
          this.#build(texture);
          this.ready = true;
        },
        undefined,
        () => console.warn('结尾照片加载失败')
      );
    }

    #build(texture) {
      const img = texture.image;
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.anisotropy = 8;
      this.aspect = img.width / img.height;

      // 带弧度的平面：中心离镜头最近、两侧向后收，形成 3D 立体感
      const geo = new THREE.PlaneGeometry(1, 1, 64, 8);
      const pos = geo.attributes.position;
      for (let i = 0; i < pos.count; i++) {
        const x = pos.getX(i);
        pos.setZ(i, -0.2 * x * x);
      }
      geo.computeVertexNormals();

      this.photo = new THREE.Mesh(
        geo,
        new THREE.MeshBasicMaterial({
          map: texture,
          transparent: true,
          opacity: 0,
          fog: false,
          depthWrite: false,
          depthTest: false, // 画在布景之上，避免被地面/树切掉下半截
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
          depthTest: false,
          blending: THREE.AdditiveBlending,
          fog: false,
        })
      );
      this.glow.position.z = -0.08;
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

    /** 设定照片的世界尺寸（宽 w，高按图比例算），弧度与柔光一起缩放 */
    setSize(w) {
      if (!this.photo) return;
      this.width = w;
      this.height = w / this.aspect;
      // z 也按宽度缩放：弧形的深度随照片尺寸一起变
      this.photo.scale.set(w, this.height, w);
      this.glow.scale.set(w * 1.12, this.height * 1.12, 1);
    }

    /**
     * 整屏覆盖：照片居中、cover 整个画面（上下左右都盖住），
     * 半透明，让天空/太阳/云/公园从后面透出来。照片带轻微透视倾斜。
     */
    placeBackdrop(camera, { overscan = 1.06 } = {}) {
      const fov = (camera.fov * Math.PI) / 180;
      const visH = 2 * DREAM_PHOTO.distance * Math.tan(fov / 2);
      const visW = visH * camera.aspect;

      // cover 两个方向：取能同时盖住宽和高的尺寸
      const width = Math.max(visW, visH * this.aspect) * overscan;
      this.setSize(width);

      const p = new THREE.Vector3(0, 0, -DREAM_PHOTO.distance);
      camera.localToWorld(p);
      this.group.position.copy(p);

      // 与画面平行：不能用 lookAt（照片巨大、中心偏离视线时会被拉歪）
      this.group.quaternion.copy(camera.quaternion);

      // 居中覆盖：照片中心对准图像中心，不需要偏移
      const map = this.photo.material.map;
      if (map) {
        map.wrapS = map.wrapT = THREE.ClampToEdgeWrapping;
        map.repeat.set(1, 1);
        map.offset.set(0, 0);
      }

      // 轻微倾斜：让相片有透视，不是一张贴死的平图
      this.group.rotateY(-0.05);
      this.group.rotateX(0.022);

      this.basePos.copy(this.group.position);
      this.baseQuat.copy(this.group.quaternion);
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

      if (this.photo) this.photo.material.opacity = this.opacity * DREAM_PHOTO.opacity;
      if (this.glow) this.glow.material.opacity = this.opacity * 0.2;

      // 悬浮：缓慢浮动 + 轻微摆动。照片铺满整屏时，弧形/倾斜的边都在屏幕外，
      // 靠这层相对场景的漂移产生前后景深，让半透明照片像浮在场景前面。
      this.group.position.y = this.basePos.y + Math.sin(this.elapsed * 0.6) * 0.5;
      this.group.position.x = this.basePos.x + Math.sin(this.elapsed * 0.45 + 1.3) * 0.35;
      this.group.quaternion.copy(this.baseQuat);
      this.group.rotateZ(Math.sin(this.elapsed * 0.5) * 0.008);
      this.group.rotateY(Math.sin(this.elapsed * 0.37) * 0.008);
    }
  }

  PP.DreamPhoto = DreamPhoto;
})(window.PP);
