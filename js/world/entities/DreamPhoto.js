/**
 * 结尾的那张照片
 * ---------------------------------------------------------------
 * 不是贴在镜头前的全屏层，也不是立起来的牌子，而是**平铺在步道尽头**的一张照片：
 * 孩子在草地上打盹、旁边躺着圆滚滚的小伙伴，装在一圈暖色相纸里，静静躺在路上。
 *
 * 相机抬起来俯视它，纸飞机贴着照片掠过去，再拉起飞进天空。
 *
 * 图片以 base64 内嵌在 js/assets/dreamScene.js（file:// 下外部图片建不出 WebGL 纹理）。
 * 这里把它画进 canvas：相纸边、圆角、内圈暖光，再铺一摊地面光池。
 */
(function (PP) {
  const { DREAM_PHOTO, pathCenterX } = PP;

  class DreamPhoto {
    constructor(scene) {
      this.scene = scene;
      this.group = new THREE.Group();
      this.group.visible = false;
      scene.add(this.group);

      this.photo = null;
      this.pool = null;

      this.ready = false;
      this.phase = 'idle';
      this.elapsed = 0;
      this.opacity = 0;

      /** 照片中心（世界坐标），相机与飞机航线都以它为准 */
      this.center = new THREE.Vector3();
      this.width = DREAM_PHOTO.width;
      this.depth = DREAM_PHOTO.width / 1.445;
      this.z = 0;

      this.#load();
    }

    #load() {
      const image = new Image();
      image.onload = () => {
        this.#build(image);
        this.ready = true;
        this.place(this.z);
      };
      image.onerror = () => console.warn('结尾照片加载失败');
      image.src = PP.DREAM_IMAGE;
    }

    #build(image) {
      const { texture, aspect } = this.#makePhotoTexture(image);
      this.depth = this.width / aspect;

      // 照片本体：平铺在地上，画面的"上边"朝远处
      this.photo = new THREE.Mesh(
        new THREE.PlaneGeometry(1, 1),
        new THREE.MeshBasicMaterial({
          map: texture,
          transparent: true,
          opacity: 0,
          alphaTest: 0.02,
          depthWrite: true,
        })
      );
      this.photo.rotation.x = -Math.PI / 2;
      this.photo.scale.set(this.width, this.depth, 1);
      this.photo.position.y = DREAM_PHOTO.lift + 0.012;
      this.group.add(this.photo);

      // 照片周围一圈地面光池，比照片大一圈，露在边缘形成暖光
      this.pool = new THREE.Mesh(
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
      this.pool.rotation.x = -Math.PI / 2;
      this.pool.scale.set(this.width * 0.5, this.depth * 0.5, 1);
      this.pool.position.y = DREAM_PHOTO.lift;
      this.group.add(this.pool);
    }

    /** 相纸：白边 + 圆角 + 内圈暖光 */
    #makePhotoTexture(image) {
      const scale = 2; // 铺到 16 个单位后仍然清晰
      const border = Math.round(image.width * 0.042) * scale;
      const w = image.width * scale + border * 2;
      const h = image.height * scale + border * 2;

      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');

      const outer = new Path2D();
      outer.roundRect(0, 0, w, h, Math.round(border * 1.5));

      ctx.save();
      ctx.clip(outer);
      ctx.fillStyle = 'rgba(255, 251, 240, 0.97)';
      ctx.fillRect(0, 0, w, h);

      const inner = new Path2D();
      inner.roundRect(border, border, image.width * scale, image.height * scale, Math.round(border * 0.8));
      ctx.save();
      ctx.clip(inner);
      ctx.drawImage(image, border, border, image.width * scale, image.height * scale);
      ctx.lineWidth = Math.max(2, border * 0.2);
      ctx.strokeStyle = 'rgba(255, 236, 190, 0.7)';
      ctx.stroke(inner);
      ctx.restore();
      ctx.restore();

      ctx.lineWidth = Math.max(1.5, border * 0.14);
      ctx.strokeStyle = 'rgba(255, 240, 205, 0.9)';
      ctx.stroke(outer);

      const texture = new THREE.CanvasTexture(canvas);
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.anisotropy = 8;
      return { texture, aspect: w / h };
    }

    /** 圆形柔光，光晕和地面光池共用 */
    #makeGlowTexture() {
      const size = 256;
      const canvas = document.createElement('canvas');
      canvas.width = canvas.height = size;
      const ctx = canvas.getContext('2d');
      const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
      g.addColorStop(0, 'rgba(255, 244, 214, 0.95)');
      g.addColorStop(0.32, 'rgba(255, 228, 172, 0.42)');
      g.addColorStop(1, 'rgba(255, 214, 150, 0)');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, size, size);

      const texture = new THREE.CanvasTexture(canvas);
      texture.colorSpace = THREE.SRGBColorSpace;
      return texture;
    }

    /**
     * 把照片铺到步道的某个 z 上（自动对齐路中心线）。
     * 画面本身再往远处推 `back`，但 center 仍锚在 z 上——
     * 相机与飞机航线都以 center 为准，所以构图不受影响。
     */
    place(z) {
      this.z = z;
      const x = pathCenterX(z);
      this.group.position.set(x, 0, z - DREAM_PHOTO.back);
      this.center.set(x, DREAM_PHOTO.lift, z);
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

      this.photo.material.opacity = this.opacity;
      this.pool.material.opacity = this.opacity * 0.22;

      // 出现时略微"长"出来，之后极缓地呼吸，避免画面死板
      const grow = 0.94 + 0.06 * this.opacity;
      const breath = 1 + Math.sin(this.elapsed * 1.1) * 0.008;
      this.photo.scale.set(this.width * grow * breath, this.depth * grow * breath, 1);

      const poolBreath = 1 + Math.sin(this.elapsed * 0.85) * 0.05;
      this.pool.scale.set(this.width * 0.5 * poolBreath, this.depth * 0.5 * poolBreath, 1);
    }
  }

  PP.DreamPhoto = DreamPhoto;
})(window.PP);
