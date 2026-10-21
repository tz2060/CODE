/**
 * 结尾的那张照片
 * ---------------------------------------------------------------
 * 它不再是"路上一张带相框的小照片"，而是**铺满整片地面的那张图**：
 * 孩子在草地上打盹的画面嵌在一大片草地里，边缘羽化、亮度对齐，
 * 于是它盖住路面和两侧草地时，看不出"一张照片"的边界。
 *
 * 相机抬起来俯视它，纸飞机贴着地面掠过去，再拉起飞进天空。
 *
 * 图在 js/assets/dreamScene.js 的 PP.DREAM_GROUND（base64，
 * file:// 下外部图片建不出 WebGL 纹理）。
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
      new THREE.TextureLoader().load(
        PP.DREAM_GROUND,
        (texture) => {
          texture.colorSpace = THREE.SRGBColorSpace;
          texture.anisotropy = 8;
          this.#build(texture);
          this.ready = true;
          this.place(this.z);
        },
        undefined,
        () => console.warn('结尾地面图加载失败')
      );
    }

    #build(texture) {
      const img = texture.image;
      this.depth = this.width / (img.width / img.height);

      // 平铺在地上，画面的"上边"朝远处
      this.photo = new THREE.Mesh(
        new THREE.PlaneGeometry(1, 1),
        new THREE.MeshBasicMaterial({
          map: texture,
          transparent: true,
          opacity: 0,
          depthWrite: true,
        })
      );
      this.photo.rotation.x = -Math.PI / 2;
      this.photo.scale.set(this.width, this.depth, 1);
      this.photo.position.y = DREAM_PHOTO.lift;
      this.group.add(this.photo);

      // 中心一小片暖光，像阳光落在这一块地上
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
      this.pool.scale.set(this.width * 0.42, this.depth * 0.42, 1);
      this.pool.position.y = DREAM_PHOTO.lift + 0.008;
      this.group.add(this.pool);
    }

    /** 圆形柔光 */
    #makeGlowTexture() {
      const size = 256;
      const canvas = document.createElement('canvas');
      canvas.width = canvas.height = size;
      const ctx = canvas.getContext('2d');
      const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
      g.addColorStop(0, 'rgba(255, 244, 214, 0.9)');
      g.addColorStop(0.35, 'rgba(255, 228, 172, 0.36)');
      g.addColorStop(1, 'rgba(255, 214, 150, 0)');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, size, size);

      const texture = new THREE.CanvasTexture(canvas);
      texture.colorSpace = THREE.SRGBColorSpace;
      return texture;
    }

    /**
     * 把地面图铺到步道的某个 z 上（自动对齐路中心线）。
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
      this.pool.material.opacity = this.opacity * 0.18;

      // 极缓的呼吸，避免画面死板
      const breath = 1 + Math.sin(this.elapsed * 0.8) * 0.004;
      this.photo.scale.set(this.width * breath, this.depth * breath, 1);
    }
  }

  PP.DreamPhoto = DreamPhoto;
})(window.PP);
