/**
 * 结尾的全屏记忆画面
 * ---------------------------------------------------------------
 * 最后两句「再靠近一点」时，孩子打盹的那张图铺满整个屏幕：
 *  - 底层：同一张图缩小 + 模糊后拉满全屏，负责"铺满"，任何屏幕比例都不留黑边
 *  - 上层：清晰图按比例完整显示，居中，边缘一圈暖光
 *
 * 两个面片都挂在相机下面（相机空间 z = -distance），所以永远正对镜头、
 * 永远铺满视口；纸飞机在世界里往前飞，自然会从这块"屏幕"中间穿过去。
 */
(function (PP) {
  const { DREAM_SCREEN } = PP;

  class DreamScreen {
    constructor(scene, camera) {
      this.scene = scene;
      this.camera = camera;

      this.group = new THREE.Group();
      this.group.position.set(0, 0, -DREAM_SCREEN.distance);
      this.group.visible = false;

      // 相机要进场景图，挂在它下面的面片才会被渲染
      camera.add(this.group);
      scene.add(camera);

      this.backdrop = null;
      this.subject = null;
      this.ready = false;
      this.phase = 'idle';
      this.elapsed = 0;
      this.opacity = 0;

      this.#load();
      window.addEventListener('resize', () => this.layout(), { passive: true });
    }

    #load() {
      const image = new Image();
      image.onload = () => {
        this.imageAspect = image.width / image.height;
        this.#build(image);
        this.ready = true;
        this.layout();
      };
      image.onerror = () => console.warn('梦境场景图加载失败');
      image.src = PP.DREAM_IMAGE;
    }

    /** 把图分别做成"清晰"和"模糊铺底"两张纹理 */
    #build(image) {
      // --- 清晰版 ---
      const sharp = document.createElement('canvas');
      sharp.width = image.width;
      sharp.height = image.height;
      sharp.getContext('2d').drawImage(image, 0, 0);
      const sharpTex = new THREE.CanvasTexture(sharp);
      sharpTex.colorSpace = THREE.SRGBColorSpace;
      sharpTex.anisotropy = 4;

      // --- 模糊铺底版：缩到很小再放大，天然柔化；有 filter 就再糊一层 ---
      const bw = 96;
      const bh = Math.max(1, Math.round(bw / this.imageAspect));
      const blur = document.createElement('canvas');
      blur.width = bw;
      blur.height = bh;
      const bctx = blur.getContext('2d');
      try {
        bctx.filter = 'blur(4px)';
      } catch (e) {
        /* 老浏览器不支持 filter，靠低分辨率放大也能糊 */
      }
      bctx.drawImage(image, 0, 0, bw, bh);
      const blurTex = new THREE.CanvasTexture(blur);
      blurTex.colorSpace = THREE.SRGBColorSpace;

      const geo = new THREE.PlaneGeometry(1, 1);

      // 铺底层：略微外扩，避免边缘露缝
      this.backdrop = new THREE.Mesh(
        geo,
        new THREE.MeshBasicMaterial({
          map: blurTex,
          transparent: true,
          opacity: 0,
          depthWrite: true,
          fog: false,
        })
      );
      this.backdrop.renderOrder = -2;

      // 清晰层：稍微靠前一点，避免 z-fighting
      this.subject = new THREE.Mesh(
        geo,
        new THREE.MeshBasicMaterial({
          map: sharpTex,
          transparent: true,
          opacity: 0,
          depthWrite: true,
          fog: false,
        })
      );
      this.subject.position.z = 0.05;
      this.subject.renderOrder = -1;

      this.group.add(this.backdrop);
      this.group.add(this.subject);
    }

    /** 按当前 fov / 宽高比铺满视口 */
    layout() {
      if (!this.ready) return;

      const d = DREAM_SCREEN.distance;
      const h = 2 * d * Math.tan((this.camera.fov * Math.PI) / 360);
      const w = h * this.camera.aspect;

      this.backdrop.scale.set(w * 1.04, h * 1.04, 1);

      // 清晰图：按比例完整显示（contain），尽量放大但不出界
      const s = DREAM_SCREEN.subjectScale;
      let sw = w * s;
      let sh = sw / this.imageAspect;
      if (sh > h * s) {
        sh = h * s;
        sw = sh * this.imageAspect;
      }
      this.subject.scale.set(sw, sh, 1);
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
        const t = Math.min(1, this.elapsed / DREAM_SCREEN.fadeIn);
        this.opacity = t * t * (3 - 2 * t);
        if (t >= 1) this.phase = 'hold';
      } else {
        this.opacity = 1;
      }

      this.backdrop.material.opacity = this.opacity;
      this.subject.material.opacity = this.opacity;
    }
  }

  PP.DreamScreen = DreamScreen;
})(window.PP);
