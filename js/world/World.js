/**
 * 三维世界：渲染器、相机、光照与整个午后公园的静态布景。
 * 只负责"场景存在"，不含任何游戏逻辑。
 */
(function (PP) {
  const {
    createSky,
    createGround,
    createParkPath,
    createPond,
    createTrees,
    createBushes,
    swayTrees,
    createBenchesAndLamps,
    createGazebo,
    createSunAndClouds,
  } = PP;

  class World {
    constructor(canvas) {
      this.canvas = canvas;
      this.trees = [];
      this.skyDome = null;
      this.sunGlow = null;
      this.sunGroup = null;
      this._sunPos = new THREE.Vector3();
      this._cloudPos = new THREE.Vector3();

      this.renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: false,
        powerPreference: 'high-performance',
      });
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.6));
      this.renderer.setSize(window.innerWidth, window.innerHeight, false);
      this.renderer.outputColorSpace = THREE.SRGBColorSpace;
      this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
      this.renderer.toneMappingExposure = 1.12;

      this.scene = new THREE.Scene();
      // 晴空雾：淡蓝，让远处景物淡入天空而不是暖黄
      this.scene.fog = new THREE.FogExp2(0xcfe8f9, 0.0027);

      this.camera = new THREE.PerspectiveCamera(57, window.innerWidth / window.innerHeight, 0.1, 900);
      this.camera.position.set(0, 5.1, 11.8);
      this.camera.lookAt(0, 2.2, -8);

      this.clock = new THREE.Clock();

      this.#addLights();
      this.#buildScenery();

      window.addEventListener('resize', () => this.resize(), { passive: true });
    }

    #addLights() {
      // 晴空下的光：天光偏冷蓝，地面回弹保留暖绿
      const hemi = new THREE.HemisphereLight(0xe6f1ff, 0x6f8a62, 2.05);
      this.scene.add(hemi);

      const sunLight = new THREE.DirectionalLight(0xfff3dd, 2.5);
      sunLight.position.set(-18, 30, 12);
      this.scene.add(sunLight);

      const fill = new THREE.DirectionalLight(0xb8dcff, 0.6);
      fill.position.set(15, 8, -35);
      this.scene.add(fill);
    }

    #buildScenery() {
      this.skyDome = createSky(this);
      createGround(this);
      createParkPath(this);
      createPond(this);
      createTrees(this);
      createBushes(this);
      createBenchesAndLamps(this);
      createGazebo(this);
      const sun = createSunAndClouds(this);
      this.sunGlow = sun.sunGlow;
      this.sunGroup = sun.sunGroup;
    }

    /** 天空穹顶与薄云永远罩住相机，避免飞行时穿出天空、或飞到没云的地方 */
    followCamera() {
      if (this.skyDome) this.skyDome.position.copy(this.camera.position);

      this.camera.updateMatrixWorld();
      const tanV = Math.tan((this.camera.fov * Math.PI) / 360);
      const tanH = tanV * this.camera.aspect;

      // 太阳锁在画面左上角（约 28% 偏左、22% 偏上）：
      // 按相机 fov / 宽高比换算成相机空间偏移，横竖屏都不会跑出画面。
      if (this.sunGroup) {
        const D = 320;
        this._sunPos.set(-0.44 * tanH * D, 0.56 * tanV * D, -D);
        this.camera.localToWorld(this._sunPos);
        this.sunGroup.position.copy(this._sunPos);
      }

      if (!this.clouds) return;
      for (const cloud of this.clouds) {
        const c = cloud;
        // 云高占屏高固定比例，宽按精灵比例算 → 任何屏上云的形状一致
        const H = c.hFrac * 2 * c.d * tanV;
        const W = H / c.aspect;
        c.mesh.scale.set(W, H, 1);
        // 屏幕归一化坐标 → 相机空间 → 世界坐标
        this._cloudPos.set(c.fx * tanH * c.d, c.fy * tanV * c.d, -c.d);
        this.camera.localToWorld(this._cloudPos);
        c.mesh.position.copy(this._cloudPos);
      }
    }

    /** 光晕随音乐呼吸 */
    pulseSun(audioEnergy, lowEnergy) {
      if (!this.sunGlow) return;
      const o = 0.78 + audioEnergy * 0.18 + lowEnergy * 0.06;
      this.sunGlow.material.opacity = Math.min(1, o);
    }

    swayTrees(now, audioEnergy) {
      swayTrees(this.trees, now, audioEnergy);
    }

    resize() {
      this.renderer.setSize(window.innerWidth, window.innerHeight, false);
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.6));
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.fov = window.innerHeight > window.innerWidth ? 57 : 47;
      this.camera.updateProjectionMatrix();
    }

    render() {
      this.renderer.render(this.scene, this.camera);
    }
  }

  PP.World = World;
})(window.PP);
