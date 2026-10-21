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
      this.sunDir = null;
      this.sunDist = 0;
      this._sunPos = new THREE.Vector3();

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

      // 太阳：固定在相机左上前方的一个世界方向上（距离恒定，大小不随飞行变化）
      if (this.sunGroup && this.sunDir) {
        const portrait = this.camera.aspect < 1;
        // 竖屏水平视角窄：把太阳的水平分量收一点，免得飞出画面
        const dx = portrait ? this.sunDir.x * 0.5 : this.sunDir.x;
        this._sunPos
          .set(dx, this.sunDir.y, this.sunDir.z)
          .normalize()
          .multiplyScalar(this.sunDist)
          .add(this.camera.position);
        this.sunGroup.position.copy(this._sunPos);
        if (this.sunGlow) {
          const s = portrait ? 150 : 280;
          if (this.sunGlow.scale.x !== s) this.sunGlow.scale.set(s, s, 1);
        }
      }

      // 云：散布在场景里，跟着相机在 z 上平移（避免片尾被甩在身后）
      if (!this.clouds) return;
      for (const cloud of this.clouds) {
        cloud.mesh.position.z = this.camera.position.z + cloud.offsetZ;
      }
    }

    /** 光晕随音乐呼吸 */
    pulseSun(audioEnergy, lowEnergy) {
      if (!this.sunGlow) return;
      const o = 0.72 + audioEnergy * 0.2 + lowEnergy * 0.06;
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
