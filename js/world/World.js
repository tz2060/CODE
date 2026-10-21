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
      this.scene.fog = new THREE.FogExp2(0xf5d2a0, 0.0027);

      this.camera = new THREE.PerspectiveCamera(57, window.innerWidth / window.innerHeight, 0.1, 900);
      this.camera.position.set(0, 5.1, 11.8);
      this.camera.lookAt(0, 2.2, -8);

      this.clock = new THREE.Clock();

      this.#addLights();
      this.#buildScenery();

      window.addEventListener('resize', () => this.resize(), { passive: true });
    }

    #addLights() {
      const hemi = new THREE.HemisphereLight(0xffe7bf, 0x6c8860, 2.15);
      this.scene.add(hemi);

      const sunLight = new THREE.DirectionalLight(0xffdfa5, 2.4);
      sunLight.position.set(-18, 30, 12);
      this.scene.add(sunLight);

      const fill = new THREE.DirectionalLight(0xb8dcff, 0.65);
      fill.position.set(15, 8, -35);
      this.scene.add(fill);
    }

    #buildScenery() {
      const before = this.scene.children.length;

      this.skyDome = createSky(this);
      createGround(this);
      createParkPath(this);
      createPond(this);
      createTrees(this);
      createBenchesAndLamps(this);
      createGazebo(this);
      this.sunGlow = createSunAndClouds(this).sunGlow;

      // 记录布景对象，结尾画面铺满时可以整体收起
      this.scenery = this.scene.children.slice(before);
    }

    /**
     * 结尾的全屏画面铺满视口后，公园其实一点都看不见了，
     * 但地面/树木仍然会在深度上挡住画面（尤其相机俯视时下缘会先打到地面），
     * 所以这时把整片布景收起来。
     */
    setSceneryVisible(visible) {
      if (!this.scenery) return;
      for (const obj of this.scenery) obj.visible = visible;
    }

    /** 天空穹顶永远罩住相机，避免飞行时穿出天空 */
    followCamera() {
      if (this.skyDome) this.skyDome.position.copy(this.camera.position);
    }

    /** 光晕随音乐呼吸 */
    pulseSun(audioEnergy, lowEnergy) {
      if (this.sunGlow) this.sunGlow.material.opacity = 0.12 + audioEnergy * 0.16 + lowEnergy * 0.06;
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
