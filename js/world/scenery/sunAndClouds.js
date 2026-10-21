/**
 * 正午的太阳（放射光线）与蓬松白云。
 * ---------------------------------------------------------------
 * 太阳锁在「相机空间的左上前方」，由 World.followCamera 每帧换算到世界坐标，
 * 因此无论飞机怎么飞，它都稳稳停在画面左上角（参考图里太阳的位置）。
 * 光晕透明度随音乐呼吸，所以把 sunGlow 返回给主循环。
 */
(function (PP) {
  /** 放射状太阳：白色核心 + 星芒 + 柔光晕 */
  function makeSunTexture() {
    const S = 512;
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = S;
    const ctx = canvas.getContext('2d');
    const cx = S / 2;
    const cy = S / 2;

    // 柔光晕：只包住核心一小圈，外面交给离散星芒，
    // 这样天空保持蓝、射线才是亮点（连续大光晕会把整片天洗白）。
    const halo = ctx.createRadialGradient(cx, cy, 0, cx, cy, S * 0.5);
    halo.addColorStop(0, 'rgba(255,255,255,0.92)');
    halo.addColorStop(0.035, 'rgba(255,253,244,0.6)');
    halo.addColorStop(0.09, 'rgba(255,247,218,0.2)');
    halo.addColorStop(0.17, 'rgba(255,242,205,0.055)');
    halo.addColorStop(0.30, 'rgba(255,242,205,0.012)');
    halo.addColorStop(1, 'rgba(255,242,205,0)');
    ctx.fillStyle = halo;
    ctx.fillRect(0, 0, S, S);

    // 星芒（放射线），加法混合叠在光晕上
    ctx.globalCompositeOperation = 'lighter';
    const RAYS = 44;
    for (let i = 0; i < RAYS; i++) {
      const ang = (i / RAYS) * Math.PI * 2 + (i % 2 ? 0.015 : -0.01);
      const long = i % 3 === 0;
      const len = S * (long ? 0.49 : 0.31) * (0.85 + Math.random() * 0.3);
      const ex = cx + Math.cos(ang) * len;
      const ey = cy + Math.sin(ang) * len;
      const g = ctx.createLinearGradient(cx, cy, ex, ey);
      g.addColorStop(0, 'rgba(255,255,255,0.55)');
      g.addColorStop(0.35, 'rgba(255,250,236,0.2)');
      g.addColorStop(1, 'rgba(255,250,236,0)');
      ctx.strokeStyle = g;
      ctx.lineWidth = S * (long ? 0.012 : 0.006);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(ex, ey);
      ctx.stroke();
    }

    // 白色核心
    ctx.globalCompositeOperation = 'source-over';
    const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, S * 0.05);
    core.addColorStop(0, 'rgba(255,255,255,1)');
    core.addColorStop(0.55, 'rgba(255,255,255,0.92)');
    core.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = core;
    ctx.beginPath();
    ctx.arc(cx, cy, S * 0.05, 0, Math.PI * 2);
    ctx.fill();

    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }

  /** 蓬松白云：多个柔边白团沿横向叠成一朵 */
  function makeCloudTexture() {
    const W = 512;
    const H = 256;
    const canvas = document.createElement('canvas');
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');

    const blobs = 18;
    for (let i = 0; i < blobs; i++) {
      const t = i / (blobs - 1);
      const bx = W * (0.1 + 0.8 * t) + (Math.random() - 0.5) * W * 0.07;
      const by = H * (0.58 + (Math.random() - 0.5) * 0.24) - Math.sin(t * Math.PI) * H * 0.2;
      const br = H * (0.15 + Math.random() * 0.2);
      const g = ctx.createRadialGradient(bx, by, 0, bx, by, br);
      g.addColorStop(0, 'rgba(255,255,255,0.95)');
      g.addColorStop(0.55, 'rgba(252,253,255,0.6)');
      g.addColorStop(1, 'rgba(248,251,255,0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(bx, by, br, 0, Math.PI * 2);
      ctx.fill();
    }

    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }

  function createSunAndClouds(world) {
    // ---- 太阳：放射光线，锁在相机空间左上前方 ----
    const sunGroup = new THREE.Group();
    world.scene.add(sunGroup);

    const sunGlow = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: makeSunTexture(),
        transparent: true,
        opacity: 0.9,
        depthWrite: false,
        fog: false,
        blending: THREE.AdditiveBlending,
        toneMapped: false,
      })
    );
    sunGlow.scale.set(330, 330, 1);
    sunGroup.add(sunGlow);

    // 位置不在这里定死：World.followCamera 会按相机 fov / 宽高比，
    // 每帧把太阳换算到「画面左上约 28% / 22%」处，横竖屏都不会跑出画面。
    world.sunGroup = sunGroup;

    // ---- 白云：精灵（永远正对相机） ----
    // 位置用「屏幕归一化坐标」存，followCamera 每帧按 fov/aspect 换算成世界坐标：
    // 这样横屏 / 竖屏都能保证云落在可见天空里，不会因竖屏窄视角被裁掉。
    const cloudTexes = [makeCloudTexture(), makeCloudTexture(), makeCloudTexture()];
    world.clouds = [];

    for (let i = 0; i < 14; i++) {
      const cloud = new THREE.Sprite(
        new THREE.SpriteMaterial({
          map: cloudTexes[i % cloudTexes.length],
          transparent: true,
          opacity: 0.92,
          depthWrite: false,
          fog: false,
          toneMapped: false,
          color: 0xffffff,
        })
      );
      const d = 180 + Math.random() * 240; // 离相机距离
      const fx = -0.92 + Math.random() * 1.87; // 屏幕水平 -0.92..0.95
      const fy = 0.28 + Math.random() * 0.57; // 屏幕垂直 0.28..0.85（天空带）
      const hFrac = 0.1 + Math.random() * 0.08; // 云高约占屏高 10%~18%
      const aspect = 0.42 + Math.random() * 0.16; // 高 / 宽
      world.scene.add(cloud);
      world.clouds.push({ mesh: cloud, d, fx, fy, hFrac, aspect });
    }

    return { sunGlow, sunGroup };
  }

  PP.createSunAndClouds = createSunAndClouds;
})(window.PP);
