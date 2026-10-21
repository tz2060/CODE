/**
 * 正午的太阳（放射光线）与蓬松白云。
 * ---------------------------------------------------------------
 * 太阳与云都是「场景里的实体」，不再按屏幕百分比钉死：
 *  - 太阳固定在相机左上前方的一个**世界方向**上（距离恒定，所以大小不随飞行变化）
 *  - 云散布在场景里，跟着相机在 z 上平移（避免飞到片尾时被甩在身后）
 * 太阳光晕随音乐呼吸，所以把 sunGlow 返回给主循环。
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

    // 柔光晕：只包住核心一小圈，外面交给离散星芒
    const halo = ctx.createRadialGradient(cx, cy, 0, cx, cy, S * 0.5);
    halo.addColorStop(0, 'rgba(255,255,255,0.82)');
    halo.addColorStop(0.03, 'rgba(255,253,244,0.45)');
    halo.addColorStop(0.08, 'rgba(255,247,218,0.14)');
    halo.addColorStop(0.16, 'rgba(255,242,205,0.04)');
    halo.addColorStop(0.28, 'rgba(255,242,205,0.008)');
    halo.addColorStop(1, 'rgba(255,242,205,0)');
    ctx.fillStyle = halo;
    ctx.fillRect(0, 0, S, S);

    // 星芒（放射线）：更细、更淡，让蓝色从缝里透出来
    ctx.globalCompositeOperation = 'lighter';
    const RAYS = 36;
    for (let i = 0; i < RAYS; i++) {
      const ang = (i / RAYS) * Math.PI * 2 + (i % 2 ? 0.015 : -0.01);
      const long = i % 3 === 0;
      const len = S * (long ? 0.49 : 0.3) * (0.85 + Math.random() * 0.3);
      const ex = cx + Math.cos(ang) * len;
      const ey = cy + Math.sin(ang) * len;
      const g = ctx.createLinearGradient(cx, cy, ex, ey);
      g.addColorStop(0, 'rgba(255,255,255,0.4)');
      g.addColorStop(0.35, 'rgba(255,250,236,0.12)');
      g.addColorStop(1, 'rgba(255,250,236,0)');
      ctx.strokeStyle = g;
      ctx.lineWidth = S * (long ? 0.009 : 0.0045);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(ex, ey);
      ctx.stroke();
    }

    // 白色核心
    ctx.globalCompositeOperation = 'source-over';
    const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, S * 0.05);
    core.addColorStop(0, 'rgba(255,255,255,1)');
    core.addColorStop(0.55, 'rgba(255,255,255,0.9)');
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
      const bx = W * (0.12 + 0.76 * t) + (Math.random() - 0.5) * W * 0.06;
      const by = H * (0.58 + (Math.random() - 0.5) * 0.22) - Math.sin(t * Math.PI) * H * 0.18;
      const br = H * (0.14 + Math.random() * 0.18);
      const g = ctx.createRadialGradient(bx, by, 0, bx, by, br);
      g.addColorStop(0, 'rgba(255,255,255,0.95)');
      g.addColorStop(0.55, 'rgba(252,253,255,0.6)');
      g.addColorStop(1, 'rgba(248,251,255,0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(bx, by, br, 0, Math.PI * 2);
      ctx.fill();
    }

    // 关键：把四条边羽化到 0。否则画布边缘的团块被硬裁断，
    // 精灵方块的直边就会露出来（之前截图里那些方块就是这么来的）。
    ctx.globalCompositeOperation = 'destination-in';
    const mx = ctx.createLinearGradient(0, 0, W, 0);
    mx.addColorStop(0, 'rgba(0,0,0,0)');
    mx.addColorStop(0.07, 'rgba(0,0,0,1)');
    mx.addColorStop(0.93, 'rgba(0,0,0,1)');
    mx.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = mx;
    ctx.fillRect(0, 0, W, H);

    const my = ctx.createLinearGradient(0, 0, 0, H);
    my.addColorStop(0, 'rgba(0,0,0,0)');
    my.addColorStop(0.1, 'rgba(0,0,0,1)');
    my.addColorStop(0.9, 'rgba(0,0,0,1)');
    my.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = my;
    ctx.fillRect(0, 0, W, H);
    ctx.globalCompositeOperation = 'source-over';

    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }

  function createSunAndClouds(world) {
    // ---- 太阳：固定在相机左上前方的一个世界方向上 ----
    const sunGroup = new THREE.Group();
    world.scene.add(sunGroup);

    const sunGlow = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: makeSunTexture(),
        transparent: true,
        opacity: 0.75,
        depthWrite: false,
        fog: false,
        blending: THREE.AdditiveBlending,
        toneMapped: false,
      })
    );
    sunGlow.scale.set(280, 280, 1);
    sunGroup.add(sunGlow);

    // 方向：左、上、前；距离恒定 → 飞行中大小不变
    // 仰角约 9°：飞机飞低时相机会俯视 ~16°，留出余量免得太阳被顶出画面
    world.sunDir = new THREE.Vector3(-0.4, 0.16, -0.9).normalize();
    world.sunDist = 540;
    world.sunGroup = sunGroup;

    // ---- 白云：散布在场景里，跟着相机在 z 上平移 ----
    const cloudTexes = [makeCloudTexture(), makeCloudTexture(), makeCloudTexture()];
    world.clouds = [];

    for (let i = 0; i < 18; i++) {
      const cloud = new THREE.Sprite(
        new THREE.SpriteMaterial({
          map: cloudTexes[i % cloudTexes.length],
          transparent: true,
          opacity: 0.88,
          depthWrite: false,
          fog: false,
          toneMapped: false,
          color: 0xffffff,
        })
      );
      const offsetZ = -210 - Math.random() * 190;
      // x 向中间聚一些（竖屏窄视角也能看到几朵），少量放到更外侧铺满横屏
      const far = Math.random() < 0.3;
      const range = far ? 340 : 180;
      const sign = Math.random() < 0.5 ? -1 : 1;
      const x = sign * Math.pow(Math.random(), 1.4) * range;
      // 尺寸按距离给，保证不同远近的云角大小接近
      const w = Math.abs(offsetZ) * (0.22 + Math.random() * 0.16);
      cloud.scale.set(w, w * (0.42 + Math.random() * 0.16), 1);
      cloud.position.set(x, 52 + Math.random() * 40, offsetZ);
      world.scene.add(cloud);
      world.clouds.push({ mesh: cloud, offsetZ });
    }

    return { sunGlow, sunGroup };
  }

  PP.createSunAndClouds = createSunAndClouds;
})(window.PP);
