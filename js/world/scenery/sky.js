/**
 * 天空穹顶：顶点着色的蓝色渐变球，永远跟随相机。
 * ---------------------------------------------------------------
 * 参考正午晴空：天顶是浓郁的蓝，越靠近地平线越淡、偏白，
 * 好让远处景物淡入雾里时与天空自然接上。
 */
(function (PP) {
  const R = 430;

  // 由地平线到天顶的色标（ny = 顶点 y / 半径，范围 -1..1）
  // 注意：相机只看到约 0–18° 仰角，对应 ny 只有 0–0.3，
  // 所以浓蓝要压在低 ny 段，否则永远看不到（之前就踩了这个坑）。
  const STOPS = [
    { y: -0.12, color: '#b7dcf6' },
    { y: 0.0, color: '#a6d3f4' },
    { y: 0.1, color: '#6fb0ef' },
    { y: 0.22, color: '#3f86e6' },
    { y: 0.45, color: '#2a6fda' },
    { y: 1.0, color: '#1d5ccb' },
  ];

  const cA = new THREE.Color();
  const cB = new THREE.Color();

  function colorAt(ny) {
    if (ny <= STOPS[0].y) return cA.set(STOPS[0].color);
    for (let i = 1; i < STOPS.length; i++) {
      if (ny <= STOPS[i].y) {
        const a = STOPS[i - 1];
        const b = STOPS[i];
        const t = (ny - a.y) / (b.y - a.y);
        cA.set(a.color);
        cB.set(b.color);
        return cA.lerp(cB, t);
      }
    }
    return cA.set(STOPS[STOPS.length - 1].color);
  }

  function createSky(world) {
    const geo = new THREE.SphereGeometry(R, 40, 24);
    const pos = geo.attributes.position;
    const cols = [];

    for (let i = 0; i < pos.count; i++) {
      const ny = pos.getY(i) / R;
      const c = colorAt(ny);
      cols.push(c.r, c.g, c.b);
    }

    geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));

    const dome = new THREE.Mesh(
      geo,
      new THREE.MeshBasicMaterial({
        vertexColors: true,
        side: THREE.BackSide,
        depthWrite: false,
        fog: false,
        // 天空不参与 ACES 色调映射：否则蓝会被压暗、发灰，
        // 这里要的就是参考图那种通透的蓝。
        toneMapped: false,
      })
    );
    dome.position.set(0, 0, -150);
    dome.renderOrder = -10; // 先画天空，其余景物再叠上去
    world.scene.add(dome);
    return dome;
  }

  PP.createSky = createSky;
})(window.PP);
