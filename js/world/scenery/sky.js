/** 天空穹顶：顶点着色的渐变色球，永远跟随相机。 */
(function (PP) {
  function createSky(world) {
    const geo = new THREE.SphereGeometry(430, 32, 18);
    const pos = geo.attributes.position;
    const cols = [];
    const top = new THREE.Color('#8dc3d8');
    const mid = new THREE.Color('#f8cf9d');
    const low = new THREE.Color('#f2dfb8');

    for (let i = 0; i < pos.count; i++) {
      const ny = pos.getY(i) / 430;
      let c;
      if (ny > 0.18) c = mid.clone().lerp(top, Math.min(1, Math.max(0, (ny - 0.18) / 0.65)));
      else c = low.clone().lerp(mid, Math.min(1, Math.max(0, (ny + 0.5) / 0.68)));
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
      })
    );
    dome.position.set(0, 0, -150);
    world.scene.add(dome);
    return dome;
  }

  PP.createSky = createSky;
})(window.PP);
