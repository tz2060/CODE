/**
 * 公园步道：一条缓缓蜿蜒的土路，贯穿整片草坪。
 * 纯色路面，不用贴图——双击即可跑，不依赖外部资源。
 */
(function (PP) {
  const { pathCenterX: centerX } = PP;

  function createParkPath(world) {
    const steps = 300;
    const zStart = 14;
    const zEnd = -850;
    const width = 11.2;

    const positions = [];
    const indices = [];

    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const z = zStart + (zEnd - zStart) * t;
      const x = centerX(z);

      positions.push(x - width / 2, 0.012, z, x + width / 2, 0.012, z);

      if (i < steps) {
        const a = i * 2;
        const b = a + 1;
        const c = a + 2;
        const d = a + 3;
        indices.push(a, c, b, b, c, d);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();

    const path = new THREE.Mesh(
      geo,
      new THREE.MeshStandardMaterial({
        color: 0xb9b09d,
        roughness: 1,
        side: THREE.DoubleSide,
      })
    );
    world.scene.add(path);
  }

  PP.createParkPath = createParkPath;
})(window.PP);
