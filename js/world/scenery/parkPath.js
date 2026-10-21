/**
 * 公园步道：一条缓缓蜿蜒的土路，贯穿整片草坪。
 * 贴图是可无缝平铺的土色纹理（同样来自照片的配色基调），沿路长方向重复。
 */
(function (PP) {
  const { pathCenterX: centerX } = PP;

  function createParkPath(world) {
    const steps = 300;
    const zStart = 14;
    const zEnd = -850;
    const width = 11.2;

    const positions = [];
    const uvs = [];
    const indices = [];
    const leftPts = [];
    const rightPts = [];

    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const z = zStart + (zEnd - zStart) * t;
      const x = centerX(z);

      positions.push(x - width / 2, 0.012, z, x + width / 2, 0.012, z);
      // u 横跨路宽，v 沿路长（每 8 个单位一个循环）
      const v = (zStart - z) / 5;
      uvs.push(0, v, 1.6, v);

      leftPts.push(new THREE.Vector3(x - width / 2, 0.024, z));
      rightPts.push(new THREE.Vector3(x + width / 2, 0.024, z));
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
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geo.setIndex(indices);
    geo.computeVertexNormals();

    const dirt = new THREE.TextureLoader().load(PP.DIRT_TILE);
    dirt.colorSpace = THREE.SRGBColorSpace;
    dirt.wrapS = THREE.RepeatWrapping;
    dirt.wrapT = THREE.RepeatWrapping;
    dirt.anisotropy = Math.min(8, world.renderer.capabilities.getMaxAnisotropy());

    const path = new THREE.Mesh(
      geo,
      new THREE.MeshStandardMaterial({
        map: dirt,
        color: 0xc7bca6,
        roughness: 1,
        side: THREE.DoubleSide,
      })
    );
    world.scene.add(path);

    // 路缘：一圈很淡的浅色描边，让土路和草地的交界利落一点
    const edgeMat = new THREE.LineBasicMaterial({
      color: 0xd8cdb2,
      transparent: true,
      opacity: 0.32,
    });
    world.scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(leftPts), edgeMat));
    world.scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(rightPts), edgeMat));
  }

  PP.createParkPath = createParkPath;
})(window.PP);
