/** 池塘：浅色石岸 + 微反光水面 + 稀疏的浮光。 */
(function (PP) {
  function createPond(world) {
    const bankShape = new THREE.Shape();
    bankShape.absellipse(0, 0, 11.1, 27.5, 0, Math.PI * 2, false, 0);
    const bank = new THREE.Mesh(
      new THREE.ShapeGeometry(bankShape, 56),
      new THREE.MeshStandardMaterial({ color: 0xc7bd9f, roughness: 1, side: THREE.DoubleSide })
    );
    bank.rotation.x = -Math.PI / 2;
    bank.position.set(18, -0.105, -125);
    world.scene.add(bank);

    const waterShape = new THREE.Shape();
    waterShape.absellipse(0, 0, 9.4, 24.6, 0, Math.PI * 2, false, 0);
    const water = new THREE.Mesh(
      new THREE.ShapeGeometry(waterShape, 56),
      new THREE.MeshStandardMaterial({
        color: 0x78b5bf,
        roughness: 0.25,
        metalness: 0.12,
        transparent: true,
        opacity: 0.9,
        side: THREE.DoubleSide,
      })
    );
    water.rotation.x = -Math.PI / 2;
    water.position.set(18, -0.075, -125);
    world.scene.add(water);

    const shorePts = [];
    for (let i = 0; i < 128; i++) {
      const a = (i / 128) * Math.PI * 2;
      shorePts.push(new THREE.Vector3(18 + 9.55 * Math.cos(a), -0.04, -125 + 24.9 * Math.sin(a)));
    }
    world.scene.add(
      new THREE.LineLoop(
        new THREE.BufferGeometry().setFromPoints(shorePts),
        new THREE.LineBasicMaterial({ color: 0xd9e5cf, transparent: true, opacity: 0.64 })
      )
    );

    // 水面浮光：几片柔软的高光，代替贴图
    const glintMat = new THREE.MeshBasicMaterial({
      color: 0xdff4e8,
      transparent: true,
      opacity: 0.38,
      depthWrite: false,
    });
    for (let i = 0; i < 8; i++) {
      const glint = new THREE.Mesh(new THREE.PlaneGeometry(1.0 + (i % 3) * 0.45, 0.035), glintMat);
      glint.rotation.x = -Math.PI / 2;
      glint.position.set(14 + (i % 4) * 2.0, -0.055, -137 + Math.floor(i / 4) * 19 + (i % 2) * 3);
      world.scene.add(glint);
    }
  }

  PP.createPond = createPond;
})(window.PP);
