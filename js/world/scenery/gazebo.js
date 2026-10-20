/** 开放式凉亭：石基、木柱、四角攒尖顶，在两排树之间露出身影。 */
(function (PP) {
  function createGazebo(world) {
    const x = -18;
    const z = -190;

    const stone = new THREE.MeshStandardMaterial({ color: 0xd2c9ae, roughness: 0.9 });
    const wood = new THREE.MeshStandardMaterial({ color: 0x896b50, roughness: 0.85 });
    const roofMat = new THREE.MeshStandardMaterial({ color: 0x718b70, roughness: 0.9 });

    const base = new THREE.Mesh(new THREE.BoxGeometry(8.2, 0.22, 6.6), stone);
    base.position.set(x, 0.015, z);
    world.scene.add(base);

    for (const dx of [-3.25, 3.25]) {
      for (const dz of [-2.45, 2.45]) {
        const col = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.17, 3.5, 8), wood);
        col.position.set(x + dx, 1.86, z + dz);
        world.scene.add(col);
      }
    }

    const roof = new THREE.Mesh(new THREE.ConeGeometry(5.55, 1.85, 4), roofMat);
    roof.rotation.y = Math.PI / 4;
    roof.position.set(x, 4.45, z);
    world.scene.add(roof);

    const cap = new THREE.Mesh(
      new THREE.SphereGeometry(0.24, 10, 8),
      new THREE.MeshStandardMaterial({ color: 0xe7d6ad, roughness: 0.7 })
    );
    cap.position.set(x, 5.42, z);
    world.scene.add(cap);
  }

  PP.createGazebo = createGazebo;
})(window.PP);
