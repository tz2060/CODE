/** 步道两侧的长椅与路灯，重复七组铺满整条路。 */
(function (PP) {
  function createBenchesAndLamps(world) {
    const wood = new THREE.MeshStandardMaterial({ color: 0x92684b, roughness: 0.9 });
    const metal = new THREE.MeshStandardMaterial({ color: 0x627777, roughness: 0.55, metalness: 0.25 });

    for (let i = 0; i < 7; i++) {
      const z = 5 - i * 35;

      for (const x of [-8.8, 8.8]) {
        const seat = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.13, 0.52), wood);
        seat.position.set(x, 0.68, z);
        world.scene.add(seat);

        const back = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.6, 0.1), wood);
        back.position.set(x, 0.98, z + 0.23);
        world.scene.add(back);

        for (const lx of [-0.68, 0.68]) {
          const leg = new THREE.Mesh(new THREE.BoxGeometry(0.11, 0.65, 0.11), metal);
          leg.position.set(x + lx, 0.34, z);
          world.scene.add(leg);
        }

        const post = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.075, 4, 7), metal);
        post.position.set(x + (x < 0 ? -1.1 : 1.1), 2, z - 3);
        world.scene.add(post);

        const bulb = new THREE.Mesh(
          new THREE.SphereGeometry(0.18, 8, 8),
          new THREE.MeshBasicMaterial({ color: 0xffe2a6 })
        );
        bulb.position.set(post.position.x, 4.03, z - 3);
        world.scene.add(bulb);
      }
    }
  }

  PP.createBenchesAndLamps = createBenchesAndLamps;
})(window.PP);
