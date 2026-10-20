/** 单棵树 + 两侧行道树。树会被注册进 world.trees，供主循环做微风摆动。 */
(function (PP) {
  const GREENS = [0x618666, 0x72956b, 0x8da477];

  function createTree(world, x, z, s) {
    const g = new THREE.Group();
    g.position.set(x, 0, z);
    g.scale.setScalar(s);

    const trunk = new THREE.Mesh(
      new THREE.CylinderGeometry(0.18, 0.27, 2.5, 7),
      new THREE.MeshStandardMaterial({ color: 0x7e6046, roughness: 1 })
    );
    trunk.position.y = 1.25;
    g.add(trunk);

    for (let j = 0; j < 3; j++) {
      const crown = new THREE.Mesh(
        new THREE.IcosahedronGeometry(1.05 - j * 0.13, 1),
        new THREE.MeshStandardMaterial({
          color: GREENS[Math.round(Math.abs(x) + j) % 3],
          roughness: 1,
        })
      );
      crown.position.set((j - 1) * 0.24, 2.5 + j * 0.55, (j % 2) * 0.15);
      crown.scale.set(1.05 - j * 0.1, 1.05 + j * 0.07, 1);
      g.add(crown);
    }

    world.scene.add(g);
    world.trees.push({ group: g, phase: Math.random() * 6.28 });
  }

  function createTrees(world) {
    for (let i = 0; i < 34; i++) {
      const z = 8 - i * 18.5;
      const gap = i < 4 ? 11 : 13;
      createTree(world, -gap - (i % 3) * 1.1, z, 0.82 + (i % 4) * 0.12);
      createTree(world, gap + (i % 2) * 1.6, z - 7, 0.88 + (i % 3) * 0.12);
    }
  }

  /** 微风摆动：音频能量越大，枝叶越活跃 */
  function swayTrees(trees, now, audioEnergy) {
    for (let i = 0; i < trees.length; i++) {
      const t = trees[i];
      const breeze = 0.022 + audioEnergy * 0.018;
      t.group.rotation.z = Math.sin(now * 0.72 + t.phase) * breeze;
      t.group.rotation.x = Math.cos(now * 0.53 + t.phase * 0.8) * breeze * 0.34;
    }
  }

  PP.createTrees = createTrees;
  PP.swayTrees = swayTrees;
})(window.PP);
