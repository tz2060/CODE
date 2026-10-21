/** 单棵树 + 两侧行道树。树会被注册进 world.trees，供主循环做微风摆动。 */
(function (PP) {
  // 取自照片里树叶的深绿，比原来的低多边形配色更沉、更接近写实
  const GREENS = [0x46663d, 0x527747, 0x3d5a36];

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
      // 用球体而不是二十面体：二十面体是非索引几何，法线按面算，看起来很"块"
      const crown = new THREE.Mesh(
        new THREE.SphereGeometry(1.05 - j * 0.13, 10, 8),
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
    // 树列要盖住整段飞行（75 秒 × 8.5 ≈ 640 单位）再加上结尾那一段，
    // 否则片尾的飞机与照片会落在光秃秃的荒地上。
    for (let i = 0; i < 48; i++) {
      const z = 8 - i * 18.5;
      const gap = i < 4 ? 11 : 13;
      createTree(world, -gap - (i % 3) * 1.1, z, 0.82 + (i % 4) * 0.12);
      createTree(world, gap + (i % 2) * 1.6, z - 7, 0.88 + (i % 3) * 0.12);
    }
  }

  /** 路边灌木丛：一簇几团低矮的球，让草地不至于空成一片 */
  function createBushes(world) {
    const shades = [0x41603a, 0x4b6b42, 0x385432];

    for (let i = 0; i < 70; i++) {
      const z = -6 - i * 12.6;
      const side = i % 2 ? 1 : -1;
      const x = side * (6.4 + (i % 4) * 1.15);
      const s = 0.72 + (i % 5) * 0.14;

      const group = new THREE.Group();
      group.position.set(x, 0, z);
      group.scale.setScalar(s);

      const mat = new THREE.MeshStandardMaterial({
        color: shades[i % 3],
        roughness: 1,
      });

      for (let j = 0; j < 3; j++) {
        const blob = new THREE.Mesh(new THREE.SphereGeometry(0.52 + (j % 2) * 0.22, 8, 6), mat);
        blob.position.set((j - 1) * 0.62, 0.34 + (j % 2) * 0.2, ((j % 2) - 0.5) * 0.7);
        blob.scale.set(1.05, 0.82, 1);
        group.add(blob);
      }

      world.scene.add(group);
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
  PP.createBushes = createBushes;
  PP.swayTrees = swayTrees;
})(window.PP);
