/** 夕阳与薄云。光晕透明度随音乐呼吸，因此把 sunGlow 返回给主循环。 */
(function (PP) {
  function createSunAndClouds(world) {
    const sun = new THREE.Mesh(
      new THREE.SphereGeometry(3.0, 20, 16),
      new THREE.MeshBasicMaterial({ color: 0xffedbf })
    );
    sun.position.set(14, 18, -1040);
    world.scene.add(sun);

    const sunGlow = new THREE.Mesh(
      new THREE.SphereGeometry(6.3, 20, 16),
      new THREE.MeshBasicMaterial({
        color: 0xffd88f,
        transparent: true,
        opacity: 0.14,
        depthWrite: false,
      })
    );
    sunGlow.position.copy(sun.position);
    world.scene.add(sunGlow);

    const cloudMat = new THREE.MeshBasicMaterial({
      color: 0xffefce,
      transparent: true,
      opacity: 0.25,
      depthWrite: false,
    });

    // 云不跟世界走，而是跟着相机在 z 上平移（见 World.followCamera）：
    // 否则飞到片尾时云早就在身后了，天空只剩一条空渐变。
    world.clouds = [];
    for (let i = 0; i < 11; i++) {
      const cloud = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 8), cloudMat);
      cloud.scale.set(7 + Math.random() * 6, 0.18 + Math.random() * 0.13, 0.8);
      const offsetZ = -72 - Math.random() * 165;
      cloud.position.set(-52 + Math.random() * 104, 28 + Math.random() * 18, offsetZ);
      world.scene.add(cloud);
      world.clouds.push({ mesh: cloud, offsetZ });
    }

    return { sun, sunGlow };
  }

  PP.createSunAndClouds = createSunAndClouds;
})(window.PP);
