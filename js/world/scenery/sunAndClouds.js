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
    for (let i = 0; i < 9; i++) {
      const cloud = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 8), cloudMat);
      cloud.scale.set(7 + Math.random() * 6, 0.18 + Math.random() * 0.13, 0.8);
      cloud.position.set(-38 + Math.random() * 76, 12 + Math.random() * 12, -100 - Math.random() * 220);
      world.scene.add(cloud);
    }

    return { sun, sunGlow };
  }

  PP.createSunAndClouds = createSunAndClouds;
})(window.PP);
