/**
 * 草地：一整片草坪，外加几条交替的浅色光带，给地面一点呼吸感。
 * 不用贴图——纯色低多边形绿，双击即可跑（不依赖任何外部资源）。
 */
(function (PP) {
  function createGround(world) {
    const lawn = new THREE.Mesh(
      new THREE.PlaneGeometry(260, 900),
      new THREE.MeshStandardMaterial({ color: 0x74a066, roughness: 1 })
    );
    lawn.rotation.x = -Math.PI / 2;
    lawn.position.set(0, -0.13, -330);
    world.scene.add(lawn);

    // 25 条交替的浅色光带，让地面不那么死板
    for (let i = 0; i < 25; i++) {
      const strip = new THREE.Mesh(
        new THREE.PlaneGeometry(260, 7.8),
        new THREE.MeshBasicMaterial({
          color: i % 2 ? 0x83a86e : 0x6f9c60,
          transparent: true,
          opacity: 0.13,
        })
      );
      strip.rotation.x = -Math.PI / 2;
      strip.position.set(0, -0.12, 12 - i * 23);
      world.scene.add(strip);
    }
  }

  PP.createGround = createGround;
})(window.PP);
