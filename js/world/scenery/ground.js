/**
 * 草地：一整片草坪。
 * 贴图来自结尾那张照片的配色（草地中位色 RGB(96,119,67)），可无缝平铺，
 * 近处看得见草叶的明暗，远处靠 mipmap 收成一片柔和的绿。
 */
(function (PP) {
  function createGround(world) {
    const maxAniso = world.renderer.capabilities.getMaxAnisotropy();

    const grass = new THREE.TextureLoader().load(PP.GRASS_TILE);
    grass.colorSpace = THREE.SRGBColorSpace;
    grass.wrapS = THREE.RepeatWrapping;
    grass.wrapT = THREE.RepeatWrapping;
    grass.repeat.set(36, 126); // 一格约 7 个单位
    grass.anisotropy = Math.min(8, maxAniso);

    const lawn = new THREE.Mesh(
      new THREE.PlaneGeometry(260, 900),
      new THREE.MeshStandardMaterial({ map: grass, roughness: 1 })
    );
    lawn.rotation.x = -Math.PI / 2;
    lawn.position.set(0, -0.13, -330);
    world.scene.add(lawn);
  }

  PP.createGround = createGround;
})(window.PP);
