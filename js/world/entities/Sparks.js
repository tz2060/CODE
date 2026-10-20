/** 光尘粒子：起飞过场和音符命中时的一次性小火花，用完即回收。 */
(function (PP) {
  const { clamp } = PP;

  class Sparks {
    constructor(scene) {
      this.scene = scene;
      this.items = [];
    }

    emit(position, velocity, life) {
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.035 + Math.random() * 0.035, 5, 5),
        new THREE.MeshBasicMaterial({
          color: Math.random() > 0.45 ? 0xfff1c6 : 0xffd28a,
          transparent: true,
          opacity: 0.9,
        })
      );
      mesh.position.copy(position);
      this.scene.add(mesh);
      this.items.push({ mesh, velocity, life, max: life });
    }

    /** 以某点为中心炸开一圈火花 */
    burst(position, count = 9, speed = 0.8) {
      for (let i = 0; i < count; i++) {
        const angle = (i / count) * Math.PI * 2;
        this.emit(
          position.clone(),
          new THREE.Vector3(
            Math.cos(angle) * (0.5 + Math.random() * speed),
            0.35 + Math.random() * 0.9,
            Math.sin(angle) * (0.5 + Math.random() * speed)
          ),
          0.55 + Math.random() * 0.35
        );
      }
    }

    /** 起飞时的升腾光尘 */
    launchVeil(count = 40) {
      for (let i = 0; i < count; i++) {
        this.emit(
          new THREE.Vector3((Math.random() - 0.5) * 5, 1.6 + Math.random() * 3, Math.random() * 2),
          new THREE.Vector3((Math.random() - 0.5) * 2, Math.random() * 1.5 - 0.2, -1 - Math.random() * 2),
          0.7 + Math.random() * 0.9
        );
      }
    }

    update(dt) {
      for (let i = this.items.length - 1; i >= 0; i--) {
        const spark = this.items[i];
        spark.life -= dt;
        spark.mesh.position.addScaledVector(spark.velocity, dt);
        spark.velocity.y -= dt * 0.35;
        spark.mesh.material.opacity = clamp(spark.life / spark.max, 0, 1);
        if (spark.life <= 0) this.#dispose(spark);
      }
    }

    clear() {
      for (const spark of this.items) this.#release(spark);
      this.items = [];
    }

    #release(spark) {
      this.scene.remove(spark.mesh);
      spark.mesh.geometry.dispose();
      spark.mesh.material.dispose();
    }

    #dispose(spark) {
      this.#release(spark);
      const index = this.items.indexOf(spark);
      if (index >= 0) this.items.splice(index, 1);
    }
  }

  PP.Sparks = Sparks;
})(window.PP);
