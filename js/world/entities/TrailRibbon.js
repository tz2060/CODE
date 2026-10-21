/**
 * 纸飞机留下的发光轨迹
 * ---------------------------------------------------------------
 * 结尾穿越画面时用。把纸飞机走过的位置记成一条带子（ribbon）：
 * 越靠近机尾越细越淡，加法混合叠在画面上，像一道划过的光。
 *
 * 用带子而不是 Line：大多数平台上 Line 的线宽被固定成 1px，太细。
 * depthTest 关掉，让它始终压在记忆画面之上。
 */
(function (PP) {
  class TrailRibbon {
    constructor(scene, camera) {
      this.camera = camera;
      this.maxPoints = 140;
      this.points = [];

      const positions = new Float32Array(this.maxPoints * 2 * 3);
      const colors = new Float32Array(this.maxPoints * 2 * 4);
      const indices = [];
      for (let i = 0; i < this.maxPoints - 1; i++) {
        const a = i * 2;
        indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
      }

      this.geometry = new THREE.BufferGeometry();
      this.geometry.setAttribute(
        'position',
        new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage)
      );
      this.geometry.setAttribute(
        'color',
        new THREE.BufferAttribute(colors, 4).setUsage(THREE.DynamicDrawUsage)
      );
      this.geometry.setIndex(indices);
      this.geometry.setDrawRange(0, 0);

      this.material = new THREE.MeshBasicMaterial({
        vertexColors: true,
        transparent: true,
        depthTest: false,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
        fog: false,
      });

      this.mesh = new THREE.Mesh(this.geometry, this.material);
      this.mesh.frustumCulled = false;
      this.mesh.renderOrder = 6;
      this.mesh.visible = false;
      scene.add(this.mesh);

      /** 整体强度 0–1，穿越之后用来让轨迹缓缓熄灭 */
      this.strength = 1;

      this._tangent = new THREE.Vector3();
      this._side = new THREE.Vector3();
      this._camDir = new THREE.Vector3();
      this._up = new THREE.Vector3(0, 1, 0);
    }

    begin() {
      this.points.length = 0;
      this.strength = 1;
      this.mesh.visible = true;
      this.geometry.setDrawRange(0, 0);
    }

    clear() {
      this.points.length = 0;
      if (this.mesh) this.mesh.visible = false;
      this.geometry.setDrawRange(0, 0);
    }

    /** 记录一个位置（太近就跳过，避免堆积） */
    push(point) {
      const last = this.points[this.points.length - 1];
      if (last && last.distanceToSquared(point) < 0.0036) return;
      this.points.push(point.clone());
      if (this.points.length > this.maxPoints) this.points.shift();
    }

    /** 重建带子 */
    update(width = 0.2) {
      const n = this.points.length;
      if (n < 2) {
        this.geometry.setDrawRange(0, 0);
        return;
      }

      const pos = this.geometry.attributes.position;
      const col = this.geometry.attributes.color;
      this.camera.getWorldDirection(this._camDir);

      for (let i = 0; i < n; i++) {
        const p = this.points[i];
        const prev = this.points[Math.max(0, i - 1)];
        const next = this.points[Math.min(n - 1, i + 1)];

        this._tangent.subVectors(next, prev);
        if (this._tangent.lengthSq() < 1e-8) this._tangent.set(0, 0, -1);
        this._tangent.normalize();

        // 面向相机的宽度方向；若与视线平行则退化，换一个参考轴
        this._side.crossVectors(this._tangent, this._camDir);
        if (this._side.lengthSq() < 1e-6) this._side.crossVectors(this._tangent, this._up);
        this._side.normalize();

        const t = n > 1 ? i / (n - 1) : 1; // 0 = 最旧，1 = 最新
        const w = width * (0.1 + 0.9 * t);
        // 只有机头附近亮，往后迅速暗下去，才像一道划过的光而不是一根白棍
        const alpha = Math.pow(t, 2.4) * this.strength;

        const i0 = i * 2;
        pos.setXYZ(i0, p.x + this._side.x * w, p.y + this._side.y * w, p.z + this._side.z * w);
        pos.setXYZ(i0 + 1, p.x - this._side.x * w, p.y - this._side.y * w, p.z - this._side.z * w);

        // 机头偏白，尾端偏金
        const warm = 1 - t * 0.35;
        col.setXYZW(i0, 1.0, 0.93 * warm + 0.07, 0.78 * warm + 0.06, alpha);
        col.setXYZW(i0 + 1, 1.0, 0.93 * warm + 0.07, 0.78 * warm + 0.06, alpha);
      }

      pos.needsUpdate = true;
      col.needsUpdate = true;
      this.geometry.setDrawRange(0, (n - 1) * 6);
    }
  }

  PP.TrailRibbon = TrailRibbon;
})(window.PP);
