/**
 * 纸飞机
 * ---------------------------------------------------------------
 * 用五个三角面拼出机身（左/右翼、左右折边、中脊），加一圈描边和一条尾迹。
 * 姿态由"实际位移速度"反推：横移带侧倾，升降带俯仰，再缓慢回正。
 */
(function (PP) {
  const { damp, clamp, lerp } = PP;
  const { PLANE_NOSE_UP, PLANE_YAW, PLANE_HIT_POINTS } = PP;

  class PaperPlane {
    constructor(scene) {
      this.group = new THREE.Group();
      this.velocityX = 0;
      this.velocityY = 0;
      this.lastX = 0;
      this.lastY = 2.35;

      /** 判定采样点的世界坐标，预分配避免每帧新建 */
      this.hitPoints = PLANE_HIT_POINTS.map(() => new THREE.Vector3());

      this.#build();
      this.group.position.set(0, 2.35, 0);
      this.group.visible = false;
      scene.add(this.group);
    }

    /**
     * 机身外轮廓的判定采样点（世界坐标）。
     * 碰撞检测不再只看机身原点，而是拿这一圈点去和音符的椭球比，
     * 所以机头先碰到、翼尖蹭到都算命中。
     */
    getHitPoints() {
      // 渲染器要到 render 时才更新矩阵，这里先自己算一次，避免慢一帧
      this.group.updateMatrixWorld();
      const m = this.group.matrixWorld;
      for (let i = 0; i < PLANE_HIT_POINTS.length; i++) {
        const p = PLANE_HIT_POINTS[i];
        this.hitPoints[i].set(p[0], p[1], p[2]).applyMatrix4(m);
      }
      return this.hitPoints;
    }

    #build() {
      const ivory = new THREE.MeshStandardMaterial({
        color: 0xfff8e7,
        roughness: 0.58,
        metalness: 0.02,
        side: THREE.DoubleSide,
        flatShading: true,
      });
      const shaded = new THREE.MeshStandardMaterial({
        color: 0xe2e4da,
        roughness: 0.72,
        side: THREE.DoubleSide,
        flatShading: true,
      });
      const fold = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        emissive: 0xc8e3e7,
        emissiveIntensity: 0.15,
        roughness: 0.45,
        side: THREE.DoubleSide,
      });

      // 机头指向 -Z：每侧机翼是一片宽的主翼面和一条窄的折返面
      const tri = (name, verts, mat) => {
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.Float32BufferAttribute(verts.flat(), 3));
        g.computeVertexNormals();
        const mesh = new THREE.Mesh(g, mat);
        mesh.name = name;
        this.group.add(mesh);
        return mesh;
      };

      tri('left-wing', [[0, 0.02, -1.16], [-1.24, 0.12, 0.58], [-0.18, 0.08, 0.83]], ivory);
      tri('right-wing', [[0, 0.02, -1.16], [0.18, 0.08, 0.83], [1.24, 0.12, 0.58]], ivory);
      tri('left-fold', [[0, 0.035, -1.16], [-0.18, 0.08, 0.83], [-0.08, 0.18, 0.64]], shaded);
      tri('right-fold', [[0, 0.035, -1.16], [0.08, 0.18, 0.64], [0.18, 0.08, 0.83]], fold);
      tri('center-ridge', [[0, 0.045, -1.18], [-0.18, 0.08, 0.83], [0, 0.11, 0.72]], fold);

      const edgePoints = [
        new THREE.Vector3(0, 0.035, -1.16),
        new THREE.Vector3(-1.24, 0.12, 0.58),
        new THREE.Vector3(-0.18, 0.08, 0.83),
        new THREE.Vector3(0, 0.035, -1.16),
        new THREE.Vector3(1.24, 0.12, 0.58),
        new THREE.Vector3(0.18, 0.08, 0.83),
        new THREE.Vector3(0, 0.035, -1.16),
      ];
      this.group.add(
        new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(edgePoints),
          new THREE.LineBasicMaterial({ color: 0xb9c5bd, transparent: true, opacity: 0.72 })
        )
      );

      const trail = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(0, 0, 0.65),
          new THREE.Vector3(0, 0, 3.7),
        ]),
        new THREE.LineBasicMaterial({ color: 0xffecc2, transparent: true, opacity: 0.27 })
      );
      trail.name = 'trail';
      this.group.add(trail);
    }

    get position() {
      return this.group.position;
    }

    get visible() {
      return this.group.visible;
    }

    set visible(value) {
      this.group.visible = value;
    }

    /** 回到起飞前的静止姿态，并清掉速度记忆（避免瞬移造成的假侧倾） */
    reset(x = 0, y = 2.35, z = 0, scale = 1) {
      this.group.position.set(x, y, z);
      this.group.scale.setScalar(scale);
      this.group.rotation.set(PLANE_NOSE_UP, PLANE_YAW, 0);
      this.velocityX = 0;
      this.velocityY = 0;
      this.lastX = x;
      this.lastY = y;
    }

    /** 只重置速度记忆，位置保持不变（用于起飞瞬间） */
    resetVelocityMemory() {
      this.lastX = this.group.position.x;
      this.lastY = this.group.position.y;
      this.velocityX = 0;
      this.velocityY = 0;
    }

    /**
     * 由位移推导姿态。
     * @param {number} dt
     * @param {{ lowEnergy: number, bankFade: number }} opts
     */
    updateAttitude(dt, { lowEnergy = 0, bankFade = 1 } = {}) {
      const pos = this.group.position;
      const prevX = Number.isFinite(this.lastX) ? this.lastX : pos.x;
      const prevY = Number.isFinite(this.lastY) ? this.lastY : pos.y;
      const step = Math.max(dt, 0.001);

      this.velocityX = damp(this.velocityX, (pos.x - prevX) / step, 8.5, dt);
      this.velocityY = damp(this.velocityY, (pos.y - prevY) / step, 8.5, dt);
      this.lastX = pos.x;
      this.lastY = pos.y;

      const targetBank = clamp(-this.velocityX * 0.038, -0.43, 0.43) * bankFade;
      const targetPitch = clamp(
        PLANE_NOSE_UP - 0.018 - lowEnergy * 0.014 + this.velocityY * 0.038,
        -0.3,
        0.3
      );

      this.group.rotation.z = damp(this.group.rotation.z, targetBank, 5.8, dt);
      // 机头跟着转向略微偏转，真正的倾斜感由机翼提供
      this.group.rotation.y = PLANE_YAW - clamp(this.velocityX * 0.012, -0.13, 0.13) * bankFade;
      this.group.rotation.x = damp(this.group.rotation.x, targetPitch, 5.8, dt);
    }

    /** 过场里的展开动画 */
    playReveal(p, reveal) {
      this.group.scale.setScalar(lerp(0.16, 1, reveal));
      this.group.position.set(0, lerp(1.6, 2.35, p), lerp(3.0, -0.7, p * p * (3 - 2 * p)));
      this.group.rotation.y = PLANE_YAW;
      this.group.rotation.z = Math.sin(p * Math.PI) * 0.06;
      this.group.rotation.x = PLANE_NOSE_UP + Math.sin(p * Math.PI * 2) * 0.035;
    }
  }

  PP.PaperPlane = PaperPlane;
})(window.PP);
