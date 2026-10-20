/**
 * 旋律引导线
 * ---------------------------------------------------------------
 * 一条穿过起飞点与每个音符中心的虚线。它既是视觉指引，
 * 也是"中立飞行路径"的数据来源（routeAtZ）。
 */
(function (PP) {
  const { clamp, lerp } = PP;

  class MelodyGuide {
    constructor(scene) {
      this.scene = scene;
      this.anchor = new THREE.Vector3(0, 2.35, -0.7);
      this.line = null;
      this.notes = [];
    }

    build(notes) {
      this.notes = notes.items;
      const points = [this.anchor.clone(), ...notes.positions];
      const geo = new THREE.BufferGeometry().setFromPoints(points);
      geo.attributes.position.setUsage(THREE.DynamicDrawUsage);

      const material = new THREE.LineDashedMaterial({
        color: 0xfff8df,
        transparent: true,
        opacity: 0.66,
        depthWrite: false,
        dashSize: 0.92,
        gapSize: 0.42,
        scale: 1,
        linewidth: 2,
      });

      this.line = new THREE.Line(geo, material);
      this.line.computeLineDistances();
      this.line.renderOrder = 1;
      this.line.visible = false;
      this.scene.add(this.line);
    }

    setAnchor(x, y, z) {
      this.anchor.set(x, y, z);
    }

    set visible(value) {
      if (this.line) this.line.visible = value;
    }

    get visible() {
      return Boolean(this.line && this.line.visible);
    }

    update(beatPulse = 0, recalculateDistances = false, audioEnergy = 0) {
      if (!this.line) return;
      const attr = this.line.geometry.attributes.position;
      attr.setXYZ(0, this.anchor.x, this.anchor.y + beatPulse * 0.014 + audioEnergy * 0.01, this.anchor.z);

      for (let i = 0; i < this.notes.length; i++) {
        const p = this.notes[i].sprite.position;
        attr.setXYZ(i + 1, p.x, p.y, p.z);
      }
      attr.needsUpdate = true;
      if (recalculateDistances) this.line.computeLineDistances();

      this.line.material.opacity = 0.56 + beatPulse * 0.19 + audioEnergy * 0.16;
      this.line.material.scale = 1 + beatPulse * 0.12;
    }

    /**
     * 沿同一条三维折线插值，返回 z 处的中立路径坐标。
     * 玩家的拖拽是叠加在这条路线上的偏移量。
     */
    routeAtZ(z) {
      let ax = this.anchor.x;
      let ay = this.anchor.y;
      let az = this.anchor.z;
      if (z >= az) return { x: ax, y: ay };

      for (const note of this.notes) {
        const p = note.sprite.position;
        if (z >= p.z) {
          const t = clamp((az - z) / Math.max(0.0001, az - p.z), 0, 1);
          return { x: lerp(ax, p.x, t), y: lerp(ay, p.y, t) };
        }
        ax = p.x;
        ay = p.y;
        az = p.z;
      }

      const last = this.notes[this.notes.length - 1];
      return { x: last.sprite.position.x, y: last.sprite.position.y };
    }
  }

  PP.MelodyGuide = MelodyGuide;
})(window.PP);
