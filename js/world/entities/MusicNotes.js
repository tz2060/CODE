/**
 * 音符群
 * ---------------------------------------------------------------
 * 每个音符是一张画好音符符号的 Sprite。它们组成一条"旋律等高线"：
 * 真实音高决定高度，音程走向决定左右偏移。
 */
(function (PP) {
  const { clamp } = PP;
  const {
    COLLISION_RADII,
    PLANE_HIT_SKIN,
    FAR_END,
    LAUNCH,
    NOTE_COUNT,
    NOTE_Y_MAX,
    NOTE_Y_MIN,
  } = PP;

  const GLYPHS = ['♪', '♫', '♬', '♪', '♫'];
  /** 旋律分析就绪前的兜底音高轮廓 */
  const FALLBACK_CONTOUR = [0, 1, 3, 2, 5, 4, 2, 0, -1, 1, 3, 5, 4, 2, 1, 3, 6, 4, 2, 0, -1, 1];

  class MusicNotes {
    constructor(scene) {
      this.scene = scene;
      this.items = [];
      this.#build();
    }

    #build() {
      const spacing = (8.5 * (FAR_END - 13.5)) / (NOTE_COUNT - 1);

      for (let i = 0; i < NOTE_COUNT; i++) {
        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 128;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, 128, 128);
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.shadowColor = '#fff1b9';
        ctx.shadowBlur = 22;
        ctx.fillStyle = i % 3 === 0 ? '#ffe9a6' : i % 3 === 1 ? '#fff8dc' : '#c5e9d7';
        ctx.font = `bold ${68 + (i % 3) * 8}px Georgia, serif`;
        ctx.fillText(GLYPHS[i % GLYPHS.length], 64, 62);

        const texture = new THREE.CanvasTexture(canvas);
        texture.colorSpace = THREE.SRGBColorSpace;

        const material = new THREE.SpriteMaterial({
          map: texture,
          transparent: true,
          opacity: 0.76,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
        });

        const scale = 1.0 + (i % 4) * 0.28;
        const contour = FALLBACK_CONTOUR[i];
        const baseX = clamp((contour - 2.4) * 0.68, -3.2, 3.2);
        const baseY = clamp(2.35 + contour * 0.22, NOTE_Y_MIN, NOTE_Y_MAX);
        const baseZ = -16 - i * spacing;

        const sprite = new THREE.Sprite(material);
        sprite.scale.set(scale, scale, 1);
        sprite.position.set(baseX, baseY, baseZ);
        this.scene.add(sprite);

        this.items.push({ sprite, baseX, baseY, baseZ, phase: i * 0.83, scale, hit: false });
      }
    }

    get positions() {
      return this.items.map((n) => n.sprite.position);
    }

    /** 起飞时把音符按飞行速度重新铺到剩余曲长上 */
    layoutForFlight({ planeStartZ, flightStartTime, planeSpeed }) {
      const playableSpan = Math.max(
        1,
        FAR_END - flightStartTime - LAUNCH.firstNoteDelay - LAUNCH.tailGap
      );
      const noteStep = (planeSpeed * playableSpan) / Math.max(1, this.items.length - 1);

      this.items.forEach((note, i) => {
        note.baseZ = planeStartZ - planeSpeed * LAUNCH.firstNoteDelay - i * noteStep;
        note.sprite.position.z = note.baseZ;
      });
    }

    /** 每帧：跟随节拍呼吸、轻微漂浮 */
    update(now, beatPulse, audioEnergy) {
      for (const note of this.items) {
        const p = note.sprite.position;
        p.y = clamp(
          note.baseY + beatPulse * 0.035 + audioEnergy * 0.02 + Math.sin(now * 1.2 + note.phase) * 0.008,
          NOTE_Y_MIN,
          NOTE_Y_MAX
        );
        p.x = note.baseX + Math.sin(now * 0.48 + note.phase) * (0.012 + audioEnergy * 0.018);
        note.sprite.material.opacity = 0.48 + 0.3 * beatPulse + audioEnergy * 0.2;

        const pulse = 1 + beatPulse * 0.18 + audioEnergy * 0.08;
        note.sprite.scale.set(note.scale * pulse, note.scale * pulse, 1);
      }
    }

    /**
     * 用真实音高重排音符位置。
     * @param {number[]} pitches 平滑后的 MIDI 音高序列
     */
    applyPitchSeries(pitches) {
      const sorted = [...pitches].sort((a, b) => a - b);
      const low = sorted[Math.floor((sorted.length - 1) * 0.1)];
      const high = sorted[Math.floor((sorted.length - 1) * 0.9)];
      const span = Math.max(5, high - low);

      for (let i = 0; i < this.items.length; i++) {
        const note = this.items[i];
        const norm = clamp((pitches[i] - low) / span, 0, 1);
        const prev = pitches[Math.max(0, i - 1)];
        const next = pitches[Math.min(pitches.length - 1, i + 1)];
        const interval = clamp((next - prev) / 8, -0.65, 0.65);

        note.baseY = clamp(NOTE_Y_MIN + norm * (NOTE_Y_MAX - NOTE_Y_MIN), NOTE_Y_MIN, NOTE_Y_MAX);
        note.baseX = clamp((norm - 0.5) * 5.6 + interval * 0.72, -3.55, 3.55);
        note.sprite.position.set(note.baseX, note.baseY, note.baseZ);
      }
    }

    /**
     * 椭球碰撞检测：纸飞机给出若干采样点，**任一点**落入椭球即算命中；
     * 距离取最近的那个点，所以越靠近音符中心得分越高。
     * @param {THREE.Vector3[]} points 纸飞机的判定采样点（世界坐标）
     * @returns {{note: object, distance: number}[]}
     */
    findHits(points) {
      const hits = [];
      const reach = 1 + PLANE_HIT_SKIN;
      const reachSq = reach * reach;

      for (const note of this.items) {
        if (note.hit || !note.sprite.visible) continue;
        const np = note.sprite.position;

        let best = Infinity;
        for (let i = 0; i < points.length; i++) {
          const p = points[i];
          const dx = (p.x - np.x) / COLLISION_RADII.x;
          const dy = (p.y - np.y) / COLLISION_RADII.y;
          const dz = (p.z - np.z) / COLLISION_RADII.z;
          const d = dx * dx + dy * dy + dz * dz; // 先比平方，省掉内层开方
          if (d < best) best = d;
        }

        if (best < reachSq) hits.push({ note, distance: Math.sqrt(best) });
      }
      return hits;
    }

    /** 结尾：让所有音符一次性消失 */
    hideAll() {
      for (const note of this.items) note.sprite.visible = false;
    }

    reset() {
      for (const note of this.items) {
        note.hit = false;
        note.sprite.visible = true;
        note.sprite.position.set(note.baseX, note.baseY, note.baseZ);
        note.sprite.material.opacity = 0.76;
      }
    }
  }

  PP.MusicNotes = MusicNotes;
})(window.PP);
