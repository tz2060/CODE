/**
 * 弹幕关「幻月」——2D 纵版弹幕
 * ---------------------------------------------------------------
 * 分数破 130 后进入的「下一关」。主角仍是纸飞机：
 *   · 拖动 / 方向键移动，自动向上发射纸屑攻击「幻月」
 *   · 躲弹幕、擦弹得分；被击中掉一条命
 *   · 击败幻月 → 通关；三条命耗尽 → 失败
 *
 * 画风沿用开场的午后公园：蓝天渐变、左上太阳、白云、草地、土路、树。
 * 背景缓存在离屏 canvas，子弹按颜色预渲染，主循环只 drawImage。
 */
(function (PP) {
  const TAU = Math.PI * 2;

  const PALETTE = {
    amber: [255, 217, 143],
    mint: [168, 236, 205],
    lavender: [201, 185, 255],
    coral: [255, 166, 144],
    white: [255, 255, 255],
  };

  /** 第二关纸飞机整体放大倍数：机身与受弹判定同比放大（用户要求大 0.5 倍） */
  const PLAYER_SCALE = 1.5;

  /**
   * 移动速度（屏高比例，每秒）。固定值，触摸与键盘一样快。
   * 弹幕最慢的一档是 0.16H，这里取 0.15H —— 比弹幕慢一点，才有躲的空间。
   */
  const MOVE_SPEED_RATIO = 0.15;

  /** 移动时机身最大倾斜角（弧度，约 19°） */
  const MAX_TILT = 0.34;

  function clamp(v, a, b) {
    return v < a ? a : v > b ? b : v;
  }
  function rand(a, b) {
    return a + Math.random() * (b - a);
  }

  class DanmakuGame {
    constructor(canvas, sfx) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.sfx = typeof sfx === 'function' ? sfx : () => {};
      this.W = 1;
      this.H = 1;
      this.dpr = 1;
      this.hitR = 4;
      this.grazeR = 16;
      this.bg = null;
      this.sprites = {};
      this.keys = { left: false, right: false, up: false, down: false };
      this.pointer = { active: false, id: null, x: 0, y: 0 };
      this.bound = false;
      this.active = false;

      this.resize();
      this.#buildSprites();
      this.reset();
      this.#bind();
    }

    // ---------------------------------------------------------------
    // 尺寸 / 预渲染
    // ---------------------------------------------------------------

    resize() {
      this.dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.W = Math.max(1, window.innerWidth);
      this.H = Math.max(1, window.innerHeight);
      this.canvas.width = Math.round(this.W * this.dpr);
      this.canvas.height = Math.round(this.H * this.dpr);
      this.canvas.style.width = this.W + 'px';
      this.canvas.style.height = this.H + 'px';
      this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

      const m = Math.min(this.W, this.H);
      this.playerScale = PLAYER_SCALE;
      this.hitR = Math.max(3, m * 0.007) * PLAYER_SCALE;
      this.grazeR = this.hitR * 4.2;
      // 固定移动速度：比最慢的一档弹幕（0.16H）略低
      this.moveSpeed = Math.max(70, this.H * MOVE_SPEED_RATIO);

      this.#buildBackground();

      if (this.player) {
        this.player.r = this.hitR;
        this.player.graze = this.grazeR;
        // 活动范围 = 整个屏幕，只内缩一个判定直径，机身贴边也不会飞出去
        const edge = this.hitR * 2;
        this.player.x = clamp(this.player.x, edge, this.W - edge);
        this.player.y = clamp(this.player.y, edge, this.H - edge);
        this.player.tx = clamp(this.player.tx, 0, this.W);
        this.player.ty = clamp(this.player.ty, 0, this.H);
      }
      if (this.boss) {
        this.boss.r = m * 0.085;
      }
    }

    #buildSprites() {
      this.sprites = {};
      for (const key in PALETTE) {
        const [r, g, b] = PALETTE[key];
        const S = 64;
        const c = document.createElement('canvas');
        c.width = c.height = S;
        const x = c.getContext('2d');
        const grd = x.createRadialGradient(S / 2, S / 2, 0, S / 2, S / 2, S / 2);
        grd.addColorStop(0, 'rgba(255,255,255,1)');
        grd.addColorStop(0.22, `rgba(${r},${g},${b},0.98)`);
        grd.addColorStop(0.48, `rgba(${r},${g},${b},0.5)`);
        grd.addColorStop(1, `rgba(${r},${g},${b},0)`);
        x.fillStyle = grd;
        x.beginPath();
        x.arc(S / 2, S / 2, S / 2, 0, TAU);
        x.fill();
        this.sprites[key] = c;
      }
    }

    #buildBackground() {
      const W = this.W;
      const H = this.H;
      const dpr = this.dpr;
      const c = document.createElement('canvas');
      c.width = Math.round(W * dpr);
      c.height = Math.round(H * dpr);
      const g = c.getContext('2d');
      g.setTransform(dpr, 0, 0, dpr, 0, 0);

      const horizon = H * 0.64;

      // 天空
      const sky = g.createLinearGradient(0, 0, 0, horizon);
      sky.addColorStop(0, '#2f6fd6');
      sky.addColorStop(0.5, '#3f86e6');
      sky.addColorStop(0.82, '#7fb6ef');
      sky.addColorStop(1, '#a6d3f4');
      g.fillStyle = sky;
      g.fillRect(0, 0, W, horizon + 1);

      // 太阳（左上）
      const sx = W * 0.2;
      const sy = H * 0.13;
      const sr = H * 0.2;
      const sun = g.createRadialGradient(sx, sy, 0, sx, sy, sr);
      sun.addColorStop(0, 'rgba(255,255,255,0.95)');
      sun.addColorStop(0.16, 'rgba(255,250,228,0.7)');
      sun.addColorStop(0.42, 'rgba(255,244,205,0.16)');
      sun.addColorStop(1, 'rgba(255,244,205,0)');
      g.fillStyle = sun;
      g.fillRect(0, 0, W, horizon);

      // 太阳星芒
      g.save();
      g.globalCompositeOperation = 'lighter';
      for (let i = 0; i < 18; i++) {
        const a = (i / 18) * TAU;
        g.strokeStyle = 'rgba(255,250,230,0.2)';
        g.lineWidth = i % 3 === 0 ? 1.5 : 0.7;
        g.beginPath();
        g.moveTo(sx, sy);
        g.lineTo(sx + Math.cos(a) * sr * 0.92, sy + Math.sin(a) * sr * 0.92);
        g.stroke();
      }
      g.restore();

      // 白云
      const cloud = (cx, cy, cw, alpha) => {
        g.save();
        g.globalAlpha = alpha;
        const blobs = 6;
        for (let i = 0; i < blobs; i++) {
          const t = i / (blobs - 1);
          const bx = cx + (t - 0.5) * cw;
          const by = cy - Math.sin(t * Math.PI) * cw * 0.1;
          const br = cw * (0.16 + Math.random() * 0.12);
          const gr = g.createRadialGradient(bx, by, 0, bx, by, br);
          gr.addColorStop(0, 'rgba(255,255,255,0.95)');
          gr.addColorStop(0.6, 'rgba(250,253,255,0.55)');
          gr.addColorStop(1, 'rgba(248,251,255,0)');
          g.fillStyle = gr;
          g.beginPath();
          g.arc(bx, by, br, 0, TAU);
          g.fill();
        }
        g.restore();
      };
      cloud(W * 0.72, H * 0.16, W * 0.34, 0.85);
      cloud(W * 0.4, H * 0.26, W * 0.26, 0.7);
      cloud(W * 0.88, H * 0.32, W * 0.2, 0.6);
      cloud(W * 0.16, H * 0.4, W * 0.22, 0.5);

      // 草地
      g.fillStyle = '#74a066';
      g.fillRect(0, horizon, W, H - horizon);
      g.fillStyle = 'rgba(131,168,110,0.18)';
      const band = (H - horizon) / 9;
      for (let i = 0; i < 9; i += 2) {
        g.fillRect(0, horizon + i * band, W, band * 0.7);
      }

      // 土路（下方中央蜿蜒到地平线）
      g.fillStyle = '#b9b09d';
      g.beginPath();
      g.moveTo(W * 0.42, H);
      g.bezierCurveTo(W * 0.4, H * 0.9, W * 0.53, H * 0.78, W * 0.5, horizon);
      g.lineTo(W * 0.55, horizon);
      g.bezierCurveTo(W * 0.58, H * 0.78, W * 0.47, H * 0.9, W * 0.58, H);
      g.closePath();
      g.fill();

      // 树（两侧，深绿）
      const tree = (x, baseY, s) => {
        g.fillStyle = '#5a4a38';
        g.fillRect(x - s * 0.06, baseY - s * 0.5, s * 0.12, s * 0.5);
        const greens = ['#46663d', '#527747', '#3d5a36'];
        for (let i = 0; i < 3; i++) {
          g.fillStyle = greens[i];
          g.beginPath();
          g.arc(x + (i - 1) * s * 0.22, baseY - s * 0.62 - i * s * 0.12, s * 0.34, 0, TAU);
          g.fill();
        }
      };
      const trees = [
        [W * 0.06, horizon + 6, H * 0.1],
        [W * 0.15, horizon + 10, H * 0.08],
        [W * 0.28, horizon + 4, H * 0.07],
        [W * 0.86, horizon + 8, H * 0.1],
        [W * 0.94, horizon + 12, H * 0.09],
        [W * 0.74, horizon + 5, H * 0.075],
      ];
      for (const [x, y, s] of trees) tree(x, y, s);

      // 地平线柔光
      const hz = g.createLinearGradient(0, horizon - H * 0.06, 0, horizon + H * 0.02);
      hz.addColorStop(0, 'rgba(255,248,224,0)');
      hz.addColorStop(0.6, 'rgba(255,248,224,0.22)');
      hz.addColorStop(1, 'rgba(255,248,224,0)');
      g.fillStyle = hz;
      g.fillRect(0, horizon - H * 0.06, W, H * 0.08);

      this.bg = c;
    }

    // ---------------------------------------------------------------
    // 输入
    // ---------------------------------------------------------------

    #bind() {
      if (this.bound) return;
      this.bound = true;
      const c = this.canvas;

      c.addEventListener(
        'pointerdown',
        (e) => {
          if (!this.active) return;
          e.preventDefault();
          this.pointer.active = true;
          this.pointer.id = e.pointerId;
          this.pointer.x = e.clientX;
          this.pointer.y = e.clientY;
          // 目标点从飞机当前位置起步：按下只负责射击，滑动才移动
          const p = this.player;
          if (p && p.alive) {
            p.tx = p.x;
            p.ty = p.y;
          }
          // 点击屏幕射击（按住连发）
          this.shooting = true;
          this.shootCd = 0;
          this.#shoot();
          c.setPointerCapture?.(e.pointerId);
        },
        { passive: false }
      );

      c.addEventListener(
        'pointermove',
        (e) => {
          if (!this.active || !this.pointer.active || e.pointerId !== this.pointer.id) return;
          const dx = e.clientX - this.pointer.x;
          const dy = e.clientY - this.pointer.y;
          this.pointer.x = e.clientX;
          this.pointer.y = e.clientY;
          // 只累积目标点，实际位移由 #updatePlayer 按固定速度推进
          const p = this.player;
          if (p && p.alive) {
            p.tx = clamp(p.tx + dx, 0, this.W);
            p.ty = clamp(p.ty + dy, 0, this.H);
          }
        },
        { passive: true }
      );

      const release = (e) => {
        if (e.pointerId === this.pointer.id) {
          this.pointer.active = false;
          this.pointer.id = null;
          // 松手即停：把目标点收回到飞机当前位置，免得它还追着旧目标滑一段
          const p = this.player;
          if (p) {
            p.tx = p.x;
            p.ty = p.y;
          }
        }
        this.shooting = false;
      };
      c.addEventListener('pointerup', release);
      c.addEventListener('pointercancel', release);

      window.addEventListener('keydown', (e) => this.#key(e, true));
      window.addEventListener('keyup', (e) => this.#key(e, false));
      window.addEventListener('resize', () => this.resize(), { passive: true });
    }

    #key(e, down) {
      if (!this.active) return;
      const k = e.key;
      if (k === 'ArrowLeft' || k === 'a' || k === 'A') this.keys.left = down;
      else if (k === 'ArrowRight' || k === 'd' || k === 'D') this.keys.right = down;
      else if (k === 'ArrowUp' || k === 'w' || k === 'W') this.keys.up = down;
      else if (k === 'ArrowDown' || k === 's' || k === 'S') this.keys.down = down;
      else if (k === ' ' || k === 'Spacebar') {
        this.keyShoot = down;
        if (down) {
          this.shootCd = 0;
          this.#shoot();
        }
      } else return;
      e.preventDefault();
    }

    // ---------------------------------------------------------------
    // 生命周期
    // ---------------------------------------------------------------

    reset() {
      const m = Math.min(this.W, this.H);
      this.active = true;
      this.state = 'intro';
      this.timer = 0;
      this.battleTime = 0;
      this.score = 0;
      this.graze = 0;
      this.lives = 3;
      this.bullets = [];
      this.pbullets = [];
      this.particles = [];
      this.hitFlash = 0;
      this.result = null;
      this.finished = false;
      this.shootCd = 0;
      this.shooting = false;
      this.keyShoot = false;
      this.grazeSfxT = 0;
      this.fireT = 0;
      this.pointer.active = false;
      this.pointer.id = null;
      this.keys.left = this.keys.right = this.keys.up = this.keys.down = false;

      this.player = {
        x: this.W * 0.5,
        y: this.H * 0.8,
        r: this.hitR,
        graze: this.grazeR,
        inv: 0,
        alive: true,
        /** 机身倾斜角（弧度）：按水平速度平滑趋近，松手回正 */
        roll: 0,
        /** 触摸目标点：手指位移累积到这里，飞机以固定速度追过去 */
        tx: this.W * 0.5,
        ty: this.H * 0.8,
      };
      this.boss = {
        x: this.W * 0.5,
        y: this.H * 0.16,
        r: m * 0.085,
        hp: 480,
        maxHp: 480,
        t: 0,
        patternT: 0,
        pattern: 0,
      };
    }

    show() {
      this.active = true;
      this.canvas.classList.add('on');
    }

    hide() {
      this.active = false;
      this.canvas.classList.remove('on');
      this.pointer.active = false;
      this.pointer.id = null;
    }

    // ---------------------------------------------------------------
    // 更新
    // ---------------------------------------------------------------

    update(dt) {
      if (!this.active) return;
      dt = Math.min(dt, 0.05);

      if (this.state === 'intro') {
        this.timer += dt;
        if (this.timer > 1.6) {
          this.state = 'battle';
          this.timer = 0;
        }
        return;
      }

      if (this.state === 'clear' || this.state === 'dead') {
        this.timer += dt;
        if (this.timer > 2.2 && !this.finished) {
          this.finished = true;
          this.result = this.state;
        }
        return;
      }

      this.battleTime += dt;
      this.#updatePlayer(dt);
      this.#updateBoss(dt);
      this.#updateBullets(dt);
      this.#updateParticles(dt);
      if (this.grazeSfxT > 0) this.grazeSfxT -= dt;
      this.#collide();

      if (this.battleTime > 70 && this.state === 'battle') this.#clear();
    }

    #updatePlayer(dt) {
      const p = this.player;
      if (!p.alive) return;

      // 固定速度移动：触摸是「以固定速度追手指」，键盘是「以固定速度走方向」。
      // 两者共用 this.moveSpeed，所以不会出现「手快飞机就快」的差别。
      const step = this.moveSpeed * dt;
      let mx = 0;
      let my = 0;

      if (this.pointer.active) {
        const dx = p.tx - p.x;
        const dy = p.ty - p.y;
        const d = Math.hypot(dx, dy);
        if (d > 0.6) {
          const s = Math.min(d, step); // 到点即停，不会过冲
          mx = (dx / d) * s;
          my = (dy / d) * s;
        }
      } else {
        let dx = (this.keys.right ? 1 : 0) - (this.keys.left ? 1 : 0);
        let dy = (this.keys.down ? 1 : 0) - (this.keys.up ? 1 : 0);
        if (dx || dy) {
          const l = Math.hypot(dx, dy);
          mx = (dx / l) * step;
          my = (dy / l) * step;
        }
      }

      p.x += mx;
      p.y += my;

      // 活动范围与屏幕对齐：上下左右都只留一个判定直径，能飞到屏幕边缘
      const m = p.r * 2;
      p.x = clamp(p.x, m, this.W - m);
      p.y = clamp(p.y, m, this.H - m);

      // 机身倾斜：按水平速度给角度，松开手指自然回正
      const target = dt > 0 ? clamp(mx / dt / this.moveSpeed, -1, 1) * MAX_TILT : 0;
      p.roll += (target - p.roll) * (1 - Math.exp(-14 * dt));

      if (p.inv > 0) p.inv -= dt;

      // 点击 / 按住屏幕射击（桌面端空格）
      if (this.shooting || this.keyShoot) {
        this.shootCd -= dt;
        if (this.shootCd <= 0) {
          this.shootCd = 0.1;
          this.#shoot();
        }
      }
    }

    #shoot() {
      const p = this.player;
      if (!p.alive) return;
      const S = this.playerScale;
      const vy = -this.H * 0.95;
      // 子弹从机头两侧射出（跟着放大后的机身走）
      for (const ox of [-7 * S, 7 * S]) {
        this.pbullets.push({
          x: p.x + ox,
          y: p.y - 20 * S,
          vx: 0,
          vy,
          r: Math.max(2.2, this.hitR * 0.55),
          dmg: 1,
          dead: false,
        });
      }
      this.sfx('shoot');
    }

    #updateBoss(dt) {
      const b = this.boss;
      b.t += dt;
      b.patternT += dt;
      if (b.patternT > 6) {
        b.patternT = 0;
        b.pattern = (b.pattern + 1) % 4;
      }
      b.x = this.W * 0.5 + Math.sin(b.t * 0.55) * this.W * 0.2;
      b.y = this.H * 0.16 + Math.sin(b.t * 0.9) * this.H * 0.02;

      this.fireT -= dt;
      if (this.fireT <= 0) this.#fire(b.pattern);
    }

    #fire(pat) {
      const b = this.boss;
      const p = this.player;
      const H = this.H;

      if (pat === 0) {
        // 环形弹幕
        this.fireT = 0.85;
        const n = 18;
        const base = b.t * 0.6;
        for (let i = 0; i < n; i++) {
          const a = base + (TAU / n) * i;
          this.#spawnBullet(b.x, b.y, Math.cos(a) * H * 0.16, Math.sin(a) * H * 0.16, 6, 'amber');
        }
      } else if (pat === 1) {
        // 螺旋
        this.fireT = 0.07;
        const a = b.t * 1.8;
        for (let i = 0; i < 2; i++) {
          const ang = a + i * Math.PI;
          this.#spawnBullet(b.x, b.y, Math.cos(ang) * H * 0.19, Math.sin(ang) * H * 0.19, 5.5, 'lavender');
        }
      } else if (pat === 2) {
        // 狙击
        this.fireT = 0.7;
        const aim = Math.atan2(p.y - b.y, p.x - b.x);
        for (let i = -2; i <= 2; i++) {
          const ang = aim + i * 0.16;
          this.#spawnBullet(b.x, b.y, Math.cos(ang) * H * 0.24, Math.sin(ang) * H * 0.24, 6, 'coral');
        }
      } else {
        // 星雨
        this.fireT = 0.13;
        const x = rand(this.W * 0.1, this.W * 0.9);
        this.#spawnBullet(x, -10, rand(-20, 20), H * 0.22, 5, 'mint');
      }
    }

    #spawnBullet(x, y, vx, vy, r, color) {
      if (this.bullets.length > 700) return;
      this.bullets.push({ x, y, vx, vy, r, color, grazed: false, dead: false });
    }

    #updateBullets(dt) {
      const W = this.W;
      const H = this.H;

      for (const b of this.bullets) {
        b.x += b.vx * dt;
        b.y += b.vy * dt;
        if (b.x < -40 || b.x > W + 40 || b.y < -60 || b.y > H + 40) b.dead = true;
      }
      if (this.bullets.length) this.bullets = this.bullets.filter((b) => !b.dead);

      for (const b of this.pbullets) {
        b.x += b.vx * dt;
        b.y += b.vy * dt;
        if (b.y < -30) b.dead = true;
      }
      if (this.pbullets.length) this.pbullets = this.pbullets.filter((b) => !b.dead);
    }

    #collide() {
      const p = this.player;
      const boss = this.boss;

      // 玩家纸屑 → 幻月
      for (const pb of this.pbullets) {
        if (pb.dead) continue;
        if (Math.hypot(pb.x - boss.x, pb.y - boss.y) < boss.r + pb.r) {
          pb.dead = true;
          boss.hp -= pb.dmg;
          this.#spark(pb.x, pb.y, 'mint');
          if (boss.hp <= 0) {
            boss.hp = 0;
            this.#clear();
            return;
          }
        }
      }

      if (!p.alive || p.inv > 0) return;

      for (const eb of this.bullets) {
        if (eb.dead) continue;
        const d = Math.hypot(eb.x - p.x, eb.y - p.y);
        if (d < eb.r * 0.62 + p.r) {
          this.#hit();
          return;
        }
        if (!eb.grazed && d < eb.r + p.graze) {
          eb.grazed = true;
          this.graze += 1;
          this.score += 10;
          this.#spark(eb.x, eb.y, 'white');
          if (this.grazeSfxT <= 0) {
            this.grazeSfxT = 0.05;
            this.sfx('graze');
          }
        }
      }
    }

    #hit() {
      const p = this.player;
      this.sfx('hit');
      this.lives -= 1;
      this.hitFlash = 1;
      p.inv = 2;
      for (const b of this.bullets) {
        if (Math.hypot(b.x - p.x, b.y - p.y) < 150) b.dead = true;
      }
      if (this.lives <= 0) {
        p.alive = false;
        this.#dead();
      }
    }

    #clear() {
      if (this.state !== 'battle') return;
      this.sfx('clear');
      this.state = 'clear';
      this.timer = 0;
      this.score += 5000;
    }

    #dead() {
      if (this.state !== 'battle') return;
      this.sfx('over');
      this.state = 'dead';
      this.timer = 0;
    }

    #spark(x, y, color) {
      for (let i = 0; i < 4; i++) {
        const a = rand(0, TAU);
        const s = rand(40, 120);
        this.particles.push({
          x,
          y,
          vx: Math.cos(a) * s,
          vy: Math.sin(a) * s,
          t: 0.4,
          max: 0.4,
          color,
        });
      }
    }

    #updateParticles(dt) {
      for (const pt of this.particles) {
        pt.x += pt.vx * dt;
        pt.y += pt.vy * dt;
        pt.t -= dt;
      }
      if (this.particles.length) this.particles = this.particles.filter((pt) => pt.t > 0);
      if (this.hitFlash > 0) this.hitFlash -= dt * 2;
    }

    // ---------------------------------------------------------------
    // 渲染
    // ---------------------------------------------------------------

    render() {
      if (!this.active) return;
      const g = this.ctx;
      const W = this.W;
      const H = this.H;

      g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      g.clearRect(0, 0, W, H);
      if (this.bg) g.drawImage(this.bg, 0, 0, W, H);

      // 粒子
      for (const pt of this.particles) {
        const spr = this.sprites[pt.color] || this.sprites.white;
        const d = 10 * (pt.t / pt.max) + 4;
        g.globalAlpha = Math.max(0, pt.t / pt.max);
        g.drawImage(spr, pt.x - d / 2, pt.y - d / 2, d, d);
      }
      g.globalAlpha = 1;

      // 敌弹
      for (const b of this.bullets) {
        const spr = this.sprites[b.color] || this.sprites.amber;
        const d = b.r * 5;
        g.drawImage(spr, b.x - d / 2, b.y - d / 2, d, d);
      }

      // 玩家纸屑
      for (const pb of this.pbullets) {
        const spr = this.sprites.white;
        const d = pb.r * 3.4;
        g.drawImage(spr, pb.x - d / 2, pb.y - d / 2, d, d);
      }

      this.#drawBoss(g);
      if (this.player.alive) this.#drawPlayer(g);
      this.#drawHud(g);
      this.#drawOverlay(g);
    }

    #drawBoss(g) {
      const b = this.boss;

      const halo = g.createRadialGradient(b.x, b.y, 0, b.x, b.y, b.r * 2.8);
      halo.addColorStop(0, 'rgba(255,250,230,0.9)');
      halo.addColorStop(0.3, 'rgba(214,204,255,0.35)');
      halo.addColorStop(1, 'rgba(200,190,255,0)');
      g.fillStyle = halo;
      g.beginPath();
      g.arc(b.x, b.y, b.r * 2.8, 0, TAU);
      g.fill();

      const body = g.createRadialGradient(
        b.x - b.r * 0.3,
        b.y - b.r * 0.3,
        b.r * 0.15,
        b.x,
        b.y,
        b.r
      );
      body.addColorStop(0, '#fffef8');
      body.addColorStop(0.65, '#f1eaff');
      body.addColorStop(1, '#cbbfff');
      g.fillStyle = body;
      g.beginPath();
      g.arc(b.x, b.y, b.r, 0, TAU);
      g.fill();

      // 旋转光环
      g.strokeStyle = 'rgba(255,240,200,0.5)';
      g.lineWidth = 1.4;
      for (let i = 0; i < 3; i++) {
        const a0 = b.t * 0.8 + (i * TAU) / 3;
        g.beginPath();
        g.arc(b.x, b.y, b.r * 1.35, a0, a0 + 0.9);
        g.stroke();
      }
    }

    #drawPlayer(g) {
      const p = this.player;
      const S = this.playerScale;

      // 地面投影：不跟着机身倾斜，单独画
      g.save();
      g.translate(p.x, p.y);
      g.scale(S, S);
      g.fillStyle = 'rgba(40,60,80,0.16)';
      g.beginPath();
      g.ellipse(0, 16, 15, 5, 0, 0, TAU);
      g.fill();
      g.restore();

      g.save();
      g.translate(p.x, p.y);
      g.rotate(p.roll); // 移动时机身倾斜
      g.scale(S, S);

      const blink = p.inv > 0 && Math.floor(p.inv * 10) % 2 === 0;
      g.globalAlpha = blink ? 0.45 : 1;

      g.fillStyle = '#fffdf6';
      g.strokeStyle = 'rgba(150,140,120,0.55)';
      g.lineWidth = 1.2;
      g.beginPath();
      g.moveTo(0, -20);
      g.lineTo(15, 11);
      g.lineTo(0, 5);
      g.lineTo(-15, 11);
      g.closePath();
      g.fill();
      g.stroke();

      g.beginPath();
      g.moveTo(0, -20);
      g.lineTo(0, 5);
      g.strokeStyle = 'rgba(150,140,120,0.35)';
      g.stroke();

      g.globalAlpha = 1;

      // 判定点 / 擦弹圈：除以 S，抵消 scale —— 画出来的半径就等于真实碰撞半径
      g.fillStyle = 'rgba(255,96,96,0.95)';
      g.beginPath();
      g.arc(0, 0, p.r / S, 0, TAU);
      g.fill();
      g.strokeStyle = 'rgba(255,255,255,0.7)';
      g.lineWidth = 1;
      g.stroke();

      g.strokeStyle = 'rgba(130,225,255,0.45)';
      g.lineWidth = 1;
      g.beginPath();
      g.arc(0, 0, p.graze / S, 0, TAU);
      g.stroke();

      g.restore();
    }

    #drawHud(g) {
      const W = this.W;
      const bw = W * 0.62;
      const bx = (W - bw) / 2;
      const by = 16;

      g.fillStyle = 'rgba(255,255,255,0.22)';
      g.fillRect(bx, by, bw, 7);
      g.fillStyle = '#ffe6a6';
      g.fillRect(bx, by, bw * Math.max(0, this.boss.hp / this.boss.maxHp), 7);
      g.strokeStyle = 'rgba(255,255,255,0.35)';
      g.lineWidth = 1;
      g.strokeRect(bx, by, bw, 7);

      g.textAlign = 'center';
      g.fillStyle = 'rgba(255,248,230,0.85)';
      g.font = '600 11px ui-sans-serif, system-ui, sans-serif';
      g.fillText('STAGE 2 · 幻月', W / 2, by + 24);

      g.textAlign = 'left';
      g.font = '600 12px ui-sans-serif, system-ui, sans-serif';
      g.fillStyle = 'rgba(255,248,230,0.9)';
      g.fillText('残机 ' + this.lives, 16, this.H - 44);
      g.fillText('擦弹 ' + this.graze, 16, this.H - 26);
    }

    #drawOverlay(g) {
      const W = this.W;
      const H = this.H;

      if (this.state === 'intro') {
        const a = Math.min(1, this.timer / 0.5) * Math.min(1, (1.6 - this.timer) / 0.5);
        if (a > 0) {
          g.fillStyle = `rgba(20,30,50,${0.32 * a})`;
          g.fillRect(0, 0, W, H);
          g.globalAlpha = a;
          g.textAlign = 'center';
          g.fillStyle = '#fff8e6';
          g.font = '700 26px ui-sans-serif, system-ui, sans-serif';
          g.fillText('STAGE 2 · 幻月弹幕', W / 2, H * 0.45);
          g.font = '500 13px ui-sans-serif, system-ui, sans-serif';
          g.fillStyle = 'rgba(255,240,210,0.9)';
          g.fillText('点击射击 · 滑动移动 · 擦弹得分', W / 2, H * 0.45 + 30);
          g.globalAlpha = 1;
        }
      } else if (this.state === 'clear' || this.state === 'dead') {
        const a = Math.min(1, this.timer / 0.4);
        g.fillStyle = `rgba(20,30,50,${0.4 * a})`;
        g.fillRect(0, 0, W, H);
        g.globalAlpha = a;
        g.textAlign = 'center';
        g.fillStyle = '#fff8e6';
        g.font = '700 28px ui-sans-serif, system-ui, sans-serif';
        g.fillText(this.state === 'clear' ? 'STAGE CLEAR' : 'GAME OVER', W / 2, H * 0.45);
        g.font = '500 14px ui-sans-serif, system-ui, sans-serif';
        g.fillStyle = 'rgba(255,240,210,0.9)';
        g.fillText(
          this.state === 'clear' ? '幻月散去，纸飞机继续飞向天空' : '纸飞机落在了这个午后',
          W / 2,
          H * 0.45 + 32
        );
        g.globalAlpha = 1;
      }

      if (this.hitFlash > 0) {
        g.fillStyle = `rgba(255,80,80,${0.28 * this.hitFlash})`;
        g.fillRect(0, 0, W, H);
      }
    }
  }

  PP.DanmakuGame = DanmakuGame;
})(window.PP);
