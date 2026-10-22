/**
 * 第三关「超级马里奥 1-1」——完整复刻横版第一关
 * ---------------------------------------------------------------
 * 第二关打败幻月后，纸飞机放大飞过屏幕，把马里奥送到这里。
 *
 * 关卡按原版 1-1 摆：孤零零的第一个问号块（出蘑菇）、五连块、
 * 四根依次变高的水管（2/3/4/4）、三处坑、几组空中砖块、三段金字塔阶梯，
 * 最后是旗杆与城堡。
 *
 * 流程：
 *   intro  纸飞机从右侧飞入 → 马里奥跳下 → 飞机拉起飞走
 *   play   跑跳、踩敌人、顶砖块、吃蘑菇变大、躲水管与坑
 *   clear  碰到旗杆 → 沿杆滑下 → 走进城堡
 *   ending 公主出现 → 纸飞机飞回 → 载着两人飞走
 *
 * 物理与手感照着 SMB 1-1 的原始参数（走 1.5625 / 跑 2.5625 px 每帧，起跳 -4.9，
 * 按住跳重力 0.22、松手 0.48）。马里奥是 16×16 像素 sprite，四帧动作。
 */
(function (PP) {
  const TAU = Math.PI * 2;

  const TILE = 16;
  const VIEW_W = 256;
  const VIEW_H = 144;
  const COLS = 212;
  const ROWS = 9;
  /** 地面顶部所在行 */
  const GROUND = 7;

  const T_EMPTY = 0;
  const T_GROUND = 1;
  const T_BRICK = 2;
  const T_QUESTION = 3;
  const T_SOLID = 4;
  const T_PIPE = 5;

  /** 旗杆与城堡 */
  const FLAG_TX = 198;
  const CASTLE_TX = 202;

  /**
   * 物理参数——照着超级马里奥 1-1 的原始手感（单位：像素/帧，1 格 = 16px）
   * 走 1.56、跑 2.56、起跳 -4.9，按住跳时重力小（跳得高），松手立刻变沉。
   */
  const GRAVITY_FALL = 0.48;
  const GRAVITY_HOLD = 0.22;
  const MAX_FALL = 4.5;
  const WALK = 1.5625;
  const RUN = 2.5625;
  const ACCEL = 0.0369;
  const RUN_ACCEL = 0.0556;
  const FRICTION = 0.052;
  const JUMP_V = -4.9;
  /** 长按方向多久从走速升到跑速（秒） */
  const DASH_RAMP = 0.65;
  /**
   * 跳跃高度随速度的增益：高度 = v²/2g，满速时希望高度是 1.3 倍，
   * 所以初速乘 √(1 + 0.3) ≈ 1.14。速度越快跳得越高，站着跳最低。
   */
  const JUMP_SPEED_GAIN = 0.3;
  /** 转向时的刹车倍率（反向输入会明显减速） */
  const TURN_BOOST = 2.4;

  /** 马里奥像素图（16×16），照着 1-1 的小马里奥画的 */
  const MARIO_PIXELS = {
    stand: [
      '......RRRRR.....',
      '.....RRRRRRRRR..',
      '.....HHHSSFS....',
      '....HSHSSSFSSS..',
      '....HSHHSSSFSSS.',
      '....HHSSSSFFFF..',
      '......SSSSSSS...',
      '.....RRBRRR.....',
      '....RRRBRRBRRR..',
      '...RRRRBBBBRRRR.',
      '..SSRBYBBYBRSS..',
      '..SSSBBBBBBSSS..',
      '..SSBBBBBBBBSS..',
      '....BBB..BBB....',
      '...BBBB..BBBB...',
      '..BBBB....BBBB..',
    ],
    walk1: [
      '......RRRRR.....',
      '.....RRRRRRRRR..',
      '.....HHHSSFS....',
      '....HSHSSSFSSS..',
      '....HSHHSSSFSSS.',
      '....HHSSSSFFFF..',
      '......SSSSSSS...',
      '.....RRBRRR.....',
      '....RRRBRRBRRR..',
      '...RRRRBBBBRRRR.',
      '..SSRBYBBYBRSS..',
      '..SSSBBBBBBSSS..',
      '...BBBBBBBBBB...',
      '..BBBB...BBBB...',
      '.BBBB......BBB..',
      '.BBB........BB..',
    ],
    walk2: [
      '......RRRRR.....',
      '.....RRRRRRRRR..',
      '.....HHHSSFS....',
      '....HSHSSSFSSS..',
      '....HSHHSSSFSSS.',
      '....HHSSSSFFFF..',
      '......SSSSSSS...',
      '.....RRBRRR.....',
      '....RRRBRRBRRR..',
      '...RRRRBBBBRRRR.',
      '..SSRBYBBYBRSS..',
      '..SSSBBBBBBSSS..',
      '....BBBBBBBB....',
      '...BBBBBBBBBB...',
      '...BBBB..BBBB...',
      '...BBB....BBB...',
    ],
    jump: [
      '......RRRRR.....',
      '.....RRRRRRRRR..',
      '.....HHHSSFS....',
      '....HSHSSSFSSS..',
      '....HSHHSSSFSSS.',
      '....HHSSSSFFFF..',
      '..S...SSSSSSS...',
      '..SS.RRBRRR.....',
      '..SSRRRBRRBRRR..',
      '...RRRRBBBBRRRR.',
      '..SSRBYBBYBRSS..',
      '..SSSBBBBBBSSS..',
      '....BBBBBBBB....',
      '...BBB....BBB...',
      '..BBB......BBB..',
      '..BB........BB..',
    ],
    /** 蹲姿：矮胖，占满 16 行（和碰撞盒 h=16 对齐） */
    duck: [
      '....RRRRRR......',
      '...RRRRRRRRR....',
      '...HHHSSSFS.....',
      '..HSHSSSSFSS....',
      '..HHSSSSFFFF....',
      '..SSRRRRRRSS....',
      '.SSRRBBBBRRSS...',
      '.SSBBBBBBBBSS...',
      '..BBBBBBBBBB....',
      '..BBBBBBBBBB....',
      '..BBBB..BBBB....',
      '..BBBB..BBBB....',
      '.HHHH...HHHH....',
      '.HHHH...HHHH....',
      '.HHHH...HHHH....',
      '.HHHH...HHHH....',
    ],
  };

  /** 超级蘑菇：红帽白点 + 米色柄 */
  const MUSHROOM_PIXELS = [
    '.....RRRRRR.....',
    '...RRWWWWWWRR...',
    '..RWWWWRRWWWWR..',
    '.RWWWWRRRRWWWWR.',
    '.RWWWRRRRRRWWWR.',
    'RWWWWRRRRRRWWWWR',
    'RWWWWWRRRRWWWWWR',
    'RRWWWWWWWWWWWWRR',
    '.RRRRRRRRRRRRRR.',
    '...SSSSSSSSSS...',
    '...SSHHSSHHSS...',
    '...SSHHSSHHSS...',
    '...SSSSSSSSSS...',
    '...SSSSSSSSSS...',
    '....SSSSSSSS....',
    '.....SSSSSS.....',
  ];

  const MARIO_COLORS = {
    R: '#e52521',
    S: '#fbc7a0',
    H: '#6b3e1b',
    B: '#2a4fd6',
    Y: '#ffd93b',
    W: '#fff8e6',
  };

  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const rand = (a, b) => a + Math.random() * (b - a);

  class MarioGame {
    constructor(canvas, sfx, onLoseLife) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.sfx = typeof sfx === 'function' ? sfx : () => {};
      /** 死亡回调：没有生命条数了，Game 收到就扣 100 分 */
      this.onLoseLife = typeof onLoseLife === 'function' ? onLoseLife : null;

      this.W = 1;
      this.H = 1;
      this.dpr = 1;
      this.view = { x: 0, y: 0, w: 1, h: 1, scale: 1 };
      this.active = false;
      this.bound = false;

      this.touches = new Map();
      this.kb = { left: false, right: false, down: false, jump: false };
      this.keys = { left: false, right: false, down: false, jump: false };

      this.tiles = [];
      this.sprites = {};

      this.#buildSprites();
      this.resize();
      this.#buildLevel();
      this.reset();
      this.#bind();
    }

    /** 像素图预渲染成 canvas，绘制时只 drawImage */
    #buildSprites() {
      const make = (rows) => {
        const c = document.createElement('canvas');
        c.width = rows[0].length;
        c.height = rows.length;
        const g = c.getContext('2d');
        for (let y = 0; y < rows.length; y++) {
          const row = rows[y];
          for (let x = 0; x < row.length; x++) {
            const ch = row[x];
            if (ch === '.') continue;
            g.fillStyle = MARIO_COLORS[ch] || '#000';
            g.fillRect(x, y, 1, 1);
          }
        }
        return c;
      };

      for (const name of Object.keys(MARIO_PIXELS)) {
        this.sprites['mario_' + name] = make(MARIO_PIXELS[name]);
      }
      this.sprites.mushroom = make(MUSHROOM_PIXELS);
    }

    // ---------------------------------------------------------------
    // 尺寸 / 按键布局
    // ---------------------------------------------------------------

    resize() {
      this.dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.W = Math.max(1, window.innerWidth);
      this.H = Math.max(1, window.innerHeight);
      this.canvas.width = Math.round(this.W * this.dpr);
      this.canvas.height = Math.round(this.H * this.dpr);
      this.canvas.style.width = this.W + 'px';
      this.canvas.style.height = this.H + 'px';

      let vw = this.W;
      let vh = (this.W * VIEW_H) / VIEW_W;
      if (vh > this.H) {
        vh = this.H;
        vw = (this.H * VIEW_W) / VIEW_H;
      }
      this.view.w = vw;
      this.view.h = vh;
      this.view.x = (this.W - vw) / 2;
      this.view.y = (this.H - vh) / 2;
      this.view.scale = vw / VIEW_W;

      this.#layoutButtons();
    }

    #layoutButtons() {
      const r = clamp(Math.min(this.W, this.H) * 0.075, 26, 42);
      const below = this.H - (this.view.y + this.view.h);
      const cy =
        below > r * 2.6
          ? this.view.y + this.view.h + below * 0.55
          : this.view.y + this.view.h - r - 12;
      this.btnR = r;
      this.btn = {
        left: { x: r + 18, y: cy, r },
        right: { x: r * 3.1 + 18, y: cy, r },
        down: { x: this.W - r * 3.1 - 20, y: cy, r },
        jump: { x: this.W - r - 20, y: cy, r },
      };
    }

    // ---------------------------------------------------------------
    // 关卡（照 1-1 摆）
    // ---------------------------------------------------------------

    #buildLevel() {
      const t = [];
      for (let y = 0; y < ROWS; y++) t.push(new Array(COLS).fill(T_EMPTY));

      const put = (x, y, v) => {
        if (x >= 0 && x < COLS && y >= 0 && y < ROWS) t[y][x] = v;
      };

      const gaps = [
        [69, 70],
        [86, 87],
        [153, 153],
      ];

      for (let x = 0; x < COLS; x++) {
        if (gaps.some(([a, b]) => x >= a && x <= b)) continue;
        t[GROUND][x] = T_GROUND;
        t[GROUND + 1][x] = T_GROUND;
      }

      // 第一个孤零零的问号块（里面是蘑菇）
      put(16, 4, T_QUESTION);

      // 砖 / 问 / 砖 / 问 / 砖
      put(20, 4, T_BRICK);
      put(21, 4, T_QUESTION);
      put(22, 4, T_BRICK);
      put(23, 4, T_QUESTION);
      put(24, 4, T_BRICK);

      // 四根依次变高的水管（1-1 的标志）
      const pipe = (x, h) => {
        for (let y = GROUND - h; y < GROUND; y++) {
          put(x, y, T_PIPE);
          put(x + 1, y, T_PIPE);
        }
      };
      pipe(28, 2);
      pipe(38, 3);
      pipe(46, 4);
      pipe(57, 4);

      put(64, 4, T_BRICK);
      put(65, 4, T_QUESTION);
      put(66, 4, T_BRICK);

      put(77, 4, T_BRICK);
      put(78, 4, T_QUESTION);
      put(79, 4, T_BRICK);
      put(80, 4, T_BRICK);

      put(91, 4, T_BRICK);
      put(92, 4, T_QUESTION);
      put(93, 4, T_BRICK);
      put(94, 4, T_BRICK);

      put(100, 4, T_BRICK);
      put(101, 4, T_BRICK);
      put(106, 4, T_BRICK);
      put(107, 4, T_BRICK);
      put(109, 3, T_QUESTION);
      put(112, 4, T_BRICK);

      put(118, 4, T_BRICK);
      put(119, 4, T_BRICK);
      put(120, 4, T_BRICK);
      put(121, 4, T_QUESTION);

      put(128, 4, T_BRICK);
      put(129, 4, T_QUESTION);
      put(130, 4, T_BRICK);
      put(131, 4, T_BRICK);

      for (let x = 134; x <= 137; x++) put(x, 4, T_BRICK);

      // 三段金字塔阶梯：上 → 下 → 上
      for (let i = 0; i < 4; i++) {
        for (let y = GROUND - 1 - i; y < GROUND; y++) put(140 + i, y, T_SOLID);
      }
      for (let i = 0; i < 4; i++) {
        for (let y = GROUND - 4 + i; y < GROUND; y++) put(148 + i, y, T_SOLID);
      }
      for (let i = 0; i < 4; i++) {
        for (let y = GROUND - 1 - i; y < GROUND; y++) put(155 + i, y, T_SOLID);
      }

      this.tiles = t;
      this.cols = COLS;
      this.rows = ROWS;
    }

    #solidAt(tx, ty) {
      if (tx < 0 || tx >= COLS) return true;
      if (ty < 0 || ty >= ROWS) return false;
      const v = this.tiles[ty][tx];
      return v === T_GROUND || v === T_BRICK || v === T_QUESTION || v === T_SOLID || v === T_PIPE;
    }

    // ---------------------------------------------------------------
    // 生命周期
    // ---------------------------------------------------------------

    reset() {
      this.#buildLevel();

      this.active = true;
      this.state = 'intro';
      this.timer = 0;
      this.camX = 0;
      this.score = 0;
      this.coins = 0;
      this.finished = false;
      this.result = null;
      this.dropped = false;
      this.mushroomSpawned = false;
      this.enemies = [];
      this.items = [];
      this.particles = [];
      this.hitFlash = 0;
      this.flagSlide = false;
      this.flagTimer = 0;

      this.player = {
        x: 40,
        y: GROUND * TILE,
        vx: 0,
        vy: 0,
        w: 12,
        h: 16,
        big: false,
        onGround: false,
        face: 1,
        walkT: 0,
        inv: 0,
        alive: true,
        jumpLock: false,
        /** 大马里奥蹲下时为 true（碰撞盒高度 32 → 16） */
        ducking: false,
        /** 长按方向的累积时间：用来从走速平滑升到跑速 */
        dashTime: 0,
      };

      this.plane = { x: 420, y: 28, s: 2.8, visible: true };

      this.princess = {
        x: (CASTLE_TX + 3) * TILE,
        y: GROUND * TILE,
        w: 12,
        h: 18,
        visible: false,
        x0: (CASTLE_TX + 3) * TILE,
        t: 0,
      };

      this.#spawnEnemies();
      this.keys.left = this.keys.right = this.keys.down = this.keys.jump = false;
      this.touches.clear();
    }

    #spawnEnemies() {
      const spots = [
        [22, 'goomba'],
        [40, 'goomba'],
        [51, 'goomba'],
        [52, 'goomba'],
        [80, 'goomba'],
        [82, 'goomba'],
        [97, 'goomba'],
        [98, 'goomba'],
        [114, 'goomba'],
        [115, 'goomba'],
        [124, 'goomba'],
        [125, 'goomba'],
        [145, 'goomba'],
        [160, 'goomba'],
      ];
      this.enemies = spots.map(([tx, type]) => ({
        x: tx * TILE + 8,
        y: GROUND * TILE,
        w: 14,
        h: 13,
        vx: (tx % 2 ? -1 : 1) * 0.72,
        type,
        alive: true,
        squish: 0,
        t: rand(0, 6),
      }));
    }

    show() {
      this.active = true;
      this.canvas.classList.add('on');
    }

    hide() {
      this.active = false;
      this.canvas.classList.remove('on');
      this.touches.clear();
      this.keys.left = this.keys.right = this.keys.down = this.keys.jump = false;
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
          const hit = this.#hitButton(e.clientX, e.clientY);
          if (hit) {
            this.touches.set(e.pointerId, hit);
            this.#syncKeys();
            c.setPointerCapture?.(e.pointerId);
          }
        },
        { passive: false }
      );

      const release = (e) => {
        if (this.touches.delete(e.pointerId)) this.#syncKeys();
      };
      c.addEventListener('pointerup', release);
      c.addEventListener('pointercancel', release);
      c.addEventListener('pointerleave', release);

      window.addEventListener('keydown', (e) => this.#key(e, true));
      window.addEventListener('keyup', (e) => this.#key(e, false));
      window.addEventListener('resize', () => this.resize(), { passive: true });
    }

    #hitButton(x, y) {
      const b = this.btn;
      if (!b) return null;
      for (const name of ['left', 'right', 'down', 'jump']) {
        const btn = b[name];
        if (Math.hypot(x - btn.x, y - btn.y) <= btn.r * 1.18) return name;
      }
      return null;
    }

    #key(e, down) {
      if (!this.active) return;
      const k = e.key;
      let handled = true;
      if (k === 'ArrowLeft' || k === 'a' || k === 'A') this.kb.left = down;
      else if (k === 'ArrowRight' || k === 'd' || k === 'D') this.kb.right = down;
      else if (k === 'ArrowDown' || k === 's' || k === 'S') this.kb.down = down;
      else if (k === ' ' || k === 'Spacebar' || k === 'ArrowUp' || k === 'w' || k === 'W')
        this.kb.jump = down;
      else handled = false;
      if (handled) {
        e.preventDefault();
        this.#syncKeys();
      }
    }

    #syncKeys() {
      const held = [...this.touches.values()];
      this.keys.left = this.kb.left || held.includes('left');
      this.keys.right = this.kb.right || held.includes('right');
      this.keys.down = this.kb.down || held.includes('down');
      this.keys.jump = this.kb.jump || held.includes('jump');
    }

    // ---------------------------------------------------------------
    // 更新
    // ---------------------------------------------------------------

    update(dt) {
      if (!this.active) return;
      dt = Math.min(dt, 0.05);
      const f = dt * 60;
      this.timer += dt;
      if (this.hitFlash > 0) this.hitFlash -= dt * 2;

      if (this.state === 'intro') return this.#updateIntro(dt, f);
      if (this.state === 'clear') return this.#updateClear(dt, f);
      if (this.state === 'ending') return this.#updateEnding(dt, f);
      if (this.state === 'dead') return this.#updateDead(dt, f);

      this.#updatePlayer(dt, f);
      if (this.state !== 'play') return;
      this.#updateEnemies(dt, f);
      this.#updateItems(dt, f);
      this.#updateParticles(dt);
      this.#checkHits();
      this.#checkFlag();
      this.#updateCamera(dt);
    }

    #updateIntro(dt, f) {
      const t = this.timer;
      const pl = this.plane;

      if (t < 1.05) {
        const k = t / 1.05;
        const e = 1 - Math.pow(1 - k, 3);
        pl.x = 420 - 272 * e;
        pl.y = 26 + 18 * e;
        pl.s = 2.9 - 1.3 * e;
      } else if (t < 1.55) {
        pl.y = 44 + Math.sin((t - 1.05) * 9) * 1.4;
        if (!this.dropped) {
          this.dropped = true;
          const p = this.player;
          p.x = pl.x + 8;
          p.y = pl.y + 10;
          p.vy = -5.4;
          p.vx = 1.2;
          this.sfx('jump');
        }
      } else if (t < 2.5) {
        const k = (t - 1.55) / 0.95;
        pl.x = 148 + 330 * k * k;
        pl.y = 44 - 70 * k;
        pl.s = 1.6 * (1 - k * 0.65);
        if (k >= 1) pl.visible = false;
      } else {
        this.state = 'play';
        this.timer = 0;
        pl.visible = false;
        this.player.vx = 0;
      }

      this.#updatePlayerPhysics(f);
      this.#updateCamera(dt);
    }

    #updateDead(dt, f) {
      const p = this.player;
      p.vy = Math.min(p.vy + GRAVITY_FALL * f, MAX_FALL);
      p.y += p.vy * f;
      this.#updateParticles(dt);
      if (this.timer > 1.3) this.#respawn();
    }

    #respawn() {
      const p = this.player;
      const x = clamp(p.x - 40, 20, COLS * TILE - 20);
      p.x = this.#safeSpawnX(x);
      p.y = GROUND * TILE - 40;
      p.vx = 0;
      p.vy = 0;
      p.inv = 1.6;
      p.alive = true;
      p.big = false;
      p.h = 16;
      this.state = 'play';
      this.timer = 0;
    }

    #safeSpawnX(x) {
      const tx = Math.floor(x / TILE);
      for (let i = 0; i < 60; i++) {
        const t = tx - i;
        if (t < 1) break;
        if (this.#solidAt(t, GROUND) && this.#solidAt(t + 1, GROUND)) return t * TILE + 8;
      }
      return 40;
    }

    #updatePlayer(dt, f) {
      const p = this.player;
      if (!p.alive) return;
      if (p.inv > 0) p.inv -= dt;
      this.#updatePlayerPhysics(f);
    }

    #updatePlayerPhysics(f) {
      const p = this.player;
      if (!p.alive) return;
      const dt = f / 60;

      // 蹲下：只有大马里奥能蹲，蹲着不能走、不能跳
      this.#updateDuck();

      const want = (this.keys.right ? 1 : 0) - (this.keys.left ? 1 : 0);

      // 长按方向自动加速：按住 0.65 秒从走速平滑升到跑速。
      // **蹲着不能加速**——dashTime 直接清零，只保留走速。
      if (want !== 0 && !p.ducking) p.dashTime = Math.min(p.dashTime + dt, 1.5);
      else p.dashTime = Math.max(p.dashTime - dt * 3, 0);
      const ratio = p.ducking ? 0 : clamp(p.dashTime / DASH_RAMP, 0, 1);
      const maxSpeed = WALK + (RUN - WALK) * ratio;
      const accel = ACCEL + (RUN_ACCEL - ACCEL) * ratio;

      if (want !== 0) {
        const turning = p.vx !== 0 && Math.sign(p.vx) !== want;
        p.vx += want * accel * (turning ? TURN_BOOST : 1) * f;
        p.vx = clamp(p.vx, -maxSpeed, maxSpeed);
        p.face = want;
        p.walkT += Math.abs(p.vx) * 0.16 * f;
      } else {
        if (Math.abs(p.vx) <= FRICTION * f) p.vx = 0;
        else p.vx -= Math.sign(p.vx) * FRICTION * f;
      }

      // 起跳：**跳跃高度跟着当前速度走**——站着跳最低，满速跳最高（高度 ×1.3）
      if (this.keys.jump && p.onGround && !p.jumpLock) {
        const speedRatio = clamp(Math.abs(p.vx) / RUN, 0, 1);
        p.vy = JUMP_V * Math.sqrt(1 + JUMP_SPEED_GAIN * speedRatio);
        p.onGround = false;
        p.jumpLock = true;
        this.sfx('jump');
      }
      if (!this.keys.jump) p.jumpLock = false;

      const g = p.vy < 0 && this.keys.jump ? GRAVITY_HOLD : GRAVITY_FALL;
      p.vy = Math.min(p.vy + g * f, MAX_FALL);

      this.#moveX(p, p.vx * f);
      this.#moveY(p, p.vy * f);

      if (p.y > (GROUND + 1) * TILE + 24) this.#kill();
    }

    /** 蹲下：大马里奥按 B 缩成 16 高；站起来前先确认头顶有空间 */
    #updateDuck() {
      const p = this.player;
      if (!p.big) {
        if (p.ducking) {
          p.ducking = false;
          p.h = 16;
        }
        return;
      }
      // 地面按 B 蹲下；已经蹲着跳起来时保持蹲姿（蹲跳）
      if (this.keys.down && (p.onGround || p.ducking)) {
        p.ducking = true;
        p.h = 16;
        return;
      }
      if (p.ducking) {
        const oldH = p.h;
        p.h = 32;
        if (this.#fits(p)) p.ducking = false;
        else p.h = oldH;
      }
    }

    /** 碰撞盒是否与实心 tile 重叠（判断能不能站起来） */
    #fits(p) {
      const x0 = Math.floor((p.x - p.w / 2 + 0.5) / TILE);
      const x1 = Math.floor((p.x + p.w / 2 - 0.5) / TILE);
      const y0 = Math.floor((p.y - p.h + 0.5) / TILE);
      const y1 = Math.floor((p.y - 0.5) / TILE);
      for (let tx = x0; tx <= x1; tx++) {
        for (let ty = y0; ty <= y1; ty++) {
          if (this.#solidAt(tx, ty)) return false;
        }
      }
      return true;
    }

    #moveX(p, dx) {
      if (!dx) return;
      p.x += dx;
      const left = p.x - p.w / 2;
      const right = p.x + p.w / 2;
      const y0 = Math.floor((p.y - p.h + 0.5) / TILE);
      const y1 = Math.floor((p.y - 0.5) / TILE);
      if (dx > 0) {
        const tx = Math.floor((right - 0.01) / TILE);
        for (let ty = y0; ty <= y1; ty++) {
          if (this.#solidAt(tx, ty)) {
            p.x = tx * TILE - p.w / 2 - 0.01;
            p.vx = 0;
            break;
          }
        }
      } else {
        const tx = Math.floor(left / TILE);
        for (let ty = y0; ty <= y1; ty++) {
          if (this.#solidAt(tx, ty)) {
            p.x = (tx + 1) * TILE + p.w / 2 + 0.01;
            p.vx = 0;
            break;
          }
        }
      }
      p.x = clamp(p.x, p.w / 2, COLS * TILE - p.w / 2);
    }

    #moveY(p, dy) {
      if (!dy) return;
      p.y += dy;
      p.onGround = false;
      const x0 = Math.floor((p.x - p.w / 2 + 0.5) / TILE);
      const x1 = Math.floor((p.x + p.w / 2 - 0.5) / TILE);
      if (dy > 0) {
        const ty = Math.floor((p.y + 0.01) / TILE);
        for (let tx = x0; tx <= x1; tx++) {
          if (this.#solidAt(tx, ty)) {
            p.y = ty * TILE;
            p.vy = 0;
            p.onGround = true;
            break;
          }
        }
      } else {
        const ty = Math.floor((p.y - p.h - 0.01) / TILE);
        for (let tx = x0; tx <= x1; tx++) {
          if (this.#solidAt(tx, ty)) {
            p.y = (ty + 1) * TILE + p.h;
            p.vy = 0;
            this.#bumpBlock(tx, ty);
            break;
          }
        }
      }
    }

    #bumpBlock(tx, ty) {
      const v = this.tiles[ty]?.[tx];
      if (v === T_QUESTION) {
        this.tiles[ty][tx] = T_SOLID;
        // 第一个问号块出蘑菇，其余出金币
        if (tx === 16 && ty === 4 && !this.mushroomSpawned) {
          this.mushroomSpawned = true;
          this.items.push({
            x: tx * TILE + 8,
            y: ty * TILE - 4,
            vx: 1.05,
            vy: -1.6,
            w: 14,
            h: 14,
            type: 'mushroom',
            emerge: 0.5,
            alive: true,
          });
          this.sfx('coin');
        } else {
          this.coins += 1;
          this.score += 200;
          this.sfx('coin');
          this.#spark(tx * TILE + 8, ty * TILE, 'coin');
        }
      } else if (v === T_BRICK) {
        if (this.player.big) {
          this.tiles[ty][tx] = T_EMPTY;
          this.score += 50;
          this.sfx('brick');
          this.#spark(tx * TILE + 8, ty * TILE + 8, 'brick');
        } else {
          this.sfx('stomp');
        }
      }
    }

    #updateEnemies(dt, f) {
      for (const e of this.enemies) {
        if (!e.alive) {
          if (e.squish > 0) e.squish -= dt;
          continue;
        }
        e.t += dt;
        e.x += e.vx * f;

        const dir = Math.sign(e.vx) || 1;
        const tx = Math.floor((e.x + dir * (e.w / 2 + 1)) / TILE);
        const bodyRow = Math.floor((e.y - e.h * 0.5) / TILE);
        const footRow = Math.floor((e.y + 1) / TILE);
        if (this.#solidAt(tx, bodyRow) || !this.#solidAt(tx, footRow)) {
          e.vx = -e.vx;
          e.x += e.vx * 0.6;
        }
        e.x = clamp(e.x, e.w / 2, COLS * TILE - e.w / 2);

        if (e.y > (GROUND + 1) * TILE + 8) e.alive = false;
      }
    }

    #updateItems(dt, f) {
      const p = this.player;
      for (const it of this.items) {
        if (!it.alive) continue;

        // 从问号块里慢慢冒出来
        if (it.emerge > 0) {
          it.emerge -= dt;
          it.y -= 20 * dt;
          continue;
        }

        it.vy = Math.min(it.vy + GRAVITY_FALL * 0.8 * f, 4);
        it.y += it.vy * f;
        it.x += it.vx * f;

        const dir = Math.sign(it.vx) || 1;
        const tx = Math.floor((it.x + dir * (it.w / 2 + 1)) / TILE);
        const ty = Math.floor((it.y - it.h * 0.5) / TILE);
        if (this.#solidAt(tx, ty)) it.vx = -it.vx;

        const footTy = Math.floor((it.y + 0.01) / TILE);
        if (this.#solidAt(Math.floor(it.x / TILE), footTy)) {
          it.y = footTy * TILE;
          it.vy = 0;
        }

        if (it.y > (GROUND + 2) * TILE) it.alive = false;

        if (p.alive && Math.abs(it.x - p.x) < 12 && Math.abs(it.y - p.y) < p.h) {
          it.alive = false;
          this.score += 1000;
          this.sfx('coin');
          this.#grow();
        }
      }
      if (this.items.length) this.items = this.items.filter((it) => it.alive);
    }

    /** 吃到蘑菇 → 变大（碰撞盒 16 → 32） */
    #grow() {
      const p = this.player;
      if (p.big) return;
      p.big = true;
      p.h = 32;
      // 头顶可能顶进砖块里，往下挪
      for (let guard = 0; guard < 8; guard++) {
        const ty = Math.floor((p.y - p.h) / TILE);
        const x0 = Math.floor((p.x - p.w / 2 + 0.5) / TILE);
        const x1 = Math.floor((p.x + p.w / 2 - 0.5) / TILE);
        let blocked = false;
        for (let tx = x0; tx <= x1; tx++) if (this.#solidAt(tx, ty)) blocked = true;
        if (!blocked) break;
        p.y += 2;
      }
    }

    /** 被敌人碰到：大马里奥变小，小马里奥死 */
    #shrink() {
      const p = this.player;
      p.big = false;
      p.h = 16;
      p.inv = 2;
      this.sfx('hit');
    }

    #checkHits() {
      const p = this.player;
      if (!p.alive || p.inv > 0) return;

      for (const e of this.enemies) {
        if (!e.alive) continue;
        if (Math.abs(p.x - e.x) < (p.w + e.w) / 2 - 1 && p.y > e.y - e.h && p.y - p.h < e.y) {
          const stomp = p.vy > 0 && p.y < e.y - e.h * 0.35;
          if (stomp) {
            e.alive = false;
            e.squish = 0.45;
            p.vy = -4.2;
            this.score += 100;
            this.sfx('stomp');
          } else if (p.big) {
            this.#shrink();
            return;
          } else {
            this.#kill();
            return;
          }
        }
      }
    }

    #checkFlag() {
      if (this.state !== 'play') return;
      const p = this.player;
      const fx = FLAG_TX * TILE + 8;
      if (p.x + p.w / 2 >= fx - 2) {
        this.state = 'clear';
        this.timer = 0;
        this.flagSlide = true;
        this.flagTimer = 0;
        this.score += 2000;
        this.sfx('clear');
      }
    }

    #kill() {
      const p = this.player;
      if (!p.alive || this.state !== 'play') return;
      p.alive = false;
      p.vy = -6.4;
      p.vx = 0;
      // 没有生命条数：上报一次，Game 扣 100 分
      if (this.onLoseLife) this.onLoseLife();
      this.hitFlash = 1;
      this.state = 'dead';
      this.timer = 0;
      this.sfx('die');
    }

    /** 通关演出 1：沿旗杆滑下 → 走进城堡 */
    #updateClear(dt) {
      const p = this.player;
      const castleDoor = (CASTLE_TX + 3) * TILE;

      if (this.flagSlide) {
        p.face = 1;
        p.x = FLAG_TX * TILE + 2;
        p.vy = 0;
        p.y += 150 * dt;
        if (p.y >= GROUND * TILE) {
          p.y = GROUND * TILE;
          this.flagSlide = false;
          this.flagTimer = 0;
        }
      } else {
        this.flagTimer += dt;
        p.face = 1;
        p.x += 80 * dt;
        p.walkT += 0.22;
        if (p.x >= castleDoor) {
          this.state = 'ending';
          this.timer = 0;
          this.princess.visible = true;
          this.princess.t = 0;
          this.princess.x0 = castleDoor - 18;
          this.princess.x = castleDoor - 18;
          this.plane = { x: 3480, y: 8, s: 2.4, visible: true };
          this.sfx('win');
        }
      }

      this.#updateParticles(dt);
      this.#updateCamera(dt);
    }

    /** 通关演出 2：公主跑过来 → 飞机载人飞走（世界坐标） */
    #updateEnding(dt, f) {
      const pr = this.princess;
      const p = this.player;
      const pl = this.plane;
      const t = this.timer;

      pr.t += dt;
      if (pr.t < 1.0) {
        const k = pr.t / 1.0;
        pr.x = pr.x0 + (p.x + 16 - pr.x0) * Math.min(1, k);
      }

      if (t < 1.2) {
        const k = t / 1.2;
        const e = 1 - Math.pow(1 - k, 2);
        pl.x = 3480 - 240 * e; // 停在城堡门口（世界坐标 3240）
        pl.y = 8 + 34 * e;
        pl.s = 2.4 - 0.9 * e;
      } else if (t < 2.2) {
        pl.y = 42 + Math.sin((t - 1.2) * 5) * 1.6;
      } else if (t < 3.2) {
        const k = (t - 2.2) / 1.0;
        pl.x += 2.4 * f * k;
        pl.y -= 1.1 * f * k;
        pl.s = 1.5 * (1 - k * 0.35);
        p.x = pl.x - 4;
        p.y = pl.y + 14;
        pr.x = pl.x + 12;
        pr.y = pl.y + 14;
      } else {
        pl.x += 5.2 * f;
        pl.y -= 2.2 * f;
        p.x = pl.x - 4;
        p.y = pl.y + 14;
        pr.x = pl.x + 12;
        pr.y = pl.y + 14;
      }

      this.#updateParticles(dt);

      if (t > 4.6 && !this.finished) {
        this.finished = true;
        this.result = 'clear';
      }
    }

    #updateParticles(dt) {
      for (const pt of this.particles) {
        pt.x += pt.vx * dt;
        pt.y += pt.vy * dt;
        pt.vy += 320 * dt;
        pt.t -= dt;
      }
      if (this.particles.length) this.particles = this.particles.filter((pt) => pt.t > 0);
    }

    #spark(x, y, kind) {
      const color = kind === 'coin' ? '#ffd93b' : '#c46a2a';
      for (let i = 0; i < 6; i++) {
        const a = rand(-Math.PI, 0);
        const s = rand(50, 140);
        this.particles.push({
          x,
          y,
          vx: Math.cos(a) * s,
          vy: Math.sin(a) * s,
          t: 0.5,
          max: 0.5,
          color,
        });
      }
    }

    #updateCamera(dt) {
      const target = clamp(this.player.x - VIEW_W * 0.42, 0, COLS * TILE - VIEW_W);
      this.camX += (target - this.camX) * Math.min(1, dt * 9);
    }

    // ---------------------------------------------------------------
    // 渲染
    // ---------------------------------------------------------------

    render() {
      if (!this.active) return;
      const g = this.ctx;
      const v = this.view;

      g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      g.imageSmoothingEnabled = false; // 像素 sprite 要锐利
      g.clearRect(0, 0, this.W, this.H);
      g.fillStyle = '#0b1526';
      g.fillRect(0, 0, this.W, this.H);

      g.save();
      g.beginPath();
      g.rect(v.x, v.y, v.w, v.h);
      g.clip();
      g.translate(v.x, v.y);
      g.scale(v.scale, v.scale);

      this.#drawSky(g);

      g.save();
      g.translate(-this.camX * 0.45, 0);
      this.#drawHills(g);
      this.#drawBushes(g);
      g.restore();

      g.save();
      g.translate(-this.camX, 0);
      this.#drawTiles(g);
      this.#drawFlag(g);
      this.#drawCastle(g);
      this.#drawItems(g);
      this.#drawEnemies(g);
      this.#drawPrincess(g);
      this.#drawPlayer(g);
      this.#drawPlane(g);
      this.#drawParticles(g);
      g.restore();

      g.restore();

      this.#drawHud(g);
      this.#drawButtons(g);
      this.#drawOverlay(g);
    }

    #drawSky(g) {
      const sky = g.createLinearGradient(0, 0, 0, VIEW_H);
      sky.addColorStop(0, '#2f6fd6');
      sky.addColorStop(0.55, '#5c94fc');
      sky.addColorStop(1, '#9ec8ff');
      g.fillStyle = sky;
      g.fillRect(0, 0, VIEW_W, VIEW_H);

      g.fillStyle = 'rgba(255,255,255,0.92)';
      const cloud = (cx, cy, s) => {
        g.beginPath();
        g.arc(cx, cy, 7 * s, 0, TAU);
        g.arc(cx + 9 * s, cy - 3 * s, 8.5 * s, 0, TAU);
        g.arc(cx + 19 * s, cy, 7 * s, 0, TAU);
        g.arc(cx + 9 * s, cy + 3.5 * s, 8 * s, 0, TAU);
        g.fill();
      };
      const off = (this.camX * 0.45) % 340;
      for (let i = -1; i < 4; i++) {
        cloud(i * 110 - off + 30, 26, 1);
        cloud(i * 150 - off + 90, 44, 0.72);
      }
    }

    #drawHills(g) {
      const base = GROUND * TILE;
      g.fillStyle = '#3f8f3f';
      for (let i = -1; i < 8; i++) {
        const x = i * 150 + 40;
        g.beginPath();
        g.moveTo(x - 60, base);
        g.quadraticCurveTo(x, base - 66, x + 60, base);
        g.closePath();
        g.fill();
      }
      g.fillStyle = 'rgba(255,255,255,0.1)';
      for (let i = -1; i < 8; i++) {
        const x = i * 150 + 40;
        g.beginPath();
        g.moveTo(x - 22, base);
        g.quadraticCurveTo(x, base - 48, x + 22, base);
        g.closePath();
        g.fill();
      }
    }

    /** 1-1 里那些绿色灌木 */
    #drawBushes(g) {
      const base = GROUND * TILE;
      g.fillStyle = '#4caf50';
      const bush = (bx, bw) => {
        g.beginPath();
        g.arc(bx - bw * 0.3, base - 5, 7, Math.PI, TAU);
        g.arc(bx, base - 7, 9, Math.PI, TAU);
        g.arc(bx + bw * 0.3, base - 5, 7, Math.PI, TAU);
        g.fill();
        g.fillRect(bx - bw * 0.45, base - 6, bw * 0.9, 6);
      };
      for (let i = -1; i < 9; i++) {
        const bx = i * 130 + 70;
        bush(bx, 34);
        bush(bx + 60, 26);
      }
    }

    #drawTiles(g) {
      const x0 = Math.max(0, Math.floor(this.camX / TILE) - 1);
      const x1 = Math.min(COLS - 1, Math.ceil((this.camX + VIEW_W) / TILE) + 1);
      for (let x = x0; x <= x1; x++) {
        for (let y = 0; y < ROWS; y++) {
          const v = this.tiles[y][x];
          if (!v) continue;
          this.#drawTile(g, x, y, v);
        }
      }
    }

    #drawTile(g, tx, ty, v) {
      const x = tx * TILE;
      const y = ty * TILE;

      if (v === T_GROUND) {
        // 地面：橙棕砖块
        g.fillStyle = '#c46a2a';
        g.fillRect(x, y, TILE, TILE);
        g.fillStyle = '#e08a45';
        g.fillRect(x + 1, y + 1, TILE - 2, 1.5);
        g.strokeStyle = 'rgba(90,45,10,0.45)';
        g.lineWidth = 0.9;
        g.beginPath();
        g.moveTo(x, y + 8);
        g.lineTo(x + TILE, y + 8);
        g.moveTo(x + 8, y);
        g.lineTo(x + 8, y + 8);
        g.moveTo(x + 4, y + 8);
        g.lineTo(x + 4, y + TILE);
        g.moveTo(x + 12, y + 8);
        g.lineTo(x + 12, y + TILE);
        g.stroke();
      } else if (v === T_BRICK) {
        // 砖块：四行砖缝，竖缝交错
        g.fillStyle = '#c46a2a';
        g.fillRect(x, y, TILE, TILE);
        g.fillStyle = '#e08a45';
        g.fillRect(x + 1, y + 1, TILE - 2, 2);
        g.strokeStyle = 'rgba(70,34,8,0.55)';
        g.lineWidth = 0.9;
        g.beginPath();
        for (let i = 1; i <= 3; i++) {
          g.moveTo(x, y + i * 4);
          g.lineTo(x + TILE, y + i * 4);
        }
        g.moveTo(x + 8, y);
        g.lineTo(x + 8, y + 4);
        g.moveTo(x + 4, y + 4);
        g.lineTo(x + 4, y + 8);
        g.moveTo(x + 12, y + 4);
        g.lineTo(x + 12, y + 8);
        g.moveTo(x + 8, y + 8);
        g.lineTo(x + 8, y + 12);
        g.moveTo(x + 4, y + 12);
        g.lineTo(x + 4, y + TILE);
        g.moveTo(x + 12, y + 12);
        g.lineTo(x + 12, y + TILE);
        g.stroke();
      } else if (v === T_QUESTION) {
        // 问号块：金色 + 四角铆钉 + 呼吸闪烁
        const pulse = 0.5 + 0.5 * Math.sin(this.timer * 5 + tx * 0.9);
        g.fillStyle = '#c98a1e';
        g.fillRect(x, y, TILE, TILE);
        g.fillStyle = `rgba(255, 214, 96, ${0.75 + pulse * 0.25})`;
        g.fillRect(x + 1.5, y + 1.5, TILE - 3, TILE - 3);
        g.fillStyle = '#8a5a10';
        for (const [ox, oy] of [
          [3, 3],
          [13, 3],
          [3, 13],
          [13, 13],
        ]) {
          g.fillRect(x + ox - 0.8, y + oy - 0.8, 1.6, 1.6);
        }
        g.fillStyle = '#7a4a08';
        g.font = 'bold 11px Georgia, serif';
        g.textAlign = 'center';
        g.textBaseline = 'middle';
        g.fillText('?', x + TILE / 2, y + TILE / 2 + 0.5);
      } else if (v === T_SOLID) {
        // 阶梯：棕色硬块
        g.fillStyle = '#b06a2c';
        g.fillRect(x, y, TILE, TILE);
        g.fillStyle = '#d08a4a';
        g.fillRect(x + 1.5, y + 1.5, TILE - 3, TILE - 3);
        g.strokeStyle = 'rgba(80,40,10,0.45)';
        g.lineWidth = 0.8;
        g.strokeRect(x + 1.5, y + 1.5, TILE - 3, TILE - 3);
      } else if (v === T_PIPE) {
        // 水管：两列拼成一根，只在最外侧画边缘；最上一格画管口
        const leftOpen = tx === 0 || this.tiles[ty][tx - 1] !== T_PIPE;
        const rightOpen = tx === COLS - 1 || this.tiles[ty][tx + 1] !== T_PIPE;
        const isTop = ty === 0 || this.tiles[ty - 1][tx] !== T_PIPE;

        g.fillStyle = '#2e9b3f';
        g.fillRect(x, y, TILE, TILE);
        g.fillStyle = '#5cd36b';
        g.fillRect(x + (leftOpen ? 2 : 1), y, leftOpen ? 5 : 4, TILE);
        if (rightOpen) {
          g.fillStyle = 'rgba(0,0,0,0.22)';
          g.fillRect(x + TILE - 4, y, 4, TILE);
        }

        if (isTop) {
          const lx = leftOpen ? x - 2 : x;
          const rx = rightOpen ? x + TILE + 2 : x + TILE;
          g.fillStyle = '#2e9b3f';
          g.fillRect(lx, y, rx - lx, 6);
          g.fillStyle = '#5cd36b';
          g.fillRect(lx + 2, y + 1, 5, 4);
          if (rightOpen) {
            g.fillStyle = 'rgba(0,0,0,0.22)';
            g.fillRect(rx - 5, y + 1, 3, 4);
          }
          g.fillStyle = 'rgba(0,0,0,0.18)';
          g.fillRect(lx, y + 6, rx - lx, 1);
        }
      }
    }

    /** 旗杆：杆 + 顶球 + 旗子（通关时旗子降下） */
    #drawFlag(g) {
      const fx = FLAG_TX * TILE + 8;
      const topY = 2 * TILE;
      const baseY = GROUND * TILE;

      g.fillStyle = '#6b6b6b';
      g.fillRect(fx - 7, baseY - 6, 14, 6);
      g.fillStyle = '#d8d8d8';
      g.fillRect(fx - 1.5, topY, 3, baseY - topY - 6);
      g.fillStyle = 'rgba(120,120,120,0.5)';
      g.fillRect(fx + 0.5, topY, 1, baseY - topY - 6);
      g.fillStyle = '#ffd93b';
      g.beginPath();
      g.arc(fx, topY - 2, 4, 0, TAU);
      g.fill();

      const lowered = this.state === 'clear' || this.state === 'ending';
      const flagY = lowered ? baseY - 26 : topY + 6;
      g.fillStyle = '#2e9b3f';
      g.beginPath();
      g.moveTo(fx - 1, flagY);
      g.lineTo(fx - 19, flagY + 7);
      g.lineTo(fx - 1, flagY + 14);
      g.closePath();
      g.fill();
    }

    /** 1-1 结尾的城堡 */
    #drawCastle(g) {
      const x0 = CASTLE_TX * TILE;
      const base = GROUND * TILE;
      const w = 6 * TILE;
      const h = 5 * TILE;

      g.fillStyle = '#b8b0a4';
      g.fillRect(x0, base - h, w, h);
      g.strokeStyle = 'rgba(90,84,76,0.45)';
      g.lineWidth = 0.8;
      for (let y = base - h + 8; y < base; y += 8) {
        g.beginPath();
        g.moveTo(x0, y);
        g.lineTo(x0 + w, y);
        g.stroke();
      }
      g.fillStyle = '#a49c90';
      for (let i = 0; i < 4; i++) g.fillRect(x0 + i * 24, base - h - 9, 14, 9);
      g.fillStyle = '#3a2a20';
      g.beginPath();
      g.moveTo(x0 + w / 2 - 12, base);
      g.lineTo(x0 + w / 2 - 12, base - 22);
      g.quadraticCurveTo(x0 + w / 2, base - 40, x0 + w / 2 + 12, base - 22);
      g.lineTo(x0 + w / 2 + 12, base);
      g.closePath();
      g.fill();
      g.fillStyle = '#2b211a';
      g.fillRect(x0 + 14, base - h + 12, 9, 12);
      g.fillRect(x0 + w - 23, base - h + 12, 9, 12);
    }

    #drawItems(g) {
      const spr = this.sprites.mushroom;
      if (!spr) return;
      for (const it of this.items) {
        if (!it.alive) continue;
        g.drawImage(spr, it.x - 8, it.y - 16, 16, 16);
      }
    }

    #drawEnemies(g) {
      for (const e of this.enemies) {
        if (!e.alive && e.squish <= 0) continue;
        const x = e.x;
        const y = e.y;

        if (!e.alive) {
          g.fillStyle = '#8b5a2b';
          g.fillRect(x - 7, y - 5, 14, 5);
          continue;
        }

        const wob = Math.sin(e.t * 8) * 1;
        g.fillStyle = '#8b5a2b';
        g.beginPath();
        g.ellipse(x, y - 7, 7, 6.5, 0, Math.PI, TAU);
        g.fill();
        g.fillRect(x - 7, y - 7, 14, 4);
        g.fillStyle = '#f0dcc0';
        g.beginPath();
        g.ellipse(x, y - 2.5, 6, 2.6, 0, 0, TAU);
        g.fill();
        g.fillStyle = '#fff';
        g.fillRect(x - 4.5, y - 10, 3, 4);
        g.fillRect(x + 1.5, y - 10, 3, 4);
        g.fillStyle = '#1a1a1a';
        g.fillRect(x - 3.8, y - 9, 1.8, 2.6);
        g.fillRect(x + 2.2, y - 9, 1.8, 2.6);
        g.fillStyle = '#5a3618';
        g.fillRect(x - 6, y - 1 + wob * 0.3, 4, 2);
        g.fillRect(x + 2, y - 1 - wob * 0.3, 4, 2);
      }
    }

    #drawPrincess(g) {
      const pr = this.princess;
      if (!pr.visible) return;
      const x = pr.x;
      const y = pr.y;
      const bob = Math.sin(this.timer * 6) * 0.8;

      g.fillStyle = '#ffd9e6';
      g.beginPath();
      g.moveTo(x, y - 12 + bob);
      g.lineTo(x + 8, y + bob);
      g.lineTo(x - 8, y + bob);
      g.closePath();
      g.fill();
      g.fillStyle = '#ffd0a8';
      g.beginPath();
      g.arc(x, y - 14 + bob, 4.2, 0, TAU);
      g.fill();
      g.fillStyle = '#f7d94c';
      g.beginPath();
      g.arc(x, y - 16 + bob, 4.6, Math.PI, TAU);
      g.fill();
      g.fillStyle = '#ffd93b';
      g.beginPath();
      g.arc(x, y - 21 + bob, 1.6, 0, TAU);
      g.fill();
    }

    #drawPlayer(g) {
      const p = this.player;
      if (!p.alive && this.state !== 'dead') return;
      if (p.inv > 0 && Math.floor(p.inv * 12) % 2 === 0) return;

      let pose = 'stand';
      if (!p.alive || !p.onGround) {
        pose = 'jump';
      } else if (Math.abs(p.vx) > 0.06 || this.state === 'clear') {
        const frame = Math.floor(p.walkT) % 3;
        pose = frame === 0 ? 'stand' : frame === 1 ? 'walk1' : 'walk2';
      }
      const spr = this.sprites['mario_' + pose];
      if (!spr) return;

      g.save();
      g.translate(p.x, p.y);
      g.scale(p.face, 1);
      if (p.ducking) {
        // 蹲姿专用 sprite（16×16，矮胖），和碰撞盒 h=16 对齐
        const duckSpr = this.sprites.mario_duck || spr;
        g.drawImage(duckSpr, -8, -16, 16, 16);
      } else {
        // 大马里奥是同一张图整体放大 2 倍（32×32），小马里奥 16×16
        const size = p.big ? 32 : 16;
        g.drawImage(spr, -size / 2, -size, size, size);
      }
      g.restore();
    }

    #drawPlane(g) {
      const pl = this.plane;
      if (!pl.visible) return;
      g.save();
      g.translate(pl.x, pl.y);
      g.scale(pl.s, pl.s);
      g.rotate(-0.12);
      g.fillStyle = '#fffdf6';
      g.strokeStyle = 'rgba(150,140,120,0.5)';
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(0, -10);
      g.lineTo(13, 7);
      g.lineTo(0, 2);
      g.lineTo(-13, 7);
      g.closePath();
      g.fill();
      g.stroke();
      g.beginPath();
      g.moveTo(0, -10);
      g.lineTo(0, 2);
      g.stroke();
      g.restore();
    }

    #drawParticles(g) {
      for (const pt of this.particles) {
        g.globalAlpha = Math.max(0, pt.t / pt.max);
        g.fillStyle = pt.color;
        g.fillRect(pt.x - 1.5, pt.y - 1.5, 3, 3);
      }
      g.globalAlpha = 1;
    }

    // ---------------------------------------------------------------
    // HUD / 按键 / 覆盖层
    // ---------------------------------------------------------------

    #drawHud(g) {
      g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      const top = Math.max(6, this.view.y + 6);
      const pad = 14;

      g.textAlign = 'left';
      g.textBaseline = 'top';
      g.font = '700 12px ui-sans-serif, system-ui, sans-serif';
      g.fillStyle = 'rgba(255,255,255,0.92)';
      g.shadowColor = 'rgba(0,0,0,0.45)';
      g.shadowBlur = 4;
      g.fillText('SCORE ' + String(this.score).padStart(5, '0'), pad, top);

      g.font = '700 11px ui-sans-serif, system-ui, sans-serif';
      g.fillStyle = 'rgba(255,240,200,0.95)';
      g.fillText('◉' + this.coins, pad, top + 16);

      const bw = Math.min(this.W * 0.34, 180);
      const bx = this.W - bw - pad;
      g.shadowBlur = 0;
      g.fillStyle = 'rgba(0,0,0,0.28)';
      g.fillRect(bx, top + 2, bw, 6);
      g.fillStyle = '#ffd93b';
      const prog = clamp(this.player.x / (FLAG_TX * TILE), 0, 1);
      g.fillRect(bx, top + 2, bw * prog, 6);
      g.strokeStyle = 'rgba(255,255,255,0.4)';
      g.lineWidth = 1;
      g.strokeRect(bx + 0.5, top + 2.5, bw - 1, 5);
      g.font = '600 9px ui-sans-serif, system-ui, sans-serif';
      g.fillStyle = 'rgba(255,255,255,0.7)';
      g.textAlign = 'right';
      g.fillText('STAGE 3 · 1-1', this.W - pad, top + 12);
    }

    #drawButtons(g) {
      if (!this.btn) return;
      const draw = (btn, label, active) => {
        g.beginPath();
        g.arc(btn.x, btn.y, btn.r, 0, TAU);
        g.fillStyle = active ? 'rgba(255,255,255,0.42)' : 'rgba(255,255,255,0.2)';
        g.fill();
        g.strokeStyle = 'rgba(255,255,255,0.55)';
        g.lineWidth = 1.5;
        g.stroke();

        g.fillStyle = 'rgba(255,255,255,0.95)';
        g.textAlign = 'center';
        g.textBaseline = 'middle';
        g.font = `700 ${Math.round(btn.r * 0.72)}px ui-sans-serif, system-ui, sans-serif`;
        g.fillText(label, btn.x, btn.y + 1);
      };

      const held = [...this.touches.values()];
      draw(this.btn.left, '‹', held.includes('left') || this.kb.left);
      draw(this.btn.right, '›', held.includes('right') || this.kb.right);
      draw(this.btn.down, 'B', held.includes('down') || this.kb.down);
      draw(this.btn.jump, 'A', held.includes('jump') || this.kb.jump);
    }

    #drawOverlay(g) {
      g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

      if (this.state === 'intro') {
        const a = clamp(this.timer / 0.5, 0, 1) * clamp((2.5 - this.timer) / 0.5, 0, 1);
        if (a > 0) {
          g.globalAlpha = a;
          g.textAlign = 'center';
          g.textBaseline = 'middle';
          g.fillStyle = 'rgba(255,255,255,0.96)';
          g.font = '700 22px ui-sans-serif, system-ui, sans-serif';
          g.shadowColor = 'rgba(0,0,0,0.5)';
          g.shadowBlur = 10;
          g.fillText('STAGE 3 · 1-1', this.W / 2, this.view.y + this.view.h * 0.3);
          g.shadowBlur = 0;
          g.globalAlpha = 1;
        }
      }

      if (this.hitFlash > 0) {
        g.fillStyle = `rgba(255,70,70,${0.3 * this.hitFlash})`;
        g.fillRect(this.view.x, this.view.y, this.view.w, this.view.h);
      }
    }
  }

  PP.MarioGame = MarioGame;
})(window.PP);
