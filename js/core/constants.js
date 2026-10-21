/**
 * 全局配置与关卡数据
 * ---------------------------------------------------------------
 * 所有"魔法数字"集中在这里，调整手感/节奏只需要改这个文件。
 *
 * 全局命名空间：window.PP —— 每个文件用 IIFE 往 PP 上挂东西，
 * 这样普通 <script> 标签就能按顺序加载（file:// 下 ES Module 会被 CORS 拦掉）。
 */
window.PP = window.PP || {};

(function (PP) {
  /** 歌曲总时长（秒），用于结束淡出曲线的归一化 */
  const SONG_DURATION = 85.86;

  /** 曲速与拍点，用于音符的呼吸律动 */
  const BPM = 136;
  const BEAT = 60 / BPM;
  const FIRST_BEAT = 1.88;

  /** 纸飞机姿态：机头指向 -Z，水平飞行 */
  const PLANE_YAW = 0;
  const PLANE_NOSE_UP = 0;

  /** 纸飞机允许的垂直飞行区间 */
  const FLIGHT_MIN_Y = 0.02;
  const FLIGHT_MAX_Y = 4.68;

  /** 音符只分布在飞行区间中间的三分之二，留出操作余量 */
  const NOTE_Y_MIN = FLIGHT_MIN_Y + (FLIGHT_MAX_Y - FLIGHT_MIN_Y) / 6;
  const NOTE_Y_MAX = FLIGHT_MAX_Y - (FLIGHT_MAX_Y - FLIGHT_MIN_Y) / 6;

  /** 告别的起点：最后两个「再靠近一点」开始，飞机抬升远去 */
  const FAR_END = 75.0;

  /** 飞行速度（单位/秒）与音符数量 */
  const PLANE_SPEED = 8.5;
  const NOTE_COUNT = 22;

  /** 起飞阶段：音符在起飞后多久开始出现、尾部留白 */
  const LAUNCH = {
    firstNoteDelay: 1.8,
    tailGap: 1.5,
  };

  /** 蓄能阶段（点击唤风）参数 */
  const AWAKENING = {
    /** 无论是否攒满，歌曲到这里自动起飞 */
    autoLaunchAt: 9.6,
    /** 首次轻触的初始能量 */
    startTapGain: 0.08,
    /** 每次轻触增加的能量 */
    tapGain: 0.135,
    /** 能量阈值 */
    energyThreshold: 0.9,
    /** 或者：在时间窗内连点次数达标 */
    burstTaps: 7,
    burstWindow: 1550,
    /** 每秒自然衰减 */
    decayPerSecond: 0.034,
  };

  /** 纸飞机现身的过场时长（秒） */
  const TRANSFORM_DURATION = 2.1;

  /** 音乐结束后，让纸飞机多飞一会儿再结算（秒） */
  const FAREWELL_LINGER = 1.25;

  /** 音符碰撞椭球半径（越大越好接） */
  const COLLISION_RADII = { x: 1.35, y: 0.78, z: 1.35 };

  /**
   * 纸飞机的判定采样点（局部坐标，沿机身外轮廓取 7 点）。
   * 纸飞机不是一个点：机头、两侧翼尖、翼缘中点、后缘都参与判定，
   * 这样用机翼蹭到音符也算命中，而不是只有机身原点那一点算数。
   * 坐标与 PaperPlane 的顶点一一对应（机头指向 -Z）。
   */
  const PLANE_HIT_POINTS = [
    [0, 0.035, -1.16], // 机头
    [-0.62, 0.08, -0.29], // 左翼前缘中点
    [0.62, 0.08, -0.29], // 右翼前缘中点
    [-1.24, 0.12, 0.58], // 左翼尖
    [1.24, 0.12, 0.58], // 右翼尖
    [-0.18, 0.08, 0.83], // 左后缘
    [0.18, 0.08, 0.83], // 右后缘
  ];

  /**
   * 采样点的额外膨胀（归一化单位）。
   * 默认 0：相邻采样点在归一化空间里只隔约 0.79，半径 1 的椭球本就互相重叠，不会漏判。
   * 调大会更宽容，但要注意算分公式是 `1 - distance`——膨胀后命中距离常驻在 1 附近，
   * 得分会被压到 1 分，所以不建议大于 0.05。
   */
  const PLANE_HIT_SKIN = 0;

  /** 单个音符的最高分（正中心命中） */
  const NOTE_MAX_SCORE = 100;

  /** 拖拽操控灵敏度 */
  const STEER = {
    xPerPixel: 10.2,
    yPerPixel: 9.36,
    maxX: 4.1,
    maxY: COLLISION_RADII.y * 3,
  };

  /** 场景中可飞行的横向边界 */
  const FLIGHT_X_LIMIT = 4.1;

  /**
   * 步道中心线：z → x。
   * 步道本身和结尾那张照片都用它，照片才会正落在路上而不是偏到草里。
   */
  const pathCenterX = (z) => 1.65 * Math.sin(-z * 0.008) + 0.48 * Math.sin(-z * 0.023);

  /** 结尾：平铺在步道尽头的一张照片，飞机掠过它飞向天空 */
  const DREAM_PHOTO = {
    /** 照片铺在飞机前方多远的路上 */
    ahead: 34,
    /**
     * 照片宽度（进深按图片比例算）。
     * 铺满整片地面（路面 11.2 宽，两侧草地也一起盖住）。
     * 图里孩子只占约 10% 宽——照片被嵌进一大片草地后再铺开，
     * 才能既盖住地面又保持孩子可辨认。
     */
    width: 46,
    /** 照片整体再往远处推多少 */
    back: 0,
    /** 离地高度，避免和路面 z-fighting */
    lift: 0.04,
    /** 淡入时长（秒） */
    fadeIn: 1.6,
    /** 相机最后停在照片后方多远、多高（要俯视才看得清平铺的画面） */
    camStop: 15,
    camHeight: 12.5,
    /** 相机推近所用的进度区间（0–1，对应整段结尾） */
    dolly: [0, 0.7],
    /** 视线在照片上方抬高多少：把照片压到画面下三分之一，避免前景大片空路面 */
    lookLift: 2.2,
    /**
     * 飞机航线：相对照片中心的偏移 [dx, y, dz]。
     * dz 为正表示在照片后方（靠相机一侧）。中间两点刻意压得很低，
     * 让飞机真的贴着照片表面横掠过去，而不是从上方飘过。
     */
    path: [
      [-1.2, 2.5, 12],
      [1.2, 1.4, 2],
      [4.2, 1.5, -3],
      [6.8, 7, -10],
      [9, 24, -17],
    ],
    /** 飞机走完这条航线的进度区间 */
    fly: [0.08, 0.9],
  };

  /**
   * 歌词字幕：[出现时间(秒), 文本]
   *
   * 时间轴来自唐宁《我和你》的官方 LRC（网易云 track 3409150942），
   * 再整体 +1.30s 校准到我们这段音频上——本音频比网易云版多了约 1.3 秒的前导静音：
   *   · 两边总时长差 1.2468s（85.86 vs 84.62）
   *   · 第一句人声实测出现在 2.50s，LRC 标 1.17s，差 1.33s
   *   · 以"每句紧跟一次换气"为准则做偏移优化，最佳偏移 1.32s（命中 14/22，不偏移仅 8/22）
   */
  const CAPTIONS = [
    [2.47, '晴天，有点孤单'],
    [5.95, '玩具，丢在旁边'],
    [9.43, '电视里没有新鲜'],
    [12.94, '球鞋跑不过时间'],
    [16.39, '我要更大的世界'],
    [19.96, '装满不同的探险'],
    [23.50, '当然你陪在身边'],
    [26.89, '每秒每天'],
    [28.84, '我和你，飞到蓝蓝的天边'],
    [32.32, '我和你，种下满满的花园'],
    [35.80, '我和你，分享暖暖的光线'],
    [39.55, '再靠近一点'],
    [40.99, '再靠近一点'],
    [42.85, '我和你，就像蓝蓝的天边'],
    [46.24, '我和你，就像满满的花园'],
    [49.72, '我和你，就像暖暖的光线'],
    [53.47, '把地球照亮'],
    [55.48, '把地球照亮'],
    [70.81, '再靠近一点'],
    [74.35, '再靠近一点'],
    [77.77, '再靠近一点'],
    [81.31, '再靠近一点'],
  ];

  PP.SONG_DURATION = SONG_DURATION;
  PP.BPM = BPM;
  PP.BEAT = BEAT;
  PP.FIRST_BEAT = FIRST_BEAT;
  PP.PLANE_YAW = PLANE_YAW;
  PP.PLANE_NOSE_UP = PLANE_NOSE_UP;
  PP.FLIGHT_MIN_Y = FLIGHT_MIN_Y;
  PP.FLIGHT_MAX_Y = FLIGHT_MAX_Y;
  PP.NOTE_Y_MIN = NOTE_Y_MIN;
  PP.NOTE_Y_MAX = NOTE_Y_MAX;
  PP.FAR_END = FAR_END;
  PP.PLANE_SPEED = PLANE_SPEED;
  PP.NOTE_COUNT = NOTE_COUNT;
  PP.LAUNCH = LAUNCH;
  PP.AWAKENING = AWAKENING;
  PP.TRANSFORM_DURATION = TRANSFORM_DURATION;
  PP.FAREWELL_LINGER = FAREWELL_LINGER;
  PP.COLLISION_RADII = COLLISION_RADII;
  PP.PLANE_HIT_POINTS = PLANE_HIT_POINTS;
  PP.PLANE_HIT_SKIN = PLANE_HIT_SKIN;
  PP.NOTE_MAX_SCORE = NOTE_MAX_SCORE;
  PP.STEER = STEER;
  PP.FLIGHT_X_LIMIT = FLIGHT_X_LIMIT;
  PP.DREAM_PHOTO = DREAM_PHOTO;
  PP.pathCenterX = pathCenterX;
  PP.CAPTIONS = CAPTIONS;
})(window.PP);
