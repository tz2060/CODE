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

  /** 歌词字幕：[出现时间(秒), 文本] */
  const CAPTIONS = [
    [0.0, '晴天，有点孤单'],
    [3.5, '玩具，丢在旁边'],
    [7.0, '电视里没有新鲜'],
    [10.5, '球鞋跑不过时间'],
    [14.0, '我要更大的世界'],
    [17.5, '装满不同的探险'],
    [21.0, '当然你陪在身边'],
    [24.5, '每秒每天'],
    [28.0, '我和你，飞到蓝蓝的天边'],
    [32.0, '我和你，种下满满的花园'],
    [36.0, '我和你，分享暖暖的光线'],
    [40.0, '再靠近一点'],
    [44.0, '再靠近一点'],
    [48.0, '我和你，就像蓝蓝的天边'],
    [52.0, '我和你，就像满满的花园'],
    [56.0, '我和你，就像暖暖的光线'],
    [60.0, '把地球照亮'],
    [65.0, '再靠近一点'],
    [70.0, '再靠近一点'],
    [75.0, '再靠近一点'],
    [80.0, '再靠近一点'],
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
  PP.NOTE_MAX_SCORE = NOTE_MAX_SCORE;
  PP.STEER = STEER;
  PP.FLIGHT_X_LIMIT = FLIGHT_X_LIMIT;
  PP.CAPTIONS = CAPTIONS;
})(window.PP);
