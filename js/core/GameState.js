/**
 * 游戏阶段（场景）枚举
 * ready → awakening → transform → flight → farewell → ended
 */
(function (PP) {
  const GameState = Object.freeze({
    /** 封面待机：纸飞机尚未出现 */
    READY: 'ready',
    /** 蓄能：轻触唤风，攒满能量或歌曲到点后起飞 */
    AWAKENING: 'awakening',
    /** 过场：纸飞机从光晕中现身 */
    TRANSFORM: 'transform',
    /** 飞行：跟随旋律收集音符 */
    FLIGHT: 'flight',
    /** 告别：最后两个「再靠近一点」，飞机抬升远去 */
    FAREWELL: 'farewell',
    /** 结算：结束封面 */
    ENDED: 'ended',
  });

  /** 是否处于可飞行的阶段 */
  function isFlying(state) {
    return state === GameState.FLIGHT || state === GameState.FAREWELL;
  }

  PP.GameState = GameState;
  PP.isFlying = isFlying;
})(window.PP);
