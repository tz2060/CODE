/** 通用数学工具 */
(function (PP) {
  function clamp(v, a, b) {
    return Math.max(a, Math.min(b, v));
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  /** 平滑阶跃：a→b 区间内的 0→1，首尾导数为 0 */
  function smooth(a, b, x) {
    const t = clamp((x - a) / (b - a), 0, 1);
    return t * t * (3 - 2 * t);
  }

  /**
   * 帧率无关的指数逼近，等价于手写的 1 - exp(-lambda * dt)。
   * lambda 越大越"跟手"。
   */
  function damp(current, target, lambda, dt) {
    return lerp(current, target, 1 - Math.exp(-lambda * dt));
  }

  PP.clamp = clamp;
  PP.lerp = lerp;
  PP.smooth = smooth;
  PP.damp = damp;
})(window.PP);
