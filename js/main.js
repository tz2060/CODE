/**
 * 入口：只负责创建 Game 并把初始化失败的情况说清楚。
 */
(function (PP) {
  try {
    const game = new PP.Game();
    // 方便在控制台调试：__paperPlane.state / .score / .audio
    window.__paperPlane = game;
    game.start();
  } catch (error) {
    console.error(error);
    const box = document.querySelector('#err');
    if (box) {
      box.textContent =
        '三维场景初始化失败，请使用较新的手机浏览器并保持网络连接，以加载 Three.js。';
      box.style.display = 'block';
    }
  }
})(window.PP);
