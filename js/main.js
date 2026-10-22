/**
 * 入口：只负责创建 Game 并把初始化失败的情况说清楚。
 */
(function (PP) {
  try {
    const game = new PP.Game();
    // 方便在控制台调试：__paperPlane.state / .score / .audio
    window.__paperPlane = game;
    game.start();

    // 直接进第二关：网址后加 ?stage=2（预览窗口键盘可能不聚焦，网址参数更稳）
    try {
      if (new URLSearchParams(location.search).get('stage') === '2') {
        document.querySelector('#startCover')?.classList.add('gone');
        const tap = document.querySelector('#tapArea');
        if (tap) tap.style.display = 'none';
        game.enterPhase(PP.GameState.DANMAKU);
      }
    } catch {
      /* 忽略 */
    }
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
