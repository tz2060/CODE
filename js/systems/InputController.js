/**
 * 输入控制器
 * ---------------------------------------------------------------
 * 把原始的指针事件翻译成游戏语义：
 *  - 待机/蓄能：轻触 = 唤风
 *  - 飞行：拖拽 = 转向
 * 只做事件分发，不持有任何游戏状态。
 */
(function (PP) {
  const { GameState } = PP;

  class InputController {
    constructor({ canvas, tapButton, onTap, onSteer, getState }) {
      this.canvas = canvas;
      this.tapButton = tapButton;
      this.onTap = onTap;
      this.onSteer = onSteer;
      this.getState = getState;

      this.pointerId = null;
      this.lastX = 0;
      this.lastY = 0;

      this.#bind();
    }

    #bind() {
      this.tapButton.addEventListener(
        'pointerdown',
        (event) => {
          event.preventDefault();
          event.stopPropagation();
          this.onTap();
        },
        { passive: false }
      );

      this.canvas.addEventListener(
        'pointerdown',
        (event) => {
          const state = this.getState();
          if (state === GameState.AWAKENING) {
            this.onTap();
            return;
          }
          if (state === GameState.FLIGHT) {
            this.pointerId = event.pointerId;
            this.lastX = event.clientX;
            this.lastY = event.clientY;
            this.canvas.setPointerCapture?.(event.pointerId);
          }
        },
        { passive: false }
      );

      this.canvas.addEventListener(
        'pointermove',
        (event) => {
          if (this.pointerId !== event.pointerId) return;
          if (this.getState() !== GameState.FLIGHT) return;

          const deltaX = event.clientX - this.lastX;
          const deltaY = event.clientY - this.lastY;
          this.lastX = event.clientX;
          this.lastY = event.clientY;
          this.onSteer(deltaX, deltaY);
        },
        { passive: true }
      );

      const release = (event) => {
        if (this.pointerId === event.pointerId) this.pointerId = null;
      };
      this.canvas.addEventListener('pointerup', release);
      this.canvas.addEventListener('pointercancel', release);
      this.canvas.addEventListener('contextmenu', (event) => event.preventDefault());
    }

    release() {
      this.pointerId = null;
    }
  }

  PP.InputController = InputController;
})(window.PP);
