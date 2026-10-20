/** 底部的「轻触，让风起」按钮 */
(function (PP) {
  class TapPad {
    constructor() {
      this.area = document.querySelector('#tapArea');
      this.button = document.querySelector('#tapButton');
      this.text = document.querySelector('#tapText');
      this.hint = document.querySelector('#startTag');
    }

    show() {
      this.area.classList.remove('hide');
    }

    hide() {
      this.area.classList.add('hide');
    }

    setPrompt(text) {
      this.text.textContent = text;
    }

    setHint(text) {
      this.hint.textContent = text;
    }

    /** 按下反馈（纯视觉，不阻塞逻辑） */
    press() {
      this.button.classList.add('pressed');
      setTimeout(() => this.button.classList.remove('pressed'), 85);
    }

    reset() {
      this.setPrompt('轻触，让风起');
      this.setHint('TAP TO BEGIN');
      this.show();
    }
  }

  PP.TapPad = TapPad;
})(window.PP);
