/** 居中的短暂提示（如「♪ +86 分」「跟着旋律，飞向远方」） */
(function (PP) {
  class Toast {
    constructor(element = document.querySelector('#toast')) {
      this.element = element;
      this.timer = null;
    }

    show(text, duration = 1600) {
      this.element.textContent = text;
      this.element.classList.add('show');
      clearTimeout(this.timer);
      this.timer = setTimeout(() => this.element.classList.remove('show'), duration);
    }

    hide() {
      clearTimeout(this.timer);
      this.element.classList.remove('show');
    }
  }

  PP.Toast = Toast;
})(window.PP);
