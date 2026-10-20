/** 开始封面 / 结束封面 */
(function (PP) {
  class Covers {
    constructor() {
      this.start = document.querySelector('#startCover');
      this.end = document.querySelector('#endCover');
      this.restartButton = document.querySelector('#restart');
    }

    hideStart() {
      this.start.classList.add('gone');
    }

    showStart() {
      this.start.classList.remove('gone');
    }

    showEnd() {
      this.end.classList.remove('gone');
    }

    hideEnd() {
      this.end.classList.add('gone');
    }

    onRestart(handler) {
      this.restartButton.addEventListener('click', handler);
    }
  }

  PP.Covers = Covers;
})(window.PP);
