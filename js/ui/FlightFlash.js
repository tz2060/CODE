/** 起飞过场的那层暖色光晕 */
(function (PP) {
  class FlightFlash {
    constructor(element = document.querySelector('#flightTransition')) {
      this.element = element;
      this.timer = null;
    }

    play(duration = 2300) {
      this.element.classList.remove('active');
      void this.element.offsetWidth;
      this.element.classList.add('active');
      clearTimeout(this.timer);
      this.timer = setTimeout(() => this.element.classList.remove('active'), duration);
    }
  }

  PP.FlightFlash = FlightFlash;
})(window.PP);
