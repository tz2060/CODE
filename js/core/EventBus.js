/** 极简事件总线：模块之间只通过事件名耦合 */
(function (PP) {
  class EventBus {
    constructor() {
      this.listeners = new Map();
    }

    on(type, handler) {
      if (!this.listeners.has(type)) this.listeners.set(type, new Set());
      this.listeners.get(type).add(handler);
      return () => this.off(type, handler);
    }

    off(type, handler) {
      this.listeners.get(type)?.delete(handler);
    }

    emit(type, payload) {
      const set = this.listeners.get(type);
      if (!set) return;
      for (const handler of [...set]) handler(payload);
    }
  }

  PP.EventBus = EventBus;
})(window.PP);
