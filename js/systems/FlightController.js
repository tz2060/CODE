/**
 * 飞行控制器
 * ---------------------------------------------------------------
 * 纸飞机的 z 由歌曲时间决定（一往无前），x/y 由"旋律路线 + 玩家偏移"决定。
 * 相机以轻微的追尾视角跟随，告别阶段逐渐脱开。
 */
(function (PP) {
  const { damp, clamp, lerp, smooth } = PP;
  const {
    FLIGHT_MAX_Y,
    FLIGHT_MIN_Y,
    FLIGHT_X_LIMIT,
    PLANE_SPEED,
    SONG_DURATION,
    STEER,
    FAR_END,
  } = PP;

  class FlightController {
    constructor({ plane, camera, guide }) {
      this.plane = plane;
      this.camera = camera;
      this.guide = guide;

      /** 玩家拖拽产生的、相对于旋律路线的偏移量 */
      this.desiredX = 0;
      this.desiredY = 0;

      this.startTime = 0;
      this.startZ = 0;
      this.speed = PLANE_SPEED;
    }

    reset() {
      this.desiredX = 0;
      this.desiredY = 0;
    }

    launch({ startTime, startZ }) {
      this.startTime = startTime;
      this.startZ = startZ;
      this.speed = PLANE_SPEED;
      this.reset();
    }

    /** 拖拽转向：位移直接映射成路线偏移 */
    steer(deltaX, deltaY) {
      this.desiredX = clamp(
        this.desiredX + (deltaX / Math.max(window.innerWidth, 1)) * STEER.xPerPixel,
        -STEER.maxX,
        STEER.maxX
      );
      this.desiredY = clamp(
        this.desiredY - (deltaY / Math.max(window.innerHeight, 1)) * STEER.yPerPixel,
        -STEER.maxY,
        STEER.maxY
      );
    }

    /**
     * @param {number} dt
     * @param {{ songTime: number, farewell: boolean, lowEnergy: number }} opts
     */
    update(dt, { songTime, farewell, lowEnergy }) {
      const time = Math.min(songTime, SONG_DURATION);
      const elapsed = Math.max(0, time - this.startTime);
      const endFade = farewell ? smooth(FAR_END, SONG_DURATION, time) : 0;

      const pos = this.plane.position;
      pos.z = this.startZ - this.speed * elapsed;

      if (farewell) {
        // 告别：抬升远去，缓缓向左上方收束
        const fadeT = smooth(FAR_END, SONG_DURATION, time);
        pos.y = 2.35 + fadeT * 10.5;
        pos.x = lerp(pos.x, 0.25, fadeT * 0.025);
      } else {
        const route = this.guide.routeAtZ(pos.z);
        const targetY = clamp(route.y + this.desiredY, FLIGHT_MIN_Y, FLIGHT_MAX_Y);
        const targetX = clamp(route.x + this.desiredX, -FLIGHT_X_LIMIT, FLIGHT_X_LIMIT);
        pos.y = damp(pos.y, targetY, 10.5, dt);
        pos.x = damp(pos.x, targetX, 10.5, dt);
      }

      this.plane.updateAttitude(dt, {
        lowEnergy,
        bankFade: farewell ? 1 - endFade * 0.58 : 1,
      });

      this.#updateCamera(time, farewell, endFade);
    }

    #updateCamera(time, farewell, endFade) {
      const camera = this.camera;
      const pos = this.plane.position;

      // 告别阶段 follow 从 1 衰减到 0，相机留在原地看飞机远去
      const follow = farewell ? 1 - smooth(FAR_END, SONG_DURATION, time) : 1;
      const camZ = pos.z + 7.8;

      camera.position.x = lerp(camera.position.x, pos.x * 0.36 - 0.55, 0.075 * follow + 0.003);
      camera.position.y = lerp(camera.position.y, 3.95 + endFade * 1.0, 0.05 * follow + 0.002);
      camera.position.z = lerp(camera.position.z, camZ, 0.08 * follow + 0.0015);

      const lookZ = pos.z - 13 - (farewell ? endFade * 30 : 0);
      const lookY = pos.y + 0.25;
      camera.lookAt(lerp(pos.x * 0.52, 0, endFade * 0.55), lookY, lookZ);
    }
  }

  PP.FlightController = FlightController;
})(window.PP);
