/**
 * 用离线解码的音高数据重排旋律路线。
 * ---------------------------------------------------------------
 * 音符的 z 已按飞行速度铺好，因此每个音符的"被掠过时刻"是可算的：
 *   time = flightStartTime + (planeStartZ - note.baseZ) / planeSpeed
 * 在这些时刻取样音高，就能得到与真实旋律同步的等高线。
 *
 * @returns {boolean} 是否成功应用
 */
(function (PP) {
  const { estimateMelodyPitch, fillAndSmooth } = PP;

  function applyMelodyRoute({ buffer, notes, guide, flightStartTime, planeStartZ, planeSpeed }) {
    if (!buffer || !notes.items.length) return false;

    const pitches = notes.items.map((note) =>
      estimateMelodyPitch(buffer, flightStartTime + (planeStartZ - note.baseZ) / planeSpeed)
    );

    const smoothed = fillAndSmooth(pitches);
    if (!smoothed) return false;

    notes.applyPitchSeries(smoothed);
    guide.update(0, true, 0);
    return true;
  }

  PP.applyMelodyRoute = applyMelodyRoute;
})(window.PP);
