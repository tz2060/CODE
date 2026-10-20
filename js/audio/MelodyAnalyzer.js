/**
 * 离线旋律分析
 * ---------------------------------------------------------------
 * 不依赖任何外部库：把解码后的 PCM 每隔固定时间取一小窗，
 * 做一次降采样的自相关，估出基频，再换算成 MIDI 音高。
 * 分析失败时上游会退回内置的兜底轮廓。
 */
(function (PP) {
  const { clamp, lerp } = PP;

  /**
   * 估计某一时刻的旋律音高。
   * @param {AudioBuffer} buffer 解码后的音频
   * @param {number} time 秒
   * @returns {number|null} MIDI 音高，无法判断时返回 null
   */
  function estimateMelodyPitch(buffer, time) {
    const channel = buffer.getChannelData(0);
    const sampleRate = buffer.sampleRate;
    const windowSize = 4096;
    const step = 4;
    const N = windowSize / step;

    let start = Math.round(time * sampleRate - windowSize / 2);
    start = clamp(start, 0, Math.max(0, channel.length - windowSize));

    // 降采样 + 去均值 + 汉宁窗
    const data = new Float32Array(N);
    let mean = 0;
    for (let i = 0; i < N; i++) {
      data[i] = channel[start + i * step] || 0;
      mean += data[i];
    }
    mean /= N;
    for (let i = 0; i < N; i++) {
      data[i] = (data[i] - mean) * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1)));
    }

    // 自相关找基频周期，搜索范围约 170Hz–1000Hz
    const effectiveRate = sampleRate / step;
    const minLag = Math.max(10, Math.floor(effectiveRate / 1000));
    const maxLag = Math.min(72, Math.floor(effectiveRate / 170));

    let bestLag = -1;
    let best = -Infinity;
    for (let lag = minLag; lag <= maxLag; lag++) {
      let corr = 0;
      let energyA = 0;
      let energyB = 0;
      for (let i = 0; i < N - lag; i++) {
        const a = data[i];
        const b = data[i + lag];
        corr += a * b;
        energyA += a * a;
        energyB += b * b;
      }
      const norm = corr / Math.sqrt(Math.max(1e-12, energyA * energyB));
      if (norm > best) {
        best = norm;
        bestLag = lag;
      }
    }

    if (bestLag < 0 || best < 0.08) return null;

    const hz = effectiveRate / bestLag;
    if (!Number.isFinite(hz) || hz < 90 || hz > 1200) return null;

    return 69 + 12 * Math.log2(hz / 440);
  }

  /**
   * 补齐空值并做三点平滑。
   * @param {(number|null)[]} pitches
   * @returns {number[]|null} 有效点太少时返回 null
   */
  function fillAndSmooth(pitches) {
    const valid = pitches.filter((v) => v !== null && Number.isFinite(v));
    if (valid.length < Math.ceil(pitches.length * 0.35)) return null;

    const filled = pitches.slice();
    for (let i = 0; i < filled.length; i++) {
      if (filled[i] !== null && Number.isFinite(filled[i])) continue;
      let lo = i - 1;
      let hi = i + 1;
      while (lo >= 0 && (filled[lo] === null || !Number.isFinite(filled[lo]))) lo--;
      while (hi < filled.length && (filled[hi] === null || !Number.isFinite(filled[hi]))) hi++;
      filled[i] =
        lo >= 0 && hi < filled.length
          ? lerp(filled[lo], filled[hi], (i - lo) / (hi - lo))
          : lo >= 0
            ? filled[lo]
            : hi < filled.length
              ? filled[hi]
              : 66;
    }

    return filled.map((v, i) => ((filled[i - 1] ?? v) + v + (filled[i + 1] ?? v)) / 3);
  }

  PP.estimateMelodyPitch = estimateMelodyPitch;
  PP.fillAndSmooth = fillAndSmooth;
})(window.PP);
