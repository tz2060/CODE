# 纸飞机 · 午后回声

一首歌，一个午后的三维纸飞机音乐飞行游戏。轻触唤起微风，折一架纸飞机，跟着旋律飞过公园的树影与湖面，把飘落的音符收进口袋。

> 从单文件原型 `纸飞机.html` 重构而来：**不需要服务器、不需要 npm install、不需要联网**，双击 `index.html` 就能玩。

## 运行

直接双击打开项目里的 `index.html` 即可（推荐 Chrome / Edge / Safari 较新版本，需要支持 WebGL）。

```
纸飞机/
└─ index.html   ← 双击它
```

> 原始原型把 Three.js 放在 CDN 上、音乐写成 base64 塞在 HTML 里。现在 Three.js 走本地 `vendor/`，音乐的 base64 挪进了 `js/audio/song-data.js`，`index.html` 只有 5KB，完全离线可用。

## 玩法

| 阶段 | 操作 | 说明 |
| --- | --- | --- |
| 待机 | 轻触圆环 | 音乐开始，进入蓄能 |
| 蓄能 | 连续轻触 | 攒满能量、或在 1.55 秒内连点 7 次即可起飞；歌曲走到 9.6 秒也会自动起飞 |
| 飞行 | 按住拖动 | 左右上下操控，纸飞机沿旋律路线前进 |
| 收集 | — | 撞到音符得分，越靠近中心分数越高（单个最高 100 分） |
| 告别 | — | 75 秒后飞机抬升远去，曲终结算 |

## 目录结构

```
纸飞机/
├─ index.html                  # 入口：纯标记，按依赖顺序引入各脚本
├─ README.md
├─ vendor/
│  └─ three.min.js             # Three.js r164（本地，非 CDN）
├─ assets/
│  └─ favicon.svg
├─ styles/                     # 按用途拆分的四份样式
│  ├─ base.css                 # 变量、重置、画布、光晕、提示、错误
│  ├─ hud.css                  # 品牌、得分、进度条、歌词、静音按钮
│  ├─ tap.css                  # 轻触按钮的呼吸光环与按下反馈
│  └─ covers.css               # 开始 / 结束封面
└─ js/
   ├─ main.js                  # 入口：创建 Game，兜底报错
   ├─ core/
   │  ├─ Game.js               # 编排者：持有全部模块，驱动主循环与阶段切换
   │  ├─ GameState.js          # 六个阶段的枚举
   │  ├─ constants.js          # 曲速、时长、歌词、飞行区间、灵敏度…全部配置
   │  ├─ math.js               # clamp / lerp / smooth / damp
   │  └─ EventBus.js           # 极简事件总线
   ├─ world/
   │  ├─ World.js              # 渲染器、相机、光照、布景装配
   │  ├─ scenery/              # 静态布景，一样一个文件
   │  │  ├─ sky.js             # 渐变天空穹顶
   │  │  ├─ ground.js          # 草地与光带
   │  │  ├─ parkPath.js        # 蜿蜒步道
   │  │  ├─ pond.js            # 池塘、石岸、水面浮光
   │  │  ├─ trees.js           # 行道树 + 微风摆动
   │  │  ├─ benchesAndLamps.js # 长椅与路灯
   │  │  ├─ gazebo.js          # 凉亭
   │  │  └─ sunAndClouds.js    # 夕阳、光晕、薄云
   │  └─ entities/             # 会动的对象
   │     ├─ PaperPlane.js      # 纸飞机几何体 + 姿态推导
   │     ├─ MusicNotes.js      # 音符群：布局、呼吸、碰撞、音高映射
   │     ├─ MelodyGuide.js     # 旋律引导虚线 + 中立路径插值
   │     └─ Sparks.js          # 光尘粒子
   ├─ audio/
   │  ├─ song-data.js          # 内置音乐的 base64 形式（自动生成，勿手改）
   │  ├─ AudioEngine.js        # <audio> 播放 + Web Audio 能量分析 + 离线解码
   │  └─ MelodyAnalyzer.js     # 自相关基频估计、缺值补齐、平滑
   ├─ systems/
   │  ├─ FlightController.js   # 前进、转向、姿态、追尾相机
   │  ├─ ScoreSystem.js        # 命中结算与反馈
   │  ├─ CaptionSystem.js      # 歌词推进
   │  ├─ InputController.js    # 指针事件 → 游戏语义
   │  └─ MelodyRoute.js        # 用真实音高重排旋律路线
   ├─ phases/                  # 六个场景，各自一个类（enter / update / exit）
   │  ├─ ReadyPhase.js
   │  ├─ AwakeningPhase.js
   │  ├─ TransformPhase.js
   │  ├─ FlightPhase.js
   │  ├─ FarewellPhase.js
   │  └─ EndedPhase.js
   └─ ui/
      ├─ Hud.js                # 得分、进度、歌词、静音
      ├─ Toast.js              # 居中提示
      ├─ Covers.js             # 开始 / 结束封面
      ├─ TapPad.js             # 轻触按钮
      └─ FlightFlash.js        # 起飞光晕
```

## 代码组织方式

`index.html` 用普通 `<script>` 标签按依赖顺序加载，每个文件是一个 IIFE，往唯一的全局命名空间 `window.PP` 上挂东西：

```js
(function (PP) {
  const { World, AudioEngine } = PP;   // 需要谁就先解构出来
  PP.Game = Game;
})(window.PP);
```

之所以不用 ES Module：**`file://` 协议下浏览器的 CORS 策略会拦截模块脚本**，双击打开就跑不起来。普通 `<script>` + 命名空间能在不引入任何构建工具的前提下实现同样的拆分。

数据流：

```
                 ┌──────────── Game ────────────┐
  用户输入 ──▶ InputController ──▶ tap()/steer() │
                 │                              │
                 ▼                              ▼
            Phase(当前场景)  ──▶  Systems  ──▶  UI
                 │                    │
                 ▼                    ▼
             Entities  ◀────────  World(scene/camera)
                 │
                 ▼
             renderer.render()
```

- **阶段（phases/）**只写"这一刻该发生什么"，不碰 DOM 与 Three.js 的创建。
- **系统（systems/）**是无状态的逻辑（计分、字幕、输入），通过构造参数拿到所需依赖。
- **实体（entities/）**自己管理几何体与生命周期，位置/姿态由系统驱动。
- **UI（ui/）**把 DOM 细节封在内部，对外只暴露语义方法（`setProgress`、`showPoem`…）。
- **配置**全部集中在 `js/core/constants.js`。

## 关于 vendor/three.min.js

Three.js r164 官方只提供 ESM 产物，而 ESM 在 `file://` 下无法加载。`vendor/three.min.js` 是把官方 `three.module.min.js` 的 `export {...}` 列表改写成 `window.THREE = {...}` 得到的等价物，内容与版本均未改动。

如需升级 Three.js 版本：

```bash
npm install three@<版本号>            # 下载到临时目录
node -e "                            # 把 ESM export 列表转成全局 THREE
const fs=require('fs');
let s=fs.readFileSync('node_modules/three/build/three.module.min.js','utf8');
const i=s.lastIndexOf('export{');
const list=s.slice(i).match(/export\{([\s\S]*?)\}/)[1];
const pairs=list.split(',').map(x=>x.trim()).filter(Boolean).map(e=>{
  const p=e.split(' as ');
  return p.length===2 ? p[1].trim()+':'+p[0].trim() : e+':'+e;
});
fs.writeFileSync('vendor/three.min.js', s.slice(0,i)+'window.THREE={'+pairs.join(',')+'};');
"
```

## 内置音乐在哪里？

音乐就是原型里那段，以 base64 形式放在 `js/audio/song-data.js` 里（约 1.83 MB），是项目里唯一的音频来源，没有 mp3 副本。`AudioEngine` 直接把它拼成 `data:` URL 交给 `<audio>` 元素播放。

### 为什么不用独立的 mp3 文件

把音频接进 Web Audio 分析链后（`createMediaElementSource`），**外部 mp3 在 file:// 下属跨域资源**，浏览器会把它静音重路由——控制台会提示：

```
MediaElementAudioSource outputs zeroes due to CORS access restrictions
```

实测：播放状态正常、`currentTime` 在走，但分析节点输出 `rms:0, peak:0`，一点声音都没有。
而 `data:` URL 与页面同源，不受此限制——原型当初把音乐写成 base64 内嵌，正好绕开了这个坑。

> 代价是 base64 比原文件大约 1/3。换来的是双击打开时，节拍律动、枝叶摆动、光晕呼吸、真实音高驱动的旋律路线全部可用。

### 换一段音乐

把新 mp3 的 base64 重新生成进去：

```bash
node -e "
const fs=require('fs');
const b64=fs.readFileSync('新的歌.mp3').toString('base64');
const out = [
  '/** 内置音乐（自动生成，请勿手改） */',
  '(function (PP) {',
  '  PP.SONG_BASE64 = \"' + b64 + '\";',
  '})(window.PP);',
  ''
].join('\n');
fs.writeFileSync('js/audio/song-data.js', out);
"
```

改曲目长度或歌词后，记得同步调整 `js/core/constants.js` 里的 `SONG_DURATION`、`BPM`、`CAPTIONS` 与 `FAR_END`。

## 已知的平台差异

| 能力 | 说明 |
| --- | --- |
| 音频自动播放 | 所有浏览器都要求首次点击后才能出声，这是策略限制；进入页面后轻触一下即可。 |
| 旋律分析 | 不依赖网络：直接解 base64 得到 PCM，真实音高驱动的旋律路线随时可用。 |

> 原型的音频用 base64 内嵌在 HTML 里（1.8MB），现在 `index.html` 只有 5KB，音乐的 base64 独立放在 `js/audio/song-data.js`（见「内置音乐在哪里？」）。
> 其余视觉、手感、计分、时间轴与原型保持一致。
