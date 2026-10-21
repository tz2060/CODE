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
| 收集 | — | 撞到音符得分，越靠近中心分数越高（单个最高 100 分）。判定用机身外轮廓，机翼蹭到也算 |
| 告别 | — | 最后几句「再靠近一点」：音符消失，步道尽头铺开一张照片，飞机掠过它飞进天空，曲终结算 |

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
   │     ├─ Sparks.js          # 光尘粒子
   │     └─ DreamPhoto.js      # 结尾平铺在路面上的那张照片（含地面光池）
   ├─ assets/
   │  └─ dreamScene.js         # 结尾那张图的 base64（自动生成，勿手改）
   ├─ audio/
   │  ├─ song-data.js          # 内置音乐的 base64 形式（自动生成，勿手改）
   │  ├─ AudioEngine.js        # <audio> 播放 + Web Audio 能量分析 + 离线解码
   │  └─ MelodyAnalyzer.js     # 自相关基频估计、缺值补齐、平滑
   ├─ systems/
   │  ├─ FlightController.js   # 前进、转向、姿态、追尾相机
   │  ├─ ScoreSystem.js        # 命中结算与反馈
   │  ├─ CaptionSystem.js      # 歌词推进（还没到第一句时不显示）
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

## 结尾：铺在路上的那张照片

曲子的最后几句「再靠近一点」，是整局的高潮：

1. 音符与旋律引导线一次性消失，画面安静下来
2. 步道尽头浮出一张**平铺在路面上的照片**——孩子趴在草地上打盹的那张
3. 相机缓缓抬起、推近，俯视着它；纸飞机从远处飞来，贴着照片表面掠过
4. 掠过之后拉起飞进天空，曲终，结算

几个实现上的关键点：

- **照片是世界里的实体，不是贴在镜头前的图层**。它躺在 `DREAM_PHOTO.ahead` 处的路面上，x 由 `pathCenterX(z)` 算出来，所以永远正落在路中间，不会偏到草里。
- **必须俯视才看得清**。平铺的画面在浅角度下会被压扁成一条，所以相机在结尾会从飞行高度（约 4）抬到 `camHeight`（12.5），同时推近到照片后方 `camStop`（15）处。
- **视线分三段**：先沿飞行时的朝向稳住 → 落到照片上（抬高 `lookLift`，把照片压到画面下三分之一，避免前景一片空路面）→ 最后跟着飞机抬头看天空。
- **飞机航线**是一串相对照片中心的偏移点（`DREAM_PHOTO.path`），中间两点刻意压到 y≈1.4，让飞机真的贴着照片横掠过去，而不是从上方飘过。

相关参数都在 `constants.js` 的 `DREAM_PHOTO` 里（距离、尺寸、相机落点、航线、视线抬升）。

### 顺带修掉的两处「场景空」

原来的树列只排到 z ≈ -602，而片尾飞机已经在 z ≈ -640 —— 照片正好落在光秃秃的荒地之外。
现在树列延伸到 48 组（约 z = -880），云也改成跟着相机在 z 上平移（`World.followCamera`），
不然飞到片尾时云早就在身后了，天空只剩一条空渐变。

### 换掉结尾那张图

图片以 base64 内嵌在 `js/assets/dreamScene.js`（原因和音乐一样：`file://` 下外部图片会被判跨域，建不出 WebGL 纹理）。原图 701×485 PNG，转存为 JPEG quality 88（约 60KB）：

```bash
node -e "
const fs=require('fs');
const b64=fs.readFileSync('新的图.png').toString('base64');
const out = [
  '/** 结尾记忆画面（自动生成，请勿手改） */',
  '(function (PP) {',
  '  PP.DREAM_IMAGE = \"data:image/png;base64,' + b64 + '\";',
  '})(window.PP);',
  ''
].join('\n');
fs.writeFileSync('js/assets/dreamScene.js', out);
"
```

图的比例任意——清晰层会按比例完整显示，模糊层负责填满剩余空间。

## 歌词时间轴是怎么校准的

原型里的字幕是按 3.5 秒均分硬编码的，和实际人声对不上（越到后面偏得越多，结尾差了 5 秒以上）。现在的时间轴来自唐宁《我和你》的官方 LRC（网易云 track `3409150942`），再整体平移校准到本项目的音频上：

| 证据 | 结果 |
| --- | --- |
| 两边总时长 | 85.86s vs 84.62s，差 **1.2468s** |
| 第一句人声实测出现 | 2.50s（LRC 标 1.17s），差 **1.33s** |
| 以「每句紧跟一次换气」做偏移优化 | 最佳偏移 **1.32s**（命中 14/22，不偏移仅 8/22） |

三条证据收敛到同一个结论：**本音频比网易云版多了约 1.3 秒的前导静音**。因此取 `+1.30s` 统一平移。

另外字幕系统现在会区分「还没到第一句」——0–2.47s 之间不显示任何歌词，而不是提前把第一句挂出来。

## 碰撞判定

音符那边是一个**椭球**，纸飞机这边是一圈**采样点**——判定语义是「机身外轮廓上任意一点是否落进椭球」。

```
constants.js
├─ COLLISION_RADII  { x: 1.35, y: 0.78, z: 1.35 }   音符的椭球半径
├─ PLANE_HIT_POINTS  7 个点：机头 / 两翼前缘中点 / 两翼尖 / 两后缘
└─ PLANE_HIT_SKIN    采样点额外膨胀，默认 0
```

`MusicNotes.findHits(points)` 遍历每个音符，对 7 个采样点算归一化距离、取**最小值**，小于 `1 + PLANE_HIT_SKIN` 即命中；这个最小距离同时决定得分 `100 × (1 - distance)`。

**为什么不给飞机一个球形半径**：机身是扁的（厚 0.16、宽 2.48、长 2.0），一个球要么把上下判得过大、要么把左右判得过小。沿外轮廓取点更贴合实际形状，代价只有每帧几十次距离计算。

**为什么 SKIN 默认是 0**：相邻采样点在归一化空间里只隔 0.27，半径 1 的椭球本就大幅重叠，不会漏判。而一旦加上膨胀，命中距离会常驻在 1 附近，`1 - distance` 被压到 1 分，得分尺度就废了。

实测效果（以机身中心为原点，飞机摆正）：

| 方向 | 改前 | 改后 |
| --- | --- | --- |
| 横向命中半宽 | 1.0 | 2.0 ~ 2.46 |
| 纵向命中半高 | 0.78 | 0.78 ~ 0.85 |

横向大约翻倍（翼尖那 1.24 个单位现在算数了），纵向基本不变——因为机身本来就很薄。

## 已知的平台差异

| 能力 | 说明 |
| --- | --- |
| 音频自动播放 | 所有浏览器都要求首次点击后才能出声，这是策略限制；进入页面后轻触一下即可。 |
| 旋律分析 | 不依赖网络：直接解 base64 得到 PCM，真实音高驱动的旋律路线随时可用。 |

> 原型的音频用 base64 内嵌在 HTML 里（1.8MB），现在 `index.html` 只有 5KB，音乐的 base64 独立放在 `js/audio/song-data.js`（见「内置音乐在哪里？」）。
> 其余视觉、手感、计分、时间轴与原型保持一致。
