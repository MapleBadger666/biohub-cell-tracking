# BioHub 14.15 恢复审计（2026-09-23）

## A. 当前真实状态

已通过现有认证 Chrome 会话访问 Kaggle Notebook notebookf25c6cd72d。不是 KAGGLE_RUNTIME_NOT_ACCESSIBLE。
仅重跑已有只读诊断：C-2-R1、首个 CUDA 可用性 Cell、A-3。A-7-R1 全 Input 递归扫描过慢，已中断，随后 A-3 定向检查成功。没有安装、构建 wheel、推理、跟踪、评分、上传或正式提交。

当前源码挂载 29 文件及 manifest 大小/SHA/文件集合 PASS；比赛 test 存在；CUDA_AVAILABLE=True，Tesla T4。
当前 torch=2.10.0+cu128、numpy=2.0.2、pandas=2.3.3、polars=1.35.2。zarr、numcodecs、geff、tracksdata、rustworkx、imagecodecs、ilpy 缺失；C-2-R1 还确认 pyscipopt 缺失。
/kaggle/working/biohub_1415、动态 split、两组预测目录缺失。Output 刷新仅有 .virtual_documents。Input 有比赛和源码 Dataset，没有 wheelhouse Dataset。

版本面板 Starting Fresh，无已保存版本；/data 页面跳回 /edit。出现过 Failed to save draft，不能依靠自动保存。此次已下载原草稿及历史输出，保存在 kaggle_draft_evidence.ipynb（244694 bytes，SHA256 4feeadee4a18850f02c2b42edcae8ecc3936d81324ab472b184672aff7f3c50a）。

搜索 Downloads、Documents、Desktop、项目、/private/tmp、uv/pip 缓存；未找到 Kaggle wheelhouse ZIP 或 13 wheel 副本。扫描两个相关 ZIP 和 pip 缓存 59 个 ZIP body，也无目标 wheel。不是声称检查了所有外接盘、云端或所有其他 Notebook。Kaggle 历史 pip cache 路径 /root/.cache/pip/wheels/e1/f2/5c/1e2520bb6c0b4aa768f9df49a682a593efc2324fbd9eff38a8 当前存在性未核实，pip wheel 可能复用缓存。

从原构建 Cell 及 stdout 重建 recovered_wheelhouse_manifest.json：3542 bytes，SHA256 9e132365f3c182a34d67a2c6b89d78f0bce246bf2bec6960d2f4ef824be3aafa，与历史记录精确匹配。仅恢复 manifest，不代表 wheel 字节恢复。

重要纠正：历史 14.15C-2 实际构建 policies 成功：gap_candidates=35680，single_policy=427，isolated_candidates=2888，strict_two_policy=195；TRACKED_GEFFS_SAVED=0。后续重跑因 split 丢失而失败。

## B. 全套检查结果

| 检查 | 状态 | 说明 |
|---|---|---|
| 本地源码 ZIP + manifest + 29 文件 | PASS | 文件 SHA、大小、ZIP 集合和 CRC 已检查 |
| 当前 Kaggle 源码挂载 29 文件 | PASS | A-3 本次只读执行通过 |
| tracksdata Git 来源与固定 commit | PASS | 本地 direct_url/uv.lock + 已恢复 Kaggle 原命令 |
| Kaggle tracksdata wheel 字节 | BLOCKED | 9e5e… 文件未恢复；本地 a12b… 是另一制品 |
| wheelhouse manifest 恢复 | PASS | 原始字节 SHA 与历史相同 |
| wheelhouse ZIP / 13 wheel 完整性 | BLOCKED | 当前无字节可读，不能逐项重新验真 |
| 全部递归依赖闭包 | NOT VERIFIED | 13 wheel 是 overlay，不是完整独立环境 |
| 从全新 Session 离线安装 | NOT VERIFIED | 历史 manifest 明示 BUILT_NOT_OFFLINE_TESTED |
| 项目 Import smoke | 历史 PASS / 当前 BLOCKED | 当前核心依赖缺失 |
| 当前 CUDA 可用性 | PASS | 本次探测 Tesla T4；未执行 CUDA 推理 |
| 全局 pip check | 历史 FAIL / 当前未重跑 | 不得称无冲突 |
| 两组全帧推理与语义检查 | 历史 PASS / 当前 BLOCKED | 八个 Kaggle GEFF 丢失 |
| policies | 历史 PASS / 当前 BLOCKED | 需对当前新预测重建 |
| Kaggle apply/save/CSV/roundtrip | NOT VERIFIED | 未执行；本地14.14有历史代码和结果 |
| 正式版本 Output / 比赛提交 | BLOCKED / 未完成 | 无提交结果或分数 |
| 新恢复 Notebook | 静态语法 PASS | 未作运行时或端到端验证，开关默认 False |

### tracksdata

Git URL https://github.com/royerlab/tracksdata，commit 7bfeaf845ceb951226f19b72fe5b80e01601018a；版本 0.1.0rc9.dev4+g7bfeaf845；wheel Requires-Python >=3.10。任务明确要求目标 Python3.12/Linux x86_64，优先于本地项目一般 Python3.11 约定。

已找回历史安装命令（本次未执行）：
```
/usr/bin/python3 -m pip install --no-deps git+https://github.com/royerlab/tracksdata.git@7bfeaf845ceb951226f19b72fe5b80e01601018a
```
已找回历史构建命令（本次未执行）：
```
/usr/bin/python3 -m pip wheel --no-deps --wheel-dir /kaggle/working/biohub_wheelhouse_py312_v1/wheels git+https://github.com/royerlab/tracksdata@7bfeaf845ceb951226f19b72fe5b80e01601018a
```
本地 wheel 297405 bytes，a12b65a10ab50f1449c532de72c8bf0c1cf0495088915dc93d4a30db93071df8；Kaggle wheel 历史297406 bytes，9e5ecf8cc4e297c55515e836109c34eaee7103a27b73563909db06e0b0bea8c4。原因未查明，不能混同。

### 13 wheel 历史清单

以下真实历史路径均为 /kaggle/working/biohub_wheelhouse_py312_v1/wheels/ 加文件名。来源是 D-1 stdout 与已恢复 manifest；当前文件 NOT FOUND，不是本次重新计算的 wheel 哈希。

| 文件 | bytes | 历史 SHA256 |
|---|---:|---|
| bidict-0.23.1-py3-none-any.whl | 32764 | `5dae8d4d79b552a71cbabc7deb25dfe8ce710b17ff41711e13010ead2abfc3e5` |
| donfig-0.8.1.post1-py3-none-any.whl | 21592 | `2a3175ce74a06109ff9307d90a230f81215cbac9a751f4d1c6194644b8204f9d` |
| geff-1.3.0.1.2-py3-none-any.whl | 69013 | `385b13b7d12a1ca9af3e21dd84eb6ac8eb25661d5764d42df7983618868c4c62` |
| geff_spec-1.2.0-py3-none-any.whl | 15233 | `875870a133e2df15d8261ba932222f5951c771c3550bee5e9e57c4a02a16b5a1` |
| ilpy-0.6.0-py3-none-any.whl | 32126 | `1c8e53144eafdaa1e59103543b9f05d38dfbf5819944f8b5766bd8a6f08fd6ab` |
| imagecodecs-2026.3.6-cp311-abi3-manylinux_2_28_x86_64.whl | 26468592 | `e30a14aa2e1c6c90e00375292726486c1d90bf003b1414d608ea4d1f62fd8a79` |
| numcodecs-0.15.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl | 8879163 | `c3a09e22140f2c691f7df26303ff8fa2dadcf26d7d0828398c0bc09b69e5efa3` |
| polars-1.43.2-py3-none-any.whl | 847150 | `22aa0cb92a1ee2d60d6a15a638b2e8e0dd99aea21ac0cd8fb29da8e382e075a9` |
| polars_runtime_32-1.43.2-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl | 57304599 | `6d5a7ae004a2723ebf4427f6d6a639f30f86af4cf077075f6b35d04711154fc3` |
| pyscipopt-6.2.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl | 17566755 | `faf86357d83772508c5b1925570ee7e08385fa1a736e0ee33bf0e5c421dd845f` |
| rustworkx-0.18.1-cp310-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl | 2366881 | `1ebd2441a51c68a784df8c62dc2e6f1a9f58c0eb1c2e21fd59776e08f55c0c0b` |
| tracksdata-0.1.0rc9.dev4+g7bfeaf845-py3-none-any.whl | 297406 | `9e5ecf8cc4e297c55515e836109c34eaee7103a27b73563909db06e0b0bea8c4` |
| zarr-3.1.6-py3-none-any.whl | 295655 | `b5a82c5079d1c3d4ee8f06746fa3b9a98a7d804300fa3f4be154362a33e1207e` |

历史 ZIP 113221626 bytes，SHA256 b25787484b0a6430caaf203aa5557ffefd4e7713c00b537accaf47c4b0a1f1bc。
第一次 D-2 在 set(archive.namelist()) 比较处失败，发生在 ZIP/外部 manifest SHA 断言之后、每个 wheel 检查之前；目录条目很可能解释集合差异，但无法在已丢失 ZIP 上实证。不能称 ZIP 损坏。后续 R1 则在 ZIP.is_file() 处失败，属于文件缺失。
新 verify_wheelhouse_cell.py 用 ZipInfo.is_dir() 区分目录/文件，检查重复路径、manifest、每个 wheel 大小/SHA及CRC，不要求外部展开目录同时存在。

### 依赖闭包

13 wheel 是12个固定索引包加1个tracksdata wheel，通过 --no-deps 下载。torch/numpy/pandas 来自镜像，本次版本吻合。
当前已确认其他基础包：scipy1.16.3，dask2026.1.1，blosc2 4.1.2，scikit-image0.25.2，networkx3.6.1，numba0.60.0，tqdm4.67.3，pyarrow24.0.0。
历史 A-8-R6 还记录：psygnal0.15.1、rich13.9.4、sqlalchemy2.0.49、typing-extensions4.15.0、pydantic2.12.3、typer0.24.2。这些当前版本及它们的所有递归依赖未全部重新验证。
numcodecs/pyscipopt 的 cp312 wheel 与目标一致；polars-runtime/rustworkx 的 cp310-abi3 和 imagecodecs 的 cp311-abi3 不应仅因 cp 数字而被误判不兼容。新 bootstrap 使用当前 sys_tags 与 Requires-Python 实查。
已知历史 pip check：cudf-polars-cu12 26.2.1 要求 polars<1.36,>=1.30，与1.43.2冲突；另有colab/pandas/jupyter-server、bigframes/google-adk依赖等镜像冲突。新流程分别记录全局 pip check 与项目递归闭包，不能把项目 smoke PASS 等同全局无错误。

### 执行链

1. 源码/依赖挂载校验：源码已有历史代码且本次PASS；wheelhouse待持久化。
2. 离线安装/smoke：新 bootstrap 完整Cell已提供，但未运行；先验证overlay+base的全部有效递归Requires-Dist，再允许--no-index --no-deps安装已逐个校验的wheel，不替换CUDA基础栈。
3. 动态测试发现：历史B-2存在；新Cell使用当前Zarr stem作为规范dataset ID，数量/名称/前缀均动态。
4–5. 0.995/0.9975 CUDA全帧：历史B-3/B-6源码与日志已恢复；新Cell复用真实predict接口、相同配置和运行时输出重定向，未运行。
6. GEFF检查：历史B-5/B-7 PASS；新Cell校验节点ID/坐标/边端点/时间方向/度数及原始预测边属性，未运行。
7. policies：历史C-2 PASS；新Cell仅对当次预测重建，train_dir=当前TEST_DIR，prefix_map=None。
8–9. apply与官方save_graph：本地14.14C有真实代码；Kaggle未完成；新Cell保留冻结配置与顺序，输出目录不存在检查，overwrite=False。
10. 官方geffs_to_csv：真实接口已有；新Cell输出暂存CSV。
11. CSV结构/官方往返：本地14.14D已有历史成功结果；新Cell检查连续id、dataset集合、整数/范围、unused=-1、图端点、行数，并调用官方csv_to_geffs(...,overwrite=False)再导出，比较坐标与有向边语义哈希，不要求重编号ID相等。
12. 正式Save Version完整运行与Output：仍待用户授权运行及确认；新Cell只生成/kaggle/working/submission.csv，不上传或提交。

阻断与风险：predict()先清理其选定目录所有*.geff，必须独立新目录；默认输出随源码ROOT指向只读Input，新Cell只改运行时PREDICTIONS_PATH/USERNAME并在finally恢复；历史.zarr.geff不能直接作为最终规范命名，否则官方转换器stem会让CSV dataset带.zarr；新run统一无扩展名ID。
全局单边候选或isolated-middle候选为空时frozen_v9主动ValueError，新Cell不捕获为成功、不发明fallback。每个样本尺度读取其Zarr元信息，不写死各向同性或固定数值。冻结预测器需要0.001/0.999 quantiles；缺失时fail closed。
官方CSV转换器仍会整体持有DataFrame；隐藏测试更大时可能内存不足。新检查尽量按dataset处理，但不宣称消除了官方转换的内存风险。
源包含Stage12文件是打包事实；新执行链没有重新导入Morph/Stage12/scorer。没有GT评估入口调用。

## C. 已执行的检查

本地执行 git status --short；rg --files --hidden 搜索目标名称；Python os.walk 有界搜索 Downloads/Documents/Desktop/tmp；读取两份相关ZIP目录和59个pip缓存ZIP body；hashlib流式SHA；zipfile.infolist/testzip/归档内容SHA；wheel METADATA和direct_url读取；Notebook JSON/AST读取；12个本地GEFF按历史目录树哈希方法检查；CSV文件哈希；新Cell ast.parse静态校验。
浏览器只读操作：现有Notebook、Versions、Output、Your Work搜索，下载现有Notebook；执行三项已说明的只读诊断。慢扫描已取消，无安装或推理动作。
新文件只写入本审计目录；原Notebook14、模型、源码、官方脚本、Bundle ZIP、既有GEFF/CSV/报告均未修改。Kaggle已有诊断Cell输出发生更新，代码未修改。

关键SHA：源码ZIP 4b7c32b08d0a12e92cacbbce846ff18e3effd1910a9dc86f2627320ca8026146；源码manifest 615c3550cba46a5032344549d2769201af22c0425beb9349726874669ae9c865；checkpoint 31d209bfde1fd5bbf3b2a661cdb4a25bb72922b832d315ee5a75cc32f44656c1；frozen_v9 486ae177e8709ddd17bf20d61dc628477a78d7434cd54a7ecf3729915b36ab17。

### 本地已找到制品（路径、大小、SHA）

GEFF为目录树SHA，算法：按相对路径排序，将relative UTF-8 + NUL + 文件SHA原始32字节 + NUL依次输入SHA256。大小为文件字节合计。所有本地GEFF与14.14 manifest匹配，仅可作为历史审计/回归材料，不能冒充丢失的Kaggle CUDA输出或隐藏测试提交。

| 路径 | bytes | SHA256 |
|---|---:|---|
| /Users/heqiuyan/Desktop/biohub-cell-tracking/predictions/heqiuyan/submission_v1_warm2_det0995/split_0/44b6_0113de3b.geff | 333239 | `648279049970253b15214a835bb165998f7b5fd4d9d5da8eb688b686f4260992` |
| /Users/heqiuyan/Desktop/biohub-cell-tracking/predictions/heqiuyan/submission_v1_warm2_det0995/split_0/44b6_0b24845f.geff | 656876 | `9927401fd6ff5cbfacffb3cc0c794e60e9c5243251895cdccd1601a272ecedf6` |
| /Users/heqiuyan/Desktop/biohub-cell-tracking/predictions/heqiuyan/submission_v1_warm2_det0995/split_0/6bba_05b6850b.geff | 111405 | `393ada9f514f98c3d9fd5bebb28b409776556bfb797e6806b99d9d2235173898` |
| /Users/heqiuyan/Desktop/biohub-cell-tracking/predictions/heqiuyan/submission_v1_warm2_det0995/split_0/6bba_05db0fb1.geff | 923762 | `98b90c664b5e46a51d5b7da2df5282377073adb2d2221688b976cbf7404602e5` |
| /Users/heqiuyan/Desktop/biohub-cell-tracking/predictions/heqiuyan/submission_v1_warm2_det09975/split_0/44b6_0113de3b.geff | 324625 | `12b5a86c1158b9361955a270db2256932f0da98163ecc9f36c9b9e9e3ad5143a` |
| /Users/heqiuyan/Desktop/biohub-cell-tracking/predictions/heqiuyan/submission_v1_warm2_det09975/split_0/44b6_0b24845f.geff | 559934 | `7d19babd318a8a1cb45cffa87eff571253faaec269b60ef991f6879ef99534cf` |
| /Users/heqiuyan/Desktop/biohub-cell-tracking/predictions/heqiuyan/submission_v1_warm2_det09975/split_0/6bba_05b6850b.geff | 103538 | `1201a71bcd1ac73998f7315d4a9768ba6fc8f14fae3aacf5a6a4aab4c3566a7b` |
| /Users/heqiuyan/Desktop/biohub-cell-tracking/predictions/heqiuyan/submission_v1_warm2_det09975/split_0/6bba_05db0fb1.geff | 888676 | `c9caf91dd27a24dc0287a21af3b264748dbfefe84d3554b5f56a6f2de54d9352` |
| /Users/heqiuyan/Desktop/biohub-cell-tracking/scratchpad/notebook14_14_14_submission_v1/tracking_geffs/44b6_0113de3b.geff | 331329 | `75c908098f449fd565df6d32fd5dfb69723c670dbd62ee372fb4db109a8b60b0` |
| /Users/heqiuyan/Desktop/biohub-cell-tracking/scratchpad/notebook14_14_14_submission_v1/tracking_geffs/44b6_0b24845f.geff | 505698 | `eda00876b75a85e21d8780203e72d043fa687a0313ca77f1fea8fa551e583f0c` |
| /Users/heqiuyan/Desktop/biohub-cell-tracking/scratchpad/notebook14_14_14_submission_v1/tracking_geffs/6bba_05b6850b.geff | 101886 | `f8a61da5e91d0fb0307a6739d07c565be5118a6ea4691bcaaabd6364a1d90bed` |
| /Users/heqiuyan/Desktop/biohub-cell-tracking/scratchpad/notebook14_14_14_submission_v1/tracking_geffs/6bba_05db0fb1.geff | 862927 | `cdb6606b2ddbc5468e054978f13626d22ca655a2337f54b621ea5f89947e3806` |
| /Users/heqiuyan/Downloads/notebookf25c6cd72d.ipynb | 244694 | `4feeadee4a18850f02c2b42edcae8ecc3936d81324ab472b184672aff7f3c50a` |
| /Users/heqiuyan/Desktop/biohub-cell-tracking/submissions/submission_v1_frozen_v9_warm2.csv | 14450896 | `eb1d384ab97e77bbd215ab902c7635cc69c8241dee10dae4a55595add785d807` |
| /Users/heqiuyan/Desktop/biohub-cell-tracking/scratchpad/notebook14_14_14_submission_v1/14_14d_submission_unverified.csv | 14450896 | `eb1d384ab97e77bbd215ab902c7635cc69c8241dee10dae4a55595add785d807` |
| /Users/heqiuyan/Desktop/biohub-cell-tracking/scratchpad/notebook14_14_14_submission_v1/14_14d_roundtrip.csv | 14437995 | `d225404ce658b83b6fa55391969fa8cd5ae27857488628993bacdec3ce08ac04` |
| /Users/heqiuyan/.cache/uv/sdists-v9/git/d52a6d697cd1c2a8/7bfeaf845ceb9512/tracksdata-0.1.0rc9.dev4+g7bfeaf845-py3-none-any.whl | 297405 | `a12b65a10ab50f1449c532de72c8bf0c1cf0495088915dc93d4a30db93071df8` |

## D. 最短恢复路径

1. 保留已恢复草稿、manifest和13 wheel清单。源码无需重打包。
2. 若无其他外部副本，使用01_wheelhouse_rebuild_NOT_RUN.ipynb，明确启用build开关后，仅重建依赖。原命令锁定Python3.12/Linux和Git commit，pip可能复用缓存。禁止覆盖现有目录/ZIP。
3. 立即校验并下载ZIP到Mac，重新计算下载后SHA；之后再由用户上传为私有依赖Dataset并挂载。只有Session中的构建成功不算持久化。新ZIP可能因时间戳改变SHA；不得强套旧ZIP SHA。wheel/manifest如果变化需要身份复核。
4. 全新Session、Internet关闭、源码+比赛+依赖Input：运行02的bootstrap。若递归闭包缺项，停止并一次性补齐所报告依赖，不零散试装。
5. 经用户批准GPU推理/跟踪后启用RUN_FROZEN_PIPELINE，一次运行动态发现→两阈值→语义检查→policies→apply/save→官方CSV→往返。所有输出新建，不删旧文件。
6. 保存包含完整运行的正式版本，确认运行完成、Output内submission.csv与run_manifest/environment_report真实存在、字节数/行数/SHA正确。自动保存草稿或仅有Cell日志不等于成功版本Output。比赛提交另行授权。

## E. 用户最少操作

- 批准或自行运行一次依赖重建，并把ZIP下载/上传为私有Dataset、挂载；Codex本次未获上传权限。
- 使用提供的新提交Notebook，在全新目标Session完成一次离线恢复；经明确批准后开启两组推理与冻结跟踪。
- 保存完整运行版本并检查Output；最后是否提交比赛由用户另行决定。

不需要用户重复运行旧的路径诊断Cell，当前缺失、GPU和源码状态已由Codex直接核实。

## F. 交接摘要

源码本地和当前Kaggle29文件PASS，CUDA Tesla T4当前PASS。Kaggle依赖缺失、八预测丢失、wheelhouse ZIP/13wheel未恢复；manifest已按原SHA精确恢复。原Kaggle Notebook40 cells与历史输出已下载。历史policies构建曾PASS但未保存跟踪GEFF。13wheel为依赖overlay，不是完整闭包，离线恢复从未验证。新重建/离线完整流程Notebook已写好且仅做静态语法验证，操作开关默认False。先持久化依赖、再全新Session离线验收，获批后一次性运行冻结链并检查版本Output。不得混用本地a12b tracksdata wheel与Kaggle9e5e wheel，不接入GT/Morph/Stage12/scorer，不修改frozen_v9空候选行为。没有正式提交或Kaggle分数。
