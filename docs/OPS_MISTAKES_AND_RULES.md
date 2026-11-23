# 运维事故复盘与硬性规则（GLM-5.3 MTP A/B 评测期间）

日期：2026-10-30
背景：36 机（A3-syn-26）dspark1 容器做 A/B 评测。**空闲卡窗口来之不易，以下每条
错误都直接烧了卡时。后续任何 agent/人操作卡资源前必须过一遍本文档清单。**

---

## 一、事故清单（时间序，根因 → 代价）

| # | 事故 | 根因 | 代价 |
|---|------|------|------|
| 1 | serve 启动 ~35min 后 graph capture 崩溃 | 把**训练**环境变量 `TASK_QUEUE_ENABLE=2` 抄进了 serving（serving 只支持 1/0） | 1 次完整启动作废 |
| 2 | speculative-config 变成裸文本，argparse 拒绝 | 在 `ssh → sudo sh -c → docker exec bash -c` 多层嵌套里手写 JSON 引号，转义塌了 | 1 次启动作废 |
| 3 | 重启后 OOM：设备只剩 8.6/61.3GiB | `pkill -f vllm`（小写）杀不掉进程名 `VLLM::Worker_TP*`（大写），残留进程占 53.8GB HBM/卡 | 1 次启动作废 |
| 4 | graph capture 报 `EE1023 Too many ACL graphs`，第二次同阶段静默死 | 自拼配置抓了 48 张图（16 worker 并发），notify 资源耗尽；且杀完进程只等了 ~10s 就重启（驱动资源未释放干净） | 2 次启动作废 |
| 5 | **首条真实请求引擎猝死**（`'function' object is not subscriptable`） | 容器 PATH 无 bisheng → triton-ascend 被禁用 → `_compute_slot_mapping_kernel` 退化为普通函数，`kernel[grid]` 语法崩。**启动成功 ≠ 能推理** | 2 次完整启动作废 |
| 6 | 把还活着的引擎误判为已死，又起了一个 → 同端口同日志互相打架，两个都死 | 上一次崩溃进程的残写让日志变成稀疏/二进制混合体（"TBE main process disappeared"+shutdown 噪音混着新 run 的正常输出），误读 | 1 次完整启动作废 |
| 7 | bench 并发了 2 个进程，计数器 delta 互相污染，结果作废 | 被中断的 bench 进程没死（本地 abort 未传导）；`pkill -9 -f bench_acceptance` 在 `bash -c` 里**自匹配**杀了自己的包装 shell，清理假成功 | 1 轮 25min bench 作废 |
| 8 | 长任务挂在 ssh/docker exec 前台，本地中断直接杀死远端评测 | 没用脱离运行 | 用户时间 ×N |
| 9 | 多条 sudo 命令 `&&` 串联，第二个 sudo 无 stdin 密码报错 | sudo -S 只读一次 stdin | 反复小失败 |
| 10 | 复杂正则穿三层引号 grep，输出被吞、误诊 | 同上引号问题 | 排障时间 ×2 |
| 11 | **训练启动即 SIGSEGV（无 Python 栈）**，16 卡首跑、1-rank warmup、faulthandler 复跑三次必现 | r89 镜像（CANN 9.2.1-20261109）自带 **torch_npu 2.10.0.post7**：`torch.npu.memory_allocated()` → `memory_stats_as_nested_dict` 的 C 层崩溃，由 fill 阶段 `[mem]` 日志调用触发。**单变量对照**：pip 降 `torch_npu==2.10.0.post4` 后同容器同代码完全正常（V2 全量 10h+ 无事故） | 3 次启动作废 + 排障 ~2h；防线 = 新环境 checklist 第 0 项 + 首启必带 `PYTHONFAULTHANDLER=1` |

**合计**：约 1.5 小时卡时浪费在部署事故上（评测本身只需 2×(8min 启动 + 25min bench)）。

**事故 11 证据链**（2026-11-13，26 机）：
- 现场：`h0-test/eval35/v2_fh.log` faulthandler 栈
  （`memory.py:353 memory_stats_as_nested_dict ← :341 memory_stats ← :462
  memory_allocated ← modeling_glm53_joint.py:321 _mem ← fill_joint`）
- 归因边界：仅验证了 **r89 镜像 + A3 driver 26.1.1 + torch 2.10.0+cpu** 组合下
  post7 必崩；未验证其他 CANN/驱动组合——新环境如用其他镜像，仍需先跑
  `python3 -c "import torch,torch_npu; torch.npu.set_device(0);
  print(torch.npu.memory_allocated())"` 冒烟
- 本会话独立发现，非官方 release note 结论；建议有条件时反查 hiascend issue 库

---

## 二、硬性规则（每次占用卡资源前的 Checklist）

### 启动服务前
0. **torch_npu 版本锁**（新环境/新镜像第 0 项）：`pip show torch_npu` 必须
   == 2.10.0.**post4**；r89 系镜像原生 post7 有 memory_allocated() 段错误
   （事故 11）。镜像不一致时先跑
   `python3 -c "import torch,torch_npu; torch.npu.set_device(0);
   print(torch.npu.memory_allocated())"` 冒烟
1. **进程清零**：`pkill -9 -fi vllm`（**必须 -i**，进程名是大写 `VLLM::`）；
   再 `ps aux | grep -c vllm` 确认 = 0
2. **HBM 清零**：`npu-smi info -t proc-mem -i <i> -c <c>` 逐卡确认无残留进程；
   只看 `ps` 不够，HBM 释放滞后于进程死亡
3. **驱动沉淀**：杀完进程**至少等 60s** 再启动（ACL/notify 资源释放滞后）
4. **配置 1:1 对齐已验证服务**，不自己拼新参数组合（cudagraph_capture_sizes [16,32]
   是防 ACL 资源耗尽的关键）

### 启动方式（不可妥协）
5. **一律脚本文件**：禁止在 ssh/sudo/docker exec 嵌套里内联复杂参数（JSON 引号必塌）。
   写 `scripts/serve_36.sh`，`docker exec -d bash <script>`
6. **每个 run 独立日志文件名**（含时间戳/pid），绝不让新旧 run 写同一路径——
   崩溃进程的残写会污染日志导致误诊（事故 #6）
7. **sudo 只包一层**：`echo pw | sudo -S sh -c '整段命令'`，禁止 `&& sudo` 串联

### 启动后（就绪 ≠ 可用）
8. **smoke 必做**：全量评测前先 `--n 2` 真实推理，确认 HTTP 200 且 spec 计数器增长。
   引擎初始化完成/HTTP started 只证明能启动，**首条请求才是真正的验收点**
9. serving 环境变量：`TASK_QUEUE_ENABLE=1`（不能 2）；
   `PATH` 必须含 `/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/bin`
   （bisheng → triton-ascend 激活，否则首请求必崩）

### 评测执行
10. **一律脱离运行**：`docker exec -d ... nohup python -u ... > <独立日志> 2>&1`，
    本地连接断/中断不影响远端；**轮询结果文件**（JSON 落盘），不持连接等待
11. **bench 单实例**：启动前确认 0 个旧 bench；清理用自不匹配 pattern
    `pkill -9 -f "bench_acceptance[.]py"`（`pkill -f bench_acceptance` 在 bash -c 里会
    **自杀**——自己的 cmdline 含该串）；启动后确认**恰好 1 个** python 进程
12. **结果以 JSON 文件为准**，日志 tail 只作辅助；切服务前必须确认上一侧 JSON 已落盘

### 排障
13. 复杂日志分析：先 `docker exec cat <log> > 本地文件`，在本地用 python/grep 分析；
    禁止在三层嵌套里写复杂正则（输出会被吞）
14. 对"进程是否还活着"的判断以 `ps aux` 为准，**日志内容不能作为死亡证据**（稀疏文件）

---

## 三、本次已固化的资产

- `scripts/serve_36.sh`：已内化规则 1/5/6/9（TASK_QUEUE=1、bisheng PATH、8007 对齐配置、独立日志由参数传入）
- `scripts/bench_acceptance.py`：支持 `--prompts` 自定义题集、`--tag` 命名、结果 JSON 落盘
- 本文档：**每次占卡前过一遍第二节 checklist**

## 四、经验记录：int8 取整地板与训练精度（2026-11-13，用户质询触发）

### 用户的问题（原话大意）

> 取整这个事情我是这么理解的：训练的时候精度很重要。本来可能每次训练提升
> 一点点，累积之后有明显提升；但如果过程都被取整了，可能完全训练不进去了。

### 复盘结论（两段式拆解，别混为一谈）

**1) 训练侧没有逐步取整——精度守住了，这是设计底线，不要动摇**

- 可训练参数（base LoRA + MTP 11 张量 + MTP-LoRA 注入对）在 fill (4) 步全部
  cast 成 fp32，AdamW 状态 CPU fp32（GLM53_ADAM_CPU=1）——每步梯度更新都在
  fp32 里累积，不存在"每步丢一点"
- int8 是冻结 base 的存储形态；训练走全精度旁路（W_deq + BA），从不写回 int8
- 判据就是 loss 曲线：V2 全量 18.4 → 9.0（step 72 仍在稳定下降）。若逐步取整
  吞掉了更新，loss 早就平了

**2) "取整吞掉累积"发生在部署合并——真实存在，但累积本身不丢**

- build_trained_cp 把**累积 N 步的全量 delta**（不是单步 delta）一次性加进
  int8 再取整 → checkpoint 存的是总和，CP 间累积不丢，delta 随训练单调增长
- 但 delta 必须超过半格点（scale/2）才在 serving 里存在。实测（V2-CP60）：
  - q_b_proj：delta 1.26e-3 > 半格点 → 25,135/33.5M 权重生效 ✅
  - o_proj：delta 2.14e-4 < 半格点 9.2e-4 → **0/1.0 亿权重改变，serving=vendor** ❌
- 这就是"训练了但评测看不见"的形态：不是训练不进去，是**部署表达不出来**
- 增长速度参照（V1：reldiff 1.6e-3@CP30→3.3e-3@CP45 ≈15步翻倍；V2 的
  MTP-LoRA 另有 10× lr）：o_proj 预计 **CP90-CP150** 自然过地板

### 沉淀出的三条规则

1. **精度问题先分层再回答**：训练域（fp32 累积）与部署域（int8 表达）是两个
   独立问题。看到"没提升"先问是哪一层丢的——loss 曲线是训练域的裁判，
   构建 reldiff/合并审计是部署域的裁判
2. **合并审计必须做去量化域**：直接比 int8 原始值无意义（delta 3.0 可能只是
   3 个 int 步）；要用 dequant(q')-dequant(q) ≈ B@A×(α/r) 对账
   （本次审计脚本范式见 V2_FULL26_EVAL_REPORT.md §5）
3. **地板不是 bug 而是进度条**：delta 从 sub-ULP 到过地板是单调过程；除非
   交付时间不允许等待，否则优先"等累积"，其次才是 o_proj bf16 serving /
   对照臂定量地板损失

### 追问（2026-11-13）：逻辑上怎么区分"噪音"还是"精度"？

**判别框架：不看幅度，看响应的方向一致性。** 三个判别器：

1. **训练域**：全精度下开/关 delta 比 held-out MTP-CE。有增益=真信号
   （问题只剩 CE→接受率的转化）；无增益=纯噪音
2. **部署域（剂量-响应）**：delta 按 k×{0.5,1,2,4} 放大构建评测。**信号的标志
   是放大后接受率单调变好；噪音是变差或不动**。bf16o 实验即 k=1 的单点
3. **时间域（最便宜）**：相邻 CP 的 delta 余弦相似度 + 范数增长律。
   信号：cos 高（→1）、范数随步数**线性**增长；随机游走：cos≈0、范数 **√步数**

**实测（V2 全量 CP30 vs CP60，判别器 3）**：

| delta | 范数增长 30→60 步 | cos(30,60) | 判读 |
|---|---|---|---|
| MTP q_b | 2.78×（线性=2.0×，游走=1.41×） | 0.717 | 信号：方向一致、幅度超线性 |
| MTP o_proj | 2.45× | 0.660 | **低成熟度信号**（非噪音；方向仍在收敛） |
| base-LoRA 54 组聚合 | 2.48× | ≈1.0 | 教科书信号：方向稳定，纯幅度增长 |

**结论修正与可翻转性**：bf16o 的负响应（§5b）是 CP60 时点判定——o_proj 的
delta 是"还没长大的信号"（CE 收益 < argmax 翻转成本），不是噪音。随方向收敛
（cos→1）+ 幅度过半格点，符号会翻正。

**沉淀为"成熟度探针"协议**：每个大 CP（90/120/150…）跑一次 bf16o 臂 vs int8 臂，
看 `EAL(bf16o) − EAL(int8)` 符号：负=地板仍在保护（挡噪声），**正=地板开始挡
真收益，此刻才是 bf16 serving 的合理介入点**。翻转判据可观测、可预期。

### cos 指标的三种 regime（2026-11-13 追问补充）

注意 cos 是**累积 delta** 之间的夹角，对"增量转向"有稀释效应。CP90 起应改算
**相邻增量的 cos**（increment_{t→t+1} 之间）：

| 增量 cos | 含义 | 后果 | 策略 |
|---|---|---|---|
| →1（高且稳） | 方向收敛，信号在走直线 | 过地板时间可预测 | 等累积成立，探针做验证 |
| →0 | 方向旋转（warmup 相位切换 / draft 追移动 target / loss 构成漂移） | 累积退化 ~√t，过地板时间不可测 | 只信探针实测，不做外推 |
| →-1 | 方向反转（早期方向作废 / 结构性重排） | 任何时点 delta 只是"当前相位最好猜测" | 早期判定全部作废，只信探针 |

V2 已知机制里旋转/反转都有真实来源（draft 的对齐目标是持续移动的 base+LoRA），
故探针协议在任何 regime 下都是最终裁判；cos 只用来预测"能不能等"。

### 关联证据

- `reports/V2_FULL26_EVAL_REPORT.md` §5（三级审计 + 数值）
- `reports/TEST_REQUIREMENTS_AND_DESIGN.md` 演进表 V2 行（⚠️ 标注）
- V1-CP30 同现象先例："int8 量化地板：早期 LoRA 更新多数层被 round 吞没，
  仅 6/18 层生效"
