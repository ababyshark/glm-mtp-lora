# NFS serving 陷阱与正确姿势(V11-A3 实测 2026-11-26)

## 症状清单(本次实测)

| 症状 | 实测数据 | 根因 |
|---|---|---|
| vLLM 加载分片极慢 | **354s/分片**(ETA 11.8h),本地加载同条件下 ~2.5s/分片 | serve 目录在 NFS:模型加载 mmap 按 4K 页缺页读,vLLM 后台 prefetch(顺序 16MB 块)与加载器竞速失败,有效读速 ~14MB/s |
| 导出 farm 写入停摆 | cp500 导出卡在 50/130 项 **1 小时+**(进程 D 状态) | 导出写 ~250G 补丁分片到 NFS,重并发写遇 NFS 抖动即 D-state 挂起 |
| safetensors 打开报 ENODEV | cp500/cp100 的 model-00121 mmap 失败 | 刚写入的 NFS 大文件 mmap 触发服务端暂存问题(间歇性) |
| bench 日志写入挂起 | cp400 bench 卡在日志写 NFS | 同导出写入停摆 |
| 历史旁证 | B1 训练 15:50 被神秘 SIGKILL、vllm server ~07:20 消失 | 平台 computin 调度器(7.221.52.112,每 10 分钟登录)在巡检点杀未注册 NPU 进程——与 NFS 无关但同期发生,注意区分 |

## 根因

1. **模型加载 = 读密集**:safetensors mmap 按页缺页读,NFS 随机小读极慢(有效 14MB/s vs 本地 NVMe ~2GB/s)。vLLM 的后台 prefetch 无法掩盖加载器的随机读。
2. **导出/日志 = 写密集**:~250G farm 写 + 持续日志写,NFS 重写抖动 → D-state 挂起(进程不死但永等)。
3. **共享 NFS 的多用户争用**:10.246.63.101:/data1 为共享存储,他人重载时段读写骤降。

## 正确姿势(已验证)

### serving(读密集)
- serve 目录**必须在 .40 本地盘**(/data2/v11_work/,NVMe)
- 构建方法 = 混合目录(`build_hybrid_a3.sh <step>`,~2 分钟):
  - 全部 vendor 文件 → 符号链接指向本地 /tmp/v103_base_bf16(NVMe)
  - 该 CP 的 layer-46 shard(~13.8G)+ config/index → 从 NFS serve 目录拷贝为本地真实文件
  - 加载 ~4 分钟(全本地读)✓ cp300/cp400 实测
- A3 特例:layer0 冻结 = factory,主模型 LoRA δ≈0.01(可忽略)→ **无需拷贝 118 个主模型补丁分片**,最小混合目录即正确服务状态(eps 0.85 bench 实测有效)

### 导出(写密集)
- 输出目录:**/data2 本地**(写 NVMe ~1GB/s)或 /mnt/share_space(仅当无本地空间且 NFS 空闲时)
- farm 构建后 NFS 大写若停摆(D 状态 >10 分钟)→ 杀掉重跑,勿等待

### 日志(持续写)
- 训练/serving 期间:日志/结果**全部走本地 /data2**(`V10_BENCH_OUT_DIR` env、LOGD 变量)
- 跑完后归档到 /mnt/share_space(供跨机查阅)

### NFS 健康检查(操作前必做)
```bash
timeout 8 ls /mnt/share_space/l00840580/ >/dev/null && echo OK || echo NFS 异常
# 重写测试(50MB):失败 = NFS 抖动期,勿启动大 IO 任务
```

## 陷阱登记

1. **写穿透符号链接**:finalize 直接 open(farm 符号链接)写 config/index → **写穿到 vendor 原件**(污染共享 vendor,已恢复)。教训:写前必须 `os.path.islink` 检查并 break 链接。
2. **/tmp ≠ overlay**:/tmp 是 bind mount(→ /data2/v11_work),写 /tmp 实际写 /data2;容器 **root overlay 才是 100% 满的盘**(Ascend plog 写失败杀进程的元凶)。清理顺序:先清 overlay(root/ascend/log 旧日志),/tmp 的清理不影响 overlay。
3. **root overlay 1.09T 为已删除但被持有的文件**:du 实际活文件仅 8.2G,df 1.1T——持有者为宿主侧进程,容器内无法释放,需容器重启(已执行,1.09T 已释放)。
