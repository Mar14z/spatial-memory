# Spatial Memory Engine

面向具身机器人的空间记忆存储引擎（C++20）。把检测观测按"对象"持久化，支持空间/时间/语义查询。

## 状态

- **v0 完成**：纯内存引擎（对象池 + 网格哈希 + 数据关联），10 单测全绿
- **v1 完成**：SQLite WAL 持久化（对象表+观测表）、异步批量写、启动恢复；12 单测全绿
- **v2 完成**：观测压缩（驻留段合并）+ 时间查询 API；16 单测全绿
- v3 计划：ROS2 节点 + 摄像头接入

详细设计（架构、数据流、设计决策、性能基线）见 [docs/DESIGN.md](docs/DESIGN.md)。

## 目标平台

- Orange Pi AIpro（昇腾 310B 4×A55 @1.6GHz / 8GB / Ubuntu 22.04）
- 开发机：WSL2 Ubuntu 22.04（x86_64）

## 结构

```
include/sm/
  types.hpp         # Observation / QueryResult / Entity / Config
  entity_pool.hpp   # 对象池（vector + free list，O(1) 分配）
  grid_index.hpp    # 网格哈希空间索引（1m 网格，u64 打包 key）
  engine.hpp        # 门面：add_observation / query_*
src/engine.cpp      # 关联 + 查询实现
tests/test_sm.cpp   # 10 个 gtest 用例
bench/bench_engine.cpp  # 4 组基准
```

## 构建与运行

```bash
sudo apt install -y cmake g++ libgtest-dev
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j3
./test_sm      # 单元测试
./bench_sm     # 基准
```

## v0 基准（Orange Pi AIpro 实测）

| 测试 | 结果 |
|---|---|
| 插入 10 万实体 | 328 k/s（引擎层） |
| 内存 | 16.6 MB / 10 万实体（≈166 B/实体） |
| 半径 1.5m 查询 | P50 37.1 µs / **P99 47.3 µs** |
| 关联去重 | 100 物体 × 10 帧 → 实体数 100，零膨胀 |
| 物体移动 | 跨格移动后旧位置无幽灵、新位置可查 |
| 持久化写入（v1/v2） | **18,717~24,155 /秒**（含批量 flush） |
| 启动恢复 | **438~452 ms**（10 万实体） |
| 观测压缩比（v2，静止场景） | **294:1**（1 万观测 → 34 段） |
| 时间范围查询（v2） | **P99 162 µs** |
| kill -9 崩溃恢复 | **5/5 轮全量恢复**（6000 条/轮，integrity ok） |
| 特征向量检索（v3a） | 1 万 × 512 维 top5：**P99 12.3 ms** |
| CLIP 视觉塔（v3b） | **18.9 ms/次**（224×224，实机） |
| CLIP 文本塔 + BPE tokenizer（v3c） | **9 ms/次**；tokenizer 与 HF 官方逐位一致 |

对照：裸网格微基准 P99 18.3 µs；目标 P99 < 500 µs。

## 设计要点

- 对象级存储：不存点云，存"物体"（位置/尺寸/类别/时间戳），数据量小 3 个数量级
- 双表模型（v1）：对象表=当前状态，观测表=历史流水
- 数据关联：网格邻域查候选 → 类别 + 时间窗 + 距离过滤 → 命中更新 / 未命中新建
- 存储层零 ROS 依赖，ROS2 节点（v3）只做消息转换
