# GUI_360 数据处理说明（处理前 / 处理中 / 处理后）

本目录脚本用于把原始 GUI_360 流程数据，整理成可直接训练图像编辑模型的数据集（`data_sample_3000`）。

处理链路：

1. 原始 GUI_360 数据（jsonl + screenshot）
2. `action_extract.py` 抽取动作并写入 action txt
3. `pair_data_sample.py` 过滤 + 采样 + 重组为 `prev/next/prompt` 训练格式
4. 输出 `data_sample_3000`（目录 + CSV）

---

## 1) 处理前：原始 GUI_360 数据格式

### 数据位置

- 原始数据根目录：`${DATA_ROOT:-./data}`

### 目录结构（原始）

```text
data/
├── train/
│   └── data/
│       ├── word/
│       │   ├── bing_search/success/*.jsonl
│       │   ├── m365/success/*.jsonl
│       │   ├── qabench/success/*.jsonl
│       │   └── wikihow/success/*.jsonl
│       ├── excel/
│       │   ├── bing_search/success/*.jsonl
│       │   ├── m365/success/*.jsonl
│       │   └── qabench/success/*.jsonl
│       └── ppt/
│           ├── bing_search/success/*.jsonl
│           ├── m365/success/*.jsonl
│           └── qabench/success/*.jsonl
└── test/
    └── data/...
```

### 原始样本（jsonl）关键字段

`action_extract.py` 实际读取以下字段：

- `step.action.control_test`
- `step.action.function`
- `step.action.args`
- `step.screenshot_clean`

其中 `step.screenshot_clean` 用来定位对应截图名（例如 `action_12.png`）。

---

## 2) 处理中：action_extract.py 输出格式

### 功能

`action_extract.py` 会遍历各个 `success/*.jsonl` 文件，把每一步的动作抽取为单独 txt 文件。

### 路径映射规则

脚本把路径中的 `data/{split}/data/...` 映射到 `data/{split}/image/...`：

- 输入：`data/train/data/word/bing_search/success/word_1_1.jsonl`
- 输出：`data/train/image/word/bing_search/success/word_1_1/action_1.txt`（示例）

即：

- 数据域从 `data` 切到 `image`
- 每个 session（如 `word_1_1`）下，按截图 step 生成同名 action txt

### action txt 文件内容

每个 txt 内是一行 JSON 字符串，结构如下：

```json
{
  "control_name": "...",
  "function": "...",
  "args": {
    "x": 123,
    "y": 456
  }
}
```

说明：

- `control_name`：控件/目标对象
- `function`：动作函数（如 click / type / drag 等）
- `args`：动作参数（坐标、文本等）

---

## 3) 处理后：pair_data_sample.py 的采样与重组

`pair_data_sample.py` 的目标是从 `data/{split}/image/...` 中构造高质量训练 pair。

### 3.1 pair 构造方式

在每个 session 目录下：

- 先按自然顺序读取 `action_*.txt`
- 用相邻 step 组成一对：
  - `prev_img = action_i.png`
  - `next_img = action_{i+1}.png`
  - `action = action_i.txt`

### 3.2 质量过滤规则（脚本中已实现）

样本会被过滤掉，若满足任一条件：

1. `action.txt` 为空或 JSON 解析失败
2. `control_name/function/args` 全空（无效动作）
3. `prev.png` 与 `next.png` 像素完全一致
4. 坐标字段为空（如 `x/y`、`start_x/start_y` 等）
5. 坐标越界（必须在 `0<=x<=1040`, `0<=y<=736`）
6. 黑边或分辨率不符合约束：
   - 黑边需为 `(top,bottom,left,right)=(7,0,7,7)`
   - 图像尺寸需为 `(736,1040)`

### 3.3 session 级采样

- 不是逐条随机，而是按 session 抽样
- 一旦抽中某个 session，该 session 下所有合法 pair 一起保留
- 脚本当前 `target=200`（每个任务域的目标规模）

---

## 4) 最终输出：data_sample_3000 格式

### 目录重组格式

脚本会复制采样结果并重排为标准训练目录：

```text
data_sample_10000/{split}/{app}/{source}/paired/
└── {session}/
    ├── pair_01/
    │   ├── prev.png
    │   ├── next.png
    │   └── action.txt
    ├── pair_02/
    │   ├── prev.png
    │   ├── next.png
    │   └── action.txt
    └── ...
```

其中：

- `prev.png`：操作前界面
- `next.png`：操作后界面（训练目标）
- `action.txt`：该步动作 JSON

> 注：脚本变量里中间目录名是 `data_sample_10000`，但最终汇总 CSV 输出到 `data_sample_3000` 路径（见下）。

### prompt 构造格式

每个 pair 会生成训练 prompt：

```text
Based on the previous UI screenshot and the described action, generate the next realistic frame showing the updated screen.
action：{action_json}
```

其中 `{action_json}` 是 `action.txt` 对应的动作对象。

### CSV 汇总文件（训练直接使用）

脚本最终生成：

- `${DATA_ROOT:-./data}/{split}/guidata_edit_{split}.csv`

CSV 列结构：

- `image`：`next.png` 路径
- `prompt`：拼好的动作描述 prompt
- `edit_image`：`prev.png` 路径
- `session`：会话 id（如 `word_1_13`）

这份 CSV 即可作为后续训练入口（例如图像编辑 LoRA 训练）的 metadata 文件。

---

## 5) 推荐执行顺序

1. 在原始 GUI_360 上运行 `action_extract.py`，产出 action txt
2. 运行 `pair_data_sample.py`，完成过滤、采样、目录重组与 CSV 导出
3. 将导出的 `guidata_edit_train.csv` / `guidata_edit_test.csv` 接到训练脚本

如果要改采样规模或规则，优先调整 `pair_data_sample.py` 中：

- `sample_rows_by_session(..., target=...)`
- 过滤条件（分辨率、黑边、坐标合法性等）
