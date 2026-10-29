# MacroStress

[English](README.md) | [Русский](README.ru.md) | [中文](README.zh.md)

[![CI](https://github.com/LeoLourson/MacroStress-DSH/actions/workflows/ci.yml/badge.svg)](https://github.com/LeoLourson/MacroStress-DSH/actions/workflows/ci.yml)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

MacroStress 回答一个问题：在给定的宏观经济情景下，投资组合会损失多少？分析师或智能体给出若干
宏观变量的路径（例如政策利率在未来一年保持 21%）以及造成这些路径的冲击。在历史上估计的 BVAR
补全其余变量的一致路径，卫星模型把路径转换成驱动因子（股价、债券利差、行业不良率、房价），再沿
同一批抽样对每个持仓重新定价。输出是损失分布：$P_5$、$P_{50}$、$P_{95}$、ES95 以及情景严苛度
$\delta$。方法见 [docs/scenario-formalization.md](docs/scenario-formalization.md)（俄文）。

服务器使用 MCP，因此任何支持的客户端都可以驱动它：Claude Desktop、Cursor、Claude Code。`dsh/`
中的 DeepSeek Harness 插件增加三个技能，并把工作区快照放进智能体的提示词。所有状态保存在一个
`workspace/` 目录；没有网页、数据库和调度器。

## 功能

| 部分 | 作用 |
|---|---|
| 数据序列 | 从 MOEX、俄罗斯央行、Rosstat 和 FRED 下载月度序列，每个数值带发布日期 |
| 宏观模型 | 带随机波动率的 BVAR：符号约束识别、条件预测、情景严苛度、历史事件检验 |
| 驱动因子 | 对象目录、分层卫星模型、样本外检验、基于同一批抽样的驱动因子路径 |
| 情景 | $(N, C, D, O)$ 文件，校验并生成路径：条件预测和驱动因子使用同一批抽样 |
| 持仓模型 | 股票、债券、信贷或存款、房地产 |
| 运行与报告 | 沿情景路径计算持仓，给出 $P_5$、$P_{50}$、$P_{95}$、ES95 和 $\delta$ |
| 扩展 | 新资产类型是插件：`python -m mst.plugin new`、契约检查、演示计算 |
| DSH 插件 | 从语料库构建情景、报告、驱动因子；由 `workspace.json` 生成提示词段落 |

MCP 工具：`jobs`、`series`、`corpus`、`scenario`、`run`、`portfolio`、`model`。参考见
[docs/tools.md](docs/tools.md)（俄文）。

## 安装

需要 Python 3.12 或更新版本，以及 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/LeoLourson/MacroStress-DSH
cd MacroStress-DSH
uv sync
uv run macrostress --workspace C:/data/macrostress
```

服务器通过 stdio 通信，只输出错误信息，等待客户端连接。Claude Desktop、Cursor 和其他 MCP 客户端的
配置见 [docs/mcp.md](docs/mcp.md)，DeepSeek Harness 插件见 [docs/install.md](docs/install.md)。

## 第一次会话

客户端连接后会看到七个工具。耗时操作以任务形式运行：工具立即返回 `job`，进度通过 `jobs` 查看。

1. `jobs(action="state")` — 工作区状态：模型版本、语料库索引、任务。
2. `series(action="update")` — 下载全部宏观序列。FRED 需要环境变量 `FRED_API_KEY` 中的密钥。
3. `jobs(action="status", id=...)` 和 `jobs(action="log", id=...)` — 等待任务结束。
4. `model(kind="macro", action="estimate", as_of="2026-10-29")` — 用该日期已知的数据估计宏观
   模型，得到带哈希的版本。
5. `series(action="update", quantity="share_price")` — 找到该数量的对象并下载序列；对
   `bond_spread`、`overdue_share`、`housing_price_primary`、`housing_price_secondary` 同理。
6. `model(kind="satellites", action="estimate", version=<宏观版本哈希>)` — 估计卫星模型。
7. `model(kind="satellites", action="describe", version=<哈希>)` — 哪些驱动因子通过样本外检验，
   其余为何被拒绝。
8. `corpus(action="search", query=..., as_of=...)` — 该日期已知的语料片段；
   `corpus(action="index")` 以任务重建索引。
9. `scenario(action="save", id=..., spec={...})` — 保存情景；
   `scenario(action="validate", id=...)` — 列出阻止计算的问题。
10. `portfolio(action="save", id=..., spec={...})` — 保存组合；
    `portfolio(action="check", id=..., scenario=...)` — 对照情景检查组合。
11. `run(action="start", scenario=..., portfolio=..., draws=..., seed=...)` — 启动运行；
    `run(action="result", id=...)` — 读取 $P_5$/$P_{50}$/$P_{95}$、$\delta$ 和覆盖率。

## 文档

文档目前为俄文，[docs/mcp.md](docs/mcp.md) 为英文。

- [docs/mcp.md](docs/mcp.md) — 任意 MCP 客户端、第一次会话、常见错误；
- [docs/install.md](docs/install.md) — Python 包和 DeepSeek Harness 插件；
- [docs/architecture.md](docs/architecture.md) — 包、工作区、版本、任务；
- [docs/tools.md](docs/tools.md) — MCP 工具和 `python -m mst.*` 命令；
- [docs/plugins.md](docs/plugins.md) — 用插件增加资产类型；
- [docs/platform.md](docs/platform.md) — 契约：持仓、路径、模型、注册表；
- [docs/scenario-formalization.md](docs/scenario-formalization.md) — 方法与代码实现的部分。

每个代码目录都有自己的 README，列出包内文件。

## 开发

```bash
uv sync
uv run pytest -q -m "not slow"
uv run ruff check .
```

带 `slow` 标记的测试要运行 BVAR 估计，耗时数分钟；夜间工作流运行全部测试。CI 在 Linux 和
Windows 上用 Python 3.12 和 3.13 运行快速测试。

## 状态

0.1.0，beta。1.0 之前接口可能变化。数据源是俄罗斯市场（MOEX、俄罗斯央行、Rosstat），全球序列来自
FRED。

## 参与和许可证

参与规则和发布流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，漏洞报告见 [SECURITY.md](SECURITY.md)。
代码使用 MIT 许可证，见 [LICENSE](LICENSE)。MOEX、俄罗斯央行、Rosstat、FRED 和 BIS 的序列不在
本仓库中：用户按各来源的条件自行加载，许可证不覆盖这些数据。引用信息见
[CITATION.cff](CITATION.cff)。
