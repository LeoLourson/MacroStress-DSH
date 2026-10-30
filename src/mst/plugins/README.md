# mst.plugins

Модели как расширения — трёх видов. Каждый подпакет — плагин: `plugin.py` с объявлением
`plugin = Plugin(...)`, модель, тесты и `README.md`. Платформа при старте импортирует все
пакеты `mst.plugins.<id>` (плюс entry points группы `mst.plugins` и каталоги
`MST_PLUGINS_PATH`), проверяет объявления вместе и строит реестр по видам: позиционные,
макро- и сателлитные. Все объявления проверяются сразу, но величину могут заявить несколько
сателлитных моделей — это не ошибка старта; ошибкой становится неоднозначное назначение, и
его разрешает явное `--model ВЕЛИЧИНА=МОДЕЛЬ`. Ядро (`mst.engine`, `mst.models`, `mst.runs`,
`mst.metrics`) получает модели только из реестра и не называет ни одной конкретной.

| Плагин | Вид | Модель | Что обслуживает |
|---|---|---|---|
| [`equity`](equity/plugin.py) | позиционная | `equity.paths.v1` | `common_stock`, драйвер `share_price` |
| [`bond`](bond/plugin.py) | позиционная | `bond.paths.v1` | `bond`, драйвер `bond_spread` |
| [`credit`](credit/plugin.py) | позиционная | `credit.paths.v1` | `loan`, `deposit`, драйвер `overdue_share` |
| [`housing`](housing/plugin.py) | позиционная | `housing.paths.v1` | `real_estate`, драйверы `housing_price_*` |
| [`bvar`](bvar/plugin.py) | макро | `bvar.sv` | уровень 0: 23 переменные, 5 именованных шоков |
| [`hier`](hier/plugin.py) | сателлитная | `hier.gibbs` | все величины реестра драйверов по умолчанию |

```powershell
python -m mst.plugin new <id> [--kind position|satellite|macro] [--quantity Q] [--label TEXT] [--dest DIR]
python -m mst.plugin check [<plugin_id> | --path DIR]
python -m mst.plugin demo-run <plugin_id> [--path DIR]
```

Новая модель любого вида — пакет здесь или рядом (`--dest`, entry point, `MST_PLUGINS_PATH`),
без правки ядра. Устройство плагина, контракты моделей и проверки — в
[../../../docs/plugins.md](../../../docs/plugins.md); контракты расчёта — в
[../../../docs/platform.md](../../../docs/platform.md).

Тесты реестра и конформанса — [../../../tests/plugins](../../../tests/plugins/);
тесты BVAR — [../../../tests/bvar](../../../tests/bvar/), иерархического сателлита —
[../../../tests/hier](../../../tests/hier/).
