# mst.plugins

Типы активов как расширения. Каждый подпакет — плагин: `plugin.py` с объявлением
`plugin = Plugin(...)`, `model.py` с моделью `CalculationModel`, демо-кейс, `tests/` и
`README.md`. Платформа при старте импортирует все пакеты `mst.plugins.<id>`, проверяет
объявления вместе и строит реестр; ядро (`mst.engine`) получает модели только из него.

| Плагин | Тип актива | Модель | Драйвер |
|---|---|---|---|
| [`equity`](equity/plugin.py) | `common_stock` | `equity.paths.v1` | `share_price` |
| [`bond`](bond/plugin.py) | `bond` | `bond.paths.v1` | `bond_spread` |
| [`credit`](credit/plugin.py) | `loan`, `deposit` | `credit.paths.v1` | `overdue_share` |
| [`housing`](housing/plugin.py) | `real_estate` | `housing.paths.v1` | `housing_price_primary` / `housing_price_secondary` |

```powershell
python -m mst.plugin new <plugin_id> [--type-id ID] [--label TEXT] [--dest DIR]
python -m mst.plugin check [<plugin_id> | --path DIR]
python -m mst.plugin demo-run <plugin_id> [--path DIR]
```

Новый тип актива — пакет здесь, без правки ядра. Устройство плагина, контракт модели,
демо-кейс и проверки — в [../../../docs/plugins.md](../../../docs/plugins.md); контракты
расчёта — в [../../../docs/platform.md](../../../docs/platform.md).

Тесты реестра и конформанса — [../../../tests/plugins](../../../tests/plugins/).
