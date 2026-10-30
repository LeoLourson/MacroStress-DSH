# MacroStress через MCP

MacroStress — stdio MCP-сервер. Его умеет подключать любой клиент, говорящий на Model Context
Protocol: Claude Desktop, Claude Code, Cursor или плагин DeepSeek Harness из `dsh/`. Инструменты
везде одни и те же; плагин DSH добавляет поверх скиллы и раздел промпта. Общая схема системы —
[../concept/system-scheme.md](../concept/system-scheme.md).

Серверу принадлежит один каталог на диске — workspace. Ряды, версии моделей, индекс корпуса,
сценарии, портфели и прогоны лежат там обычными файлами, поэтому прогон воспроизводится из
workspace и источников.

## Требования

- Python 3.12 или новее.
- [uv](https://docs.astral.sh/uv/) для установки и запуска пакета, либо обычный `pip install`.
- Бесплатный [ключ к API FRED](https://fred.stlouisfed.org/docs/api/api_key.html), если нужны
  глобальные ряды (`gl.brent`, `gl.fed_rate`, `gl.ust_10y`, `gl.vix`). Остальные источники
  работают без ключа.

## Из репозитория

```powershell
git clone https://github.com/LeoLourson/MacroStress-DSH
cd MacroStress-DSH
uv sync
```

Дальше сервер — одна команда:

```powershell
uv run macrostress --workspace C:/data/macrostress
```

Сервер говорит по stdio и до подключения клиента ничего не печатает. Workspace создаётся при
первом запуске вместе с `workspace.json`.

## Настройка клиента

Клиент запускает команду сам. Передайте абсолютный путь к workspace, а ключ FRED положите в
`env`, чтобы сервер передавал его задачам загрузки.

### Claude Desktop

Отредактируйте `claude_desktop_config.json`:

- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "macrostress": {
      "command": "uv",
      "args": [
        "run",
        "--project", "C:/src/MacroStress-DSH",
        "macrostress",
        "--workspace", "C:/data/macrostress"
      ],
      "env": {
        "FRED_API_KEY": "your-key"
      }
    }
  }
}
```

### Cursor

Добавьте тот же блок в `mcpServers` в `~/.cursor/mcp.json` или в `.cursor/mcp.json` внутри
проекта.

### Любой клиент, без выкачивания репозитория

`uvx` запускает пакет из PyPI:

```json
{
  "mcpServers": {
    "macrostress": {
      "command": "uvx",
      "args": ["macrostress", "--workspace", "C:/data/macrostress"]
    }
  }
}
```

То же работает прямо из репозитория — без клонирования и публикации:

```json
{
  "mcpServers": {
    "macrostress": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/LeoLourson/MacroStress-DSH",
        "macrostress",
        "--workspace", "C:/data/macrostress"
      ]
    }
  }
}
```

В путях JSON на Windows используйте прямые слеши: обратный слеш начинает
escape-последовательность.

### DeepSeek Harness

Плагин в `dsh/` запускает тот же сервер, регистрирует три скилла и добавляет состояние workspace
в промпт агента. Шаги подключения — в [install.md](install.md).

## Первый сеанс

Сеанс повторяет схему: заготовить историю по рядам, оценить модели, записать сценарий, посчитать.

1. `series(action="update")` — ряды истории.
2. `model(kind="macro", action="estimate", as_of=...)` — макромодель, уровень 0.
3. `series(action="update", quantity=...)` и `model(kind="satellites", action="estimate", version=...)` — сателлиты, уровни выше.
4. `scenario(action="save")` и `scenario(action="validate")` — сценарий $(N, D, C, O)$.
5. `run(action="start")` и `run(action="result")` — расчёт: условный прогноз целевых показателей
   и отчёт портфеля.

Пошаговый сеанс с полными вызовами — в [first-session.md](first-session.md). Полный справочник —
[../reference/tools.md](../reference/tools.md), контракты чисел —
[../reference/platform.md](../reference/platform.md), метод —
[../concept/scenario-formalization.md](../concept/scenario-formalization.md).

## Что где лежит

- `workspace/series/` — по одному CSV на ряд; каждое значение — с датой публикации.
- `workspace/models/` — версии моделей, названные хешем.
- `workspace/corpus/` — тексты в Markdown, карточки эпизодов, индекс.
- `workspace/scenarios/`, `workspace/portfolios/`, `workspace/runs/` — файлы и результаты прогона.
- `workspace/workspace.json` — снимок, который читает раздел промпта.

## Типовые ошибки

- **В клиенте нет инструментов.** Сервер не запустился. Запустите команду из конфигурации клиента
  вручную в терминале: ошибку импорта или неверный путь будет видно сразу.
- **`FRED needs an API key ...`** в логе задачи. Добавьте `FRED_API_KEY` в блок `env` или в
  системное окружение клиента.
- **Ошибка сертификата на `rosstat.gov.ru`.** Росстат подписан корнем Минцифры; бандл в
  `src/mst/data/mintsifry-roots.pem` подключается только для этого хоста. Если Минцифры сменило
  сертификат, файл надо обновить.
- **Задача зависла или ресурс занят.** `jobs(action="cancel", id=...)` освобождает ресурс;
  второе обновление не начнётся, пока не кончилось первое.
