# Установка

Плагин состоит из двух частей. Python-пакет `mst` из `src/` — это MCP-сервер и все
расчёты. Папка `dsh/` — пакет плагина DSH: он запускает этот сервер, регистрирует скиллы и
добавляет в системный промпт агента снимок workspace.

## 1. Python-часть

Нужны Python 3.12 или новее и [uv](https://docs.astral.sh/uv/); минимальную версию Python
фиксирует `requires-python` в `pyproject.toml`.

```powershell
cd <путь к репозиторию>
uv sync
.venv\Scripts\python.exe -m pytest -q -m "not slow"
```

`uv sync` создаёт `.venv` и ставит туда пакет `mst` вместе с зависимостями из `uv.lock`.
Проверить, что сервер запускается:

```powershell
uv run macrostress --workspace workspace
```

Сервер говорит по stdio и ждёт клиента, поэтому после запуска ничего не печатает; выход —
Ctrl+C. При запуске он создаёт `workspace/` и пишет туда `workspace.json`.

## 2. Где лежат данные

В `dsh/cordis.patch.yml` нет путей: `uvx` запускает пакет с PyPI, а сервер и плагин находят один
каталог данных по одному правилу: переменная окружения `MACROSTRESS_WORKSPACE`, иначе
`~/.macrostress/workspace`. Для обычной установки ничего настраивать не нужно.

Всё, что относится к одной машине, пишется в патче профиля
`~/.dsh/profiles/<имя>/cordis.patch.yml`: он применяется после бандла и заменяет `config` записи
целиком. Так подключают свой каталог данных или разработческий checkout:

```yaml
- id: mcp-macrostress
  config:
    serverName: macrostress
    transport: stdio
    command: '<путь к репозитории>/.venv/Scripts/python.exe'
    args: ['-u', '-m', 'mst.mcp', '--workspace', '<каталог данных>']
    env:
      PYTHONUTF8: '1'
    failOnStartupError: false
- id: dsh-macrostress
  config:
    workspace: '<каталог данных>'
```

## 3. Ключ FRED

Глобальные ряды (`gl.brent`, `gl.fed_rate`, `gl.ust_10y`, `gl.vix`) загружаются из FRED, и для
них нужен бесплатный ключ `FRED_API_KEY`. Остальные источники работают без ключа. DSH убирает из
окружения дочерних процессов имена, похожие на секреты, поэтому системная переменная до сервера
не дойдёт: ключ задаётся в поле `env` записи `mcp-macrostress` патча профиля, как в примере выше.
В репозиторий ключ не попадает. Задачи загрузки наследуют окружение сервера.

## 4. Подключение к профилю DSH

Профиль DSH — каталог `~/.dsh/profiles/<имя>/`. Плагин ставится из репозитория одной командой:

```powershell
dsh plugin add "github:LeoLourson/MacroStress-DSH#path:/dsh"
```

Менеджеру плагинов нужен `pnpm`. Без него, или чтобы менять плагин на месте, он подключается
вручную из клона репозитория.

1. Ссылка на папку `dsh/` из каталога плагинов DSH (в Windows — junction):

   ```powershell
   New-Item -ItemType Junction -Path "$HOME\.dsh\plugins\dsh-macrostress" `
       -Target "<путь к репозиторию>\dsh"
   ```

2. Зависимость и бандл в `~/.dsh/profiles/web/package.json`:

   ```json
   {
     "dependencies": {
       "dsh-macrostress": "link:../../plugins/dsh-macrostress"
     },
     "dsh": {
       "profile": {
         "bundles": ["@deepseek-ai/dsh-base", "...", "dsh-macrostress"]
       }
     }
   }
   ```

   Имя в `bundles` — поле `name` из `dsh/package.json`. Бандл подключает
   `dsh/cordis.patch.yml` (поле `dsh.bundle.patch`), а патч вставляет в дерево профиля MCP-клиент
   `@deepseek-ai/dsh-mcp-client` с сервером `macrostress` и сам плагин.

3. Ссылка `~/.dsh/profiles/web/node_modules/dsh-macrostress` →
   `~/.dsh/plugins/dsh-macrostress`. Если после перезапуска её нет, создайте её так же, как
   в п. 1.

4. Перезапустить DSH с этим профилем.

Общий патч машины `~/.dsh/cordis.patch.yml` менять не нужно: плагин, подключённый ссылкой,
описывается в манифесте и патче профиля.

## 5. Проверка

В новом сеансе DSH:

- в системном промпте есть раздел MacroStress из пяти строк с путём workspace и версиями
  моделей (до первой оценки — прочерки); четвёртая строка — число портфелей, сценариев,
  расчётов и задач в работе, пятая — с чего начать;
- в списке скиллов три плагинных: `scenario-from-corpus`, `run-report`, `drivers`;
- в карточке плагина (Settings → Plugins) видны название и описание на языке интерфейса;
- вызов `jobs(action="state")` возвращает словарь с ключами `workspace`, `macro`,
  `satellites`, `corpus_index`, `portfolios`, `scenarios`, `runs`, `jobs`.

Проверка без запуска DSH:

```powershell
node dsh/test/smoke.mjs
```

Он подставляет фейковый контекст, регистрирует скиллы, строит пять строк раздела промпта из
временного `workspace.json`, проверяет однострочный запасной вариант без файла, перечитывание
файла и снятие скиллов при выключении; карточка проверяется по обеим локалям.

Если инструментов нет, сервер, скорее всего, не стартовал. Патч ставит
`failOnStartupError: false`, поэтому DSH при этом не падает. Запустите команду из `command`
и `args` патча вручную в терминале: ошибку импорта или неверный путь будет видно сразу.

## Типовые ошибки

| Симптом | Причина |
|---|---|
| `resource 'series' is busy with job ...` | Задача на этом ресурсе ещё идёт; дождитесь её или отмените `jobs(action="cancel", id=...)` |
| `FRED needs an API key ...` в логе загрузки | Нет `FRED_API_KEY` в окружении сервера |
| Ошибка сертификата на rosstat.gov.ru | Росстат подписан корнем Минцифры; корни лежат в `src/mst/data/mintsifry-roots.pem` и подключаются только для этого хоста. Если Минцифры сменило сертификат, файл надо обновить |
| Оценка сателлитов идёт часами | Команда запущена не через `python -m mst.drivers`, и BLAS работает в много потоков; см. [../reference/tools.md](../reference/tools.md) |
