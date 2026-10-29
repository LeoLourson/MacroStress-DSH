# Установка

Плагин состоит из двух частей. Python-пакет `mst` из `src/` — это MCP-сервер и все
расчёты. Папка `dsh/` — пакет плагина DSH: он запускает этот сервер, регистрирует скиллы и
добавляет в системный промпт агента снимок workspace.

Порядок подключения к DSH ниже восстановлен по рабочей установке на машине автора
(`~/.dsh`). Официальной инструкции DSH по локальным плагинам в репозитории нет, поэтому
шаги, которые на этой машине не проверялись, помечены.

## 1. Python-часть

Нужны Python 3.12 (ровно 3.12, см. `requires-python` в `pyproject.toml`) и
[uv](https://docs.astral.sh/uv/).

```powershell
cd <путь к репозиторию>
uv sync
.venv\Scripts\python.exe -m pytest -q
```

`uv sync` создаёт `.venv` и ставит туда пакет `mst` вместе с зависимостями из `uv.lock`.
Проверить, что сервер запускается:

```powershell
.venv\Scripts\python.exe -m mst.mcp --workspace workspace
```

Сервер говорит по stdio и ждёт клиента, поэтому после запуска ничего не печатает; выход —
Ctrl+C. При запуске он создаёт `workspace/` и пишет туда `workspace.json`.

## 2. Пути в `dsh/cordis.patch.yml`

В патче записаны абсолютные пути машины автора:

```yaml
command: 'C:\Users\LeoLeon\Documents\MacroStress\macrostress-dsh\.venv\Scripts\python.exe'
args: [..., '--workspace', 'C:\Users\LeoLeon\Documents\MacroStress\macrostress-dsh\workspace']
```

На другой машине оба пути надо заменить на свои: интерпретатор из `.venv` этого репозитория
и каталог данных. Workspace можно держать где угодно, но код плагина (`dsh/index.js`) читает
снимок из `<репозиторий>/workspace`, если ему не передан параметр `workspace`. Если каталог
данных лежит в другом месте, передайте плагину тот же путь в конфиге записи
`dsh-macrostress`:

```yaml
- id: dsh-macrostress
  name: 'dsh-macrostress'
  config:
    workspace: 'D:\data\macrostress'
```

Иначе сервер будет работать с одним каталогом, а промпт описывать другой.

## 3. Ключ FRED

Глобальные ряды (`gl.brent`, `gl.fed_rate`, `gl.ust_10y`, `gl.vix`) загружаются из FRED, и
для них нужен бесплатный ключ в переменной окружения `FRED_API_KEY`. Остальные источники
работают без ключа. Задачи загрузки наследуют окружение сервера. Ключ можно задать в
системных переменных Windows или в `env` записи `mcp-macrostress` в патче; передаёт ли
клиент DSH серверу окружение самого DSH, не проверено.

## 4. Подключение к профилю DSH

Профиль DSH — каталог `~/.dsh/profiles/<имя>/`. На машине автора плагин подключён к
профилю `web` так:

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
   `~/.dsh/plugins/dsh-macrostress`. На машине автора она есть, но lock-файла пакетного
   менеджера в профиле нет, поэтому неизвестно, создал её DSH сам при запуске или её
   поставили вручную. Если после перезапуска DSH ссылки нет, создайте её так же, как в
   п. 1.

4. Перезапустить DSH с этим профилем (как именно выбирается профиль при запуске, здесь не
   проверялось).

Общий патч машины `~/.dsh/cordis.patch.yml` трогать не нужно: в его комментарии сказано,
что плагины, подключённые ссылкой, описываются в манифесте и патче самого профиля.

## 5. Проверка

В новом сеансе DSH:

- в системном промпте есть раздел MacroStress из пяти строк с путём workspace и версиями
  моделей (до первой оценки — прочерки);
- вызов `jobs(action="state")` возвращает словарь с ключами `workspace`, `macro`,
  `satellites`, `corpus_index`, `portfolios`, `scenarios`, `runs`, `jobs`.

Если инструментов нет, сервер, скорее всего, не стартовал. Патч ставит
`failOnStartupError: false`, поэтому DSH при этом не падает. Запустите команду из `command`
и `args` патча вручную в терминале: ошибку импорта или неверный путь будет видно сразу.

## Типовые ошибки

| Симптом | Причина |
|---|---|
| `resource 'series' is busy with job ...` | Задача на этом ресурсе ещё идёт; дождитесь её или отмените `jobs(action="cancel", id=...)` |
| `FRED needs an API key ...` в логе загрузки | Нет `FRED_API_KEY` в окружении сервера |
| Ошибка сертификата на rosstat.gov.ru | Росстат подписан корнем Минцифры; корни лежат в `src/mst/data/mintsifry-roots.pem` и подключаются только для этого хоста. Если Минцифры сменило сертификат, файл надо обновить |
| Оценка сателлитов идёт часами | Команда запущена не через `python -m mst.drivers`, и BLAS работает в много потоков; см. [tools.md](tools.md) |
