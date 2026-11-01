# Документация

Документация отвечает на три вопроса; у каждого своя папка.

- **[concept/](concept/)** — что система делает и почему так устроена: общая схема, метод,
  устройство кода.
- **[howto/](howto/)** — как её поставить и пользоваться: установка, подключение, первый сеанс,
  плагин.
- **[reference/](reference/)** — что чему соответствует: контракты платформы и справочник вызовов.

| Файл | Вопрос, на который отвечает |
|---|---|
| [concept/formalization.md](concept/formalization.md) | Общая схема: уровни, сценарий $(N,D,C,O)$, наложение, условный прогноз, метрики качества — что система делает и в каком порядке |
| [concept/scenario-formalization.md](concept/scenario-formalization.md) | Как устроен метод: переменные, сценарий, макромодель, условный прогноз, неправдоподобность, драйверы, отчёт; что из этого делает код |
| [concept/architecture.md](concept/architecture.md) | Как схема реализована в коде: пакеты и поток данных, workspace, версии моделей, задачи, раздел системного промпта |
| [howto/install.md](howto/install.md) | Как поставить Python-часть, подключить плагин к профилю DSH и проверить, что всё видно |
| [howto/mcp.md](howto/mcp.md) | Как подключить сервер к любому MCP-клиенту и разобрать типовые ошибки |
| [howto/first-session.md](howto/first-session.md) | Как провести первый сеанс: от рядов до отчёта, с вызовами инструментов |
| [howto/plugins.md](howto/plugins.md) | Как добавить плагином модель позиции, макромодель или сателлит: объявление, модель, демо-кейс, проверки |
| [reference/tools.md](reference/tools.md) | Какие есть MCP-инструменты и команды `python -m mst.*`, что они принимают и возвращают |
| [reference/platform.md](reference/platform.md) | Контракты расчёта: слои, позиция, пути, модель, диспетчер, реестр, прогон и отчёт |

Порядок чтения для нового читателя: formalization → scenario-formalization → architecture →
install → mcp → first-session → tools → platform → plugins.

Описание каждого пакета лежит в README его папки (`src/mst/*/README.md`), корпуса — в
[../corpus/README.md](../corpus/README.md), плагина DSH — в [../dsh/README.md](../dsh/README.md).
