# Документация

[English version](README.en.md)

Документация отвечает на три вопроса читателя и хранит принятые решения; у каждого своя
папка.

- **[concept/](concept/)** — что система делает и почему так устроена: общая схема, метод,
  устройство кода.
- **[howto/](howto/)** — как её поставить и пользоваться: установка, подключение, первый сеанс,
  плагин.
- **[reference/](reference/)** — что чему соответствует: контракты платформы и справочник вызовов.
- **[adr/](adr/)** — принятые решения с их контекстом и последствиями: почему именно так.

| Файл | Вопрос, на который отвечает |
|---|---|
| [concept/formalization.md](concept/formalization.md) | Общая схема: уровни, сценарий, наложение, условный прогноз, метрики качества — что система делает и в каком порядке |
| [concept/bvar-method-legacy.md](concept/bvar-method-legacy.md) | Устаревшее описание первой версии метода на BVAR, не канон: подробности оценки BVAR и сателлитов |
| [concept/architecture.md](concept/architecture.md) | Как схема реализована в коде: пакеты и поток данных, workspace, версии моделей, задачи, раздел системного промпта |
| [howto/install.md](howto/install.md) | Как поставить Python-часть, подключить плагин к профилю DSH и проверить, что всё видно |
| [howto/mcp.md](howto/mcp.md) | Как подключить сервер к любому MCP-клиенту и разобрать типовые ошибки |
| [howto/first-session.md](howto/first-session.md) | Как провести первый сеанс: от рядов до отчёта, с вызовами инструментов |
| [howto/plugins.md](howto/plugins.md) | Как добавить плагином модель позиции, макромодель или сателлит: объявление, модель, демо-кейс, проверки |
| [reference/tools.md](reference/tools.md) | Какие есть MCP-инструменты и команды `python -m mst.*`, что они принимают и возвращают |
| [reference/platform.md](reference/platform.md) | Контракты расчёта: слои, позиция, пути, модель, диспетчер, реестр, расчёт и отчёт |
| [adr/0001-tool-schema-descriptions.md](adr/0001-tool-schema-descriptions.md) | Почему каждый параметр инструмента описан, а каждый инструмент аннотирован |
| [adr/0002-job-wait.md](adr/0002-job-wait.md) | Почему ожидание задачи держит вызов сервер, а не опрашивает клиент |
| [adr/0003-plugin-stdio-isolation.md](adr/0003-plugin-stdio-isolation.md) | Почему печать плагина уходит в stderr на уровне Python, а не дескриптора |
| [adr/0004-atomic-writes-windows.md](adr/0004-atomic-writes-windows.md) | Почему атомарная запись повторяется, когда читатель держит файл на Windows |

Порядок чтения для нового читателя: formalization → architecture →
install → mcp → first-session → tools → platform → plugins.

Описание каждого пакета лежит в README его папки (`src/mst/*/README.md`), корпуса — в
[../corpus/README.md](../corpus/README.md), плагина DSH — в [../dsh/README.md](../dsh/README.md).
