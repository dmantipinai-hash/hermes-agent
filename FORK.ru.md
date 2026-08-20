# Этот форк: установка, обновление и история версий

**English version: [FORK.md](FORK.md).**

Этот репозиторий — **форк dmantipinai-hash** проекта
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent).
Он несёт **Architecture 2.0** — систему памяти (v2, с поиском FTS5), конвейер
быстрой доставки сообщений и мульт-агентную оркестрацию (`/team`, kanban-доска,
agent mailbox). Этого нет **ни в одном релизе апстрима**.

## Линия версий — прочитайте, прежде чем трогать версии

- Архитектура создана на линии **v0.16.0 этого форка** (автор переработал
  архитектуру агента) и продолжается **только на `main` этого репозитория**.
- **«Мои обновления» = свежий tip ветки `main` репозитория
  dmantipinai-hash/hermes-agent.** Они не привязаны к номерам версий
  глобального Hermes: когда автор дорабатывает форк и пушит в `main`,
  пользователь получает именно эти доработки командой обновления ниже.
- Релизы апстрима **0.17 / 0.18 / 0.19 / 0.20** (пакет PyPI, установщики
  NousResearch, архивы GitHub NousResearch) архитектуру **не содержат**.
  Установка или «обновление» на них означает потерю памяти, доставки
  сообщений и оркестрации. Обратной миграции нет — только переустановка
  форка.
- Номер версии форка (например, `0.20.4`) **наследуется** от апстрим-базы,
  которую сопровождающий вручную влил *под* архитектуру. Это номер базы
  слияния, а не «код апстрима 0.20.4». Форк определяется по репозиторию,
  никогда — по числу версии.
- Изменения апстрима попадают в форк **только вручную и осознанно** —
  слияние с разрешением конфликтов делает сопровождающий. Автоматически —
  никогда, и не через `hermes update`.
- **`hermes update` в форке отключён.** Все его автоматические каналы
  вели на апстрим (PyPI-пакет, ZIP-архив NousResearch, синхронизация с
  upstream-remote). Теперь команда вместо обновления печатает правильную
  команду обновления форка.

## Установка — одна команда на платформу

**Windows 10/11 (PowerShell):**

```powershell
iex (irm https://raw.githubusercontent.com/dmantipinai-hash/hermes-agent/main/scripts/install.ps1)
```

**macOS / Linux:**

```bash
curl -fsSL https://raw.githubusercontent.com/dmantipinai-hash/hermes-agent/main/scripts/install.sh | bash
```

**Любая ОС, если уже стоит [uv](https://docs.astral.sh/uv/):**

```bash
uv tool install --from "git+https://github.com/dmantipinai-hash/hermes-agent.git" "hermes-agent[all]"
```

Установщики клонируют **этот репозиторий** (не NousResearch) и сами ставят
uv, Python и Node по мере необходимости. После установки выполните
`hermes setup` — мастер настройки модели и ключей.

## Обновление — одна команда

Форк обновляется **только из этого репозитория**. Когда автор пушит
доработки в `main` форка:

```bash
uv tool install --force --from "git+https://github.com/dmantipinai-hash/hermes-agent.git" "hermes-agent[all]"
```

Если вы ставили платформенным установщиком (он оставляет git-чекаут),
`hermes update` вам тоже подходит — он тянет репозиторий форка. Команда
`uv` выше покрывает любой тип установки независимо от способа.

## Ловушки — НЕ делайте так

- **Не ставите и не обновляйтесь с PyPI.** Пакет `hermes-agent` на PyPI —
  апстрим-релиз без архитектуры. `pip install --upgrade hermes-agent` и
  `uv tool upgrade hermes-agent` ведут именно туда.
- **Не используйте апстрим-установщики.** Всё, что указывает на
  `NousResearch/hermes-agent` или `hermes-agent.nousresearch.com`, ставит
  апстрим. Используйте команды выше — они указывают на этот форк.
- **Не смешивайте установки.** Запуск апстрим-hermes поверх `~/.hermes`,
  созданного форком (база памяти v2, конфиг-ключи форка), не тестировался
  и может потерять или исказить данные. Делайте резервную копию `~/.hermes`
  перед переключением в любую сторону.
- **Не используйте editable-дев-чекаут как рантайм круглосуточного
  gateway.** Чекаут посреди merge или rebase скармливает поломанный код
  работающему агенту.

## Продвинутые пути

**Pinned wheel-snapshot** (самый стабильный — его использует сопровождающий;
установка не меняется, пока вы явно не пересоберёте):

```bash
git clone https://github.com/dmantipinai-hash/hermes-agent.git
cd hermes-agent && git log --oneline -1        # запомните коммит
HERMES_NIX_BUILD=1 uv build --wheel -o /tmp/hermes-dist
uv tool install --force 'hermes-agent[all] @ file:///tmp/hermes-dist/hermes_agent-<ВЕРСИЯ>-py3-none-any.whl'
```

`HERMES_NIX_BUILD=1` обязателен — апстрим намеренно валит сборку wheel без
этой переменной.

**Дев-чекаут (editable venv):**

```bash
git clone https://github.com/dmantipinai-hash/hermes-agent.git
cd hermes-agent
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[all]"
```

## Заметки

- Данные пользователя живут в `~/.hermes/` (config.yaml, ключи в .env,
  skills, память, сессии). Профили изолированы в `~/.hermes/profiles/<имя>/`.
- После обновления глобальной установки, работающей как сервис, перезапустите
  сервис (например, `launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway`).
  Уведомление «Gateway shutting down» в домашний канал штатно приходит при
  каждом чистом рестарте — независимо от того, идёт ли задача.
