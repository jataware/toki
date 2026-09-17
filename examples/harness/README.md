# toki harness — a Cursor-style coding agent in the terminal

A small but complete coding agent built on toki: it browses a workspace, edits
files, runs commands under a permission policy, writes plans to disk, and has
Cursor's three modes (agent / ask / plan). The engine is frontend-agnostic; the
terminal UI is one thin consumer of it.

```bash
pip install 'toki[openrouter]' rich prompt_toolkit      # or any other toki backend; in this repo: uv sync --group examples
export OPENROUTER_API_KEY=...
python -m examples.harness            # from the toki repo root; uses cwd as the workspace
python -m examples.harness ~/code/app --model anthropic:claude-sonnet-4-5
python -m examples.harness --model demo                 # scripted tour, no API key needed
pip install textual && python -m examples.harness --tui  # full-screen UI with a sidebar and clickable rewind
```

Two frontends share one engine: the plain CLI below leaves a normal scrollback
and works anywhere; `--tui` is a Textual app with a sidebar, clickable rewind
on every message, collapsible tool cards, and diff cards with open and revert
buttons (see [The Textual UI](#the-textual-ui)). Type `/help` (or press F1 in
the TUI) inside for everything below.

## The tour

```
╭─ toki harness ──────────────────────────────────────────────────╮
│ workspace  /home/me/code/app  git                               │
│ model      openrouter:anthropic/claude-sonnet-4.5               │
│ mode       agent · shell: ask                                   │
│                                                                 │
│ Type a message. @file attaches context, /help lists commands,   │
│ /agent /ask /plan switch modes.                                 │
╰─────────────────────────────────────────────────────────────────╯
agent ❯ why does @src/auth.py retry twice?
```

Replies stream as markdown. Every tool call is a one-line card; edits also
print a diff:

```
  ▤ list_dir  .  (14 entries, depth 2)
  ▢ read_file  src/auth.py  (120 lines)
  ✎ edit_file  src/auth.py  +3 -1
╭─ src/auth.py +3 -1 · modified ──────────────────────────────────╮
│ @@ -40,7 +40,9 @@                                               │
│ -    for _ in range(2):                                         │
│ +    for attempt in range(MAX_RETRIES):                         │
╰─────────────────────────────────────────────────────────────────╯
```

### Modes

| mode    | what the model can do                                                     |
|---------|---------------------------------------------------------------------------|
| `agent` | read, edit, delete, run commands (subject to permissions), update the plan |
| `ask`   | read only. The edit and shell tools are not even in its schema            |
| `plan`  | read, then write `.harness/plan.md` for you to review; `/build` implements |

Switch with `/agent`, `/ask`, `/plan`, or `/mode <name>`. The prompt and the
bottom toolbar always show the current mode.

**Plan flow.** In plan mode, describe what you want. The agent explores and
writes a markdown plan with `- [ ]` steps. `/plan show` prints it, `/plan edit`
opens it in `$EDITOR` (or Cursor), `/build` switches to agent mode and asks the
model to implement it, ticking each checkbox as it goes. `/plan clear` archives
it to `.harness/plans/`.

### Reviewing edits in Cursor

Edits are written straight to disk. There is no accept/reject step in the
terminal on purpose: Cursor's Source Control view already shows every changed
file with gutter markers and a side-by-side diff, and it lets you keep or
discard per file or per hunk ("Discard Changes", or "Revert this change" from
the gutter). Commit before a session and the review is clean.

The terminal keeps a light safety net:

- `/changes` lists files touched this session
- `/diff [path]` prints the diff against the pre-session content
- `/revert <path|all>` restores the pre-session content (works without git too)
- `/open <path[:line]>` jumps to a file in Cursor (falls back to VS Code)

### Undo

`/undo` removes the last turn (your message and everything the model did in
response) and puts your message back in the input box so you can edit and
resend it. If the turn changed files, it first asks:

```
  ↶ turn 4  add retries to the client
  this turn changed: src/client.py, tests/test_client.py
  [k] keep the file changes   [u] undo them   (Ctrl-C cancels) ›
```

Undoing files restores each one to its content before that turn. Shell
commands can't be undone, and the harness says so. `/undo 3` goes back three
turns, `/history` lists the turns, and `/redo` brings the last undone turn back
(including its file changes) until you send a new message. Turns loaded from a
saved session can be undone conversationally, but their file changes predate
this process and stay put.

### Shell permissions

Four levels, independent of the mode. Set with `/permissions <level>` or
`--permissions`; saved in `.harness/config.json`.

| level       | behaviour                                                    |
|-------------|--------------------------------------------------------------|
| `none`      | no shell tool at all; the model only has the file tools      |
| `ask`       | every command prompts `[y] once [n] deny`                    |
| `allowlist` | commands matching a saved prefix run silently, others prompt with an extra `[a] always` that adds the prefix |
| `open`      | nothing prompts                                              |

`/allow pytest`, `/allow git status`, `/disallow ...` manage the list. A
pipeline or `&&` chain only runs silently if every segment matches.

File edits never prompt, since they are reviewed in your editor afterwards.
Most of the model's work goes through the dedicated tools; `run_command` is
described to it as being for tests, builds, git, and formatters.

### Internet access

Two read-only tools, available in every mode: `web_search` (DuckDuckGo, no API
key) and `fetch_url` (page text with HTML stripped, capped in length). The
system prompt tells the model to prefer the workspace and cite the URL it used.
Since these send data off the machine, `/web off` removes them; the setting is
saved with the shell permissions and shown in the banner and toolbar.

### Sessions

Every turn is saved to `.harness/sessions/<timestamp>-<title>.json` with the
messages, mode, and token usage. On startup, if the workspace has saved
conversations, the five most recent are listed; Enter starts fresh, a number
resumes one and replays its last few exchanges. `/sessions` lists them any time,
`/resume <n>` switches, `/new` starts a fresh file. `--new` and `--resume [N]`
skip the picker. The system prompt is regenerated on resume, so the current mode
and permissions apply to the old conversation.

### Everything else

- `@path` anywhere in a message attaches that file (or a directory tree). Tab
  completes paths and `/commands`.
- Alt+Enter inserts a newline; Enter sends. History lives in `.harness/history`.
- Ctrl+C stops a running turn cleanly (dangling tool calls are answered with
  "cancelled by user" so the conversation stays valid).
- `/model provider:name` switches models mid-conversation. `/new` clears the
  conversation. `/tokens` shows usage. `--think` or `/think on` streams
  reasoning when the backend exposes it.

## The Textual UI

`python -m examples.harness --tui` runs the same engine inside a
[Textual](https://textual.textualize.io/) app:

```
┌ toki harness — /path/to/repo · anthropic:claude-sonnet-4-5 ──────────────────┐
│ mode        │ you · turn 3                                        ↶ rewind    │
│  ● agent    │ add retries to the client                                       │
│  ○ ask      │                                                                 │
│  ○ plan     │ Reading the client first.                                       │
│ shell  ask ▾│  ▶ ▢ read_file  src/client.py  (88 lines)                       │
│ [x] web     │  ▼ src/client.py  +6 -1 · modified                              │
│ plan        │    @@ -12,7 +12,12 @@ …                          open   revert  │
│  build clear│                                                                 │
│ changed     │ I added a retry loop with backoff …                             │
│  M client.py│                                                                 │
│ conversations ├───────────────────────────────────────────────────────────────┤
│  ● add retri…│ agent │ shell: ask │ web: on │ 12,400 tokens                   │
│  new        │ Message the agent… (Enter sends, Shift+Enter newline)          │
└──────────────┴────────────────────────────────────────────────────────────────┘
```

- **↶ rewind** on any of your messages undoes everything from that turn on. If
  files changed, a dialog asks *keep file changes* or *undo file changes*, and
  Escape cancels. Your message comes back in the input box.
- Tool cards are collapsed one-liners; click to see the full output. Failed
  calls expand automatically.
- Diff cards show the change with **open** (jumps to the file in Cursor) and
  **revert**. Large diffs start collapsed.
- The plan card has a **build** button. The sidebar shows the current plan,
  changed files (click to open; **revert all**), and saved conversations
  (click to resume, **new** to start over).
- Shell approval is a modal with `y` / `n` / `a` shortcuts. Escape stops a
  running turn. Ctrl+B hides the sidebar. F1 is help. Ctrl+P opens Textual's
  command palette.
- Slash commands still work in the input for parity with the CLI.

The engine runs in a worker thread; every event is marshalled to the UI thread,
and the approver blocks the worker on a modal, which is the same shape a
desktop GUI would use.

## Layout

```
examples/harness/
  core/            engine, no UI dependencies
    harness.py     Harness: modes, tool loop, plan, changes, undo/redo, cancel → run_turn() yields events
    events.py      TextDelta, ThinkingDelta, ToolStarted, ToolArgDelta, ToolFinished,
                   FileChanged, PlanUpdated, Notice, TurnEnded
    tools.py       list_dir, read_file, grep, find_files, edit_file, create_file,
                   delete_file, write_plan, run_command, web_search, fetch_url
    web.py         DuckDuckGo search and HTML-to-text, stdlib only
    sessions.py    SessionStore: save/list/load conversations as JSON
    permissions.py PermissionLevel, Permissions (allowlist, web flag, persistence), Approver
    workspace.py   path jail, ignore rules, snapshots, diff/revert, subprocess
    models.py      `provider:name` → toki model
  tui/             the Textual frontend (`--tui`): app.py holds the widgets, modals, and app
  cli/             the plain terminal frontend
    app.py         loop: prompt → slash command or run_turn → render events
    ui.py          Rich rendering: streamed markdown, tool cards, diff panels, help
    input.py       prompt_toolkit: history, multiline, completion, toolbar
    commands.py    one table drives dispatch, /help, and completion
  testing.py       ScriptedModel (a fake toki backend) and the `demo` tour
  tests/           pytest, no network:  pytest examples/harness/tests -q  (includes headless TUI tests)
```

## Using the engine from a GUI

The core never touches stdin or stdout. A frontend does three things:

```python
from examples.harness.core import Harness, Mode, PermissionLevel, make_model
from examples.harness.core import TextDelta, ToolStarted, ToolFinished, FileChanged, TurnEnded

def approve(request):                 # called only when a shell command needs a decision
    return "allow" if my_dialog(request.detail) else "deny"     # or "allow_always"

harness = Harness(make_model("anthropic:claude-sonnet-4-5"), root="/path/to/repo",
                  approver=approve, mode=Mode.AGENT, permission_level=PermissionLevel.ALLOWLIST)

for event in harness.run_turn("add input validation to the signup form", attachments=["src/signup.py"]):
    match event:
        case TextDelta(text):            append_to_chat(text)
        case ToolStarted(name=name):     show_tool_card(name)
        case ToolFinished(summary=s):    finish_tool_card(s)
        case FileChanged(path, kind, diff): show_diff(path, diff)
        case TurnEnded():                enable_input()
```

Mode, permissions, plan, changes, and sessions are plain methods on the same
object (`set_mode`, `set_permission_level`, `set_web`, `allow`, `plan_text`,
`build_prompt`, `changes`, `diff`, `revert`, `undo_preview`, `undo`, `redo`, `history`,
`list_sessions`, `resume_session`, `cancel`, `reset`), so every slash command in the CLI maps to one call. The
harness autosaves after each turn; a GUI's conversation list is
`harness.list_sessions()`. Run the generator in a worker thread and marshal events
to the UI thread; call `harness.cancel()` from a stop button. An async twin on
`Agent.aexecute` is a straightforward port of `run_turn`.

## How it uses toki

- `Agent` holds the conversation; `Agent.tools` is rebound whenever the mode or
  permission level changes, so ask mode literally cannot edit.
- `edit_file`, `create_file`, and `write_plan` are `StreamingToolSchema`s, so
  the replacement text streams in as `ToolArgDelta` events while the model is
  still writing it (the CLI shows a live character count).
- Tool results are capped before entering history. There is no automatic
  compaction; see [`compact.py`](../compact.py) for that policy and `/new` to
  start over.
- `usage_metadata` is summed per turn for `/tokens`; hosted backends are
  created with `cache="rolling"` where supported.
