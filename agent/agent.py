"""Conversational ReAct-style CLI agent that clones the Scaler Academy site.

Architecture
------------
The agent runs in a while loop. On each iteration the LLM emits exactly one
JSON step:

    { "step": "START|THINK|TOOL|OBSERVE|OUTPUT",
      "content": "...",
      "tool_name": "...",   # only for TOOL
      "tool_args": "..." }  # only for TOOL

* We append the assistant's step to the message history.
* If the step is TOOL, we run the tool and inject an OBSERVE message
  (as a `user` role) carrying the tool's output, then loop again.
* If the step is OUTPUT, we stop the inner loop and hand control back to
  the interactive prompt for follow-up questions.

The agent never completes the task in a single LLM call — each call advances
exactly one step.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from tools import call_tool


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    print("ERROR: GROQ_API_KEY is not set.")
    print("Copy .env.example to .env and paste your Groq API key.")
    sys.exit(1)

client = Groq(api_key=API_KEY)
# Model is overridable via env var so you can switch to a smaller / less
# rate-limited model (e.g. "llama-3.1-8b-instant") without editing code.
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

MAX_ITERS_PER_TURN = 40  # safety cap on loop iterations per user message

# When a write_file payload exceeds this many characters, we keep the FULL
# payload only long enough to dispatch the tool and store a stub in history
# instead. This is the single biggest token-saver across long sessions.
WRITE_FILE_HISTORY_STUB_THRESHOLD = 600

SYSTEM_PROMPT = """You are a step-by-step ReAct agent that builds and edits websites.

PROTOCOL (strict). Every response = ONE JSON object, no fences, no prose:
{"step":"START|THINK|TOOL|OBSERVE|OUTPUT","content":"...","tool_name":"...","tool_args":"..."}
tool_name and tool_args only when step == TOOL.

STEPS
- START: acknowledge a new user task + brief plan.
- THINK: reason about next action; multiple THINKs in a row are fine.
- TOOL: invoke ONE tool. Do NOT emit OBSERVE yourself.
- OBSERVE: system-injected after each TOOL. Never produce it yourself.
- OUTPUT: final answer; only when the task is fully done.
RULE: emit exactly ONE step per response. Do not chain steps.

TOOLS
- execute_command -> tool_args = shell command string.
- write_file      -> tool_args = JSON {"filepath":"...","content":"..."}. CRITICAL: `content` is the COMPLETE file written verbatim to disk. NEVER use "...", "[unchanged]", "[existing content]", or any placeholder shorthand — they will be saved literally and corrupt the file. write_file is NOT a patch/diff tool; every call REPLACES the entire file from scratch. To edit a file, read_file first then write_file the FULL updated content. Escape \\n and \\" in content.
- read_file       -> tool_args = path string.
- fetch_url       -> tool_args = URL string.

BUILD GUIDANCE
- Save the site under scaler_clone/ as THREE separate files:
    scaler_clone/index.html   (markup only — link styles.css and script.js)
    scaler_clone/styles.css   (all CSS, properly formatted)
    scaler_clone/script.js    (interactivity)
  Splitting across files gives each write_file call a full output budget so the
  page can be rich and readable. Do NOT inline CSS or JS into index.html.
- After fetch_url, do NOT copy raw HTML; synthesize a polished modern recreation.
- Build order: write index.html first, then styles.css, then script.js — one
  write_file call per file. After all three succeed, go straight to OUTPUT.
- For follow-up edits in later turns: read_file first, then write_file the FULL
  updated version of that one file (every byte, not just the changed parts).

FORMATTING RULES (these directly affect the demo grade — do not ignore)
- HTML: one tag per line where reasonable, 2-space indent, blank lines between
  major sections. NEVER cram the whole document onto one line.
- CSS: one selector per block, one declaration per line, 2-space indent. Group
  rules by section with /* === Header === */ comments. NEVER minify.
- JS: readable, indented, with brief comments on each handler.
- NO lorem ipsum, NO "Program 1 / Program 2" placeholders, NO empty href="#" with
  no label. Use real, Scaler-flavored copy (see CONTENT below).

VISUAL SPEC (non-negotiable)
Fonts/color: load Inter via Google Fonts <link> in index.html <head>. Bg #0a0b0f,
text #fff, muted #9aa0aa, accent gradient #4f46e5 -> #7c3aed -> #06b6d4 (apply to
key headline word + primary CTA).
Header: sticky, dark, backdrop-filter:blur(12px), faint bottom border. Brand
"Scaler" as styled text (no <img>) with second half in the gradient. Nav
(Courses, Events, Blog, For Business) with hover-underline animation. Right-side
primary "Login" button (gradient bg, hover lift).
Hero: min-height 80vh, max-width ~1100px container. Headline clamp(2.5rem,5vw,
4.5rem) weight 700+, key phrase wrapped in <span> using background-clip:text
gradient. Muted subheadline max-width 640px, line-height 1.6. Two CTAs: primary
(gradient) + secondary (outlined), hover transitions. Decorative radial-gradient
blob absolutely positioned behind the headline. 3-4 trust pills below the CTAs
(e.g. "1.5L+ learners", "Top tech mentors", "Industry-vetted curriculum").
Mid section (REQUIRED): a 4-card "Our Programs" grid with real course names —
"DSA & System Design", "Data Science & ML", "DevOps & Cloud", "Full-Stack Web".
Each card: title, 2-3 sentence description, "Explore →" link.
Stats strip BELOW programs: 4 stat boxes (e.g. "20K+ learners placed",
"1.2 Cr highest CTC", "650+ hiring partners", "9.5/10 mentor rating").
Footer: 4-col grid (Company / Programs / Resources / Connect, 4-6 real links
each), top border, bottom row with copyright + Terms/Privacy.
Responsive: @media (max-width:768px) stacks nav, shrinks hero font, single-col
footer and programs grid.
JS: scroll handler that adds .scrolled class to header when scrollY>30 and
reduces padding; smooth-scroll for in-page anchor links.
Size target: index.html ~120-180 lines, styles.css ~250-400 lines, script.js
~30-80 lines. Anything dramatically smaller is under-delivered.
"""

INITIAL_TASK = (
    "Clone the Scaler Academy website (scaler.com). Build a polished page "
    "with Header, Hero, Programs grid, Stats strip, and Footer. Save THREE "
    "files: scaler_clone/index.html (markup), scaler_clone/styles.css (all "
    "styling), scaler_clone/script.js (interactivity). Follow the FORMATTING "
    "RULES and VISUAL SPEC strictly — properly indented, no minification, no "
    "lorem ipsum, real Scaler-style content."
)


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Remove ``` or ```json fences if the model added them despite instructions."""
    s = text.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _parse_step(raw: str) -> dict:
    """Parse one LLM response into a step dict, tolerating fenced code blocks."""
    return json.loads(_strip_fences(raw))


# ---------------------------------------------------------------------------
# History compaction
# ---------------------------------------------------------------------------

# OBSERVE bodies older than the most recent one get stubbed if longer than this.
# Mostly targets large fetch_url dumps whose research value has already been
# consumed by the THINKs that followed.
OBSERVE_HISTORY_STUB_THRESHOLD = 400


def _compact_history(messages: list[dict]) -> None:
    """Shrink older OBSERVE payloads in place to stay under tight TPM limits.

    The most recent OBSERVE is kept verbatim — the model has just acted on
    it and may still reference it. Older OBSERVE bodies are replaced with a
    short stub. write_file payloads are already stubbed at insert time, so
    the only large items left in older history are fetch_url responses.
    """
    observe_indices: list[int] = []
    for i, m in enumerate(messages):
        if m.get("role") != "user":
            continue
        try:
            obs = json.loads(m.get("content", ""))
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obs, dict) and obs.get("step") == "OBSERVE":
            observe_indices.append(i)

    # Preserve the most recent OBSERVE; stub earlier large ones.
    for i in observe_indices[:-1]:
        try:
            obs = json.loads(messages[i]["content"])
        except (json.JSONDecodeError, TypeError):
            continue
        body = obs.get("content", "")
        if isinstance(body, str) and len(body) > OBSERVE_HISTORY_STUB_THRESHOLD:
            obs["content"] = f"<elided earlier observation: {len(body)} chars>"
            messages[i] = {"role": "user", "content": json.dumps(obs)}


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _llm_step(messages: list[dict]) -> str:
    """Make a single LLM call and return the raw assistant content."""
    _compact_history(messages)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

def _truncate(s: str, n: int = 500) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + f"...[+{len(s) - n} more]"


def _print_step(step: dict, tool_result: str | None = None) -> None:
    kind = step.get("step", "?").upper()
    content = step.get("content", "")
    if kind == "START":
        print(f"\n[START] {content}")
    elif kind == "THINK":
        print(f"\n[THINK] {content}")
    elif kind == "TOOL":
        name = step.get("tool_name", "?")
        args_preview = _truncate(step.get("tool_args", ""), 240)
        print(f"\n[TOOL ] {name}({args_preview})")
        if tool_result is not None:
            print(f"   ->  {_truncate(tool_result, 400)}")
    elif kind == "OUTPUT":
        print(f"\n[OUT  ] {content}")
    else:
        print(f"\n[{kind}] {content}")


# ---------------------------------------------------------------------------
# The ReAct loop — one user turn
# ---------------------------------------------------------------------------

def run_turn(messages: list[dict]) -> None:
    """Drive the agent until it emits OUTPUT (or hits the iteration cap).

    `messages` is mutated in place so follow-up turns inherit history.
    """
    for _ in range(MAX_ITERS_PER_TURN):
        try:
            raw = _llm_step(messages)
        except Exception as e:
            print(f"\n[llm error] {e}")
            return

        try:
            step = _parse_step(raw)
        except json.JSONDecodeError as e:
            # Tell the model its output was malformed and let it self-correct.
            print(f"\n[parse error] {e} :: {raw[:200]}")
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON. Reply with "
                    "exactly one JSON object matching the protocol."
                ),
            })
            continue

        kind = step.get("step", "").upper()

        if kind == "TOOL":
            tool_name = step.get("tool_name", "")
            tool_args = step.get("tool_args", "")

            # Persist a compacted version of write_file in history: the file is
            # already on disk, so the model doesn't need its own giant payload
            # echoed back. If it needs the content again it can call read_file.
            # This is the single biggest token-saver across long sessions.
            if (
                tool_name == "write_file"
                and isinstance(tool_args, str)
                and len(tool_args) > WRITE_FILE_HISTORY_STUB_THRESHOLD
            ):
                stubbed = dict(step)
                stubbed["tool_args"] = (
                    "[history compressed: write succeeded — the full content "
                    "you sent was saved to disk verbatim. Call read_file if "
                    "you need to inspect it before editing.]"
                )
                messages.append({"role": "assistant", "content": json.dumps(stubbed)})
            else:
                messages.append({"role": "assistant", "content": json.dumps(step)})

            result = call_tool(tool_name, tool_args)
            _print_step(step, tool_result=result)
            # Inject an OBSERVE message as the next user turn.
            observe = {"step": "OBSERVE", "content": result}
            messages.append({"role": "user", "content": json.dumps(observe)})
            continue

        # Non-TOOL steps: persist verbatim so the model sees its own trace.
        messages.append({"role": "assistant", "content": json.dumps(step)})

        if kind == "OBSERVE":
            # Model should not emit OBSERVE itself. Nudge it back on protocol.
            print(f"\n[warn] model emitted OBSERVE; nudging back to protocol")
            messages.append({
                "role": "user",
                "content": "Do not emit OBSERVE yourself — the system injects it. Continue with THINK, TOOL, or OUTPUT.",
            })
            continue

        _print_step(step)
        if kind == "OUTPUT":
            return
        if kind not in {"START", "THINK"}:
            messages.append({
                "role": "user",
                "content": "Unknown step. Use START, THINK, TOOL, or OUTPUT.",
            })

    print("\n[loop hit MAX_ITERS_PER_TURN; pausing for next user input]")


# ---------------------------------------------------------------------------
# Entry point — initial task, then interactive chat
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 64)
    print(" Scaler Clone Agent")
    print(" model: " + MODEL)
    print("=" * 64)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Skip the initial build if scaler_clone/index.html already exists — lets us
    # demo follow-up edits on a polished pre-built page without the agent
    # rebuilding (and possibly degrading) it on every startup. Force a rebuild
    # by setting FORCE_REBUILD=1 or deleting scaler_clone/.
    existing = Path("scaler_clone/index.html")
    force_rebuild = os.environ.get("FORCE_REBUILD") == "1"

    if existing.exists() and not force_rebuild:
        print(
            f"\n[skip] {existing} already exists — jumping to interactive mode."
        )
        print("       (set FORCE_REBUILD=1 to rebuild from scratch)")
        # Seed the agent with context about what's already on disk so it can
        # answer follow-ups intelligently without re-reading the files first.
        messages.append({
            "role": "user",
            "content": (
                "The scaler_clone/ site has already been built (index.html, "
                "styles.css, script.js exist). Wait for my next instruction — "
                "do not rebuild. When I ask for an edit, read_file the relevant "
                "file first, then write_file the FULL updated content."
            ),
        })
    else:
        messages.append({"role": "user", "content": INITIAL_TASK})
        # Phase 1: hard-coded initial task.
        run_turn(messages)

    # Phase 2: interactive follow-up loop.
    print("\n" + "-" * 64)
    print(" Ready for follow-ups (e.g. 'add a testimonials section between")
    print(" programs and stats', 'change the accent color to green').")
    print(" Type 'exit' or 'quit' to leave.")
    print("-" * 64)

    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", ":q"}:
            print("bye.")
            break
        messages.append({"role": "user", "content": user_input})
        run_turn(messages)


if __name__ == "__main__":
    main()
