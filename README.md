# Google TTS MCP Server (`google-tts-mcp`)

A Python-based Model Context Protocol (MCP) server designed to automate Text-to-Speech (TTS) audio generation via the Google AI Studio / Gemini TTS API from `.tts` script files (such as `samples/aula04-script-duo.tts`).

## Key Features

- **Smart Paragraph Partitioning**: Splits the input script into continuous partitions up to a maximum character limit (default 1300), first respecting existing `---` section delimiters. Sections within 1300 characters are kept intact; larger sections are re-partitioned while strictly preserving paragraphs and sentence-ending punctuation (`.`, `!`, `?`, `:`).
- **Quota Protection & Rate Limiting**: Built on the official `google-genai` SDK using `aiolimiter` and `asyncio.Semaphore` to manage rate limits (15 RPM) and concurrency with exponential backoff retries.
- **Direct Video Editor Compatibility (48000 Hz, pcm_s16le, Mono)**: Automatic resampling from native 24kHz PCM to 48,000 Hz `pcm_s16le` Mono (1 channel). Ensures 100% audio clock sync in video editing software (DaVinci Resolve, Premiere, CapCut, Kdenlive, Final Cut) without bloated stereo file sizes or manual `ffmpeg` commands.
- **Real-Time Job Progress & File Logging**: Automatically writes `generation.log` and `progress.json` to the output directory during batch operations. Clients can track live progress via the `check_job_progress` tool.
- **Secure `config.yaml`**: No API keys are stored in configuration files (safe for git repository sharing). API keys are loaded dynamically from environment variables (`GEMINI_API_KEY` or `GOOGLE_API_KEY` or `.env`).
- **Automatic Part Merging**: Generates individual partition `.wav` files (e.g., `aula04-script-duo_part01.wav`) and a full concatenated file `aula04-script-duo_complete.wav` with customizable silence pauses (300ms).
- **Exposed Config Resources**: MCP clients can dynamically fetch `config://schema` and `config://template` to auto-generate or validate custom configurations.

---

## Project Structure

```
google-tts-mcp/
├── pyproject.toml                 # Package configuration and dependencies
├── config.yaml                    # Voices, model, rate limit, and audio config
├── README.md                      # Documentation
├── samples/
│   └── aula04-script-duo.tts      # Provided sample script
├── src/
│   └── google_tts_mcp/
│       ├── __init__.py
│       ├── server.py              # FastMCP server, resources, tools, and logging
│       ├── config.py              # config.yaml loader
│       ├── partitioner.py         # Text partition engine (<1300 chars)
│       ├── api_client.py          # Google GenAI client with rate limiters
│       ├── audio.py              # 48kHz pcm_s16le resampling and RIFF WAV writer
│       └── utils.py              # File naming and formatting utilities
└── tests/
    ├── test_partitioner.py        # Partitioner tests (paragraphs, sections, sentences)
    ├── test_config.py             # Configuration loader, MCP resources, and progress tests
    ├── test_audio.py              # 48kHz resampling and WAV header tests
    └── test_samples.py            # Automated tests on all samples/ files
```

---

## Configuration (`config.yaml`)

```yaml
generator:
  provider: "google-ai-studio"
  model: "gemini-2.5-flash-preview-tts"

rate_limit:
  max_requests_per_minute: 15
  max_concurrent_requests: 2
  retry_attempts: 3
  backoff_factor: 2.0

partitioning:
  max_chars_per_partition: 1300
  respect_existing_delimiters: true

voices:
  language_code: "pt-BR"
  scene: "A high-quality recording studio, two friends talking casually into dynamic mics."
  context: "Podcast style. Fast, slightly overlapping pacing. Tone is energetic, conversational, and warm."
  speakers:
    "Speaker 1":
      voice_name: "Aoede"
      profile: "An authoritative main news anchor."
      directors_note: "Style: Vocal Smile. Pace: Natural conversational pace. Accent: American (Gen)."
    "Speaker 2":
      voice_name: "Puck"
      profile: "A professional field correspondent."
      directors_note: "Style: Newscaster. Pace: Rapid Fire. Accent: American (Gen)."

audio:
  format: "wav"
  sample_rate: 48000
  sample_width_bytes: 2
  channels: 1
  output_dir: "output"
  inter_partition_pause_ms: 300
  naming_pattern: "{input_name}_part{part_num:02d}.wav"
  combine_full: true
```

---

## Required Environment Variable

Set your API key in the environment or in a `.env` file prior to generating speech:

```bash
export GEMINI_API_KEY="your_google_ai_studio_api_key"
```

---

## MCP Resources & Tools

### Exposed MCP Resources

- **`config://schema`**: Returns the complete JSON schema definition and field documentation for building custom `config.yaml` files.
- **`config://template`**: Returns the actual content of `config.yaml` as a YAML string that clients can save directly to disk.

### Exposed MCP Tools

1. **`generate_tts_from_file`**:
   - `file_path`: (Required) Path to the `.tts` script file.
   - `output_dir`: (Optional) Output directory path for generated `.wav` files (defaults to `'output'`).
   - `dry_run`: (Optional) If `True`, simulates speech generation with synthetic audio without calling Google APIs.
   - `resume`: (Optional) If `True`, skips already generated valid partition files from a previous run.
   - `config_path`: (Optional) Path to a custom `config.yaml`.

2. **`check_job_progress`**:
   - `output_dir`: (Optional) Target output directory path (defaults to `'output'`). Reads real-time `progress.json` and recent lines of `generation.log`.

---

## Installation & Usage

### 1. Installation using `uv`
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 2. Running Automated Tests
```bash
.venv/bin/python -m pytest -v
```

### 3. Registering the MCP Server
In your MCP client settings (Gemini / Claude / VSCode MCP settings):

```json
{
  "mcpServers": {
    "google-tts-mcp": {
      "command": "/home/einstein/projects/google-tts-mcp/.venv/bin/python",
      "args": ["-m", "google_tts_mcp.server"],
      "env": {
        "GEMINI_API_KEY": "YOUR_KEY_HERE"
      }
    }
  }
}
```
