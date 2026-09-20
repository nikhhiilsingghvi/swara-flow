# SwaraFlow

A multimodal AI coaching system for Indian classical vocal practice.
SwaraFlow analyzes a singer's live voice (pitch, swara, intonation,
rhythm) and body (posture, motion) against a chosen raga's actual
melodic rules, and turns the measured results into raga-correct
practice exercises -- not generic pitch-tuner feedback and not an LLM
chatbot with a music-themed skin.

**Read `docs/testing.md` before trusting any "done" claim in this
repo.** This project was built under a no-network-access constraint;
that file is the honest, itemized breakdown of what was actually
executed and verified versus written-correctly-but-unverified. Short
version: the audio/vision/music-theory/coaching-agent core (backend/)
is real, tested, and passing (195/195 tests at last run). The FastAPI
transport layer and the Next.js frontend are written to the same
standard but need `pip install`/`npm install` and a real
browser/webcam to verify -- which this development environment did not
have.

## Quick start

```
python scripts/setup.py
python scripts/download_models.py   # optional, enables posture/hand tracking
python scripts/run-dev.py
```
Then open `http://localhost:3000`. See `docs/setup.md` for details,
Windows-native commands, and Docker.

## What makes this different from a pitch tuner

- **Sa-relative, just-intonation pitch model** (`music/shruti.py`,
  `music/swara_mapping.py`) -- not 12-TET, not hardcoded to any
  Western note.
- **Data-driven raga rule engine** (`music/raga_rules.py`) -- adding a
  raga means adding a JSON file, not editing code. Ships with Bhupali,
  Yaman, and Bhairav.
- **Continuous pitch trajectory, not discrete notes**
  (`audio/phrase.py`) -- meend/glide, stable notes, and jumps are
  classified explicitly, matching how the tradition actually treats
  pitch.
- **A real agent pipeline that works with zero LLM calls**
  (`agents/`) -- Analysis -> Critic -> Practice -> Progress, all
  deterministic, an LLM only ever rephrases an already-decided
  finding into friendlier prose.
- **OpenCV-core computer vision** (`vision/`) -- real Haar-cascade
  face detection and Farneback optical-flow motion analysis, both
  verified against real images/known ground truth, not fabricated
  metrics.

## Project structure

```
swara-flow/
├── backend/        FastAPI + audio/vision/music/agents/sessions (Python)
├── frontend/        Next.js/React/TypeScript/Tailwind (the SwaraFlow visualization)
├── configs/raga/     Raga definitions (JSON) -- Bhupali, Yaman, Bhairav
├── data/               Session storage, feature/dataset directories
├── docker/              Dockerfiles for backend + frontend
├── docs/                 Architecture, protocol, privacy, research, testing
└── scripts/               setup / run-dev / run-tests / download_models (.py + .bat + .sh)
```

## Documentation

- [`docs/setup.md`](docs/setup.md) -- installation and running
- [`docs/architecture.md`](docs/architecture.md) -- module boundaries, agent graph, data flow
- [`docs/websocket-protocol.md`](docs/websocket-protocol.md) -- the live-session message protocol
- [`docs/testing.md`](docs/testing.md) -- what's verified vs. what needs your machine
- [`docs/privacy.md`](docs/privacy.md) -- local-first design, what's stored, what's never sent anywhere
- [`docs/research-notes.md`](docs/research-notes.md) -- problem definition, limitations, future work

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
