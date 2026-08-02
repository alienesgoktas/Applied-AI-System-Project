# AI Interactions Log

> **Stretch features only.** Only fill in the sections that apply to stretch features you attempted. If you did not attempt a stretch feature, leave its section blank or delete it. This file is not required for the core project.

---

## Phase 1–2: Concept Research and Data Design

> Not a stretch feature — logged here for transparency about how AI was used
> during the design phases. Full research output lives in [`research.md`](research.md).

### Prompt 1 — How real recommenders work

> *"Act as a senior machine learning engineer specializing in recommendation
> systems. Provide a deep-dive breakdown comparing Collaborative Filtering and
> Content-Based Filtering, covering high-level architecture, technical mechanisms
> and data vectors, cold-start behavior, and the real-world hybrid approach used
> by platforms like Spotify and YouTube."*

**What it produced:** A structured comparison saved to `research.md` — the
implicit-feedback matrix factorization objective, spectrogram CNNs predicting CF
latent vectors, a cold-start matrix, and the two-stage
candidate-generation → ranking → re-ranking pipeline.

**What I took from it:** The realization that my project has *no interaction
data at all*, so collaborative filtering isn't a design option I'm rejecting —
the data to build it does not exist. That reframed the whole project as
content-based by necessity, which is now the opening argument in my README.

**What I verified:** Checked the four cited papers are real and correctly
attributed. Noted in `research.md` that published architectures describe these
systems as of publication date; current production internals are proprietary.

### Prompt 2 — Data types in recommender systems

> *"Identify the main data types involved in these systems, such as likes, skips,
> playlists, tempo, or mood."*

**What it produced:** A split between *signal categories* (explicit feedback,
implicit feedback, sequence/relational, item content, context, user attributes,
derived) and *representation types* (continuous normalized vs. unbounded,
categorical low- vs. high-cardinality, boolean, ordinal, sequential, text).

**What I took from it:** The warning that unbounded numeric features must be
normalized before entering a weighted sum — directly relevant to `tempo_bpm`
(60–168) sitting alongside 0–1 features.

### Prompt 3 — Feature selection and the scoring rule

> Analyze the available data in songs.csv and suggest which features would be
> most effective for a simple content-based recommender. Then explain how to
> score a numerical feature like energy so that songs closer to the user's
> preference are rewarded, rather than just higher values. Should a matching
> genre be worth more than a matching mood? And why do we need both a Scoring
> Rule and a Ranking Rule?"

**What it produced:** The AI ran an actual correlation analysis on the CSV rather
than guessing, and found that in the 10-song starter catalog `tempo_bpm`
(r = +0.96), `danceability` (+0.86) and `acousticness` (−0.99) were nearly
perfectly correlated with `energy`.

**What I took from it:** This changed my feature list. Scoring all four numerics
would have been counting one signal four times at an effective weight of ~4.0,
drowning out genre and mood. I dropped `tempo_bpm` and `danceability` from the
scoring rule for this reason, and kept `valence` because it was the only numeric
that was *not* correlated with energy (r = +0.19).

Also adopted: `closeness = 1 − |song.energy − user.target_energy|` so the score
peaks at a match and decays in both directions, and the scoring/ranking split
(per-song pure function vs. list-level policy that owns `k`, tie-breaks and
diversity).

**What I pushed back on / verified manually:** I re-ran the correlation numbers
myself against the CSV rather than trusting the claim. I also had to decide
whether `target_valence` was worth adding to `UserProfile`, since the starter
class did not have that field.

### Prompt 4 — Expanding the dataset

> *"Generate 5–10 additional songs in valid CSV format matching the existing
> headers. Cover a diverse range of genres and moods not already present in the
> starter file."*

**What it produced:** 10 songs (ids 11–20) adding 9 genres (hip hop, folk, metal,
classical, house, bluegrass, r&b, afrobeats, country) and 5 moods (confident,
melancholy, euphoric, romantic, nostalgic).

**What I took from it:** Beyond variety, several rows were designed to break the
`energy`/`acousticness` correlation found in Prompt 3 — a loud acoustic bluegrass
track (#16, energy 0.87 / acousticness 0.90) and a quiet electronic one (#17,
energy 0.32 / acousticness 0.07). Two `ambient` songs now sit at opposite ends of
acousticness, so genre can no longer predict production style.

**Measured effect after expansion:**

| correlation with energy | before | after |
|---|---|---|
| acousticness | −0.99 | −0.72 |
| danceability | +0.86 | +0.69 |
| valence | +0.19 | +0.38 |
| tempo_bpm | +0.96 | +0.91 |

**Limitations I noted rather than hid:** `tempo_bpm` is still redundant
(+0.91), and genre is now sparse at 16 genres across 20 songs, so exact genre
matches will rarely fire. Both are documented as known trade-offs instead of
being engineered away.

---

## Agentic Workflow (SF8)

> Document your experience using an AI agent (e.g., Cursor Agent, Claude, Copilot) to make multi-step changes autonomously.

**What task did you give the agent?**

The assignment asked me to learn more about how streaming platforms like spotify youtube etc. predicts what the user will like next. So I provided a prompt for it to make a plan.

**Prompts used:**

Act as a senior machine learning engineer specializing in recommendation systems. I want to understand exactly how major music and video streaming platforms like Spotify and YouTube predict what a user will love next. 

Please provide a deep-dive breakdown that compares and contrasts the two core pillars of recommendation engines: **Collaborative Filtering** and **Content-Based Filtering**.

To keep this highly structured and actionable, please organize your response into the following sections:

1. **High-Level Architectural Breakdown:** Define Collaborative Filtering (leveraging collective user behavior) vs. Content-Based Filtering (analyzing intrinsic item attributes) in simple, architectural terms. How do they fundamentally differ in the data inputs they require?
2. **Technical Mechanisms & Data Vectors:** 
   * For Collaborative Filtering, explain how user-item interaction matrices, matrix factorization, or embeddings map similarities between users (e.g., "Users who liked X also liked Y").
   * For Content-Based Filtering, explain how platforms analyze raw assets—such as audio features (tempo, valence, timbre via spectrograms) or video data (tags, transcripts, visual framing)—to create an item profile.
3. **The "Cold Start" and Lifecycle Matrix:** Create a text-based matrix or comparison table illustrating how each approach handles:
   * A brand-new user with zero history.
   * A brand-new song/video with zero plays.
   * Niche or hyper-specific tastes.
4. **Real-World Synergy (The Hybrid Approach):** Explain how platforms like Spotify (e.g., Discover Weekly) or YouTube blend these two approaches to overcome individual limitations. What does the final "re-ranking" or hybrid pipeline look like in practice?

Keep your tone analytical, technical yet accessible, and use concrete examples from modern streaming platforms where applicable.

**What did the agent generate or change?**

The research prompt itself produced no code — it generated a written breakdown
that the agent saved to a new file, `research.md`. That file covers the
architectural split between collaborative and content-based filtering, the
implicit-feedback matrix factorization used on play counts, spectrogram CNNs that
give brand-new songs a starting point, a cold-start comparison matrix, and the
two-stage candidate-generation → ranking → re-ranking pipeline. It ends with a
section on what this means for my project, and four cited papers.

From there the agent worked autonomously across the rest of the build:

**Files created**

| File | What it holds |
|---|---|
| `research.md` | The recommender systems research above |

**Files edited**

| File | Change |
|---|---|
| `src/recommender.py` | Implemented `load_songs`, `score_song`, `recommend_songs`, plus `Recommender.recommend` and `explain_recommendation`. Added `target_valence` to `UserProfile` with a `0.5` default |
| `src/main.py` | Fixed the broken import, printed the catalog size, and formatted the ranked output with per-song reasons |
| `tests/test_recommender.py` | Grew from 2 starter tests to 10 |
| `data/songs.csv` | Added songs 11–20 |
| `README.md` | Project summary, system design, data-flow diagram, algorithm recipe, sample output, experiments, limitations, reflection |
| `model_card.md` | All nine sections |

**Code the agent wrote that I kept**

- The scoring rule, where energy and valence are scored by *closeness*
  (`1 - |difference|`) rather than by magnitude, so a user asking for calm music
  is not handed the loudest track in the catalog.
- A partial genre match, so "indie pop" earns half credit for someone who asked
  for "pop".
- Reasons carrying their own point values, e.g. `genre match: pop (+2.0)`, so the
  output explains itself.
- `sorted()` rather than `.sort()` in `recommend_songs`, so asking for a
  recommendation does not silently reorder the caller's catalog. There is a
  regression test for this.

**Analysis scripts it wrote and ran** (kept out of the repo — these were
throwaway working files, not part of the project)

- A correlation analysis of `songs.csv`, which found `tempo_bpm` at +0.96,
  `danceability` at +0.86 and `acousticness` at −0.99 against `energy`. This is
  why I excluded tempo and danceability from scoring.
- A weight-sketch prototype that tested the proposed profile before any real code
  existed, confirming a rock profile and a lofi profile shared no results.
- A five-profile evaluation harness that ran both Phase 4 experiments by
  overriding the weight constants at runtime and restoring them afterwards, so no
  experimental edits were left behind in `recommender.py`.

**Commands run:** `git clone`, `python -m pytest -q` before and after every
change, `python -m src.main`, and `git add` / `commit` / `push origin main`
across 8 commits.

**What did you verify or fix manually?**

I verified by running tests with claude. I asked it to create tests first then ran them. 

---

## Design Pattern (SF10)

> Document how AI helped you choose or implement a design pattern.

**Which design pattern did you use?**

**Strategy.** The scoring weights are now an object that can be swapped out,
instead of constants baked into the scoring function.

**How did AI help you brainstorm or implement it?**

I asked gemini for a prompt at first then did the research with claude on how the streaming services work and what can I do that they don't. 

The pattern itself came out of a mistake rather than a plan.

During Phase 4 I needed to run two experiments — halve the genre weight while
doubling energy, and remove the mood check entirely. The way the AI did this was
to reach into the module at runtime and overwrite the weight constants
(`R.W_GENRE = 1.0`), run the comparison, then set them back afterwards. It
produced correct numbers, so I nearly left it at that.

When I later asked about design patterns, the AI pushed back on documenting one
retroactively. Its point was that my code did not actually contain a deliberate
pattern — the structure it had (`Recommender.score` delegating to `score_song`)
was an accident of keeping two APIs in sync, not a choice, and writing that up as
a pattern would be inventing history.

What it suggested instead was that the monkey-patching hack was itself the
argument for Strategy. Mutating global state to answer a question is fragile: if
one experiment forgets to restore a value, every result after it is silently
wrong and nothing in the code flags it. Making each weight configuration a proper
object removes that whole class of bug.

That reasoning is what convinced me. I did not want a pattern for its own sake,
but "my experiments were editing global variables and could have corrupted each
other" is a real problem worth fixing.

**How does the pattern appear in your final code?**

In `src/recommender.py`:

- **`ScoringStrategy`** — a frozen dataclass holding the six weights. Frozen
  means an experiment physically cannot modify a shared strategy, which is the
  exact bug the old approach risked. There is a test for this.
- **Four named strategies** — `BALANCED` (the recipe in my README),
  `ENERGY_FIRST` and `MOOD_BLIND` (my two Phase 4 experiments), and
  `GENRE_PURIST` (an exaggerated genre weight, to demonstrate the filter-bubble
  effect).
- **`score_song(user_prefs, song, strategy=BALANCED)`** — takes the strategy as
  an argument, so the algorithm is chosen by the caller rather than hard-coded.
  It also skips any term whose weight is zero, which is why `MOOD_BLIND` produces
  no "mood match" line in the explanation instead of a `+0.0` one.
- **`Recommender(songs, strategy=...)`** — the class-based entry point accepts
  one too.

Both default to `BALANCED`, so nothing that already called these functions needed
to change, and the CLI output is identical to before.

The payoff shows up in `tests/test_recommender.py`. My Phase 4 experiments used
to live in a throwaway script; they are now real tests —
`test_strategies_can_produce_different_rankings` pins that the balanced strategy
puts *Gym Hero* in the top 3 while the energy-first strategy puts *Ocean Bus
Route* there instead. That finding is now protected against regression rather
than being a number I wrote down once.

Test count went from 10 to 17.

## Module 4 — RAG + Reliability Layer (SF)

This module was built in a pair-programming session with an AI coding agent. The plan,
code, and tests were produced collaboratively; the load-bearing interactions are below.

### Prompt — designing the AI feature

> *"I want to add a genuine, fully-integrated AI feature to the recommender."*

**What it produced:** a plan to turn the recommender into the **retriever** of a RAG pipeline
(NL request → taste profile → retrieve top-k → grounded explanation), wrapped in a reliability
layer (confidence + honesty note + grounding guardrail + tests), provider-neutral and
local-first with a deterministic offline fallback.

**What I took from it:** the whole architecture — retriever-as-RAG, the guardrail, strategy-aware
confidence, and the offline fallback for reproducibility.

**What I verified:** 50 tests (mocked, no key/server), the offline CLI, and a live run against my
local `gemma4-12b-says-v2` server — the guardrail held (every pick was a real catalog song).

### Prompt — backend shape (a flawed assumption I caught)

**What it produced:** an initial design assuming my local server was OpenAI-compatible
(`/v1/chat/completions` via the `openai` SDK).

**What I took from it:** nothing — it was wrong for my setup.

**What I verified:** I shared my actual `curl` (a custom `POST /api/v1/chat` with a
`{model, system_prompt, input}` body). We replaced the OpenAI assumption with a small stdlib
`urllib` client tuned to the real contract, and dropped the `openai` dependency.

### Prompt — confidence denominator (a helpful catch)

**What it produced:** a code review flagging that a hardcoded max score (6.5) would make
confidence falsely saturate at 1.0 for non-default scoring strategies.

**What I took from it:** deriving the denominator from the active strategy
(`ScoringStrategy.max_score()`), so confidence stays honest for any strategy.

**What I verified:** a regression test, `test_confidence_is_strategy_aware_and_does_not_saturate`,
confirming GENRE_PURIST no longer saturates.

Test count went from 17 to 50.

