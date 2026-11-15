# Answer quality

How often the default local model (Gemma 4 E4B, 4-bit, on MLX) declines or gets
an ordinary question wrong, measured with `bench/answer_quality.py` over the
40 questions in [`answer_questions.json`](answer_questions.json): 24 facts with
a known answer ("Who painted the Mona Lisa?") and 16 explanations and everyday
questions ("Why do cats purr?", "How do I hard boil an egg?"). An answer counts
as declined when it says it does not have or know the answer.

## November 14, 2026: why it declined, and the fixes

The roadmap recorded that the model declined about one general question in
six. Measured on the owner's own setup (a knowledge base of workplace notes,
the default tools and persona; two rounds, 80 answers), it was far worse:

| Setup | Declined | Facts right | Median first words |
| --- | ---: | ---: | ---: |
| As shipped | 43 of 80 | 23 of 48 | 0.97 s |
| No tools at all | 1 of 80 | 43 of 48 | 0.67 s |
| Tools, knowledge off | 12 of 80 | 40 of 48 | 1.10 s |
| New persona and note wording | 6 of 80 | 44 of 48 | 1.08 s |
| Prompt reuse fix only | 30 of 80 | 33 of 48 | 1.56 s |
| **All fixes** | **2 of 80** | **48 of 48** | 1.47 s |

The model knew the answers; three things stopped it giving them:

1. **Knowledge notes told it to refuse.** Notes are found for most requests,
   usually unrelated, and came with "if the snippets do not answer the
   question, say you are not sure". So "How many legs does a spider have?"
   got "I don't have that information" whenever a workplace note was attached.
   The notes now say they may be unrelated, to use them when they answer the
   question and otherwise to answer from the model's own knowledge. An empty
   tool result likewise points general questions back to the model.
2. **The persona leaned the same way.** "If you are unsure, say so plainly"
   and "keep answers grounded in what the system can really do" read, to a
   small model, as "only answer what you were given". The default `SOUL.md`
   now tells it to answer general knowledge and everyday questions from what
   it knows, and to say so briefly only when it really does not know.
3. **Prompt reuse corrupted what the model read.** Gemma 4 E2B and E4B add a
   per-layer embedding for every token. mlx-vlm 0.7.12 computes these once for
   all the tokens a generation is given and slices them at the cache's
   absolute offset, which is wrong when a cached prompt prefix is reused: every
   new token got another token's embedding. At temperature 0, a reused prompt
   dropped the first letters of an email address it was copying from a note
   and garbled a phone number, while the same prompt processed from scratch,
   whole or in chunks of any size, was right. `llm.py` now drops the precomputed copy so
   the model computes them from each pass's own tokens. On the owner's
   knowledge questions this took correct answers from 12 of 16 to 16 of 16.

The two remaining "declines" are "What should I make for dinner tonight?",
answered with a question about what the owner likes.

The repetition penalty (1.2) was suspected too, but with the reuse fix every
local fact was right at both 1.0 and 1.2, so it stays and is now a setting
(`models.repetition_penalty`).

On the isolated shipped setup (`bench/answer_quality.py`, public starter notes
only, two rounds): 0 of 80 declined, 47 of 48 facts right, 1.05 s median to the
first words. The miss was the calculator speaking its "isn't an expression"
error when the model sent it "the hardest natural substance"; it now says
nothing for text that is not arithmetic, so the model answers.

### Limits

Questions are easy and one at a time, with no conversation history; answers
are judged by a pattern, not read for quality. Checked on the M4/32 GB Mac.

## November 14, 2026: the optional smarter model

`[smarter_model]` (off by default) hands a question to a bigger model. Measured
on the owner's setup with `gemma4:31b-cloud` served by Ollama (two rounds, 80
answers):

| Mode | Declined | Facts right | Handed over | Median first words |
| --- | ---: | ---: | ---: | ---: |
| Off (local only) | 2 of 80 | 48 of 48 | 0 | 1.47 s |
| `when = "unsure"` | 0 of 80 | 48 of 48 | 4 of 80 | 1.28 s |
| `when = "always"` (one round) | 0 of 40 | 24 of 24 | 39 of 40 | 0.77 s |

In "unsure" mode the cloud model took the four answers the local model began by
declining (two of them the dinner question) and the rest stayed local, so
speed was unchanged. With the local model now answering ordinary questions,
the main value of a smarter model is harder questions than these.

"Always" was first measured at 6.84 s to the first words: the local model began
each answer and was stopped once the smarter model took over, and stopping a
generation part way discards the cached prompt, so every turn started from
scratch. Now the local model writes only its first six tokens, enough to see
whether it starts a tool call, which finishes normally and keeps the cache;
the cloud model then answers in about 0.3 s. The one answer kept local was the
local model asking for a tool.

## November 14, 2026: notes in a conversation

Answers drawing on notes were the slowest in a live session (2.3 to 4.8 s to
first audio, against 1.1 to 1.7 s for tool answers). Two causes, measured with
`--conversation 6` (questions asked six in a row with history, as voice does):

1. **Notes went before the history.** They were placed right after the
   persona, so every question with notes, and the one after it, changed the
   start of the prompt and the model read the whole conversation again. They
   now go after the history, next to the question, and the fixed instruction on
   how to use notes moved into the system prompt, which is reused. With
   [made-up household notes](fixture_notes/) and
   [their questions](notes_questions.json) (two rounds, 24 answers, all right
   before and after), the median time to the first clause fell from 0.96 s to
   0.81 s for questions with notes and from 0.64 s to 0.54 s for the rest; the
   slowest fell from 1.38 s to 1.13 s.
2. **Notes were attached to most questions.** Hybrid search attached up to
   three notes whenever any came within the vector floor (0.3), which nearly
   every question does, and with no usable index it fell back to 40% of the
   keyword bar. The owner's index was from before the index format changed,
   so it had been ignored. A note now needs a cosine similarity of at least
   `min_similarity` (0.6) to be attached, or the full `min_score` without an
   index, and a missing or outdated index is built when the assistant starts.
   On the owner's notes, all eight questions about them scored 0.63 to 0.83 and
   no general question above 0.59.

The owner's setup, the 40 general questions six at a time (one round):

| | Notes attached | Median first clause | p95 | Facts right |
| --- | ---: | ---: | ---: | ---: |
| Before | 17 of 40 | 1.21 s | 2.89 s | 24 of 24 |
| Notes after the history only | 17 of 40 | 1.44 s | 2.70 s | 24 of 24 |
| Both fixes | 0 of 40 | 0.66 s | 1.09 s | 24 of 24 |

The owner's 16 questions about the notes stay right (16 of 16; the two
"declines" are a question with no answer in the notes, asked on purpose), at a
median of 2.08 s. The model reads about 4 ms a token on this Mac and three notes
add about 300 tokens. Attaching fewer notes was tried: with two or with one,
two answers went wrong, because the fact was in the third.

### Limits

Timings are to the first clause of text, without speech recognition or
synthesis. One round on the owner's setup; the fixture notes are short.
