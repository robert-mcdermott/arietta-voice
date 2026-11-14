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

In "unsure" mode the cloud model took the four answers the local model began by
declining (two of them the dinner question) and the rest stayed local, so
speed was unchanged. With the local model now answering ordinary questions,
the main value of a smarter model is harder questions than these.
