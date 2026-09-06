# 📘 MODULE 14: Attention — Complete Visual + Dry-Run Edition

**Difficulty:** 🟠 Medium-Hard
**Time:** 300 minutes
**Prerequisite:** Modules 1-13 + Projects #1, #2, #3
**Tools:** Google Colab, PyTorch

---

## 📖 How to Read These Notes

- Everything in **simple English**, like teaching a 10-year-old who has never seen this before
- **Every variable has a self-explaining name** — the same name is used from start to finish
- **Real numbers at every step** so you can check with a calculator
- **Pictures everywhere** (ASCII diagrams)
- 🔍 **Dry runs** of every line of code — what goes IN, what comes OUT

---

## 📕 THE NAME LIST (used everywhere in these notes)

### 🔧 THINGS THAT GET UPDATED (the "settings" / the recipe 📖)

| Name | What it is |
|------|-----------|
| `english_word_table` | English word → numbers lookup table |
| `encoder_lstm` | the encoder's knobs and dials |
| `attention_mixer` | first half of the attention judge |
| `attention_judge` | second half — writes the final score |
| `hindi_word_table` | Hindi word → numbers lookup table |
| `decoder_lstm` | the decoder's knobs and dials |
| `word_scorer` | the final layer that scores all Hindi words |

### 📄 THINGS THAT ARE JUST RESULTS (thrown away after / today's dish 🍲)

| Name | Value in our example |
|------|---------------------|
| `memory_after_I` | `[0.12, 0.31, -0.05, 0.18]` |
| `memory_after_am` | `[0.28, 0.19, 0.34, 0.25]` |
| `memory_after_happy` | `[0.71, 0.83, 0.35, 0.66]` |
| `belt_after_happy` | `[1.12, 1.31, 0.58, 1.04]` |
| `all_memories` | all three of the above, stacked |
| `look_percent_step_1..4` | the attention percentages |
| `blended_look_step_1..4` | the blended context |
| `decoder_memory_step_1..4` | the decoder's memories |
| `decoder_belt_step_1..4` | the decoder's cell states |
| `word_scores` | the 9 raw scores |

### 📚 OUR TWO DICTIONARIES

```
  ENGLISH:  <pad>0  <unk>1  I=2  am=3  happy=4  sad=5  you=6  are=7
  HINDI:    <pad>0  <sos>1  <eos>2  main=3  khush=4  hoon=5  aap=6  ho=7  udaas=8

  INPUT  = "I am happy"                  → [2, 3, 4]
  TARGET = "<sos> main khush hoon <eos>"  → [1, 3, 4, 5, 2]
```

---

# PART A: THE PROBLEM WE ARE FIXING

## A.1 What Module 13 Did Wrong

In Module 13 we built a translator with an encoder and a decoder. The encoder read the English sentence and made three memories. But then:

```
   "I"      "am"     "happy"
    │        │         │
    ▼        ▼         ▼
  [enc 1] [enc 2]  [enc 3]
    │        │         │
    ▼        ▼         ▼
   memory_after_I  memory_after_am  memory_after_happy
       ✗                ✗                  ✓
      BIN 🗑️          BIN 🗑️            KEEP!

   Only memory_after_happy became "the context vector."
```

**In code it was ONE line:**
```python
return final_memory, final_belt      # ← all_memories was THROWN AWAY! 🗑️
```

## A.2 Why That Was Bad — Problem 1: The Squeeze 🍾

The context vector is always the **same size**, no matter how long the sentence:

```
  "I am happy"                     (3 words)  →  4 numbers   ← plenty of room ✅

  "I went to the market to buy vegetables because my
   mother asked me to and then I met my old friend..."   (50 words)
                                              →  4 numbers   ← SQUEEZED! 😰
```

```
        word 1  ─┐
        word 2  ─┤
        word 3  ─┼──▶  ╔════════════════╗  ──▶  the decoder must
        word 4  ─┤     ║  ONE fixed     ║       rebuild EVERYTHING
         ...    ─┤     ║  size vector   ║       from just this!
        word 49 ─┤     ╚════════════════╝
        word 50 ─┘            ↑
                     everything squeezes
                       through this hole! 🍾
```

> 🧒 **Imagine this:** you must summarize an ENTIRE BOOK in ONE sentence. Then someone else must rebuild the whole book from just that sentence. For a short story? Fine. For a 400-page novel? Impossible — too much is lost! 📚

## A.3 Why That Was Bad — Problem 2: The Decoder Was Blind

```
   When the decoder writes word 20, what can it see?

      ONLY the one squeezed vector. That's it.

   It CANNOT say "let me check English word 7 again."
   It only has the squished summary. 🙈
```

## A.4 The Real-World Symptom

```
  Translation quality vs sentence length (old Module 13 models):

   good  │████████████
         │███████████
         │██████████
         │████████
         │██████
         │████
   bad   │██
         └─────────────────────────────────────
          5    10   20   30   40   50   words

          Short sentences: great!
          Long sentences: worse and worse 😞
```

---

# PART B: THE BIG IDEA — A FLASHLIGHT 🔦

## B.1 The Question That Fixed Everything

> **"What if we KEEP all the memories, and let the decoder shine a flashlight on whichever English word matters right now?"**

That's it. That's attention. One idea.

## B.2 The Kid Version

```
  ❌ MODULE 13 WAY:
     Read the whole English sentence.
     Memorize it perfectly.
     Put the paper AWAY. 📄🚫
     Now write the Hindi from memory alone.

  ✅ MODULE 14 WAY:
     Read the whole English sentence.
     KEEP THE PAPER IN FRONT OF YOU. 📄👀
     As you write each Hindi word, glance back at the part you need!
```

```
  Writing "main"  (= "I")      →  eyes look at  "I"      👀
  Writing "khush" (= happy)    →  eyes look at  "happy"  👀
  Writing "hoon"  (= am)       →  eyes look at  "am"     👀
```

> 🧒 **You don't need to memorize everything perfectly — you can just LOOK BACK!** That is the entire invention. 🎉

## B.3 Old Way vs New Way — Side By Side

```
  ═══════════ MODULE 13 (no attention) ═══════════

     memory_after_I        memory_after_am     memory_after_happy
       🗑️ thrown             🗑️ thrown              ✓ kept
                                                       │
                                                       ▼
                                                   [DECODER]
     ← the decoder only ever sees ONE thing


  ═══════════ MODULE 14 (with attention) ═══════════

     memory_after_I        memory_after_am     memory_after_happy
        ✓ kept                ✓ kept                ✓ kept
          16.9%                 20.6%                 62.5%
          ▂▂                    ▂▂▂                   ████
           └────────────────────┼────────────────────┘
                                ▼
                      [ BLEND THEM BY WEIGHT ]
                                │
                                ▼
                            [DECODER]
     ← the decoder sees a CUSTOM MIX, fresh every step! 🥤
```

## B.4 The Single Most Important Difference

```
   ╔══════════════════════════════════════════════════════════════╗
   ║  MODULE 13:  ONE context vector  →  used for ALL steps   😴  ║
   ║  MODULE 14:  a FRESH context     →  recomputed EVERY step 🔥 ║
   ╚══════════════════════════════════════════════════════════════╝
```

---

# PART C: THE 4 STEPS OF ATTENTION

Attention always does exactly 4 things. Learn these 4 and you know attention.

```
  ┌────────────────────────────────────────────────────────────────┐
  │  STEP 1 — SCORE                                                 │
  │     "How useful is each English word to me RIGHT NOW?"          │
  │     → gives one raw number per English word                     │
  ├────────────────────────────────────────────────────────────────┤
  │  STEP 2 — SOFTMAX                                               │
  │     "Turn those into percentages that add to 100%"              │
  │     → gives look_percent                                        │
  ├────────────────────────────────────────────────────────────────┤
  │  STEP 3 — BLEND                                                 │
  │     "Mix the memories using those percentages"                  │
  │     → gives blended_look (fresh, for THIS step only!)           │
  ├────────────────────────────────────────────────────────────────┤
  │  STEP 4 — USE IT                                                │
  │     "Combine with my own memory and predict the Hindi word"     │
  └────────────────────────────────────────────────────────────────┘
```

## 🧒 The Smoothie Analogy 🥤

```
  You are making juice:

  STEP 1 — decide how much you like each fruit      (score)
  STEP 2 — turn that into percentages: 60% mango,
           30% apple, 10% lemon                      (softmax)
  STEP 3 — pour them into the blender in those
           exact amounts and blend                   (blend)
  STEP 4 — drink it!                                 (use it)

  DIFFERENT RECIPE EACH TIME = DIFFERENT JUICE EACH TIME 🥤
```

---

# PART D: STEP 1 — THE SCORE (the dot product)

## D.1 What a Score Means

The score answers one question: **"how well do these two things match?"**

The simplest way to measure "matching" is the **dot product** — multiply matching numbers and add them all up.

## D.2 A Tiny Example (2 numbers, so you can SEE it)

Say the decoder is currently "looking for" something described by `[1.0, 0.2]`.

```
  ┌──────────────────────────────────────────────────────────────┐
  │ memory A = [ 0.9,  0.1]                                       │
  │   score = (1.0 × 0.9) + (0.2 × 0.1)                          │
  │         =     0.90     +     0.02      =  0.92               │
  │   ✅ HIGH — they point the SAME direction!                     │
  ├──────────────────────────────────────────────────────────────┤
  │ memory B = [ 0.2,  0.9]                                       │
  │   score = (1.0 × 0.2) + (0.2 × 0.9)                          │
  │         =     0.20     +     0.18      =  0.38               │
  │   😐 LOW — they point different ways                          │
  ├──────────────────────────────────────────────────────────────┤
  │ memory C = [-0.8,  0.3]                                       │
  │   score = (1.0 × -0.8) + (0.2 × 0.3)                         │
  │         =     -0.80     +     0.06     = -0.74               │
  │   ❌ NEGATIVE — they point OPPOSITE ways!                      │
  └──────────────────────────────────────────────────────────────┘
```

## D.3 The Dot Product Is a "Do We Agree?" Meter 📏

```
  BIG POSITIVE  →  "yes, we're pointing the same way!"   👍
  NEAR ZERO     →  "we're unrelated"                     😐
  NEGATIVE      →  "we're opposites!"                    👎
```

> 🧒 **Imagine two people pushing a box** 📦:
> - Pushing the SAME direction → they help each other → **big number**
> - Pushing SIDEWAYS to each other → no help → **zero**
> - Pushing OPPOSITE ways → they fight → **negative**

## D.4 Doing It With Our Real Numbers (4 numbers each)

At decoder step 1, the query is `memory_after_happy` = `[0.71, 0.83, 0.35, 0.66]` (we'll explain WHY in Part H).

```
  ── score for memory_after_I = [0.12, 0.31, -0.05, 0.18] ──

     0.71 × 0.12  =  0.0852
     0.83 × 0.31  =  0.2573
     0.35 × -0.05 = -0.0175
     0.66 × 0.18  =  0.1188
                    ────────
        score_I  =   0.4438


  ── score for memory_after_am = [0.28, 0.19, 0.34, 0.25] ──

     0.71 × 0.28  =  0.1988
     0.83 × 0.19  =  0.1577
     0.35 × 0.34  =  0.1190
     0.66 × 0.25  =  0.1650
                    ────────
        score_am =   0.6405


  ── score for memory_after_happy = [0.71, 0.83, 0.35, 0.66] ──
                    (comparing it with ITSELF!)

     0.71 × 0.71  =  0.5041
     0.83 × 0.83  =  0.6889
     0.35 × 0.35  =  0.1225
     0.66 × 0.66  =  0.4356
                    ────────
     score_happy  =  1.7511    ← the biggest!
```

## D.5 ⚠️ A REAL WEAKNESS You Should Know About

**"happy" got the biggest score. Why?** Because we compared `memory_after_happy` with **itself**, and anything dotted with itself gives a big number!

```
   The query IS memory_after_happy, so:
   score_happy = memory_after_happy · memory_after_happy
               = "how similar is this to itself?"  =  VERY! 😅
```

> 🧒 **Like a singing contest where one judge is also a contestant.** Of course they vote for themselves! You need a **fair, trained judge**. 👨‍⚖️

**How real models fix this:**

| Fix | How it works |
|-----|--------------|
| **Bahdanau (what our code uses)** | a small LEARNED network scores the pair, so it can learn "don't just favour yourself" |
| **Transformers (Module 15)** | project into two DIFFERENT learned versions, so a vector is never compared to its raw self |

**And after training, the weights DO become sensible** — you'll see them shift in Part L!

---

# PART E: STEP 2 — SOFTMAX → THE LOOK PERCENTAGES

## E.1 Why We Need It

Raw scores can be anything: `0.4438`, `0.6405`, `1.7511`. We need **percentages that add to 100%**.

Softmax does exactly that. (It's the SAME softmax from Module 9's temperature lesson! 🎯)

## E.2 The Full Calculation — Check It With a Calculator!

```
  scores = [0.4438, 0.6405, 1.7511]

  ── STEP A: take e^score for each ──

     e^0.4438  =  1.5587
     e^0.6405  =  1.8974
     e^1.7511  =  5.7605
                  ───────
        SUM    =  9.2166

  ── STEP B: divide each by the sum ──

     look_percent for I     = 1.5587 / 9.2166 = 0.1691  →  16.9%
     look_percent for am    = 1.8974 / 9.2166 = 0.2059  →  20.6%
     look_percent for happy = 5.7605 / 9.2166 = 0.6250  →  62.5%
                                                           ──────
                                                 TOTAL  =  100.0%  ✓
```

## E.3 Picture It

```
   memory_after_I      16.9%  ███████
   memory_after_am     20.6%  ████████
   memory_after_happy  62.5%  █████████████████████████
                              └──── always adds to 100% ────┘
```

> 🧒 **These are the LOOK PERCENTAGES** — how much of your looking goes to each word. Like sharing 100 candies among 3 friends: 17 to one, 21 to another, 62 to the last. You always give away exactly 100! 🍬

## E.4 Why `e^x` and Not Just Divide by the Sum?

```
  BRUTE FORCE WAY (just divide by sum):
     scores = [0.4438, 0.6405, 1.7511],  sum = 2.8354
     → 15.7%, 22.6%, 61.8%

  PROBLEM 1: what if a score is NEGATIVE? You'd get a negative percent! ❌
  PROBLEM 2: differences don't get emphasized enough

  THE e^x WAY:
     e^x is ALWAYS positive ✅  (no negative percentages possible!)
     e^x makes the big scores stand out MORE ✅
```

---

# PART F: STEP 3 — BLEND (the weighted sum)

## F.1 The Formula

```
  blended_look = (look_percent_I     × memory_after_I)
               + (look_percent_am    × memory_after_am)
               + (look_percent_happy × memory_after_happy)
```

## F.2 Doing It Number By Number

```
  0.1691 × [0.12, 0.31, -0.05, 0.18] = [0.0203, 0.0524, -0.0085, 0.0304]
  0.2059 × [0.28, 0.19,  0.34, 0.25] = [0.0577, 0.0391,  0.0700, 0.0515]
  0.6250 × [0.71, 0.83,  0.35, 0.66] = [0.4438, 0.5188,  0.2188, 0.4125]
                                        ─────────────────────────────────
  blended_look_step_1                = [0.5218, 0.6103,  0.2803, 0.4944]
```

**Let me show one position completely** (the first number):
```
  0.1691 × 0.12  =  0.02029     ← from memory_after_I
  0.2059 × 0.28  =  0.05765     ← from memory_after_am
  0.6250 × 0.71  =  0.44375     ← from memory_after_happy
                    ─────────
                     0.52169   →  0.5218  ✓
```

> 🧒 **It's a smoothie!** 🥤 Pour in 17% of the first memory, 21% of the second, 62% of the third, and blend. The result tastes mostly like the third one — because it got the biggest share!

## F.3 ⭐ A Fresh Blend Every Step

```
  blended_look_step_1 = [0.5218, 0.6103, 0.2803, 0.4944]   ← mostly "happy"
  blended_look_step_2 = [0.4820, 0.5660, 0.2630, 0.4600]   ← different!
  blended_look_step_3 = [0.4660, 0.5470, 0.2580, 0.4460]   ← different again!
  blended_look_step_4 = [0.4580, 0.5380, 0.2550, 0.4390]   ← different again!
```

**In Module 13, all four steps got the SAME vector. Now each step gets what it needs!** 🎉

---

# PART G: STEP 4 — USE IT

## G.1 The Blend Is Used TWICE!

```
  USE 1 — it goes INTO the decoder LSTM (glued to the previous word):
     lstm_input = [ previous word's numbers ; blended_look ]

  USE 2 — it goes INTO the final scorer (glued to the new memory):
     for_scoring = [ new decoder memory ; blended_look ]
```

> 🧒 **Why twice?** Once to help the decoder THINK (the LSTM), once to help it DECIDE (the scorer). Two chances to use the information! 🧠👨‍⚖️
>
> 📌 Remember this — it matters a lot in backprop (Part L)!

---

# PART H: ⭐ THE BATON — What Is the Decoder's FIRST Memory?

## H.1 The Question

```
  At decoder step 1, we do the dot product with "decoder_memory".
  But we haven't written ANY Hindi words yet!

  So... what IS decoder_memory at step 1? 🤔
```

## H.2 THE ANSWER

```
   ╔═══════════════════════════════════════════════════════════════╗
   ║  decoder_memory_step_0  =  memory_after_happy                  ║
   ║  decoder_belt_step_0    =  belt_after_happy                    ║
   ║                                                                 ║
   ║  The decoder starts with the ENCODER'S LAST MEMORY!             ║
   ╚═══════════════════════════════════════════════════════════════╝
```

**Why?** Because `memory_after_happy` means *"I have read the whole English sentence."* That's the perfect starting thought for someone about to translate it!

> 🧒 **The relay race!** 🏃‍♂️➡️🏃‍♀️ The encoder runs the first leg and hands over the baton. The baton is `memory_after_happy`. The decoder doesn't start from nothing — it starts holding everything the encoder learned!

## H.3 So `memory_after_happy` Has TWO JOBS

```
                  "I"      "am"     "happy"
                   │         │         │
                   ▼         ▼         ▼
              [enc 1] ──▶ [enc 2] ──▶ [enc 3]
                   │         │         │
                   ▼         ▼         ▼
          memory_after_I  memory_after_am  memory_after_happy
                   │         │              │       │
                   └─────────┴──────────────┘       │
                             │                      │
                    JOB 1: all three are      JOB 2: this one is
                    attention targets 🔦      also the BATON 🏃
                             │                      │
                             ▼                      ▼
                       ┌──────────────────────────────────┐
                       │  DECODER STEP 1                  │
                       │  starting memory = the baton     │
                       │  looks at all three              │
                       └──────────────────────────────────┘
```

## H.4 What Changes Each Step, What Doesn't

```
  ┌──────────────────────────┬────────────┬──────────────────────────────┐
  │ all_memories             │ SAME ♻️     │ computed once, reused forever │
  │ decoder_memory           │ CHANGES 🔄 │ the LSTM updates it each step │
  │ the word going in        │ CHANGES 🔄 │ previous output / correct word│
  │ look_percent             │ CHANGES 🔄 │ because the memory changed!   │
  │ blended_look             │ CHANGES 🔄 │ because the percentages did!  │
  └──────────────────────────┴────────────┴──────────────────────────────┘
```

> 🧒 **The encoder memories are like a book lying open on your desk** 📖 — always the same page. **Your understanding (decoder_memory) changes** as you write. So each time you glance at the book, **your eyes land somewhere different!** 👀

---

# PART I: ⭐ THE FULL FORWARD PASS — "I am happy"

## I.1 The Setup

```
  INPUT  = "I am happy"                  → [2, 3, 4]
  TARGET = "<sos> main khush hoon <eos>"  → [1, 3, 4, 5, 2]
```

## I.2 The Encoder (runs ONCE)

```
  "I"     (2) → english_word_table → [0.2,  0.5, -0.1, 0.3] → encoder_lstm
                                    → memory_after_I     = [0.12, 0.31, -0.05, 0.18]

  "am"    (3) → english_word_table → [0.4, -0.2,  0.6, 0.1] → encoder_lstm
                                    → memory_after_am    = [0.28, 0.19,  0.34, 0.25]

  "happy" (4) → english_word_table → [0.9,  0.7,  0.2, 0.8] → encoder_lstm
                                    → memory_after_happy = [0.71, 0.83,  0.35, 0.66]
                                      belt_after_happy   = [1.12, 1.31,  0.58, 1.04]
```

**All three memories are KEPT** in a stack we call `all_memories`. 📚

## I.3 DECODER STEP 1 — generating "main"

```
  ┌─────────────────────────────────────────────────────────────────┐
  │ MEMORY IN:  decoder_memory_step_0 = [0.71, 0.83, 0.35, 0.66]   │
  │             (this IS memory_after_happy — the baton! 🏃)         │
  │ WORD IN:    <sos>  →  hindi_word_table  →  [0.10,0.10,0.10,0.10]│
  ├─────────────────────────────────────────────────────────────────┤
  │ SCORES:       [0.4438, 0.6405, 1.7511]                          │
  │ LOOK PERCENT: [ 16.9%,  20.6%,  62.5%]                          │
  │ BLENDED LOOK: [0.5218, 0.6103, 0.2803, 0.4944]                  │
  ├─────────────────────────────────────────────────────────────────┤
  │ LSTM INPUT = [0.10,0.10,0.10,0.10, 0.5218,0.6103,0.2803,0.4944] │
  │               └── the word ────┘   └──── the blend ──────────┘  │
  │ decoder_memory_step_1 = [0.55, 0.61, 0.22, 0.48]                │
  │ decoder_belt_step_1   = [0.88, 0.97, 0.36, 0.79]                │
  ├─────────────────────────────────────────────────────────────────┤
  │ SCORER INPUT = [0.55,0.61,0.22,0.48, 0.5218,0.6103,0.2803,0.4944]│
  │ word_scores (9 numbers) → softmax → "main" wins with 40% ✅     │
  └─────────────────────────────────────────────────────────────────┘
```

## I.4 DECODER STEP 2 — generating "khush"

```
  ┌─────────────────────────────────────────────────────────────────┐
  │ MEMORY IN:  decoder_memory_step_1 = [0.55, 0.61, 0.22, 0.48]   │
  │             ⭐ DIFFERENT from step 1! The LSTM changed it!       │
  │ WORD IN:    "main"  →  [0.30, 0.60, 0.10, 0.40]                │
  ├─────────────────────────────────────────────────────────────────┤
  │ SCORES CALCULATION:                                              │
  │   score_I     = 0.55×0.12 + 0.61×0.31 + 0.22×(-0.05) + 0.48×0.18│
  │               = 0.0660 + 0.1891 - 0.0110 + 0.0864 = 0.3305      │
  │   score_am    = 0.55×0.28 + 0.61×0.19 + 0.22×0.34 + 0.48×0.25   │
  │               = 0.1540 + 0.1159 + 0.0748 + 0.1200 = 0.4647      │
  │   score_happy = 0.55×0.71 + 0.61×0.83 + 0.22×0.35 + 0.48×0.66   │
  │               = 0.3905 + 0.5063 + 0.0770 + 0.3168 = 1.2906      │
  ├─────────────────────────────────────────────────────────────────┤
  │ SOFTMAX:  e^0.3305=1.3917, e^0.4647=1.5915, e^1.2906=3.6350     │
  │           SUM = 6.6182                                           │
  │ LOOK PERCENT: [ 21.0%,  24.0%,  54.9%]   ← MOVED! 🔦            │
  │ BLENDED LOOK: [0.4820, 0.5660, 0.2630, 0.4600]                  │
  ├─────────────────────────────────────────────────────────────────┤
  │ decoder_memory_step_2 = [0.42, 0.53, 0.31, 0.39]                │
  │ → "khush" wins with 30% ✅                                       │
  └─────────────────────────────────────────────────────────────────┘
```

## I.5 DECODER STEP 3 — generating "hoon"

```
  ┌─────────────────────────────────────────────────────────────────┐
  │ MEMORY IN:  decoder_memory_step_2 = [0.42, 0.53, 0.31, 0.39]   │
  │ WORD IN:    "khush"                                              │
  ├─────────────────────────────────────────────────────────────────┤
  │   score_I     = 0.42×0.12 + 0.53×0.31 + 0.31×(-0.05) + 0.39×0.18│
  │               = 0.0504 + 0.1643 - 0.0155 + 0.0702 = 0.2694      │
  │   score_am    = 0.42×0.28 + 0.53×0.19 + 0.31×0.34 + 0.39×0.25   │
  │               = 0.1176 + 0.1007 + 0.1054 + 0.0975 = 0.4212      │
  │   score_happy = 0.42×0.71 + 0.53×0.83 + 0.31×0.35 + 0.39×0.66   │
  │               = 0.2982 + 0.4399 + 0.1085 + 0.2574 = 1.1040      │
  ├─────────────────────────────────────────────────────────────────┤
  │ SOFTMAX:  e^0.2694=1.3092, e^0.4212=1.5238, e^1.1040=3.0163     │
  │           SUM = 5.8493                                           │
  │ LOOK PERCENT: [ 22.4%,  26.0%,  51.6%]   ← MOVED again! 🔦      │
  │ BLENDED LOOK: [0.4660, 0.5470, 0.2580, 0.4460]                  │
  ├─────────────────────────────────────────────────────────────────┤
  │ decoder_memory_step_3 = [0.35, 0.47, 0.28, 0.33]                │
  │ → "hoon" wins with 35% ✅                                        │
  └─────────────────────────────────────────────────────────────────┘
```

## I.6 DECODER STEP 4 — generating `<eos>`

```
  ┌─────────────────────────────────────────────────────────────────┐
  │ MEMORY IN:  decoder_memory_step_3 = [0.35, 0.47, 0.28, 0.33]   │
  │ WORD IN:    "hoon"                                               │
  │ LOOK PERCENT: [ 23.0%,  27.0%,  50.0%]                          │
  │ BLENDED LOOK: [0.4580, 0.5380, 0.2550, 0.4390]                  │
  │ decoder_memory_step_4 = [0.29, 0.41, 0.25, 0.30]                │
  │ → <eos> wins with 45% ✅  🏁 STOP!                               │
  └─────────────────────────────────────────────────────────────────┘
```

## I.7 The Complete Result

```
   INPUT:   "I am happy"
   OUTPUT:  main  khush  hoon   =  "मैं खुश हूँ"  ✅ CORRECT!
```

## I.8 Watch the Flashlight MOVE 🔦

```
                    "I"      "am"    "happy"
   STEP 1:         16.9%    20.6%    62.5%
   STEP 2:         21.0%    24.0%    54.9%
   STEP 3:         22.4%    26.0%    51.6%
   STEP 4:         23.0%    27.0%    50.0%
                     ▲        ▲        ▼
                  rising   rising   falling

   The percentages CHANGE every step — because the memory changed!
```

---

# PART J: THE ALIGNMENT GRID — The Beautiful Payoff

## J.1 What a Trained Model Looks Like

After training, the percentages become SHARP and meaningful:

```
                 "I"      "am"    "happy"
              ┌────────┬────────┬────────┐
      main    │  85%   │  10%   │   5%   │   ← looks at "I"
              │ ██████ │   ▌    │   ▏    │
              ├────────┼────────┼────────┤
      khush   │   5%   │  10%   │  85%   │   ← looks at "happy"!
              │   ▏    │   ▌    │ ██████ │
              ├────────┼────────┼────────┤
      hoon    │  10%   │  75%   │  15%   │   ← looks at "am"!
              │   ▌    │ █████  │   ▊    │
              └────────┴────────┴────────┘
```

## J.2 🎉 LOOK AT THE CROSSING PATTERN!

Follow the dark cells:
```
   main   →  "I"        (position 1)
   khush  →  "happy"    (position 3)   ← JUMPED to the end!
   hoon   →  "am"       (position 2)   ← came BACK to the middle!

   The path CROSSES itself: 1 → 3 → 2
```

## J.3 Why This Is Amazing

**Remember Module 13's Problem #1?** Word-by-word translation failed because Hindi puts the verb LAST.

```
  ENGLISH:  I    am    happy       (subject - verb  - adjective)
  HINDI:    मैं   खुश   हूँ          (subject - adjective - verb!)
```

**Attention figured that out BY ITSELF!** 🤯

> 🧒 Nobody programmed "Hindi verbs go last." The model learned it from examples, and you can literally SEE it in the grid. 🎯

## J.4 Why the Grid Is Useful

```
   🧒 The alignment grid is a WINDOW INTO THE MODEL'S MIND. 🔍

   You can watch where it was "looking" for every word it wrote.
   Before attention, networks were black boxes. Now you can peek inside!
```

**That's why the code returns `look_percent`** — we don't need it for the math, but we want to draw this grid! 📊

---

# PART K: THREE WAYS TO COMPUTE THE SCORE

| Name | Formula | 🧒 What it means | Has learned weights? |
|------|---------|-----------------|:--------------------:|
| **Dot** (Luong) | `query · memory` | just multiply and add — fastest | ❌ no |
| **Multiplicative** | `query · W · memory` | multiply through a learned grid first | ✅ yes |
| **Additive** (Bahdanau) | `judge(tanh(mixer[query ; memory]))` | a tiny network decides the match | ✅ yes |

```
  DOT — simplest:                     ADDITIVE — a mini brain:

    query ─┐                            query  ─┐
           ├──▶ multiply & add            memory ┤──▶ [ tape together ]
   memory ─┘         │                           │
                     ▼                           ▼
                   score                  [ attention_mixer + tanh ]
                                                 │
    ⚡ fast, no weights                          ▼
    but can favour itself                 [ attention_judge → 1 number ]
    (see Part D.5!)                              │
                                                 ▼  score
                                          🧠 slower but SMARTER
```

> 🧒 **Dot product** is like judging a friendship by how many hobbies you share — quick, but crude. **Additive** is like a matchmaker who has learned from thousands of friendships what really matters. Slower, but much smarter! 💘

**Where each is used:**
- **Additive (Bahdanau, 2014)** — the ORIGINAL attention paper. **Our code uses this.**
- **Dot product (scaled)** — what **Transformers** use! (Module 15) 🚀

---

# PART L: ⭐⭐ BACKPROPAGATION — THE COMPLETE STEP-BY-STEP

This is the hardest part. Let's go very slowly.

## L.1 First — The Question That Blocks Everyone

> **"How do we UPDATE the memory?"**

```
   ╔════════════════════════════════════════════════════════╗
   ║  ANSWER:  WE DON'T! WE NEVER UPDATE ANY MEMORY.        ║
   ║                                                          ║
   ║  ❌ Memory is not a setting.                             ║
   ║  ✅ Memory is a RESULT — an answer.                      ║
   ╚════════════════════════════════════════════════════════╝
```

> 🧒 **Imagine a calculator.** 🔢 You type `3 × 4` and get `12`. If the 12 is wrong, do you "fix the 12"? **NO!** You fix the numbers you typed in. The 12 just gets thrown away and recalculated next time.
>
> **Memory is the 12.** Blame passes THROUGH it to reach the things that made it. 🎯

## L.2 The Two Kinds of Things

```
  ┌───────────────────────────────┬───────────────────────────────┐
  │ SETTINGS (kept and updated)   │ RESULTS (computed, thrown away)│
  │ 📥 they COLLECT gradient       │ 📨 they PASS gradient along    │
  ├───────────────────────────────┼───────────────────────────────┤
  │ ✅ english_word_table          │ ❌ memory_after_I              │
  │ ✅ encoder_lstm                │ ❌ memory_after_am             │
  │ ✅ attention_mixer             │ ❌ memory_after_happy          │
  │ ✅ attention_judge             │ ❌ all look_percent            │
  │ ✅ hindi_word_table            │ ❌ all blended_look            │
  │ ✅ decoder_lstm                │ ❌ all decoder_memory          │
  │ ✅ word_scorer                 │ ❌ all word_scores             │
  └───────────────────────────────┴───────────────────────────────┘
     these survive between sentences   these are rebuilt every time
```

## L.3 🧒 The Cooking Analogy 🍳

```
   SETTINGS  =  your RECIPE 📖      ← you improve this over time
   RESULTS   =  today's DISH 🍲     ← you eat it, then it's gone

   THE CAKE IS TOO FLAT:            (loss = 0.9922)

   YOU INVESTIGATE:                 (backward pass)
      "the cake was flat"        → the cake is a RESULT, can't fix it 📨
      "so the batter was wrong"  → batter is a RESULT too 📨
      "so the RECIPE said too little baking powder"  → RECIPE! 📥 FIX IT ✅

   YOU FIX THE RECIPE:              (optimizer.step())
      baking powder: 1 tsp → 1.2 tsp ✅

   YOU THROW AWAY THE CAKE 🗑️. Tomorrow you bake a better one! 🍰
```

## L.4 The Loss (all 4 steps)

Each step is a "which of 9 Hindi words?" question. Loss = `−ln(probability we gave the correct word)`:

```
  STEP  CORRECT   WE SAID   LOSS
  ────  ───────   ───────   ────────────────────
   1     main      40%      −ln(0.40) = 0.9163
   2     khush     30%      −ln(0.30) = 1.2040   ← worst! least sure
   3     hoon      35%      −ln(0.35) = 1.0498
   4     <eos>     45%      −ln(0.45) = 0.7985   ← best
                            ──────────────────
                   TOTAL  =  3.9686
                   AVERAGE = 3.9686 ÷ 4 = 0.9922
```

**Sanity check:** random guessing with 9 words = `−ln(1/9) = 2.197`. We're at 0.99, so the model has learned something — but there's room to improve. 📉

## L.5 ⭐ THE ONE FORMULA THAT MAKES BACKPROP EASY

When you combine **softmax + cross-entropy loss**, the gradient on the scores becomes ridiculously simple:

```
   ╔══════════════════════════════════════════════════════════════╗
   ║      grad_on_scores  =  what_we_predicted  −  the_truth       ║
   ╚══════════════════════════════════════════════════════════════╝
```

### 🔍 Doing it for STEP 4 (correct word = `<eos>`, index 2)

```
  ── WHAT WE PREDICTED (9 probabilities, adding to 1.00) ──

     index 0  <pad>    0.02
     index 1  <sos>    0.03
     index 2  <eos>    0.45   ← the CORRECT one
     index 3  main     0.10
     index 4  khush    0.12
     index 5  hoon     0.08
     index 6  aap      0.07
     index 7  ho       0.08
     index 8  udaas    0.05
                       ────
                       1.00 ✓

  ── THE TRUTH (a 1 on the correct word, 0 everywhere else) ──

     [0, 0, 1, 0, 0, 0, 0, 0, 0]
            ↑ index 2 = <eos>

  ── SUBTRACT! ──

     index 0  <pad>:  0.02 − 0 =  +0.02
     index 1  <sos>:  0.03 − 0 =  +0.03
     index 2  <eos>:  0.45 − 1 =  −0.55   ⭐ NEGATIVE!
     index 3  main:   0.10 − 0 =  +0.10
     index 4  khush:  0.12 − 0 =  +0.12
     index 5  hoon:   0.08 − 0 =  +0.08
     index 6  aap:    0.07 − 0 =  +0.07
     index 7  ho:     0.08 − 0 =  +0.08
     index 8  udaas:  0.05 − 0 =  +0.05

  grad_on_scores = [0.02, 0.03, −0.55, 0.10, 0.12, 0.08, 0.07, 0.08, 0.05]
```

### 🧒 What Do These Numbers MEAN?

```
   NEGATIVE gradient  →  "this score was TOO LOW,  RAISE it!"  ⬆️
   POSITIVE gradient  →  "this score was TOO HIGH, LOWER it!"  ⬇️

   <eos> got −0.55  →  raise it a LOT!  (it's the right answer!)
   khush got +0.12  →  lower it a bit   (wrong, but we liked it too much)
   <pad>  got +0.02 →  lower it a tiny bit (we barely liked it anyway)
```

> 🧒 **It's a tug-of-war!** 🪢 The correct word gets pulled UP hard. All wrong words get pushed DOWN, and the ones we wrongly liked most get pushed hardest.
>
> **And the SIZE of the pull IS how wrong we were!**
> ```
>    said <eos> with 90% →  0.90 − 1 = −0.10   gentle nudge
>    said <eos> with 45% →  0.45 − 1 = −0.55   big shove! 💪
>    said <eos> with 10% →  0.10 − 1 = −0.90   HUGE shove! 💥
> ```

---

## L.6 BACKWARD STEP 4 (the LAST word, `<eos>`) — Every Sub-Step

We go backward, so the LAST word is FIRST.

### 🔹 4a. What went INTO the word_scorer?

```
  word_scorer took 8 numbers:
     [ decoder_memory_step_4 ; blended_look_step_4 ]

   = [0.29, 0.41, 0.25, 0.30, 0.458, 0.538, 0.255, 0.439]
      └── the memory (4) ──┘  └── the blend (4) ────────┘
   position:  0     1     2     3     4      5      6      7

  And it produced 9 scores. So:
     word_scorer.weight  is a grid of 9 rows × 8 columns  (72 numbers)
     word_scorer.bias    is 9 numbers
```

### 🔹 4b. Gradient for the BIAS — the easiest one!

```
   ╔═══════════════════════════════════════════╗
   ║   grad_bias[row]  =  grad_on_scores[row]  ║
   ╚═══════════════════════════════════════════╝

   grad for bias[2] (<eos>)  = −0.55
   grad for bias[4] (khush)  = +0.12
   ... and so on for all 9 rows
```

> 🧒 The bias is ADDED straight to the score. So blaming it is a **direct copy** — if the score should go up by 0.55, the bias should go up by 0.55. One-to-one! ✅

### 🔹 4c. Gradient for the WEIGHTS — multiply two things

```
   ╔══════════════════════════════════════════════════════════════╗
   ║  grad_weight[row][col] = grad_on_scores[row] × input[col]    ║
   ╚══════════════════════════════════════════════════════════════╝
```

**🔍 Let's do ONE number completely:**
```
  I want the gradient for word_scorer.weight[2][0]
                                          ↑  ↑
                              row 2 = <eos>  column 0 = first input number

  grad_on_scores[2] = −0.55         (the <eos> blame)
  input[0]          =  0.29         (the first input number)

  grad_weight[2][0] = −0.55 × 0.29 = −0.1595
```

**🔍 Now the WHOLE `<eos>` row (all 8 columns):**
```
    col 0:  −0.55 × 0.290 = −0.1595
    col 1:  −0.55 × 0.410 = −0.2255
    col 2:  −0.55 × 0.250 = −0.1375
    col 3:  −0.55 × 0.300 = −0.1650
    col 4:  −0.55 × 0.458 = −0.2519
    col 5:  −0.55 × 0.538 = −0.2959   ← biggest push (input was biggest)
    col 6:  −0.55 × 0.255 = −0.1403
    col 7:  −0.55 × 0.439 = −0.2415

  ALL NEGATIVE  →  all these weights will INCREASE  →  <eos> scores higher! ✅
```

**🔍 And a WRONG word's row — "khush" (row 4, grad +0.12):**
```
    col 0:  +0.12 × 0.290 = +0.0348
    col 1:  +0.12 × 0.410 = +0.0492
    col 2:  +0.12 × 0.250 = +0.0300
    ...

  ALL POSITIVE  →  these weights will DECREASE  →  khush scores lower! ✅
```

> 🧒 **Why multiply by the input?** 🤔
>
> Because a weight only matters if its input was BIG!
>
> Imagine a volume knob connected to a **silent** speaker (input = 0). Turning the knob does nothing — so don't bother adjusting it. But a knob on a **loud** speaker (input = 0.538) matters a lot — adjust that one! 🔊
>
> **Big input → big blame. Zero input → zero blame.** 🎯

### 🔹 4d. Pass the blame BACKWARD to the 8 inputs

Now we ask: "what should those 8 input numbers have been?"

```
   ╔═══════════════════════════════════════════════════════════════╗
   ║  grad_input[col] = SUM over all 9 rows of                      ║
   ║                    ( weight[row][col] × grad_on_scores[row] )  ║
   ╚═══════════════════════════════════════════════════════════════╝
```

**🔍 One column, fully worked (column 0):**
```
  Using the CURRENT weight values in column 0:

   row 0 <pad>:   weight[0][0]= 0.11 × grad  0.02 =  0.0022
   row 1 <sos>:   weight[1][0]= 0.08 × grad  0.03 =  0.0024
   row 2 <eos>:   weight[2][0]=−0.42 × grad −0.55 =  0.2310  ← biggest!
   row 3 main:    weight[3][0]= 0.31 × grad  0.10 =  0.0310
   row 4 khush:   weight[4][0]= 0.27 × grad  0.12 =  0.0324
   row 5 hoon:    weight[5][0]= 0.19 × grad  0.08 =  0.0152
   row 6 aap:     weight[6][0]= 0.14 × grad  0.07 =  0.0098
   row 7 ho:      weight[7][0]=−0.09 × grad  0.08 = −0.0072
   row 8 udaas:   weight[8][0]= 0.05 × grad  0.05 =  0.0025
                                                    ────────
                                    grad_input[0] =  0.3193 ≈ 0.31
```

**Doing that for all 8 columns:**
```
  grad_input = [0.31, −0.18, 0.22, −0.09,  0.24, −0.15, 0.19, −0.07]
                └─ for decoder_memory_step_4 ─┘ └─ for blended_look_step_4 ─┘

  grad_decoder_memory_step_4 = [0.31, −0.18, 0.22, −0.09]
  grad_blended_look_step_4   = [0.24, −0.15, 0.19, −0.07]
```

> 🧒 **We SPLIT the message in two**, because the input was two things taped together! First 4 numbers = advice for the memory. Last 4 = advice for the blend. ✂️

### ⚠️ 4d-WARNING: We Do NOT Change the Memory!

```
  decoder_memory_step_4 was  [0.29,  0.41, 0.25,  0.30]
  its gradient is            [0.31, −0.18, 0.22, −0.09]

  ❌ We do NOT do:  memory − learning_rate × gradient
  ✅ We just CARRY the message further backward!
```

> 🧒 The gradient on the memory is a **letter being delivered** 📨, not a change being made. It says *"whoever MADE this memory, you should have made it differently."* We carry the letter to whoever made it — the decoder_lstm! 📮

### 🔹 4e. `grad_decoder_memory_step_4` → into the decoder_lstm

The LSTM made `decoder_memory_step_4` out of **three** things:
```
   decoder_memory_step_3      (the previous memory)
   decoder_belt_step_3        (the previous belt)
   its input = [ hindi_word_table["hoon"] ; blended_look_step_4 ]
```

So the blame splits into **four** places:
```
  ✅ KEEP  → grad for decoder_lstm weights                (a setting! 📥)
  ✅ KEEP  → grad for hindi_word_table["hoon"] row        (a setting! 📥)
  📨 PASS  → grad_decoder_memory_step_3 += [0.19, −0.10, 0.14, −0.06]
  📨 PASS  → grad_blended_look_step_4   += (a bit more!)
```

### ⚠️ Why `+=` and not `=`?

**Because these things get blame from MORE THAN ONE place!**

```
  blended_look_step_4 was used TWICE in the forward pass (Part G!):
     USE 1 → it went into the word_scorer   → gave grad [0.24, −0.15, 0.19, −0.07]
     USE 2 → it went into the LSTM input    → gives MORE grad

  So we ADD both:
     grad_blended_look_step_4 = [from use 1] + [from use 2]
```

> 🧒 **If you lend your bike to two friends and BOTH scratch it, you count BOTH scratches!** 🚲 A thing used twice gets blamed twice, and we add them up. ➕

### 🔹 4f. `grad_blended_look_step_4` → into attention

Remember how the blend was made?
```
  blended_look_step_4 = 0.230 × memory_after_I
                      + 0.270 × memory_after_am
                      + 0.500 × memory_after_happy
                        └── the look percentages at step 4 ──┘
```

Going backward, the blame splits by those SAME percentages:

```
   ╔══════════════════════════════════════════════════════════════╗
   ║  grad going to each memory  =  its look_percent × grad_blend ║
   ╚══════════════════════════════════════════════════════════════╝
```

Using the overall size of grad_blended_look_step_4 = `0.31`:
```
  grad_memory_after_I     += 0.230 × 0.31 = 0.0713
  grad_memory_after_am    += 0.270 × 0.31 = 0.0837
  grad_memory_after_happy += 0.500 × 0.31 = 0.1550   ← biggest! looked at most
```

**And attention_mixer + attention_judge KEEP their own gradients** ✅ 📥
> They get told *"you should have given DIFFERENT percentages"* — that's how the judge learns to focus better next time! 🔦

---

## L.7 BACKWARD STEP 3 (the word "hoon") — Watch the ADDING!

### 🔹 3a. Its own gradient (SAME formula!)

```
  We said "hoon" with 35%.  Correct = hoon (index 5).

  predicted = [0.03, 0.04, 0.10, 0.15, 0.18, 0.35, 0.06, 0.05, 0.04]
  truth     = [0,    0,    0,    0,    0,    1,    0,    0,    0   ]
                                              ↑ index 5 = hoon
  SUBTRACT:
  grad_on_scores = [0.03, 0.04, 0.10, 0.15, 0.18, −0.65, 0.06, 0.05, 0.04]
                                                    ↑ raise "hoon" a LOT!
```

> 🧒 Notice `−0.65` is bigger than step 4's `−0.55` — because we were *less* sure (35% vs 45%). **More wrong = bigger push!** 💪

### 🔹 3b. Through the word_scorer — ⭐ IT'S THE SAME SCORER!

```
  ✅ KEEP → MORE gradient for word_scorer   📥📥 ADDED to step 4's!
  📨 PASS → grad_decoder_memory_step_3 from HERE = [0.28, −0.15, 0.20, −0.11]
  📨 PASS → grad_blended_look_step_3
```

**⭐ The same `word_scorer` was used at ALL 4 steps**, so it collects gradient 4 times:
```
  final grad_word_scorer = (step 1) + (step 2) + (step 3) + (step 4)
```

### 🔹 3c. ⭐ NOW WATCH `decoder_memory_step_3` GET ITS TOTAL

```
  from step 4's LSTM (part 4e):     [0.19, −0.10, 0.14, −0.06]
  from step 3's word_scorer (3b):   [0.28, −0.15, 0.20, −0.11]
                                    ──────────────────────────  ADD! ➕
  TOTAL grad_decoder_memory_step_3 = [0.47, −0.25, 0.34, −0.17]  ⭐
```

**Why two?** Because `decoder_memory_step_3` did **TWO JOBS** in the forward pass:
```
  JOB 1: it predicted the word "hoon"            (used by word_scorer at step 3)
  JOB 2: it helped make decoder_memory_step_4    (used by the LSTM at step 4)
```

> 🧒 **Like a student who did two assignments.** 📋 They get feedback on BOTH, and both go on the report card. You don't pick one — **you add them!** ➕
>
> **This is exactly BPTT from Module 11** — the same "sum the gradients" rule. 🔁

### 🔹 3d. The rest, same pattern

```
  grad_decoder_memory_step_3 → decoder_lstm:
     ✅ KEEP more grad for decoder_lstm                    📥
     ✅ KEEP grad for hindi_word_table["khush"] row        📥
     📨 PASS grad_decoder_memory_step_2 += [...]

  grad_blended_look_step_3 → attention:
     ✅ KEEP more grad for attention_mixer + attention_judge  📥
     📨 PASS to the three memories:
        grad_memory_after_I     += 0.224 × 0.48 = 0.1075
        grad_memory_after_am    += 0.260 × 0.48 = 0.1248
        grad_memory_after_happy += 0.516 × 0.48 = 0.2477
```

---

## L.8 BACKWARD STEPS 2 AND 1 — Exactly the Same Pattern

```
┌── BACKWARD STEP 2 ("khush", we said 30%) ────────────────────────┐
│  grad_on_scores = predicted − truth, with −0.70 on "khush"        │
│  ✅ word_scorer += more grad                                       │
│  📨 grad_decoder_memory_step_2 += [from word_scorer]               │
│      (it already had some from step 3's LSTM → ADD them! ➕)       │
│  ✅ decoder_lstm += more                                           │
│  ✅ hindi_word_table["main"] row                                   │
│  ✅ attention_mixer + attention_judge += more                      │
│  📨 grad_memory_after_I     += 0.210 × 0.55 = 0.1155              │
│     grad_memory_after_am    += 0.240 × 0.55 = 0.1320              │
│     grad_memory_after_happy += 0.549 × 0.55 = 0.3020              │
└───────────────────────────────────────────────────────────────────┘

┌── BACKWARD STEP 1 ("main", we said 40%) ─────────────────────────┐
│  grad_on_scores = predicted − truth, with −0.60 on "main"         │
│  ✅ word_scorer += more                                            │
│  ✅ decoder_lstm += more                                           │
│  ✅ hindi_word_table["<sos>"] row                                  │
│  ✅ attention_mixer + attention_judge += more                      │
│  📨 grad_memory_after_I     += 0.169 × 0.42 = 0.0710              │
│     grad_memory_after_am    += 0.206 × 0.42 = 0.0865              │
│     grad_memory_after_happy += 0.625 × 0.42 = 0.2625              │
│                                                                    │
│  ⭐ AND: decoder_memory_step_0 IS memory_after_happy!              │
│     So memory_after_happy gets EXTRA blame from being the baton!   │
└───────────────────────────────────────────────────────────────────┘
```

## L.9 📊 THE GRAND TOTALS

```
  grad_memory_after_I     = 0.0710 + 0.1155 + 0.1075 + 0.0713 = 0.3653
  grad_memory_after_am    = 0.0865 + 0.1320 + 0.1248 + 0.0837 = 0.4270
  grad_memory_after_happy = 0.2625 + 0.3020 + 0.2477 + 0.1550 = 0.9672
                            └step1┘  └step2┘  └step3┘  └step4┘
                                                          ↑
                                          2.6× more than memory_after_I!
```

> 🧒 **ATTENTION DECIDED WHO GETS BLAMED!** ⚖️
>
> "happy" was looked at ~55% of the time, so it gets ~55% of the blame.
> "I" was barely looked at, so it barely gets blamed.
>
> **Like a group project: did 85% of the work? Get 85% of the credit — and 85% of the blame.** Totally fair! 🎯

## L.10 Into the Encoder (BPTT again — Module 11!)

```
  grad_memory_after_happy (0.9672)
        │
        ├──✅ KEEP grad for encoder_lstm                     📥
        ├──✅ KEEP grad for english_word_table["happy"]      📥
        └──📨 PASS to memory_after_am  (ADD to its 0.4270!) ➕
                  │
                  ├──✅ KEEP more encoder_lstm grad          📥
                  ├──✅ KEEP grad for english_word_table["am"] 📥
                  └──📨 PASS to memory_after_I (ADD to its 0.3653!) ➕
                            │
                            ├──✅ KEEP more encoder_lstm      📥
                            └──✅ KEEP english_word_table["I"] 📥
```

**So each encoder memory gets blame from TWO sources:**
```
  SOURCE 1: directly from attention (all 4 decoder steps)
  SOURCE 2: from the memory AFTER it in the encoder chain (BPTT!)
```

> 🧒 **Blamed for two things:** what you did yourself, AND for teaching your friend who then messed up! 😅

## L.11 🔧 THE UPDATE (finally!)

Every SETTING that collected gradient now moves:

```
   ╔══════════════════════════════════════════════════════════╗
   ║   new_value  =  old_value  −  learning_rate × gradient   ║
   ╚══════════════════════════════════════════════════════════╝
```

**🔍 One complete example:**
```
  word_scorer.weight[2][0]   (the <eos> row, first column)

  old value       =  −0.4200
  gradient        =  −0.1595       (computed in part 4c!)
  learning rate   =   0.001

  new = −0.4200 − 0.001 × (−0.1595)
      = −0.4200 + 0.0001595
      = −0.4198405
```

> 🧒 **Why the MINUS sign?** The gradient points **uphill** (toward MORE loss). We want to go **downhill**! So we move the OPPOSITE way. A negative gradient means we ADD → the value goes UP. ✅ Exactly what we wanted for `<eos>`!

**🔍 A few more:**
```
                                       OLD      GRAD      NEW
  ──────────────────────────────────────────────────────────────
  word_scorer.bias[2] (<eos>)        0.0310   −0.550   0.03155   ✅
  word_scorer.bias[4] (khush)        0.0880   +0.120   0.08788   ✅ goes down
  attention_judge.weight[0]         −0.2130   −4.300  −0.20870   ✅
  hindi_word_table["hoon"][0]        0.2000   −1.900   0.20190   ✅
  english_word_table["happy"][0]     0.9000   −1.900   0.90190   ✅
  english_word_table["I"][0]         0.2000   −0.700   0.20070   ✅ smaller!

  decoder_memory_step_4              ❌ NOT UPDATED — recomputed next time!
  memory_after_happy                 ❌ NOT UPDATED — recomputed next time!
  blended_look_step_4                ❌ NOT UPDATED — recomputed next time!
```

**"happy" moved MORE than "I"** because it collected 2.6× more gradient. **Attention shaped the learning!** 🔦

## L.12 🔄 What Happens Next Round

```
  ROUND 1                              ROUND 2 (after the update)
  ─────────                            ──────────────────────────
  P(main)  = 40%                       P(main)  = 43%     📈
  P(khush) = 30%                       P(khush) = 34%     📈
  P(hoon)  = 35%                       P(hoon)  = 38%     📈
  P(<eos>) = 45%                       P(<eos>) = 48%     📈

  LOSS = 0.9922                        LOSS = 0.8734      📉

  look_percent on "I" at step 1:
      16.9%                       →       18.2%           🔦 focusing!
```

**Everything improved a little, all at once** — because EVERY setting got nudged in the right direction simultaneously. Repeat 1,000 times:
```
  P(main) → 95%   |   look_percent on "I" → 85%   |   LOSS → 0.05  🎉
```

## L.13 ⭐ THE TWO RULES THAT EXPLAIN EVERYTHING

```
   ╔═══════════════════════════════════════════════════════════════╗
   ║  RULE 1 — Is it a SETTING or a RESULT?                         ║
   ║     SETTING (weights, bias, tables)  →  KEEP the gradient 📥   ║
   ║     RESULT  (memory, blend, scores)  →  PASS it along     📨   ║
   ║                                                                 ║
   ║  RULE 2 — Was it used more than ONCE?                          ║
   ║     Then ADD UP all the gradients it receives             ➕   ║
   ╚═══════════════════════════════════════════════════════════════╝
```

**That's the ENTIRE backward pass.** Every single sub-step is one of those two rules! 🎉

## L.14 Forward and Backward, Hand in Hand

```
  ┌──────────────────────────────────────────────────────────────┐
  │  ➡️  FORWARD:  build things                                  │
  │                                                               │
  │     settings ──▶ compute memories ──▶ compute attention      │
  │     ──▶ compute blends ──▶ compute scores ──▶ LOSS           │
  │                                                               │
  │     (PyTorch secretly RECORDS every step — like a camera 📹)  │
  └──────────────────────────────────────────────────────────────┘
                             ⬇
  ┌──────────────────────────────────────────────────────────────┐
  │  ⬅️  BACKWARD:  assign blame                                 │
  │                                                               │
  │     LOSS ──▶ REWIND the recording ⏪                          │
  │     ──▶ RESULTS pass the blame along  📨                      │
  │     ──▶ SETTINGS collect the blame    📥                      │
  └──────────────────────────────────────────────────────────────┘
                             ⬇
  ┌──────────────────────────────────────────────────────────────┐
  │  🔧 UPDATE:  fix the recipe                                   │
  │     every setting moves a tiny step in the better direction   │
  └──────────────────────────────────────────────────────────────┘
                             ⬇
                    🔁 repeat with the next sentence
```

---

# PART M: ⭐⭐ THE CODE — LINE-BY-LINE DRY RUN

## M.1 First: The 5 Tensor Operations

The code uses 5 little operations. If you don't know them, EVERYTHING looks confusing. Let's learn them with tiny examples.

### 1️⃣ `.unsqueeze(n)` — add an empty layer of brackets

```
  BEFORE:  [1, 2, 3]                      shape (3,)

  .unsqueeze(0):  [[1, 2, 3]]             shape (1, 3)   ← bracket at position 0
  .unsqueeze(1):  [[1], [2], [3]]         shape (3, 1)   ← bracket at position 1
```
> 🧒 **Putting your items into a box** 📦. Same items — just one more layer of packaging!

### 2️⃣ `.squeeze(n)` — remove an empty layer

```
  BEFORE:  [[1, 2, 3]]                    shape (1, 3)
  .squeeze(0):  [1, 2, 3]                 shape (3,)
```
> 🧒 **The opposite — take the items OUT of the box** 📦➡️. Only works if the box holds exactly 1 thing!

### 3️⃣ `.repeat(a, b, c)` — make photocopies

```
  BEFORE:  [[[5, 6]]]                     shape (1, 1, 2)

  .repeat(1, 3, 1):
           [[[5, 6],
             [5, 6],
             [5, 6]]]                     shape (1, 3, 2)   ← 3 identical copies!
```
> 🧒 **The photocopier** 📄📄📄. `repeat(1,3,1)` means: don't copy in slot 1, make **3** copies in slot 2, don't copy in slot 3.

### 4️⃣ `torch.cat((a, b), dim=n)` — tape two things together

```
  a = [[1, 2]]           shape (1, 2)
  b = [[7, 8, 9]]        shape (1, 3)

  torch.cat((a, b), dim=1)  →  [[1, 2, 7, 8, 9]]      shape (1, 5)
                                      ↑ taped at the seam
```
> 🧒 **Taping two strips of paper end to end** 📎. `dim=1` says which end to tape.

### 5️⃣ `torch.bmm(a, b)` — multiply-and-add for batches

```
  a shape (1, 1, 3)  @  b shape (1, 3, 4)  →  result shape (1, 1, 4)
                ↑           ↑
           these MUST match (both 3) — and the 3 DISAPPEARS!
```
> 🧒 **The 3 vanishing IS the adding-up happening!** 3 things go in, they get blended into 1. That's exactly our weighted sum! 🥤

---

## M.2 THE ENCODER — Dry Run

```python
class Encoder(nn.Module):
    def __init__(self, english_vocab_size, numbers_per_word, memory_size):
        super().__init__()
        self.english_word_table = nn.Embedding(english_vocab_size, numbers_per_word)
        self.encoder_lstm = nn.LSTM(numbers_per_word, memory_size, batch_first=True)

    def forward(self, english_sentence):
        word_vectors = self.english_word_table(english_sentence)
        all_memories, (final_memory, final_belt) = self.encoder_lstm(word_vectors)
        return all_memories, final_memory, final_belt
```

**Setup:**
```python
encoder = Encoder(english_vocab_size=8, numbers_per_word=4, memory_size=4)
english_sentence = torch.tensor([[2, 3, 4]])       # "I am happy"
```
```
  english_sentence  =  [[2, 3, 4]]        shape (1, 3)
                          ↑  ↑  ↑
                         I  am happy
```

---

### 🔍 LINE: `word_vectors = self.english_word_table(english_sentence)`

```
  IN:   [[2, 3, 4]]                            shape (1, 3)

  WHAT IT DOES: look up row 2, row 3, row 4 of the table

  OUT:  [[[ 0.2,  0.5, -0.1,  0.3],            ← row 2 = "I"
          [ 0.4, -0.2,  0.6,  0.1],            ← row 3 = "am"
          [ 0.9,  0.7,  0.2,  0.8]]]           ← row 4 = "happy"
                                                shape (1, 3, 4)
```
> 🧒 **Each single number turned into 4 numbers!** The shape GAINED a dimension. 📈

---

### 🔍 LINE: `all_memories, (final_memory, final_belt) = self.encoder_lstm(word_vectors)`

```
  IN:   word_vectors                           shape (1, 3, 4)

  WHAT IT DOES: reads the 3 word-vectors one at a time, building memory

  OUT 1 — all_memories:                        shape (1, 3, 4)
        [[[0.12, 0.31, -0.05, 0.18],           ← memory_after_I
          [0.28, 0.19,  0.34, 0.25],           ← memory_after_am
          [0.71, 0.83,  0.35, 0.66]]]          ← memory_after_happy

  OUT 2 — final_memory:                        shape (1, 1, 4)
        [[[0.71, 0.83, 0.35, 0.66]]]           ← just the LAST one

  OUT 3 — final_belt:                          shape (1, 1, 4)
        [[[1.12, 1.31, 0.58, 1.04]]]           ← the cell state
```

### ⭐ LINE: `return all_memories, final_memory, final_belt`

```
   ╔═══════════════════════════════════════════════════════════════╗
   ║  THIS IS THE ENTIRE ATTENTION CHANGE!                          ║
   ║                                                                 ║
   ║  MODULE 13:  return final_memory, final_belt                   ║
   ║              ← all_memories was THROWN AWAY 🗑️                  ║
   ║                                                                 ║
   ║  MODULE 14:  return all_memories, final_memory, final_belt     ║
   ║              ← we KEEP everything! ⭐                            ║
   ╚═══════════════════════════════════════════════════════════════╝
```

> 🧒 **One extra word in a `return` statement — that's the whole revolution!** 🎉

---

## M.3 ⭐ THE ATTENTION — Dry Run (the important one!)

```python
class Attention(nn.Module):
    def __init__(self, memory_size):
        super().__init__()
        self.attention_mixer = nn.Linear(memory_size * 2, memory_size)
        self.attention_judge = nn.Linear(memory_size, 1, bias=False)

    def forward(self, decoder_memory, all_memories):
        how_many_words = all_memories.shape[1]
        memory_copies  = decoder_memory.unsqueeze(1).repeat(1, how_many_words, 1)
        pairs          = torch.cat((memory_copies, all_memories), dim=2)
        thinking       = torch.tanh(self.attention_mixer(pairs))
        scores         = self.attention_judge(thinking).squeeze(2)
        look_percent   = torch.softmax(scores, dim=1)
        blended_look   = torch.bmm(look_percent.unsqueeze(1), all_memories).squeeze(1)
        return blended_look, look_percent
```

**What goes IN:**
```python
decoder_memory = torch.tensor([[0.71, 0.83, 0.35, 0.66]])   # shape (1, 4)
all_memories   = (the (1,3,4) tensor from the encoder)
```
> 🧒 At decoder step 1, `decoder_memory` IS `memory_after_happy` — the baton! 🏃

---

### 🔍 LINE 1: `how_many_words = all_memories.shape[1]`

```
  IN:   all_memories.shape = (1, 3, 4)
                                 ↑ position 1
  OUT:  how_many_words = 3
```
> 🧒 "How many English words are there?" **3**. Just reading a label off the box. 🏷️

---

### 🔍 LINE 2: `memory_copies = decoder_memory.unsqueeze(1).repeat(1, how_many_words, 1)`

**This is TWO operations — let's split them:**

```
  ── PART A: .unsqueeze(1) ──
  IN:   [[0.71, 0.83, 0.35, 0.66]]              shape (1, 4)
  OUT:  [[[0.71, 0.83, 0.35, 0.66]]]            shape (1, 1, 4)
                                                 ↑ added an empty box

  ── PART B: .repeat(1, 3, 1) ──
  IN:   [[[0.71, 0.83, 0.35, 0.66]]]            shape (1, 1, 4)
  OUT:  [[[0.71, 0.83, 0.35, 0.66],             ← copy 1
          [0.71, 0.83, 0.35, 0.66],             ← copy 2
          [0.71, 0.83, 0.35, 0.66]]]            ← copy 3
                                                 shape (1, 3, 4)
```

> 🧒 **WHY?** We need to compare the decoder's memory against EACH of the 3 English memories. So we make **3 photocopies** — one for each comparison! 📄📄📄
>
> ⚠️ Notice: **all 3 rows are IDENTICAL.** That's on purpose!

---

### 🔍 LINE 3: `pairs = torch.cat((memory_copies, all_memories), dim=2)`

```
  IN 1:  memory_copies    shape (1, 3, 4)
  IN 2:  all_memories     shape (1, 3, 4)

  dim=2 means "tape along the LAST dimension" → 4 + 4 = 8

  OUT:   shape (1, 3, 8)

    row 0: [0.71, 0.83, 0.35, 0.66,  0.12, 0.31, -0.05, 0.18]
            └─ decoder memory ────┘  └─ memory_after_I ────┘

    row 1: [0.71, 0.83, 0.35, 0.66,  0.28, 0.19,  0.34, 0.25]
            └─ decoder memory ────┘  └─ memory_after_am ───┘

    row 2: [0.71, 0.83, 0.35, 0.66,  0.71, 0.83,  0.35, 0.66]
            └─ decoder memory ────┘  └─ memory_after_happy ┘
```

> 🧒 **Each row is now a PAIR taped together** — "here's what I'm looking for, and here's what this word offers." Three pairs ready to be judged! 👥📎

---

### 🔍 LINE 4: `thinking = torch.tanh(self.attention_mixer(pairs))`

```
  ── PART A: self.attention_mixer(pairs) ──
  attention_mixer is nn.Linear(8, 4)   ← squeezes 8 numbers down to 4

  IN:   shape (1, 3, 8)
  OUT:  shape (1, 3, 4)
        [[[ 0.2465, -0.1113,  0.4694,  0.0893],
          [ 0.2972, -0.0655,  0.4253,  0.1254],
          [ 0.4030,  0.0217,  0.3527,  0.2137]]]

  ── PART B: torch.tanh(...) ──
  squashes everything to between -1 and +1

  OUT:  shape (1, 3, 4)
        [[[ 0.2416, -0.1108,  0.4374,  0.0891],
          [ 0.2888, -0.0654,  0.4013,  0.1247],
          [ 0.3826,  0.0217,  0.3382,  0.2105]]]
```

> 🧒 This is the judge **THINKING about each pair** 🤔. She reads the pair (mixer), then forms an opinion (tanh keeps the opinion in a sensible range). She hasn't given a score yet — she's just thinking!

---

### 🔍 LINE 5: `scores = self.attention_judge(thinking).squeeze(2)`

```
  ── PART A: self.attention_judge(thinking) ──
  attention_judge is nn.Linear(4, 1)   ← turns 4 thinking-numbers into ONE score

  IN:   shape (1, 3, 4)
  OUT:  shape (1, 3, 1)
        [[[0.4438],
          [0.6405],
          [1.7511]]]

  ── PART B: .squeeze(2) ──
  removes the useless last box (each holds just 1 number)

  OUT:  shape (1, 3)
        [[0.4438, 0.6405, 1.7511]]
```

> 🧒 **The judge writes one score per word!** 👨‍⚖️
> ```
>    "I"     → 0.4438
>    "am"    → 0.6405
>    "happy" → 1.7511   ← she likes this one most
> ```
> Then `.squeeze(2)` just unwraps the pointless single-item boxes. 📦➡️

**🔍 Why `bias=False` on the judge?**
```
  A bias would add the SAME amount to every score.
  Softmax only cares about DIFFERENCES between scores.
  Adding the same thing to all of them changes NOTHING!

  🧒 If everyone in class gets +5 marks, the ranking doesn't change. 📋
```

---

### 🔍 LINE 6: `look_percent = torch.softmax(scores, dim=1)`

```
  IN:   [[0.4438, 0.6405, 1.7511]]              shape (1, 3)

  THE MATH (check with a calculator!):
     e^0.4438 = 1.5587
     e^0.6405 = 1.8974
     e^1.7511 = 5.7605
                ───────
        SUM   = 9.2166

     1.5587 / 9.2166 = 0.1691
     1.8974 / 9.2166 = 0.2059
     5.7605 / 9.2166 = 0.6250
                       ──────
                       1.0000  ✓

  OUT:  [[0.1691, 0.2059, 0.6250]]              shape (1, 3)
         └ 16.9%   20.6%   62.5% ┘
```

**⚠️ WHY `dim=1` AND NOT `dim=0`?** (the #1 attention bug!)
```
  shape is (1, 3) = (batch, source words)

  dim=1  → softmax ACROSS the 3 WORDS        ✅ CORRECT
           "share 100% among the input words"

  dim=0  → softmax across the BATCH          ❌ WRONG!
           "share 100% among sentences" — meaningless!

  🧒 We're splitting attention among WORDS, not among sentences! 🎯
```

---

### 🔍 LINE 7: `blended_look = torch.bmm(look_percent.unsqueeze(1), all_memories).squeeze(1)`

**THREE operations — let's split:**

```
  ── PART A: look_percent.unsqueeze(1) ──
  IN:   [[0.1691, 0.2059, 0.6250]]              shape (1, 3)
  OUT:  [[[0.1691, 0.2059, 0.6250]]]            shape (1, 1, 3)

  ── PART B: torch.bmm(..., all_memories) ──
  shapes:  (1, 1, 3)  @  (1, 3, 4)  →  (1, 1, 4)
                  ↑        ↑
             both 3 — they match, and the 3 DISAPPEARS ✓

  THE MATH — first number:
     0.1691 × 0.12  =  0.02029     (from memory_after_I)
     0.2059 × 0.28  =  0.05765     (from memory_after_am)
     0.6250 × 0.71  =  0.44375     (from memory_after_happy)
                       ─────────
                        0.52169  →  0.5217

  Doing that for all 4 positions:
  OUT:  [[[0.5217, 0.6103, 0.2803, 0.4944]]]    shape (1, 1, 4)

  ── PART C: .squeeze(1) ──
  OUT:  [[0.5217, 0.6103, 0.2803, 0.4944]]      shape (1, 4)
```

> 🧒 **The smoothie is made!** 🥤 16.9% of "I", 20.6% of "am", 62.5% of "happy", all blended into 4 numbers.

**🔍 PROOF that `bmm` is just multiply-and-add:**
```python
the_easy_way = torch.bmm(look_percent.unsqueeze(1), all_memories).squeeze(1)
the_slow_way = (look_percent.unsqueeze(2) * all_memories).sum(dim=1)

print(the_easy_way)   # tensor([[0.5217, 0.6103, 0.2803, 0.4944]])
print(the_slow_way)   # tensor([[0.5217, 0.6103, 0.2803, 0.4944]])
print(torch.allclose(the_easy_way, the_slow_way))   # True ✓
```
> 🧒 **Same answer!** `bmm` is just the fast shortcut for "multiply each, then add." **Brute force proven!** ✅

---

### 🔍 LINE 8: `return blended_look, look_percent`

```
  Why return look_percent too? We don't NEED it for the math!

  🧒 We return it so we can DRAW THE ALIGNMENT GRID! 📊
     It's how we peek inside the model's mind. Free explainability! 🔍
```

### 📊 THE WHOLE ATTENTION JOURNEY

```
  decoder_memory     (1, 4)      what I'm looking for
       ↓ unsqueeze + repeat
  memory_copies      (1, 3, 4)   3 photocopies
       ↓ cat with all_memories
  pairs              (1, 3, 8)   3 pairs taped together
       ↓ attention_mixer + tanh
  thinking           (1, 3, 4)   the judge thinking
       ↓ attention_judge
  scores             (1, 3)      one score per word
       ↓ softmax
  look_percent       (1, 3)      percentages adding to 100%
       ↓ bmm with all_memories
  blended_look       (1, 4)      the smoothie! 🥤
```

---

## M.4 THE DECODER — Dry Run

```python
class Decoder(nn.Module):
    def __init__(self, hindi_vocab_size, numbers_per_word, memory_size):
        super().__init__()
        self.hindi_word_table = nn.Embedding(hindi_vocab_size, numbers_per_word)
        self.attention        = Attention(memory_size)
        self.decoder_lstm     = nn.LSTM(numbers_per_word + memory_size,
                                        memory_size, batch_first=True)
        self.word_scorer      = nn.Linear(memory_size * 2, hindi_vocab_size)

    def forward(self, previous_word, decoder_memory, decoder_belt, all_memories):
        word_vector = self.hindi_word_table(previous_word).unsqueeze(1)
        blended_look, look_percent = self.attention(decoder_memory.squeeze(0), all_memories)
        lstm_input  = torch.cat((word_vector, blended_look.unsqueeze(1)), dim=2)
        new_output, (decoder_memory, decoder_belt) = self.decoder_lstm(
                                       lstm_input, (decoder_memory, decoder_belt))
        for_scoring = torch.cat((new_output.squeeze(1), blended_look), dim=1)
        word_scores = self.word_scorer(for_scoring)
        return word_scores, decoder_memory, decoder_belt, look_percent
```

### 🔍 The TWO size changes from Module 13 — explained

```
  ┌───────────────────────────────────────────────────────────────────┐
  │ nn.LSTM(numbers_per_word + memory_size, ...)  =  nn.LSTM(4+4=8,...)│
  │                       ↑                                             │
  │ Module 13 was just nn.LSTM(4, ...). Why bigger?                    │
  │                                                                     │
  │ 🧒 Because we feed it TWO things taped together:                    │
  │    the word AND the blend! Bigger door needed! 🚪                    │
  ├───────────────────────────────────────────────────────────────────┤
  │ nn.Linear(memory_size * 2, hindi_vocab_size)  =  nn.Linear(8, 9)   │
  │                       ↑                                             │
  │ Module 13 was nn.Linear(4, 9). Why doubled?                        │
  │                                                                     │
  │ 🧒 Because the judge sees TWO things:                               │
  │    the memory AND the blend! Two views = better decision! 👨‍⚖️        │
  └───────────────────────────────────────────────────────────────────┘
```

**Dry run setup — decoder step 1:**
```python
previous_word  = torch.tensor([1])          # <sos>
decoder_memory = final_memory               # shape (1,1,4) — the baton!
decoder_belt   = final_belt                 # shape (1,1,4)
```

---

### 🔍 LINE 1: `word_vector = self.hindi_word_table(previous_word).unsqueeze(1)`

```
  ── PART A: hindi_word_table(previous_word) ──
  IN:   [1]                                shape (1,)      ← <sos>
  OUT:  [[0.10, 0.10, 0.10, 0.10]]         shape (1, 4)    ← row 1 of the table

  ── PART B: .unsqueeze(1) ──
  OUT:  [[[0.10, 0.10, 0.10, 0.10]]]       shape (1, 1, 4)
```
> 🧒 **Why unsqueeze?** The LSTM ALWAYS wants `(batch, how_many_words, numbers)`. We're feeding ONE word, so "how many words" = 1. We wrap it in a list of length 1! 🧺

---

### 🔍 LINE 2: `blended_look, look_percent = self.attention(decoder_memory.squeeze(0), all_memories)`

```
  ── decoder_memory.squeeze(0) ──
  IN:   [[[0.71, 0.83, 0.35, 0.66]]]       shape (1, 1, 4)
  OUT:  [[0.71, 0.83, 0.35, 0.66]]         shape (1, 4)

  ── then attention runs (everything from M.3!) ──
  blended_look  = [[0.5217, 0.6103, 0.2803, 0.4944]]    shape (1, 4)
  look_percent  = [[0.1691, 0.2059, 0.6250]]            shape (1, 3)
```
> 🧒 `.squeeze(0)` removes the layer-count box because attention wants a plain `(batch, numbers)`. Just unpacking! 📦

---

### 🔍 LINE 3: `lstm_input = torch.cat((word_vector, blended_look.unsqueeze(1)), dim=2)`

```
  IN 1:  word_vector                       shape (1, 1, 4)
  IN 2:  blended_look.unsqueeze(1)         shape (1, 1, 4)

  OUT:   shape (1, 1, 8)
     [[[0.10, 0.10, 0.10, 0.10,  0.5217, 0.6103, 0.2803, 0.4944]]]
       └── the word "<sos>" ──┘  └──── where to look ─────────┘
```
> 🧒 **The LSTM now gets a package with TWO things:** "here's the last word I said, AND here's what I should be looking at." 📦🎁

---

### ⭐🔍 LINE 4: `new_output, (decoder_memory, decoder_belt) = self.decoder_lstm(lstm_input, (decoder_memory, decoder_belt))`

```
  IN 1:  lstm_input                        shape (1, 1, 8)
  IN 2:  (decoder_memory, decoder_belt)    ← ⭐ THE STARTING MEMORY!

  OUT:   new_output      = [[[0.55, 0.61, 0.22, 0.48]]]    shape (1, 1, 4)
         decoder_memory  = [[[0.55, 0.61, 0.22, 0.48]]]    shape (1, 1, 4)
         decoder_belt    = [[[0.88, 0.97, 0.36, 0.79]]]    shape (1, 1, 4)
```

```
   ╔═══════════════════════════════════════════════════════════════╗
   ║  THE SECOND ARGUMENT IS THE KEY LINE!                          ║
   ║                                                                 ║
   ║  PROJECT #3:  self.lstm(x)                                     ║
   ║               → memory starts at ZERO                          ║
   ║                                                                 ║
   ║  HERE:        self.lstm(x, (decoder_memory, decoder_belt))     ║
   ║               → "start with THIS memory!" ⭐                     ║
   ╚═══════════════════════════════════════════════════════════════╝
```
> 🧒 At step 1 that's the **baton from the encoder** 🏃. At later steps it's the **previous step's memory**. That's how the chain continues! 🔗

---

### 🔍 LINE 5: `for_scoring = torch.cat((new_output.squeeze(1), blended_look), dim=1)`

```
  ── PART A: new_output.squeeze(1) ──
  IN:   [[[0.55, 0.61, 0.22, 0.48]]]       shape (1, 1, 4)
  OUT:  [[0.55, 0.61, 0.22, 0.48]]         shape (1, 4)

  ── PART B: cat with blended_look ──
  OUT:  [[0.55, 0.61, 0.22, 0.48,  0.5217, 0.6103, 0.2803, 0.4944]]
          └── the new memory ───┘  └──── the blend ──────────────┘
                                    shape (1, 8)
```
> 🧒 **Giving the judge BOTH sources:** "here's my thinking (memory) AND here's what I looked at (blend)." Two views = a better decision! 👨‍⚖️👀

---

### 🔍 LINE 6: `word_scores = self.word_scorer(for_scoring)`

```
  word_scorer is nn.Linear(8, 9)

  IN:   shape (1, 8)
  OUT:  shape (1, 9)      ← one score per Hindi word!

  [[-2.10, -1.80, -0.90,  3.20,  0.80,  0.40, -0.50, -1.20, -1.50]]
    <pad>  <sos>  <eos>   main  khush  hoon   aap    ho    udaas
                          ↑ biggest!
```
> 🧒 **These are RAW scores, not percentages.** We do NOT apply softmax here — `CrossEntropyLoss` does it inside! (Project #2's lesson! 🎯)

---

## M.5 THE TRAINING LOOP — Dry Run

```python
english_sentence = torch.tensor([[2, 3, 4]])        # I am happy
hindi_target     = torch.tensor([[1, 3, 4, 5, 2]])  # <sos> main khush hoon <eos>

loss_function = nn.CrossEntropyLoss()
optimizer     = torch.optim.Adam(model.parameters(), lr=0.001)

optimizer.zero_grad()

all_memories, decoder_memory, decoder_belt = encoder(english_sentence)

previous_word = hindi_target[:, 0]
total_loss = 0

for step in range(1, hindi_target.shape[1]):
    word_scores, decoder_memory, decoder_belt, look_percent = decoder(
        previous_word, decoder_memory, decoder_belt, all_memories)

    correct_word = hindi_target[:, step]
    total_loss = total_loss + loss_function(word_scores, correct_word)
    previous_word = correct_word

average_loss = total_loss / 4
average_loss.backward()
optimizer.step()
```

### 🔍 `optimizer.zero_grad()`
```
  Sets every stored gradient to 0.
  🧒 Wiping the whiteboard clean before today's lesson! 🧹
     (Without it, yesterday's blame would pile onto today's.)
```

### 🔍 `all_memories, decoder_memory, decoder_belt = encoder(english_sentence)`
```
  Runs the encoder ONCE.
  all_memories   (1, 3, 4)   ← the three memories 📚
  decoder_memory (1, 1, 4)   ← the baton 🏃
  decoder_belt   (1, 1, 4)
```

### 🔍 `previous_word = hindi_target[:, 0]`
```
  hindi_target = [[1, 3, 4, 5, 2]]
                    ↑ column 0

  OUT: tensor([1])   =  <sos>   ← the start gun 🔫
```

### 🔍 `for step in range(1, 5):`
```
  step = 1, 2, 3, 4        ← starts at 1, NOT 0!

  🧒 Why skip 0? Because position 0 is <sos>, which goes IN.
     No word comes OUT there. 📥
```

### 🔍 THE LOOP, ITERATION BY ITERATION

```
  ┌─ step = 1 ───────────────────────────────────────────────┐
  │  previous_word = [1] (<sos>)                             │
  │  decoder runs → word_scores (1,9), "main" wins with 40%  │
  │  correct_word = hindi_target[:, 1] = [3] ("main")        │
  │  loss = −ln(0.40) = 0.9163                               │
  │  total_loss = 0 + 0.9163 = 0.9163                        │
  │  previous_word = [3]  ← TEACHER FORCING!                 │
  └───────────────────────────────────────────────────────────┘
  ┌─ step = 2 ───────────────────────────────────────────────┐
  │  previous_word = [3] ("main")                            │
  │  "khush" wins with 30%                                   │
  │  correct_word = [4] ("khush")                            │
  │  loss = −ln(0.30) = 1.2040                               │
  │  total_loss = 0.9163 + 1.2040 = 2.1203                   │
  │  previous_word = [4]                                     │
  └───────────────────────────────────────────────────────────┘
  ┌─ step = 3 ───────────────────────────────────────────────┐
  │  "hoon" wins 35% → loss 1.0498 → total_loss = 3.1701     │
  │  previous_word = [5]                                     │
  └───────────────────────────────────────────────────────────┘
  ┌─ step = 4 ───────────────────────────────────────────────┐
  │  <eos> wins 45% → loss 0.7985 → total_loss = 3.9686      │
  └───────────────────────────────────────────────────────────┘
```

### ⭐🔍 `previous_word = correct_word` — THIS IS TEACHER FORCING!
```
  🧒 We feed the CORRECT word next, NOT our guess.
     ONE LINE OF CODE = the whole trick! 👩‍🏫

     Without it, one early mistake would poison the whole sentence.
```

### 🔍 `average_loss = total_loss / 4`
```
  IN:   3.9686
  OUT:  0.9922    ← the average per word ✓
```

### ⭐🔍 `average_loss.backward()`
```
  🧒 Presses REWIND on the recording tape ⏪
     Walks EVERY step backward, collecting gradients on all settings.

     THIS ONE LINE IS ALL OF PART L! 🤯

  Nothing visible happens — gradients quietly appear:
     word_scorer.weight.grad             is now filled in ✅
     attention_judge.weight.grad         filled ✅
     attention_mixer.weight.grad         filled ✅
     decoder_lstm.weight_ih_l0.grad      filled ✅
     hindi_word_table.weight.grad        filled ✅
     encoder_lstm.weight_ih_l0.grad      filled ✅
     english_word_table.weight.grad      filled ✅
```

### 🔍 `optimizer.step()`
```
  🧒 Applies:  new = old − 0.001 × gradient   to EVERY setting.

  word_scorer.weight[2][0]:       −0.4200  →  −0.4198  ✅
  hindi_word_table[5][0]:          0.2000  →   0.2019  ✅
  english_word_table[4][0]:        0.9000  →   0.9019  ✅

  decoder_memory, blended_look, all_memories:  ❌ untouched (they're RESULTS!)
```

---

## M.6 🎯 THE COMPLETE SHAPE JOURNEY

```
  ┌──────────────────────────────────────────────────────────────┐
  │  english_sentence   (1, 3)          [[2, 3, 4]]              │
  │        ↓ english_word_table                                   │
  │  word_vectors       (1, 3, 4)       each word → 4 numbers     │
  │        ↓ encoder_lstm                                         │
  │  all_memories       (1, 3, 4)   ⭐ KEPT for attention!        │
  │  decoder_memory     (1, 1, 4)   ⭐ the baton 🏃               │
  ├──────────────────────────────────────────────────────────────┤
  │  ── FOR EACH OF 4 DECODER STEPS ──                            │
  │                                                                │
  │  memory_copies      (1, 3, 4)       3 photocopies 📄📄📄       │
  │        ↓ cat                                                  │
  │  pairs              (1, 3, 8)       3 pairs 📎                │
  │        ↓ attention_mixer + tanh + attention_judge             │
  │  scores             (1, 3)          one score per word 👨‍⚖️     │
  │        ↓ softmax                                              │
  │  look_percent       (1, 3)          adds to 100% 💯           │
  │        ↓ bmm                                                  │
  │  blended_look       (1, 4)          the smoothie 🥤           │
  │        ↓ cat with word, into decoder_lstm                     │
  │  decoder_memory     (1, 1, 4)       new memory                │
  │        ↓ cat with blend, into word_scorer                     │
  │  word_scores        (1, 9)          one score per Hindi word  │
  │        ↓ CrossEntropyLoss                                     │
  │  loss               a single number                           │
  ├──────────────────────────────────────────────────────────────┤
  │  average_loss.backward()   ⏪ rewind, collect gradients        │
  │  optimizer.step()          🔧 move every setting a tiny bit    │
  └──────────────────────────────────────────────────────────────┘
```

## M.7 🔑 THE 3 LINES THAT *ARE* ATTENTION

If you remember nothing else from the code:

```python
  return all_memories, final_memory, final_belt      # 1. KEEP everything ⭐
  look_percent = torch.softmax(scores, dim=1)        # 2. decide where to look 🔦
  blended_look = torch.bmm(...)                      # 3. blend by those % 🥤
```

**Everything else is just RESHAPING** (unsqueeze, squeeze, repeat, cat) to make the shapes line up! 📐

---

# PART N: SELF-ATTENTION vs CROSS-ATTENTION

## N.1 Two Flavours of the Same Machine

Everything in this module used **CROSS-attention** — two different sentences. But there's another flavour!

```
  ═══════ CROSS-ATTENTION (this module) ═══════

     ENGLISH: "I am happy"        HINDI: "main khush hoon"
       └─ gives the MEMORIES ─┘     └─ gives the QUERY ─┘
                    ▲                        │
                    └────────────────────────┘
              the Hindi word looks ACROSS at the English words


  ═══════ SELF-ATTENTION (Module 15) ═══════

     "The cat sat because it was tired"
                              ↑
                    what does "it" mean?

       "it" looks at every OTHER word in its OWN sentence
       → it discovers "it" = "cat"! 🐱

        the    cat    was     it
         │      │      │      │
         ▏    ████     ▎      ⬅ strongest link to "cat"
```

## N.2 The Comparison

| | **Cross-attention** (Module 14) | **Self-attention** (Module 15) |
|---|---|---|
| How many sentences? | TWO (English + Hindi) | **ONE** |
| Query comes from | the decoder (Hindi) | **the same sentence** |
| Memories come from | the encoder (English) | **the same sentence** |
| Question | "which English word helps me write this Hindi word?" | **"which other words explain this word?"** |
| Used in | translation | **GPT, BERT, Claude** |
| Is there an RNN? | YES (LSTM) | **NO — attention only!** |

```
   BOTH use the SAME 4 STEPS:  score → softmax → blend → use

   The ONLY difference is WHERE the query and memories come from:

     cross-attention:  query from sentence A, memories from sentence B  🔀
     self-attention:   query and memories BOTH from sentence A          🔄
```

> 🧒 **Cross-attention** = reading an English book while writing Hindi notes 📖✍️ (two documents).
> **Self-attention** = re-reading YOUR OWN sentence to understand it better 🔄 (one document).

## N.3 Query, Key, Value — The Bridge to Transformers 🌉

Modern papers use three fancy words. **They're just new names for what you already know!**

| Fancy name | 🧒 What it really is | In our code |
|------------|---------------------|-------------|
| **Query** (Q) | "what am I looking for?" | `decoder_memory` |
| **Key** (K) | "what do I have to offer?" | `all_memories` |
| **Value** (V) | "what do I actually give you?" | `all_memories` (again!) |

### 🧒 The Library Analogy 📚

```
   You walk into a library:

   QUERY  🙋 = what you ask for         →  "books about dinosaurs"
   KEY    🏷️ = the label on each shelf   →  "Dinosaurs", "Space", "Cooking"
   VALUE  📕 = the actual book you take  →  the dinosaur book itself

   HOW IT WORKS:
     1. Compare your QUERY to every KEY       → how well does each match?
     2. Softmax                                → how much from each shelf?
     3. Take a mix of the VALUES               → your blended answer!

   🎯 THAT'S EXACTLY OUR 4 STEPS! Just different words! 🎉
```

### The Famous Transformer Formula — You Can Now Read It!

```
     Attention(Q, K, V) = softmax( Q·Kᵀ / √d ) · V
                          └──┬──┘  └─┬─┘  └┬┘   └┬┘
                          our step 2  step 1  ⬇   step 3
                                             scaling
```

- `Q·Kᵀ` = our **score** (dot product!) — **Step 1** ✓
- `/√d` = divide by √(size) — keeps scores from getting huge, so softmax isn't too spiky (**exactly like TEMPERATURE from Module 9!** 🌡️)
- `softmax(...)` = our **look_percent** — **Step 2** ✓
- `· V` = our **blended_look** — **Step 3** ✓

> 🧒 **You already understand the Transformer's core equation!** 🎉

---

# 📋 MODULE 14 MASTER RECAP

```
   1. THE PROBLEM: Module 13 squeezed everything into ONE fixed
      context vector → long sentences lost information 🍾

   2. THE FIX: keep ALL encoder memories, let the decoder shine
      a flashlight on whichever one matters right now 🔦

   3. THE 4 STEPS:  score → softmax → blend → use

   4. SCORE = "how well do these match?" — dot product is the
      simplest ("do we agree?" meter 📏)

   5. SOFTMAX turns scores into percentages that ALWAYS add to 100%

   6. BLEND = weighted sum of memories = a fresh context 🥤

   7. ⭐ A NEW blend at EVERY step (vs ONE for all steps in Module 13)

   8. ⭐ THE BATON: the decoder's FIRST memory = memory_after_happy
      (the encoder's last memory) 🏃

   9. THE ALIGNMENT GRID shows where the model looked — it even
      learns word order by itself (the CROSSING pattern!) 🎉

  10. SCORE TYPES: dot (fast), multiplicative, additive/Bahdanau (learned)

  11. BACKPROP RULE 1: settings KEEP gradient 📥, results PASS it 📨
      BACKPROP RULE 2: used more than once? ADD the gradients ➕

  12. THE MAGIC FORMULA: grad_on_scores = predicted − truth

  13. Each memory gets blame × its look_percent — attention decides
      who gets credited AND blamed ⚖️

  14. WE NEVER UPDATE MEMORY. Only settings get updated. 🍳

  15. THE 3 CODE LINES THAT ARE ATTENTION:
        return all_memories, ...      (keep everything)
        softmax(scores, dim=1)        (decide where to look)
        bmm(...)                      (blend)

  16. Q, K, V are just new names: query = what I want,
      key = what's offered, value = what I take 📚

  17. Self-attention (same sentence) → Transformers, Module 15! 🚀
```

---

# 🤔 COMMON DOUBTS

**Q1: What is the decoder's memory at step 1?**
> 🧒 It's `memory_after_happy` — the encoder's LAST memory, handed over like a baton in a relay race 🏃. It means "I've read the whole English sentence."

**Q2: Does attention replace the LSTM?**
> 🧒 Not here — attention HELPS the LSTM by giving it a better blend each step. In Transformers (Module 15), attention replaces the RNN completely!

**Q3: Why must look_percent sum to 100%?**
> 🧒 Because it's a SHARE of your looking. Softmax guarantees it. If it didn't sum to 1, the blend's size would swing wildly and training would break.

**Q4: Are the look percentages learned directly?**
> 🧒 NO! They're computed FRESH every step from the current situation. What's LEARNED is the JUDGE (`attention_mixer` + `attention_judge`) — the thing that decides how to score.

**Q5: How do we update the memory?**
> 🧒 **WE DON'T!** Memory is a RESULT, like the "12" from `3 × 4`. Blame passes THROUGH it to reach the settings that made it. 🔢

**Q6: Why does the decoder LSTM's input_size grow?**
> 🧒 Because we feed it TWO things taped together: the previous word AND the blend. A wider door! 🚪

**Q7: Why does the word_scorer's input double?**
> 🧒 Because the final judge sees TWO things: the new memory AND the blend. Two sources = a better decision! 👨‍⚖️

**Q8: Why is `+=` used for gradients and not `=`?**
> 🧒 Because a thing used more than once gets blamed more than once — and you ADD all the blames. Like a bike lent to two friends who both scratched it! 🚲

**Q9: Can I see the attention weights in a real model?**
> 🧒 YES! That's why we return `look_percent`. Plotting the alignment grid is standard practice — and it's a big reason attention made models less of a black box. 🔍

**Q10: Is attention slow?**
> 🧒 It compares every output step against every input word, so cost grows as (input length × output length). Fine for sentences, expensive for very long documents.

**Q11: What's the difference between cross-attention and self-attention?**
> 🧒 Cross = the decoder looks at the ENCODER's words (two sentences). Self = words in the SAME sentence look at each other (one sentence). Same 4 steps!

**Q12: Why `dim=1` in softmax and not `dim=0`?**
> 🧒 Shape is (batch, words). We split 100% among WORDS (dim 1), not among sentences (dim 0). Using dim=0 is the classic attention bug! 🐛

---

# ✅ QUICK PRACTICE

**Q1:** What exactly does attention fix from Module 13?
<details><summary>Answer</summary>
The bottleneck. Instead of squeezing the whole input into ONE fixed context vector, the decoder keeps all encoder memories and builds a FRESH blend at every step by weighting them.
</details>

**Q2:** Name the 4 steps of attention.
<details><summary>Answer</summary>
(1) Score — how well does each input word match my current need. (2) Softmax — turn scores into percentages summing to 100%. (3) Blend — weighted sum of the memories. (4) Use it — combine with the decoder's memory to predict the word.
</details>

**Q3:** Scores are `[1.0, 2.0, 0.0]`. Which word gets the most attention and roughly how much?
<details><summary>Answer</summary>

```
e^1 = 2.718,  e^2 = 7.389,  e^0 = 1.000,  sum = 11.107
→ 24.5%, 66.5%, 9.0%
```
The second word, with about 66.5%.
</details>

**Q4:** What is the decoder's starting memory, and why?
<details><summary>Answer</summary>
`memory_after_happy` — the encoder's final memory. It means "I've read the whole English sentence," which is the perfect starting thought for translating it. Like a baton handed over in a relay race.
</details>

**Q5:** Your untrained model gives look_percent `[0.33, 0.34, 0.33]`. Bug or normal?
<details><summary>Answer</summary>
Totally normal! Random weights mean the model has no idea what matters, so it spreads attention evenly (a shrug 🤷). After training these sharpen to something like [0.85, 0.10, 0.05].
</details>

**Q6:** We predicted `<eos>` with 45% and it was correct. What is the gradient on the `<eos>` score?
<details><summary>Answer</summary>
`predicted − truth = 0.45 − 1 = −0.55`. Negative means "raise this score!" The size (0.55) shows how wrong we were.
</details>

**Q7:** How do we update `decoder_memory_step_4`?
<details><summary>Answer</summary>
We DON'T! It's a RESULT, not a setting. Its gradient is just a message passed backward to whoever made it (the decoder_lstm). Only settings get updated.
</details>

**Q8:** Why does `decoder_memory_step_3` receive gradient from TWO places?
<details><summary>Answer</summary>
Because it did two jobs: (1) it predicted the word "hoon" at step 3, and (2) it helped make `decoder_memory_step_4` at step 4. Used twice → blamed twice → ADD the gradients. That's BPTT from Module 11!
</details>

**Q9:** What does `.repeat(1, 3, 1)` do and why do we need it?
<details><summary>Answer</summary>
It makes 3 identical photocopies of the decoder's memory along dimension 1. We need one copy to pair against each of the 3 encoder memories for scoring.
</details>

**Q10:** In `torch.bmm((1,1,3), (1,3,4))`, what happens to the 3?
<details><summary>Answer</summary>
It DISAPPEARS — that's the summing-up happening. 3 memories go in weighted, 1 blended memory comes out. The result is shape (1,1,4).
</details>

**Q11:** Which single line of Module 13's code WAS the bottleneck?
<details><summary>Answer</summary>
`return final_memory, final_belt` in the encoder — because it discarded `all_memories`. Module 14 changes it to `return all_memories, final_memory, final_belt`.
</details>

**Q12:** What do Query, Key, and Value correspond to in our code?
<details><summary>Answer</summary>
Query = `decoder_memory` (what I'm looking for). Key = `all_memories` (what's on offer). Value = `all_memories` again (what I actually take). Library: your request, the shelf labels, the books.
</details>

---

# 🎬 WHAT'S NEXT: MODULE 15 — TRANSFORMERS! 🚀

```
   You now understand attention completely.

   Transformers ask one bold question:

     "If attention is SO good... why do we need the RNN at all?
      Let's DELETE it and use ONLY attention!"

   That's it. That's the Transformer.

   And it's what powers GPT, BERT, Claude, and every modern AI.

   YOU ARE ONE MODULE AWAY! 🔥
```

---

*Module 14 Complete! Attention understood — concept, math, backprop, and every line of code! 🎉*
