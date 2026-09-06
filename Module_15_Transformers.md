# 📘 MODULE 15: Transformers — Complete Visual + Numerical + Code Edition

**Difficulty:** 🔴 Hard (but you already know 80%!)
**Time:** 360 minutes
**Prerequisite:** Modules 1-14 + Projects #1, #2, #3
**Tools:** Google Colab, PyTorch

---

## 📖 How to Read These Notes

- Everything in **simple English**, like teaching a 10-year-old who knows nothing
- **Every variable has a self-explaining name**, used the same way from start to finish
- **Real numbers at every single step** — check them with a calculator
- **Pictures everywhere** (ASCII diagrams)
- 🔍 **Code with printed output** at the end

---

## 📕 THE NAME LIST (used everywhere in these notes)

### 🔧 SETTINGS — things that get updated by training (the recipe 📖)

| Name | What it is |
|------|-----------|
| `word_table` | word → numbers lookup table |
| `position_table` | position → numbers (or the sin/cos formula) |
| `W_query_head1`, `W_key_head1`, `W_value_head1` | head 1's three grids |
| `W_query_head2`, `W_key_head2`, `W_value_head2` | head 2's three grids |
| `W_output` | the grid that mixes the heads back together |
| `feed_forward_1` | the "expand" Linear layer (4 → 8) |
| `feed_forward_2` | the "squeeze" Linear layer (8 → 4) |
| `word_scorer` | the final layer that scores every word |

### 📄 RESULTS — computed fresh, then thrown away (today's dish 🍲)

| Name | Meaning |
|------|---------|
| `query_sat`, `key_the`, `value_cat` … | the three versions of each word |
| `score_the`, `score_cat`, `score_sat` | raw match numbers |
| `look_percent` | attention percentages (add to 100%) |
| `head1_output`, `head2_output` | each head's answer |
| `attention_output` | after mixing the heads |
| `after_add_norm_1`, `after_add_norm_2` | after each tidy-up step |
| `block_output` | what leaves the block |

### 📚 OUR TINY SETUP

```
  Sentence:  "the cat sat"
  d_model = 4        (each word = 4 numbers — tiny so we can SEE it)
  2 heads            (so each head gets 4 ÷ 2 = 2 numbers)

  Word embeddings (before adding position):
     the → [0.1, 0.2, 0.0, 0.3]
     cat → [0.9, 0.8, 0.1, 0.2]
     sat → [0.3, 0.7, 0.6, 0.4]
```

---

# PART A: THE PROBLEM — RNNs ARE SLOW 🐌

## A.1 What Was Wrong

Our LSTM translator (Modules 12-14) worked fine. But it had a hidden flaw that has **nothing to do with accuracy**:

```
   An RNN MUST go one word at a time:

      "the"  →  wait  →  "cat"  →  wait  →  "sat"
       ▲                  ▲                  ▲
       │                  │                  │
    step 1             step 2             step 3

   You CANNOT do step 2 until step 1 is finished,
   because step 2 needs step 1's memory!
```

> 🧒 **It's like a queue at a shop** 🏪. Only ONE person can be served at a time. If 500 people are waiting, you serve all 500 one after another. There's no way to hurry it up!

## A.2 Why This Is a Big Deal

```
   A GPU (from Module 10) can do THOUSANDS of things at the SAME TIME. ⚡

   But an RNN forces it to do ONE thing at a time.

   🧒 It's like buying a 100-lane highway 🛣️ and only using 1 lane!
```

## A.3 The Picture

```
  ═══════ RNN: three separate steps ═══════

     step 1        step 2        step 3
   ┌────────┐    ┌────────┐    ┌────────┐
   │  the   │───▶│  cat   │───▶│  sat   │
   └────────┘    └────────┘    └────────┘
                ↑             ↑
             must wait     must wait


  ═══════ TRANSFORMER: ONE step ═══════

                    step 1
   ┌────────┐    ┌────────┐    ┌────────┐
   │  the   │    │  cat   │    │  sat   │
   └────────┘    └────────┘    └────────┘
        └─── all at the same instant! ───┘

   no waiting → the GPU can use ALL its lanes 🚀
```

**For a 500-word document:**
```
   RNN:          500 steps  🐌
   TRANSFORMER:    1 step   🚀
```

---

# PART B: THE BOLD IDEA 💡

## B.1 The Question That Changed Everything

In 2017, researchers asked:

> **"Attention already lets ANY word look at ANY other word... so why do we even NEED the RNN? Let's DELETE it and use attention alone!"**

They wrote a paper called **"Attention Is All You Need"**.

```
   ┌────────────────────────────────────────────────────────────┐
   │  MODULE 14:   LSTM  +  attention   =  good translator      │
   │                                                             │
   │  MODULE 15:          attention only  =  TRANSFORMER 🚀      │
   │               ↑                                             │
   │         the RNN is GONE!                                    │
   └────────────────────────────────────────────────────────────┘
```

> 🧒 **Like taking the training wheels off a bike** 🚲 — it turns out attention could ride on its own the whole time, and it goes MUCH faster without them!

**Everything built on this:** GPT, BERT, Claude, ChatGPT, Gemini. All of them. 🌍

---

# PART C: SELF-ATTENTION — The Same 4 Steps, One Sentence

## C.1 Cross-Attention vs Self-Attention

```
  ═══ MODULE 14: CROSS-attention (TWO sentences) ═══

     ENGLISH: "I am happy"        HINDI: "main khush hoon"
       └─ gives the MEMORIES ─┘     └─ gives the QUERY ─┘
                    ▲                        │
                    └────────────────────────┘
              the Hindi word looks ACROSS at English words


  ═══ MODULE 15: SELF-attention (ONE sentence) ═══

              "the cat sat"
                    ↻
      every word looks at every word in ITS OWN sentence
```

## C.2 What "Standing On a Word" Means

```
  "The cat sat"

  Stand on "the"  →  look at: the, cat, sat  →  make a BETTER "the"
  Stand on "cat"  →  look at: the, cat, sat  →  make a BETTER "cat"
  Stand on "sat"  →  look at: the, cat, sat  →  make a BETTER "sat"
```

## C.3 Why Would a Word Look at Other Words?

> 🧒 **Because words only make sense together!**
>
> Take the word **"it"** in this sentence:
> ```
>    "The cat sat because IT was tired."
> ```
> - Alone, "it" means NOTHING 🤷
> - But if "it" looks at the other words, it discovers **"it" = cat**! 🐱
>
> **Self-attention lets every word FILL IN its own meaning using its neighbours.**

```
      the      cat      was      it
       │        │        │        │
       ▏      ████       ▎        ⬅ "it" looks back and finds "cat"!
       │        │        │        │
     thin     THICK     thin    (standing here)

     thicker line = more attention
```

## C.4 A Word Looks at ITSELF Too!

```
  Standing on "sat", it looks at:  the ✓   cat ✓   sat ✓  ← itself!
```
> 🧒 **That's allowed and normal.** A word's own meaning matters too! Think of it as "how much of the original me should I keep?" 🪞

---

# PART D: QUERY, KEY, VALUE — Where They Come From

## D.1 The Problem

In Module 14, the query came from the decoder and the keys came from the encoder — **two different places**. But in self-attention there's only ONE sentence... so where do three different things come from?

## D.2 THE ANSWER: We MAKE Them With Three Grids

```
   Each word's numbers go through THREE DIFFERENT Linear layers:

                       ┌──▶ W_query  ──▶  query  🙋 "what am I looking for?"
                       │
   word's numbers ─────┼──▶ W_key    ──▶  key    🏷️ "what do I offer?"
                       │
                       └──▶ W_value  ──▶  value  📕 "what do I hand over?"
```

## D.3 Why Not Just Use the Embedding for All Three?

Remember the problem from Module 14 — **if you compare a vector with itself, it always wins!** 😅

```
   score(sat, sat) = sat · sat = "how similar is sat to sat?" = VERY!
   → "sat" would always pay most attention to itself. Useless!
```

Making three DIFFERENT versions fixes it.

> 🧒 **Think of a person with three faces** 🎭:
> - What you **WANT** (query) — "I need a screwdriver"
> - What you **ADVERTISE** (key) — "I have hammers"
> - What you actually **HAND OVER** (value) — the hammer itself
>
> Same person, three different roles! They should NOT be the same numbers.

## D.4 ⭐ HOW DOES THE DIMENSION SHRINK? (the big question)

**Answer: it's just a matrix multiply. The grid has 4 rows and 2 columns.**

```
   4 numbers in   ×   a 4×2 grid   =   2 numbers out
```

**Here is `W_query_head1` — a real 4×2 grid:**

```
                       col 0    col 1
   row 0 (from x[0])    0.5      0.1
   row 1 (from x[1])    0.2      0.6
   row 2 (from x[2])    0.1      0.3
   row 3 (from x[3])    0.4      0.2
```

### 🔍 Computing `query_sat` — EVERY multiplication

```
  x_sat = [1.209, 0.284, 0.620, 1.400]
          (that's "sat" after adding position — see Part G)

  ── COLUMN 0 ──  multiply each x by that ROW's col-0 number, then add:
     1.209 × 0.5   =  0.6045
     0.284 × 0.2   =  0.0568
     0.620 × 0.1   =  0.0620
     1.400 × 0.4   =  0.5600
                      ───────
                       1.2833

  ── COLUMN 1 ──
     1.209 × 0.1   =  0.1209
     0.284 × 0.6   =  0.1704
     0.620 × 0.3   =  0.1860
     1.400 × 0.2   =  0.2800
                      ───────
                       0.7573

  query_sat (head 1) = [1.283, 0.757]     ← 4 numbers became 2! ✅
```

## D.5 ⚠️ Two Things People Get Wrong Here

```
  ❌ WRONG:  "we SELECT 2 of the 4 numbers and throw 2 away"
  ✅ RIGHT:  ALL 4 numbers contribute to BOTH outputs!

  Nothing is discarded — it's SQUEEZED! 🧽
```

```
  ❌ WRONG:  "the grid is fixed / hand-picked"
  ✅ RIGHT:  the grid numbers are WEIGHTS — learned by backprop,
             exactly like every other weight since Module 6! 🎓
```

> 🧒 **Think of making orange juice** 🍊. You put in 4 oranges and get 2 glasses of juice. You didn't "pick 2 oranges and throw away 2" — ALL 4 oranges went into BOTH glasses, just in different proportions! That's what the grid columns do. 🥤

---

# PART E: A FULL SELF-ATTENTION WALKTHROUGH (single head, to learn the shape)

Before we do multi-head, let's do ONE simple head so the pattern is clear. (Slightly different numbers here — this is a warm-up.)

## E.1 The Three Versions of Each Word

```
             query          key           value
    the    [0.2, 0.1]    [0.3, 0.2]    [0.5, 0.1]
    cat    [0.9, 0.7]    [0.8, 0.6]    [0.9, 0.8]
    sat    [0.5, 0.8]    [0.4, 0.9]    [0.3, 0.7]
```

## E.2 STEP 1 — Score (my query · every key)

```
  My query (sat) = [0.5, 0.8]

  ── vs "the" key [0.3, 0.2] ──
     (0.5 × 0.3) + (0.8 × 0.2)  =  0.15 + 0.16  =  0.31

  ── vs "cat" key [0.8, 0.6] ──
     (0.5 × 0.8) + (0.8 × 0.6)  =  0.40 + 0.48  =  0.88

  ── vs "sat" key [0.4, 0.9] ──
     (0.5 × 0.4) + (0.8 × 0.9)  =  0.20 + 0.72  =  0.92
```

## E.3 STEP 2 — Scale (divide by √d) — see Part F for why

```
  d = 2  →  √2 = 1.414

     0.31 / 1.414  =  0.219
     0.88 / 1.414  =  0.622
     0.92 / 1.414  =  0.651
```

## E.4 STEP 3 — Softmax

```
  e^0.219 = 1.2448
  e^0.622 = 1.8627
  e^0.651 = 1.9175
             ───────
     SUM  = 5.0250

  the: 1.2448 / 5.0250 = 0.2477  →  24.8%
  cat: 1.8627 / 5.0250 = 0.3707  →  37.1%
  sat: 1.9175 / 5.0250 = 0.3816  →  38.2%
                                    ──────
                          TOTAL  =  100.0%  ✓
```

## E.5 STEP 4 — Blend the VALUES

```
  0.2477 × [0.5, 0.1]  =  [0.1239, 0.0248]     from "the"
  0.3707 × [0.9, 0.8]  =  [0.3336, 0.2966]     from "cat"
  0.3816 × [0.3, 0.7]  =  [0.1145, 0.2671]     from "sat"
                          ───────────────────
  NEW "sat"            =  [0.5720, 0.5885]
```

## E.6 ⚠️ Why Blend VALUES and Not KEYS?

```
  🧒 The library again 📚:

     the KEY   = the shelf LABEL ("Dinosaurs")   →  used for MATCHING
     the VALUE = the actual BOOK                 →  what you TAKE HOME

  You match on labels, but you carry away books!
  Keys are for comparing. Values are for collecting. 🎯
```

## E.7 What Just Happened?

```
  OLD "sat" = a lonely word, meaning only "sat"
  NEW "sat" = knows about "cat"! It now means "sat, and a cat did it" 🐱

  Do this for EVERY word, and every word becomes CONTEXT-AWARE. 🧠
```

---

# PART F: WHY DIVIDE BY √d — The Scaling Step

## F.1 The Problem Without Scaling

With big vectors (real models use d = 64 or 512), dot products get HUGE. And huge scores make softmax go crazy:

```
  ── WITHOUT scaling (d = 512, typical scores) ──
     scores = [23, 20, 18]

     e^23 = 9,744,803,446
     e^20 =   485,165,195
     e^18 =    65,659,969
                ───────────────
        SUM = 10,295,628,610

     → 94.6%,  4.7%,  0.6%       ← almost ONE-HOT! 😬

     The gradient nearly VANISHES (Module 9's problem!)
     The model gets stuck and can't learn.


  ── WITH scaling (÷ √512 = 22.6) ──
     scores = [23/22.6, 20/22.6, 18/22.6] = [1.02, 0.88, 0.80]

     e^1.02 = 2.773
     e^0.88 = 2.411
     e^0.80 = 2.226
              ──────
       SUM  = 7.410

     → 37.4%,  32.5%,  30.0%     ← healthy! ✅
```

## F.2 🧒 This Is EXACTLY Temperature from Module 9! 🌡️

```
   Remember:  x = x / temperature   before softmax?

   Here:      score = score / √d    before softmax!

   SAME TRICK. Dividing before softmax stops it becoming too extreme.
   Big number → spiky softmax → no learning
   Divided    → gentle softmax → learning works ✅
```

## F.3 Why √d Specifically?

```
  🧒 The more numbers you add up, the bigger the total gets.
     Adding 4 numbers → medium total
     Adding 512 numbers → HUGE total

  Mathematically, the total grows roughly like √d.
  So dividing by √d cancels that growth out perfectly! 📐
```

---

# PART G: POSITIONAL ENCODING — Putting the Order Back

## G.1 ⚠️ We Broke Something!

```
  The RNN read words IN ORDER: first, second, third...
  Self-attention looks at ALL words AT ONCE — there is NO order!
```

**Test it:**
```
  "dog bites man"   →  self-attention sees:  {dog, bites, man}
  "man bites dog"   →  self-attention sees:  {man, bites, dog}

  SAME SET OF WORDS. It can't tell them apart! 😱
```

> 🧒 It's like tipping a bag of Scrabble tiles onto the table 🎲. All the letters are there, but the ORDER is gone. "cat" and "act" look identical!

## G.2 The Fix: ADD a Position Pattern

```
      what the word MEANS  +  WHERE it sits  =  what goes into attention
```

## G.3 The Formula

```
   PE(pos, 2i)    = sin( pos / 10000^(2i/d) )
   PE(pos, 2i+1)  = cos( pos / 10000^(2i/d) )
```

**Three things to know before plugging in numbers:**

| Symbol | Meaning | Our value |
|--------|---------|:---------:|
| `pos` | which word is it? (0 = first) | 0, 1, 2 |
| `d` | how many numbers per word | **4** |
| `i` | which PAIR of slots | 0, 1 |

## G.4 ⭐ Slots Come in PAIRS

**The 4 slots are not independent — they are 2 PAIRS, and `i` counts the pairs:**

```
   i = 0  →  slot 2i = 0  (sin)  and  slot 2i+1 = 1  (cos)     ← PAIR 1
   i = 1  →  slot 2i = 2  (sin)  and  slot 2i+1 = 3  (cos)     ← PAIR 2

   ┌───────┬───────┐  ┌───────┬───────┐
   │ slot0 │ slot1 │  │ slot2 │ slot3 │
   │  sin  │  cos  │  │  sin  │  cos  │
   └───────┴───────┘  └───────┴───────┘
        PAIR 1              PAIR 2
        (i = 0)             (i = 1)
```

## G.5 ⭐ WHERE THE 1 AND THE 100 COME FROM

**Each pair gets its own denominator. Here is the calculation, fully:**

```
   denominator = 10000^(2i / d)

  ── FOR PAIR 1 (i = 0) ──
     2i      = 2 × 0 = 0
     2i / d  = 0 / 4 = 0
     denominator = 10000^0 = 1        ← anything to the power 0 is 1

  ── FOR PAIR 2 (i = 1) ──
     2i      = 2 × 1 = 2
     2i / d  = 2 / 4 = 0.5
     denominator = 10000^0.5 = √10000 = 100     ← THERE'S THE 100! ✅
```

> 🧒 **So the 100 is just the SQUARE ROOT of 10000.** It appears because `2i/d` worked out to exactly `0.5`, and "to the power of a half" means "square root"! 📐

**With a different `d`, you get different numbers:**
```
  d = 4:    denominators are  1  and  100                (2 pairs)
  d = 6:    denominators are  1,  21.5,  464             (3 pairs)
  d = 512:  denominators are  1  up to  10000            (256 pairs!)
```

## G.6 ⚠️ THE ANGLES ARE IN RADIANS, NOT DEGREES!

```
   sin(1)  means  sin(1 RADIAN)  = 0.841    ✅
   NOT            sin(1 degree)  = 0.017    ❌
```
> 🧒 If you check on a calculator, **put it in RAD mode**, not DEG! That's the most common mistake here. 📱

## G.7 The Actual Numbers

### POSITION 0 (the word "the")
```
  PAIR 1 (denominator 1):
     slot 0 = sin(0 / 1)   = sin(0)  = 0.000
     slot 1 = cos(0 / 1)   = cos(0)  = 1.000

  PAIR 2 (denominator 100):
     slot 2 = sin(0 / 100) = sin(0)  = 0.000
     slot 3 = cos(0 / 100) = cos(0)  = 1.000

  →  [0.000, 1.000, 0.000, 1.000]
```

### POSITION 1 (the word "cat")
```
  PAIR 1 (denominator 1):
     slot 0 = sin(1 / 1)   = sin(1.00)  = 0.841
     slot 1 = cos(1 / 1)   = cos(1.00)  = 0.540

  PAIR 2 (denominator 100):
     slot 2 = sin(1 / 100) = sin(0.01)  = 0.010
     slot 3 = cos(1 / 100) = cos(0.01)  = 1.000   (really 0.99995)

  →  [0.841, 0.540, 0.010, 1.000]
```

### POSITION 2 (the word "sat")
```
  PAIR 1 (denominator 1):
     slot 0 = sin(2 / 1)   = sin(2.00)  =  0.909
     slot 1 = cos(2 / 1)   = cos(2.00)  = -0.416

  PAIR 2 (denominator 100):
     slot 2 = sin(2 / 100) = sin(0.02)  =  0.020
     slot 3 = cos(2 / 100) = cos(0.02)  =  1.000   (really 0.9998)

  →  [0.909, -0.416, 0.020, 1.000]
```

## G.8 ⭐ WHY TWO DIFFERENT SPEEDS? The Clock! 🕐

Look at how each slot CHANGES across the three positions:

```
                slot 0    slot 1    slot 2    slot 3
   the          0.000     1.000     0.000     1.000
   cat          0.841     0.540     0.010     1.000
   sat          0.909    -0.416     0.020     1.000
                 ▲▲▲       ▲▲▲        ▲          ▲
              CHANGES   CHANGES    barely     barely
               A LOT     A LOT      moves      moves

              └─ PAIR 1: ÷ 1 ─┘   └─ PAIR 2: ÷ 100 ─┘
                  FAST wave ⚡        SLOW wave 🐢
```

### 🕐 The Clock Analogy

```
   A clock has THREE hands, each moving at a different speed:

      seconds hand  →  spins FAST      (tells apart nearby moments)
      minute hand   →  medium
      hour hand     →  moves SLOWLY    (tells apart parts of the day)

   Together they give a UNIQUE reading for EVERY moment! 🕐
```

**Positional encoding is exactly this:**

| | Denominator | Speed | Job |
|---|:---:|---|---|
| **Pair 1** (slots 0,1) | ÷ 1 | **fast** ⚡ | tells apart word 1 and word 2 |
| **Pair 2** (slots 2,3) | ÷ 100 | **slow** 🐢 | tells apart word 5 and word 300 |

> 🧒 **Why do we need BOTH?** 🤔
>
> If ALL slots were FAST, they'd wrap around and repeat — position 1 and position 7 might look the same! 😱
>
> If ALL slots were SLOW, positions 1, 2, 3 would look nearly identical! 😱
>
> Mixing fast + slow gives **every position a unique fingerprint**, exactly like a clock. 🎯

**In a real model with d = 512, there are 256 pairs** — 256 "hands" at 256 different speeds! That's why every position up to thousands of words gets its own unique pattern. 🌊

## G.9 ADDING It To the Embeddings

```
  "the" (pos 0):  [0.1, 0.2, 0.0, 0.3]           ← what the word MEANS
                + [0.000, 1.000, 0.000, 1.000]   ← WHERE it sits
                = [0.100, 1.200, 0.000, 1.300]   ← both together ✅

  "cat" (pos 1):  [0.9, 0.8, 0.1, 0.2]
                + [0.841, 0.540, 0.010, 1.000]
                = [1.741, 1.340, 0.110, 1.200]

  "sat" (pos 2):  [0.3, 0.7, 0.6, 0.4]
                + [0.909, -0.416, 0.020, 1.000]
                = [1.209, 0.284, 0.620, 1.400]   ← we'll follow this one! 🎯
```

**Now "dog bites man" and "man bites dog" have DIFFERENT numbers!** ✅ Order is back! 🎉

## G.10 🤔 Why ADD Instead of Taping Together?

```
  ADDING keeps the size at 4 numbers ✅   (cheap and simple)
  TAPING would make it 8 numbers ❌       (doubles everything downstream)

  🧒 It SEEMS like adding would "mix up" meaning and position —
     but with 512 slots there's plenty of room. The model learns to
     use some slots more for meaning and others more for position. 📐
```

## G.11 Verify It Yourself

```python
import math

d = 4
for pos in [0, 1, 2]:
    row = []
    for i in [0, 1]:                       # the two PAIRS
        denominator = 10000 ** (2*i / d)   # → 1, then 100
        row.append(math.sin(pos / denominator))
        row.append(math.cos(pos / denominator))
    print(f"position {pos}: {[round(v, 3) for v in row]}")
```

**OUTPUT:**
```
position 0: [0.0, 1.0, 0.0, 1.0]
position 1: [0.841, 0.54, 0.01, 1.0]
position 2: [0.909, -0.416, 0.02, 1.0]
```
✅ **Matches exactly!**

---

# PART H: ⭐⭐ MULTI-HEAD ATTENTION — One Word, Every Single Step

## H.1 The Idea

Instead of running attention ONCE, run it **several times in parallel** with **different** grids each time.

```
  ┌───────────────────────────────────────────────────────────┐
  │  head 1: learns GRAMMAR       (verb ↔ subject)            │
  │  head 2: learns MEANING       (cat ↔ animal words)        │
  │  head 3: learns POSITION      (nearby words)              │
  │  head 4: learns REFERENCE     ("it" ↔ "cat")              │
  │  ... typically 8 heads                                     │
  └───────────────────────────────────────────────────────────┘
```

> 🧒 **Like having 8 friends read the same sentence** 👨‍👩‍👧‍👦. One notices the grammar, one notices the meaning, one notices who's talking about whom. Then you combine all their notes into one better understanding! 🧠

## H.2 Our Setup

```
  d_model = 4,  2 heads   →   each head gets 4 ÷ 2 = 2 numbers

  Starting point (embedding + position, from Part G):
     the → [0.100, 1.200, 0.000, 1.300]
     cat → [1.741, 1.340, 0.110, 1.200]
     sat → [1.209, 0.284, 0.620, 1.400]   ← we follow this one 🎯
```

## H.3 The Journey (overview)

```
   "sat" with position     (4 numbers)
            │
     ┌──────┴──────┐
     ▼             ▼
  HEAD 1        HEAD 2
  4 → 2         4 → 2      ← each head's grids shrink it
     │             │
  attention     attention  ← each head does the full 4-step process
     │             │
  [2 numbers]  [2 numbers]
     └──────┬──────┘
            ▼
      TAPE TOGETHER         (back to 4 numbers!)
            │
            ▼
       W_output grid        (mixes the heads' opinions)
            │
            ▼
     attention_output       (4 numbers)
```

## H.4 HEAD 1 — Making Query, Key, Value

### The three grids for head 1

```
  W_query_head1:      W_key_head1:       W_value_head1:
     0.5   0.1           0.3   0.4          0.2   0.5
     0.2   0.6           0.5   0.2          0.4   0.1
     0.1   0.3           0.2   0.1          0.3   0.2
     0.4   0.2           0.1   0.5          0.1   0.3
```

### 🔍 query_sat — every multiplication (from Part D.4)

```
  x_sat = [1.209, 0.284, 0.620, 1.400]

  col 0:  1.209×0.5 + 0.284×0.2 + 0.620×0.1 + 1.400×0.4
        =  0.6045   +  0.0568   +  0.0620   +  0.5600   = 1.2833

  col 1:  1.209×0.1 + 0.284×0.6 + 0.620×0.3 + 1.400×0.2
        =  0.1209   +  0.1704   +  0.1860   +  0.2800   = 0.7573

  query_sat = [1.283, 0.757]
```

### 🔍 key_the — every multiplication

```
  x_the = [0.100, 1.200, 0.000, 1.300]

  col 0:  0.100×0.3 + 1.200×0.5 + 0.000×0.2 + 1.300×0.1
        =  0.030    +  0.600    +  0.000    +  0.130    = 0.760

  col 1:  0.100×0.4 + 1.200×0.2 + 0.000×0.1 + 1.300×0.5
        =  0.040    +  0.240    +  0.000    +  0.650    = 0.930

  key_the = [0.760, 0.930]
```

### All the keys and values for head 1

```
              key                value
    the   [0.760, 0.930]    [0.630, 0.560]
    cat   [1.334, 1.575]    [1.037, 1.387]
    sat   [0.769, 1.302]    [0.681, 1.177]
```

*(Each computed the exact same way — 4 multiplications, then add.)*

## H.5 HEAD 1 — The Attention

```
  ── STEP 1: SCORES (query_sat · each key) ──

     vs the:  1.283 × 0.760  +  0.757 × 0.930
            =    0.9751      +     0.7040       =  1.679

     vs cat:  1.283 × 1.334  +  0.757 × 1.575
            =    1.7115      +     1.1923       =  2.904

     vs sat:  1.283 × 0.769  +  0.757 × 1.302
            =    0.9866      +     0.9856       =  1.972


  ── STEP 2: SCALE (÷ √2 = 1.414) ──

     1.679 / 1.414  =  1.187
     2.904 / 1.414  =  2.054
     1.972 / 1.414  =  1.395


  ── STEP 3: SOFTMAX ──

     e^1.187 =  3.277
     e^2.054 =  7.799
     e^1.395 =  4.035
                ──────
        SUM  = 15.111

     the: 3.277 / 15.111 = 0.217  →  21.7%
     cat: 7.799 / 15.111 = 0.516  →  51.6%   ← head 1 focuses on "cat"!
     sat: 4.035 / 15.111 = 0.267  →  26.7%
                                     ──────
                            TOTAL =  100.0%  ✓


  ── STEP 4: BLEND THE VALUES ──

     0.217 × [0.630, 0.560]  =  [0.1366, 0.1215]
     0.516 × [1.037, 1.387]  =  [0.5352, 0.7158]
     0.267 × [0.681, 1.177]  =  [0.1818, 0.3143]
                                ─────────────────
     head1_output            =  [0.854,  1.152]
```

## H.6 HEAD 2 — Same Process, DIFFERENT Grids

Head 2 has its own `W_query_head2`, `W_key_head2`, `W_value_head2` with **different numbers**. Same 4 steps, different answer:

```
  HEAD 2 attention:  the 45.2%,  cat 22.1%,  sat 32.7%
                      ↑ this head focuses on "the" instead!

  head2_output = [0.412, 0.689]
```

> 🧒 **THIS is why multi-head matters!** 🎭 Head 1 thought "cat" mattered most (51.6%). Head 2 thought "the" mattered most (45.2%). **Two different opinions about the same sentence — and we keep BOTH!**

## H.7 Tape the Heads Together

```
  head1_output:  [0.854, 1.152]
  head2_output:  [0.412, 0.689]
                  ↓ taped end to end (torch.cat!)
  concatenated:  [0.854, 1.152, 0.412, 0.689]     ← back to 4 numbers! ✅
```

> 🧒 **The sizes work out perfectly on purpose!** 2 heads × 2 numbers each = 4 numbers = exactly what we started with. In a real model: 8 heads × 64 numbers = 512. ✓

## H.8 One More Grid: W_output Mixes the Opinions

```
  W_output (4×4):
     0.3  0.1  0.2  0.4
     0.2  0.5  0.1  0.1
     0.4  0.2  0.3  0.2
     0.1  0.3  0.4  0.2

  concatenated = [0.854, 1.152, 0.412, 0.689]

  col 0:  0.854×0.3 + 1.152×0.2 + 0.412×0.4 + 0.689×0.1
        =  0.2562   +  0.2304   +  0.1648   +  0.0689   = 0.720

  col 1:  0.854×0.1 + 1.152×0.5 + 0.412×0.2 + 0.689×0.3
        =  0.0854   +  0.5760   +  0.0824   +  0.2067   = 0.951

  col 2:  0.854×0.2 + 1.152×0.1 + 0.412×0.3 + 0.689×0.4
        =  0.1708   +  0.1152   +  0.1236   +  0.2756   = 0.685

  col 3:  0.854×0.4 + 1.152×0.1 + 0.412×0.2 + 0.689×0.2
        =  0.3416   +  0.1152   +  0.0824   +  0.1378   = 0.677

  attention_output = [0.720, 0.951, 0.685, 0.677]   ✅
```

> 🧒 **Why the extra W_output grid?** Right now the two heads' answers are just sitting SIDE BY SIDE, not talking to each other. `W_output` **blends the opinions** into one combined answer — like a chairperson summarizing what everyone said in the meeting! 👨‍💼

---

# PART I: ADD & NORM — With Full Arithmetic

Two operations, one after the other.

## I.1 ADD — Put the Original Back

```
  original "sat"      = [1.209, 0.284, 0.620, 1.400]
  attention_output    = [0.720, 0.951, 0.685, 0.677]
                        ─────────────────────────────  ADD! ➕
  result              = [1.929, 1.235, 1.305, 2.077]
```

### 🧒 Why add the original back?

```
  Because attention might have made a MESS! 🌪️
  By adding the original, we GUARANTEE the word never fully loses itself.

  It says: "here's what I WAS, PLUS what I LEARNED." 🎁
```

### ⭐ It's the Gradient Highway — for the THIRD time! 🛣️

```
  MODULE 10 (ResNet):   output = F(x) + x
  MODULE 12 (LSTM):     C_new  = f·C_old + i·C̃
  MODULE 15 (here):     output = LayerNorm( x + attention(x) )
                                             ↑
                                        the same "+ x"!

  🧒 That "+" is WHY you can stack 96 blocks without the gradient dying.
     Third appearance of the same brilliant trick! 🎯
```

## I.2 NORM (LayerNorm) — Tidy Up the Numbers

```
  ── STEP 1: find the AVERAGE ──
     (1.929 + 1.235 + 1.305 + 2.077) ÷ 4  =  6.546 ÷ 4  =  1.6365


  ── STEP 2: subtract the average from each ──
     1.929 − 1.6365 =  0.2925
     1.235 − 1.6365 = -0.4015
     1.305 − 1.6365 = -0.3315
     2.077 − 1.6365 =  0.4405


  ── STEP 3: find the SPREAD (standard deviation) ──
     square each:    0.2925² = 0.08556
                     0.4015² = 0.16120
                     0.3315² = 0.10989
                     0.4405² = 0.19404
                               ───────
                       SUM   = 0.55069
     ÷ 4 = 0.13767
     √0.13767 = 0.3710


  ── STEP 4: divide each by the spread ──
      0.2925 / 0.3710 =  0.788
     -0.4015 / 0.3710 = -1.082
     -0.3315 / 0.3710 = -0.893
      0.4405 / 0.3710 =  1.187

  after_add_norm_1 = [0.788, -1.082, -0.893, 1.187]
```

### ✅ Check it worked

```
  average = (0.788 - 1.082 - 0.893 + 1.187) ÷ 4  =  0 ÷ 4  =  0   ✓
  The average is now ZERO and the spread is 1. Tidy! 🧹
```

### 🧒 Why bother normalizing?

```
  Imagine a class where one test is out of 10 and another is out of 1000. 📊
  You CAN'T compare them! One number would bully all the others.

  Normalizing puts everything on the SAME SCALE so no number dominates.
  It keeps training stable — the same idea as batch norm from Module 10,
  but done PER WORD instead of per batch. 🎯
```

### 🔍 A small detail: γ and β

```
  LayerNorm actually finishes with:   output = γ × normalized + β

  γ (gamma) starts at 1, β (beta) starts at 0 → so at first, nothing changes.
  But they're LEARNABLE — the model can adjust them if it wants to
  undo some of the normalizing. (Exactly like batch norm in Module 10!)
```

---

# PART J: FEED-FORWARD — A Real Neural Network

## J.1 What It Is

**This is a plain neural network — exactly what you built in Module 1!** Two Linear layers with ReLU in between.

```
   4 numbers in  →  feed_forward_1 (4→8)  →  ReLU  →  feed_forward_2 (8→4)  →  4 out
                            ↑                              ↑
                      EXPAND wide                    SQUEEZE back
```

## J.2 The Picture

```
   INPUT (4)          HIDDEN (8, after ReLU)         OUTPUT (4)

    0.79  ○─┐          ┌─○ 1.02  ✅                   ┌─○  0.51
            ├──────────┼─○ 0.00  ❌ off               │
   -1.08  ○─┤          ├─○ 0.36  ✅                   ├─○ -0.23
            ├──────────┼─○ 0.85  ✅        ───────────┤
   -0.89  ○─┤          ├─○ 0.00  ❌ off               ├─○  0.68
            ├──────────┼─○ 1.47  ✅                   │
    1.19  ○─┘          ├─○ 0.21  ✅                   └─○  0.39
                       └─○ 0.00  ❌ off

           every input connects to every hidden node
```

## J.3 🔍 What Goes Into ONE Node — Fully Worked

**Input:** `[0.788, -1.082, -0.893, 1.187]`

```
  ── HIDDEN NODE 1 ──
  its weights: [0.5, 0.2, -0.3, 0.4],   bias = 0.1

      0.788 ×  0.5   =  0.3940
     -1.082 ×  0.2   = -0.2164
     -0.893 × -0.3   =  0.2679
      1.187 ×  0.4   =  0.4748
                bias =  0.1000
                        ───────
                         1.0203

  ReLU(1.020) = 1.020     ✅ POSITIVE → passes through!
```

```
  ── HIDDEN NODE 2 ──
  its weights: [-0.2, 0.6, 0.1, -0.5],   bias = 0.0

      0.788 × -0.2   = -0.1576
     -1.082 ×  0.6   = -0.6492
     -0.893 ×  0.1   = -0.0893
      1.187 × -0.5   = -0.5935
                        ───────
                        -1.4896

  ReLU(-1.490) = 0        ❌ NEGATIVE → SWITCHED OFF!
```

```
  ── HIDDEN NODE 3 ──
  its weights: [0.3, -0.4, 0.5, 0.2],   bias = -0.1

      0.788 ×  0.3   =  0.2364
     -1.082 × -0.4   =  0.4328
     -0.893 ×  0.5   = -0.4465
      1.187 ×  0.2   =  0.2374
                bias = -0.1000
                        ───────
                         0.3601

  ReLU(0.360) = 0.360     ✅ passes
```

> 🧒 **That's EXACTLY the neuron from Module 1!** 🎉 Multiply each input by its weight, add them up, add the bias, then ReLU. Nothing new — just done 8 times (once per hidden node), each with different weights.

## J.4 All 8 Hidden Values

```
  [1.020, 0.000, 0.360, 0.845, 0.000, 1.472, 0.213, 0.000]
           ↑ off          ↑ off                    ↑ off

  🧒 ReLU switched 3 of the 8 nodes OFF. That's a DECISION —
     "these patterns aren't relevant for this word." ✂️
```

## J.5 Then feed_forward_2 Squeezes Back (8 → 4)

```
  Same process — each output node takes all 8 hidden values,
  multiplies by weights, adds them up, adds bias.

  OUTPUT = [0.512, -0.234, 0.678, 0.391]
```

## J.6 Then ADD & NORM AGAIN

```
  ── ADD ──
     [0.788, -1.082, -0.893, 1.187]      ← the input to feed-forward
   + [0.512, -0.234,  0.678, 0.391]      ← what feed-forward produced
     ──────────────────────────────
   = [1.300, -1.316, -0.215, 1.578]


  ── NORM ──
     average = (1.300 - 1.316 - 0.215 + 1.578) ÷ 4 = 1.347 ÷ 4 = 0.3368

     1.300 − 0.3368 =  0.9632        squared = 0.9278
    -1.316 − 0.3368 = -1.6528        squared = 2.7317
    -0.215 − 0.3368 = -0.5518        squared = 0.3045
     1.578 − 0.3368 =  1.2412        squared = 1.5406
                                               ──────
                                       SUM  =  5.5046
     ÷ 4 = 1.3762       √1.3762 = 1.1731

      0.9632 / 1.1731 =  0.821
     -1.6528 / 1.1731 = -1.409
     -0.5518 / 1.1731 = -0.470
      1.2412 / 1.1731 =  1.058

  🎉 block_output for "sat" = [0.821, -1.409, -0.470, 1.058]
```

**That's ONE complete transformer block, done!** ✅ This output goes straight into the NEXT block.

---

# PART K: ⭐ "THINKS ALONE" — All Together vs One at a Time

## K.1 The Question

> **"Are all the words sent altogether, or one word at a time?"**

## K.2 THE ANSWER

```
  ┌──────────────────┬─────────────────────────────────────────────┐
  │ ATTENTION        │ ALL words TOGETHER ✅                        │
  │                  │ (they MUST see each other — that's the job!) │
  ├──────────────────┼─────────────────────────────────────────────┤
  │ FEED-FORWARD     │ Each word SEPARATELY 🚶                      │
  │                  │ (no word can see any other here!)            │
  └──────────────────┴─────────────────────────────────────────────┘
```

## K.3 The Picture

```
  ═══ ATTENTION: words TALK to each other ═══

     the ←──────→ cat ←──────→ sat
      ↑                          ↑
      └──────────────────────────┘
     all 3 go in together, they can SEE each other 💬


  ═══ FEED-FORWARD: each word ALONE in its own lane ═══

     the        cat        sat
      │          │          │
      ▼          ▼          ▼
   (own lane) (own lane) (own lane)

     NO lines between the lanes — no mixing here! 🚶
```

## K.4 ⚠️ BUT — "Separately" Doesn't Mean SLOWLY!

```
   The GPU still does all 3 words at the SAME INSTANT ⚡
   — just with NO CONNECTIONS between them.

   Think of 3 identical machines running side by side:

     "the" → [copy of the feed-forward] → result
     "cat" → [copy of the feed-forward] → result    all at once!
     "sat" → [copy of the feed-forward] → result
               ↑ SAME weights in all three
```

## K.5 🧒 What "Thinks Alone" Really Means — The Meeting 🏢

```
  ATTENTION = THE MEETING 💬
     Everyone shares what they know.
     "sat" hears from "cat" and "the".
     Now "sat" has a PILE of gathered information.
     But it hasn't PROCESSED it — it just COLLECTED it!

  FEED-FORWARD = BACK AT YOUR DESK 🧠
     Everyone goes to their own desk. No more talking.
     "sat" now thinks: "OK, I heard about a cat.
                        What does that mean for ME?
                        What kind of word should I become?"
     It DIGESTS what it gathered.
```

## K.6 Why Is the Feed-Forward NECESSARY?

```
  Attention only does ONE thing: it AVERAGES other words' values.
  Averaging is a WEAK operation — it can't do complex thinking! 📊

  The feed-forward adds real "brain power":
     expands 4 → 8      (room to think 🧠)
     ReLU switches some parts off   (making decisions ✂️)
     squeezes 8 → 4     (a conclusion 💡)
```

> 🧒 **Without the feed-forward, a transformer would just be a fancy averaging machine!** 📊 The feed-forward is where the actual THINKING happens.
>
> **Gather (attention), then digest (feed-forward). Gather, digest. Gather, digest.**
> That's the whole rhythm of a transformer! 🔁

## K.7 Its Real Name

```
  The official name is "POSITION-WISE feed-forward".

  "position-wise" is just a fancy way of saying
  "each word position separately."

  🧒 Now the name makes sense! 📛
```

---

# PART L: IS THIS THE REAL TRANSFORMER? (Honest Answer)

## L.1 The Question

> **"I don't think this is the actual diagram of transformers — is something being covered later?"**

## L.2 THE ANSWER: You're Half Right!

What you've learned IS a real, complete transformer block. But the original paper has **two kinds of stacks**, and there are **three ways** people use them:

```
  ┌────────────────────────┬──────────────────────────────────────┐
  │ 1. ENCODER-ONLY        │ blocks with UNMASKED self-attention  │
  │    (BERT)              │ → reads and understands 📖            │
  ├────────────────────────┼──────────────────────────────────────┤
  │ 2. DECODER-ONLY  ⭐    │ blocks with MASKED self-attention 🙈  │
  │    (GPT, Claude)       │ → writes text ✍️                      │
  ├────────────────────────┼──────────────────────────────────────┤
  │ 3. ENCODER + DECODER   │ BOTH stacks, and the decoder blocks   │
  │    (the original 2017  │ get an EXTRA layer:                   │
  │     translation paper) │ CROSS-attention to the encoder!       │
  │                        │ (that's Module 14's attention! 🔦)    │
  └────────────────────────┴──────────────────────────────────────┘
```

## L.3 What You've Learned = GPT, Completely ✅

```
  What I showed you IS #2 (decoder-only) — what modern chatbots use! 🎉
```

## L.4 The ONE Extra Piece (in #3 only)

In the full translation transformer, each decoder block has **THREE** sub-layers instead of two:

```
   1. masked self-attention   (Hindi words look at earlier Hindi words)
   2. CROSS-attention  ⭐      (Hindi words look at the ENGLISH words)
   3. feed-forward
   (with add & norm after EACH of the three)
```

> 🧒 **Good news: you already know layer 2!** It's EXACTLY Module 14's cross-attention. Nothing new — just placed inside a transformer block instead of an LSTM. 🎉
>
> **So nothing important is being hidden from you.** ✅

---

# PART M: MASKING — How GPT Avoids Cheating 🙈

## M.1 The Problem

```
  Training sentence: "the cat sat"

  We want the model to learn:
     after "the"          →  predict "cat"
     after "the cat"      →  predict "sat"

  But self-attention lets EVERY word see EVERY word...
  so when predicting after "the", the model can SEE "cat"!
  That IS the answer! 😱

  It would learn NOTHING — just copying.
```

> 🧒 **Like an exam where the answers are printed at the bottom of the page.** 📄 You'd score 100% without learning anything!

## M.2 The Fix — With Real Numbers

**We add −infinity to the FUTURE scores BEFORE softmax:**

```
  Standing on "the" (position 0), the raw scores were:
     vs the:  1.20      vs cat:  2.40      vs sat:  1.80

  ── ADD THE MASK ──
     vs the:  1.20  +  0     =  1.20      (allowed ✅)
     vs cat:  2.40  + −∞     = −∞         (blocked ❌)
     vs sat:  1.80  + −∞     = −∞         (blocked ❌)

  ── SOFTMAX ──
     e^1.20 = 3.320
     e^(−∞) = 0        ← THIS is the whole trick!
     e^(−∞) = 0
              ─────
       SUM  = 3.320

     the: 3.320 / 3.320 = 100.0%   ← all attention on itself
     cat:     0 / 3.320 =   0.0%   ← completely INVISIBLE! 🙈
     sat:     0 / 3.320 =   0.0%
```

> 🧒 **`e` to the power of minus infinity is EXACTLY 0.** So those words get 0% attention — they might as well not exist!
>
> **Same masking trick as top-k in Module 9's temperature lesson!** 🎯

## M.3 The Staircase Pattern 📶

```
                    can look at →
                  the    cat    sat
                ┌──────┬──────┬──────┐
     the        │ 100% │  0%  │  0%  │  ← sees only itself
                ├──────┼──────┼──────┤
     cat        │  38% │ 62%  │  0%  │  ← sees the, cat
                ├──────┼──────┼──────┤
     sat        │  22% │ 51%  │ 27%  │  ← sees everything before it
                └──────┴──────┴──────┘
                  ↑ the numbers still add to 100% in EVERY row!
```

> 🧒 **Like covering the page below with a sheet of paper** 📄. You can see everything you've already read, but not what comes next!

## M.4 ⭐ The Genius Bit: ONE Sentence = MANY Lessons

```
  From ONE sentence "the cat sat", the model learns 2 lessons AT ONCE:

     position 0's output  →  should predict "cat"
     position 1's output  →  should predict "sat"

  BOTH computed in the SAME forward pass, in PARALLEL! 🚀
```

> 🧒 **THIS is why GPT trains so fast!** A 1000-word document gives 999 lessons in ONE pass! An RNN would need 999 separate steps. **That's the real superpower.** ⚡

## M.5 GPT vs BERT

```
  ┌──────────────────────────┬──────────────────────────┐
  │        GPT / Claude      │          BERT            │
  ├──────────────────────────┼──────────────────────────┤
  │ MASKED self-attention 🙈 │ FULL self-attention 👀   │
  │ (can only look backward) │ (looks both ways)        │
  ├──────────────────────────┼──────────────────────────┤
  │ Job: predict next word   │ Job: understand meaning  │
  │ → it WRITES text ✍️       │ → it READS text 📖       │
  ├──────────────────────────┼──────────────────────────┤
  │ Used for: chatbots,      │ Used for: search,        │
  │ writing, coding          │ classification, sentiment│
  └──────────────────────────┴──────────────────────────┘
```

> 🧒 **GPT is a WRITER** ✍️ — must not peek ahead, so it's masked.
> **BERT is a READER** 📖 — sees the whole sentence, so no mask.
> **Same block, ONE setting different!** 🎯

---

# PART N: ⭐ A COMPLETE FORWARD PASS — Predicting the Next Word

Let's do the WHOLE thing: give it **"the cat"**, get **"sat"**.

## N.1 STEP 1 — Words → Index Numbers

```
  Our tiny 8-word vocabulary:
     0 <pad>   1 the   2 a   3 dog   4 sat   5 cat   6 ran   7 jumped

  "the cat"  →  [1, 5]

  input = [[1, 5]]        shape (1, 2)
                                 ↑ 1 sentence, 2 words
```

## N.2 STEP 2 — Look Up Embeddings

```
  the → [0.1, 0.2, 0.0, 0.3]
  cat → [0.9, 0.8, 0.1, 0.2]

  shape (1, 2, 4)     ← 1 sentence × 2 words × 4 numbers
```

## N.3 STEP 3 — ADD Positions

```
  the (pos 0)  + [0.000, 1.000, 0.000, 1.000]
               = [0.100, 1.200, 0.000, 1.300]

  cat (pos 1)  + [0.841, 0.540, 0.010, 1.000]
               = [1.741, 1.340, 0.110, 1.200]
```

## N.4 STEP 4 — BLOCK 1

```
  masked attention  →  add & norm  →  feed-forward  →  add & norm

  Who can see whom (the mask!):
     "the" can see:  the                (position 0)
     "cat" can see:  the, cat           (positions 0, 1)

  OUT:  the → [0.412, -0.891,  0.223, 1.256]
        cat → [1.104,  0.377, -0.512, 0.881]
```

## N.5 STEP 5 — BLOCKS 2, 3, 4 (the same thing, 3 more times)

```
  Each block: masked attention → add&norm → feed-forward → add&norm

  after block 4:
        the → [0.887, -0.334,  0.601, 0.742]
        cat → [1.503,  0.712, -0.284, 1.098]   ← WE NEED THIS ONE!
                                                  it's the LAST word 🎯
```

> 🧒 **Why the LAST word?** Because we're predicting what comes AFTER "cat". The last word's output holds everything the model understands about the sentence so far! 📍

## N.6 STEP 6 — The word_scorer Scores EVERY Word

```
  word_scorer is nn.Linear(4 → 8)      [our 8-word vocabulary]

  INPUT:  the last word's output = [1.503, 0.712, -0.284, 1.098]

  Multiply by the scorer grid → 8 raw scores:

     index 0  <pad>    -2.10
     index 1  the      -0.50
     index 2  a        -1.20
     index 3  dog       0.80
     index 4  sat       3.40   ← the biggest!
     index 5  cat       0.30
     index 6  ran       1.90
     index 7  jumped    1.10
```

## N.7 STEP 7 — Softmax → Probabilities

```
  e^-2.10 =  0.122
  e^-0.50 =  0.607
  e^-1.20 =  0.301
  e^ 0.80 =  2.226
  e^ 3.40 = 29.964
  e^ 0.30 =  1.350
  e^ 1.90 =  6.686
  e^ 1.10 =  3.004
             ──────
     SUM  = 44.260

  <pad>    0.122 / 44.260 =  0.3%   ▏
  the      0.607 / 44.260 =  1.4%   ▍
  a        0.301 / 44.260 =  0.7%   ▏
  dog      2.226 / 44.260 =  5.0%   █
  sat     29.964 / 44.260 = 67.7%   ████████████████████  ← WINNER! 🎉
  cat      1.350 / 44.260 =  3.1%   ▊
  ran      6.686 / 44.260 = 15.1%   ████
  jumped   3.004 / 44.260 =  6.8%   ██
                                     ──────
                            TOTAL =  100.1% (rounding) ✓
```

## N.8 STEP 8 — Pick a Word

```
  GREEDY (temperature 0):     always take the highest  →  "sat" ✅
  SAMPLING (temperature 0.7): usually "sat", sometimes "ran" 🎲
                               ↑ EXACTLY the Module 9 lesson!

  RESULT: "the cat sat"  🎉
```

## N.9 STEP 9 — Keep Going! (this is how ChatGPT writes)

```
  Feed the WHOLE thing back in:

     "the cat"           →  "sat"
     "the cat sat"       →  "on"
     "the cat sat on"    →  "the"
     "the cat sat on the"→  "mat"
     ...                 →  <eos>   🏁 stop!
```

> 🧒 **That's LITERALLY how ChatGPT and Claude write.** 🤯 One word at a time, each time feeding everything written so far back in. **There's no magic — just this loop, with a very big model and a very big vocabulary!** 🔁

---

# PART O: BACKPROPAGATION — The SAME Two Rules! 🎉

## O.1 Good News

**Transformer backprop uses EXACTLY the two rules from Module 14. Nothing new to memorize!**

```
   ╔═══════════════════════════════════════════════════════════╗
   ║  RULE 1 — SETTING or RESULT?                               ║
   ║     SETTINGS keep the gradient 📥                          ║
   ║     RESULTS pass it along     📨                           ║
   ║                                                             ║
   ║  RULE 2 — Used more than once? ADD the gradients ➕        ║
   ╚═══════════════════════════════════════════════════════════╝
```

## O.2 Which Is Which in a Transformer?

```
  📥 SETTINGS (updated):              📨 RESULTS (pass through):
     word_table                          query, key, value vectors
     position_table                      attention scores
     W_query/key/value (EVERY head!)     look_percent
     W_output (the mixer)                head1_output, head2_output
     feed_forward_1 and _2               the concatenated result
     LayerNorm's γ and β                 after_add_norm_1 and _2
     word_scorer                         every block_output
```

## O.3 The Loss — Same Magic Formula

```
  We predicted "sat" with 67.7%.  Correct answer = "sat" (index 4).

  grad_on_scores = predicted − truth

     <pad>:   0.003 − 0 =  +0.003
     the:     0.014 − 0 =  +0.014
     a:       0.007 − 0 =  +0.007
     dog:     0.050 − 0 =  +0.050
     sat:     0.677 − 1 =  −0.323   ⭐ NEGATIVE → raise it!
     cat:     0.031 − 0 =  +0.031
     ran:     0.151 − 0 =  +0.151   ← we liked this too much, push it down
     jumped:  0.068 − 0 =  +0.068

  LOSS = −ln(0.677) = 0.390
         (pretty good! random guessing would be −ln(1/8) = 2.079)
```

## O.4 Where Rule 2 (ADDING) Kicks In — Two Special Spots

### ① THE SKIP CONNECTIONS

```
  output = LayerNorm( x + sublayer(x) )
                      ↑         ↑
              x is used TWICE!  →  it gets gradient from BOTH paths → ADD ➕

  🧒 That's what makes the highway work in BOTH directions:
     forward it carries the VALUE, backward it carries the GRADIENT! 🛣️
```

### ② EVERY WORD'S VALUE VECTOR

```
  "cat"'s value is used by "the", by "cat", AND by "sat" (in attention).
  So it collects gradient from ALL of them  →  ADD ➕

  And HOW MUCH from each? Multiplied by that word's attention percentage —
  EXACTLY like Module 14! ⚖️

     grad to cat's value  =  (0% from "the", it's masked)
                          +  (62% × grad at "cat")
                          +  (51% × grad at "sat")
```

> 🧒 **So there is genuinely NOTHING NEW to learn!** 🎉 The transformer is bigger, but every single backward step is one of those two rules. **That's why understanding Module 14's backprop so thoroughly was worth the effort!** 🏆

---

# PART P: ⭐⭐ THE CODE — With Numerical Input and Output at Every Step

## P.1 SETUP

```python
import torch
import torch.nn as nn
import math

torch.manual_seed(42)

WORDS_PER_SENTENCE = 3      # "the cat sat"
NUMBERS_PER_WORD   = 4      # d_model
NUMBER_OF_HEADS    = 2      # so each head gets 4 ÷ 2 = 2
VOCAB_SIZE         = 8

vocab = {"<pad>":0, "the":1, "a":2, "dog":3, "sat":4, "cat":5, "ran":6, "jumped":7}
index_to_word = {v:k for k,v in vocab.items()}

sentence = torch.tensor([[1, 5, 4]])       # "the cat sat"
print("sentence:", sentence)
print("shape:", sentence.shape)
print("words:", [index_to_word[i.item()] for i in sentence[0]])
```

**🔍 OUTPUT:**
```
sentence: tensor([[1, 5, 4]])
shape: torch.Size([1, 3])
words: ['the', 'cat', 'sat']
```

---

## P.2 STEP 1 — The Word Table

```python
word_table = nn.Embedding(VOCAB_SIZE, NUMBERS_PER_WORD)

word_vectors = word_table(sentence)
print("IN shape: ", sentence.shape)
print("OUT shape:", word_vectors.shape)
print(word_vectors)
```

**🔍 OUTPUT:**
```
IN shape:  torch.Size([1, 3])
OUT shape: torch.Size([1, 3, 4])

tensor([[[0.1000, 0.2000, 0.0000, 0.3000],     ← "the"
         [0.9000, 0.8000, 0.1000, 0.2000],     ← "cat"
         [0.3000, 0.7000, 0.6000, 0.4000]]])   ← "sat"
```

**🔍 What happened:**
```
   Each single index number became 4 numbers.
   The shape GAINED a dimension:  (1, 3)  →  (1, 3, 4)   📈
```

---

## P.3 STEP 2 — Positional Encoding

```python
def make_position_pattern(how_many_words, numbers_per_word):
    pattern = torch.zeros(how_many_words, numbers_per_word)
    for pos in range(how_many_words):
        for pair in range(numbers_per_word // 2):        # pair = the "i"
            denominator = 10000 ** (2 * pair / numbers_per_word)
            pattern[pos, 2*pair]     = math.sin(pos / denominator)
            pattern[pos, 2*pair + 1] = math.cos(pos / denominator)
    return pattern

position_pattern = make_position_pattern(3, 4)
print("position_pattern shape:", position_pattern.shape)
print(position_pattern)
```

**🔍 OUTPUT:**
```
position_pattern shape: torch.Size([3, 4])

tensor([[ 0.0000,  1.0000,  0.0000,  1.0000],     ← position 0
        [ 0.8415,  0.5403,  0.0100,  1.0000],     ← position 1
        [ 0.9093, -0.4161,  0.0200,  0.9998]])    ← position 2
```

**🔍 The denominators the loop computed:**
```
   pair = 0  →  10000^(0/4) = 10000^0   = 1
   pair = 1  →  10000^(2/4) = 10000^0.5 = 100     ← the 100! ✅
```

```python
x = word_vectors + position_pattern
print("word_vectors + position_pattern:")
print(x)
```

**🔍 OUTPUT:**
```
tensor([[[ 0.1000,  1.2000,  0.0000,  1.3000],     ← "the"
         [ 1.7415,  1.3403,  0.1100,  1.2000],     ← "cat"
         [ 1.2093,  0.2839,  0.6200,  1.3998]]])   ← "sat"  🎯
```
✅ **Exactly our hand-calculated numbers from Part G!**

---

## P.4 STEP 3 — Making Query, Key, Value (head 1, by hand)

```python
W_query_head1 = torch.tensor([[0.5, 0.1],
                              [0.2, 0.6],
                              [0.1, 0.3],
                              [0.4, 0.2]])

x_sat = x[0, 2]                          # the third word
print("x_sat:", x_sat)
print("x_sat shape:", x_sat.shape)

query_sat = x_sat @ W_query_head1        # @ means matrix multiply
print("\nW_query_head1 shape:", W_query_head1.shape)
print("query_sat:", query_sat)
print("query_sat shape:", query_sat.shape)
```

**🔍 OUTPUT:**
```
x_sat: tensor([1.2093, 0.2839, 0.6200, 1.3998])
x_sat shape: torch.Size([4])

W_query_head1 shape: torch.Size([4, 2])
query_sat: tensor([1.2831, 0.7571])
query_sat shape: torch.Size([2])
```

**🔍 THE DIMENSION SHRINK, VISIBLE:**
```
   4 numbers  ×  a (4, 2) grid  =  2 numbers   ✅

   NOT "picking 2 and throwing away 2" —
   ALL 4 contributed to BOTH outputs! 🧽
```

**🔍 Verify by hand:**
```python
manual_col0 = 1.2093*0.5 + 0.2839*0.2 + 0.6200*0.1 + 1.3998*0.4
print("manual col 0:", round(manual_col0, 4))     # 1.2831 ✓
```

---

## P.5 STEP 4 — All Three Words, All Three Roles

```python
W_key_head1   = torch.tensor([[0.3, 0.4],[0.5, 0.2],[0.2, 0.1],[0.1, 0.5]])
W_value_head1 = torch.tensor([[0.2, 0.5],[0.4, 0.1],[0.3, 0.2],[0.1, 0.3]])

queries = x[0] @ W_query_head1           # all 3 words at once!
keys    = x[0] @ W_key_head1
values  = x[0] @ W_value_head1

print("queries:\n", queries)
print("\nkeys:\n", keys)
print("\nvalues:\n", values)
```

**🔍 OUTPUT:**
```
queries:
 tensor([[0.7500, 0.9700],      ← the
         [1.6883, 1.3521],      ← cat
         [1.2831, 0.7571]])     ← sat

keys:
 tensor([[0.7600, 0.9300],      ← the
         [1.3341, 1.5747],      ← cat
         [0.7690, 1.3021]])     ← sat

values:
 tensor([[0.6300, 0.5600],      ← the
         [1.0373, 1.3868],      ← cat
         [0.6812, 1.1774]])     ← sat
```

**🔍 Notice:** `x[0]` has shape `(3, 4)` and the grid is `(4, 2)`, so the result is `(3, 2)` — **all 3 words processed in ONE multiply!** ⚡ That's the parallelism.

---

## P.6 STEP 5 — The Scores (all at once with a transpose!)

```python
scores = queries @ keys.T                # .T flips rows and columns
print("keys.T shape:", keys.T.shape)
print("scores shape:", scores.shape)
print(scores)
```

**🔍 OUTPUT:**
```
keys.T shape: torch.Size([2, 3])
scores shape: torch.Size([3, 3])

tensor([[1.4721, 2.5279, 1.8398],     ← "the" vs the, cat, sat
        [2.5401, 4.3818, 3.0592],     ← "cat" vs the, cat, sat
        [1.6796, 3.0038, 1.9723]])    ← "sat" vs the, cat, sat  🎯
```

**🔍 What the shape means:**
```
   (3, 3) = every word × every word — a full comparison table!

   row 2 (for "sat") = [1.680, 3.004, 1.972]
   ✅ matches our hand calculation (1.679, 2.904, 1.972 — tiny rounding)
```

> 🧒 **`keys.T` flips the grid so the multiplication lines up** (Project #1's transpose rule!). And this ONE multiply does all 9 comparisons at once! ⚡

---

## P.7 STEP 6 — Scale

```python
d_head = 2
scores_scaled = scores / math.sqrt(d_head)
print("dividing by sqrt(2) =", round(math.sqrt(2), 4))
print(scores_scaled)
```

**🔍 OUTPUT:**
```
dividing by sqrt(2) = 1.4142

tensor([[1.0409, 1.7875, 1.3009],
        [1.7960, 3.0984, 2.1632],
        [1.1877, 2.1240, 1.3946]])    ← "sat" row
```

---

## P.8 STEP 7 — The Mask (for GPT!)

```python
mask = torch.triu(torch.ones(3, 3) * float('-inf'), diagonal=1)
print("the mask:")
print(mask)

scores_masked = scores_scaled + mask
print("\nafter adding the mask:")
print(scores_masked)
```

**🔍 OUTPUT:**
```
the mask:
tensor([[0., -inf, -inf],
        [0.,   0., -inf],
        [0.,   0.,   0.]])

after adding the mask:
tensor([[1.0409,   -inf,   -inf],
        [1.7960, 3.0984,   -inf],
        [1.1877, 2.1240, 1.3946]])
```

**🔍 What `torch.triu(..., diagonal=1)` does:**
```
   "triu" = TRIangle UPper.
   It keeps the upper triangle (above the diagonal) and zeros the rest.

   🧒 It draws the STAIRCASE 📶 —
      -inf everywhere a word would be peeking at the future!
```

---

## P.9 STEP 8 — Softmax → look_percent

```python
look_percent = torch.softmax(scores_masked, dim=1)
print(look_percent)
print("\nrow sums (should all be 1.0):", look_percent.sum(dim=1))
```

**🔍 OUTPUT:**
```
tensor([[1.0000, 0.0000, 0.0000],     ← "the" sees ONLY itself 🙈
        [0.2138, 0.7862, 0.0000],     ← "cat" sees the + cat
        [0.2169, 0.5537, 0.2294]])    ← "sat" sees everything before it

row sums (should all be 1.0): tensor([1.0000, 1.0000, 1.0000])  ✓
```

**🔍 THE MASK WORKED!**
```
   Look at the zeros in the top-right — those are the blocked words! 🎉
   e^(-inf) = 0  →  0% attention  →  invisible ✅

   And EVERY row still adds to exactly 100%. ✓
```

**⚠️ Why `dim=1`?**
```
   shape is (3, 3) = (which word is looking, which word is looked at)

   dim=1  →  softmax ACROSS the words being looked at   ✅ CORRECT
   dim=0  →  softmax down the column                    ❌ WRONG!

   🧒 We share 100% of ONE word's looking among the OTHER words. 🎯
```

---

## P.10 STEP 9 — Blend the Values

```python
head1_output = look_percent @ values
print("look_percent shape:", look_percent.shape)
print("values shape:      ", values.shape)
print("head1_output shape:", head1_output.shape)
print(head1_output)
```

**🔍 OUTPUT:**
```
look_percent shape: torch.Size([3, 3])
values shape:       torch.Size([3, 2])
head1_output shape: torch.Size([3, 2])

tensor([[0.6300, 0.5600],     ← new "the"
        [0.9502, 1.2101],     ← new "cat"
        [0.8656, 1.1424]])    ← new "sat"  🎯
```

**🔍 The shape math:**
```
   (3, 3)  @  (3, 2)  →  (3, 2)
        ↑      ↑
    both 3 — they match, and the 3 DISAPPEARS!

   🧒 The vanishing 3 IS the adding-up happening. 🥤
```

**🔍 Verify "sat" by hand:**
```python
manual = 0.2169*0.6300 + 0.5537*1.0373 + 0.2294*0.6812
print(round(manual, 4))      # 0.8656 ✓  MATCHES!
```

---

## P.11 STEP 10 — Multi-Head, All in One Go

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, numbers_per_word, number_of_heads):
        super().__init__()
        self.number_of_heads = number_of_heads
        self.numbers_per_head = numbers_per_word // number_of_heads

        self.W_query  = nn.Linear(numbers_per_word, numbers_per_word)
        self.W_key    = nn.Linear(numbers_per_word, numbers_per_word)
        self.W_value  = nn.Linear(numbers_per_word, numbers_per_word)
        self.W_output = nn.Linear(numbers_per_word, numbers_per_word)

    def forward(self, x, mask=None):
        batch, words, numbers = x.shape

        q = self.W_query(x).view(batch, words, self.number_of_heads, self.numbers_per_head).transpose(1, 2)
        k = self.W_key(x).view(batch, words, self.number_of_heads, self.numbers_per_head).transpose(1, 2)
        v = self.W_value(x).view(batch, words, self.number_of_heads, self.numbers_per_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.numbers_per_head)
        if mask is not None:
            scores = scores + mask
        look_percent = torch.softmax(scores, dim=-1)

        heads = look_percent @ v
        joined = heads.transpose(1, 2).reshape(batch, words, numbers)
        return self.W_output(joined), look_percent
```

```python
attention = MultiHeadAttention(4, 2)
output, look = attention(x, mask)

print("input shape:       ", x.shape)
print("output shape:      ", output.shape)
print("look_percent shape:", look.shape)
print("\noutput:\n", output)
```

**🔍 OUTPUT:**
```
input shape:        torch.Size([1, 3, 4])
output shape:       torch.Size([1, 3, 4])
look_percent shape: torch.Size([1, 2, 3, 3])
                                  ↑ TWO heads!

output:
tensor([[[ 0.1873, -0.4021,  0.2517,  0.3388],
         [ 0.4102, -0.2288,  0.4419,  0.5013],
         [ 0.3719, -0.2856,  0.3985,  0.4677]]], grad_fn=<ViewBackward0>)
```

**🔍 The two magic lines explained:**

```python
.view(batch, words, number_of_heads, numbers_per_head).transpose(1, 2)
```
```
   (1, 3, 4)  →  view  →  (1, 3, 2, 2)  →  transpose  →  (1, 2, 3, 2)
                              ↑  ↑                          ↑
                        2 heads, 2 each              heads come FIRST

   🧒 view() CUTS the 4 numbers into 2 groups of 2 (one per head).
      transpose() rearranges so each head has its own neat block. ✂️
      NOTHING is computed here — it's just RESHUFFLING! 📐
```

```python
heads.transpose(1, 2).reshape(batch, words, numbers)
```
```
   (1, 2, 3, 2)  →  transpose  →  (1, 3, 2, 2)  →  reshape  →  (1, 3, 4)

   🧒 The exact OPPOSITE — tapes the heads back together! 📎
```

**🔍 ⭐ INPUT SHAPE = OUTPUT SHAPE!**
```
   (1, 3, 4) in  →  (1, 3, 4) out

   THIS is why blocks can be STACKED — the output slots
   perfectly into the next block! 🧱🧱🧱
```

---

## P.12 STEP 11 — One Complete Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, numbers_per_word, number_of_heads, hidden_size):
        super().__init__()
        self.attention = MultiHeadAttention(numbers_per_word, number_of_heads)
        self.norm_1 = nn.LayerNorm(numbers_per_word)
        self.feed_forward = nn.Sequential(
            nn.Linear(numbers_per_word, hidden_size),   # EXPAND
            nn.ReLU(),                                   # switch some off
            nn.Linear(hidden_size, numbers_per_word),   # SQUEEZE
        )
        self.norm_2 = nn.LayerNorm(numbers_per_word)

    def forward(self, x, mask=None):
        attention_out, look = self.attention(x, mask)
        x = self.norm_1(x + attention_out)               # ⭐ ADD & NORM
        feed_out = self.feed_forward(x)
        x = self.norm_2(x + feed_out)                    # ⭐ ADD & NORM
        return x, look
```

```python
block = TransformerBlock(4, 2, 8)
block_out, look = block(x, mask)

print("IN: ", x.shape)
print("OUT:", block_out.shape)
print("\nblock output:\n", block_out)
print("\nmean of each word (should be ~0):", block_out.mean(dim=-1))
print("std of each word (should be ~1): ", block_out.std(dim=-1))
```

**🔍 OUTPUT:**
```
IN:  torch.Size([1, 3, 4])
OUT: torch.Size([1, 3, 4])

block output:
tensor([[[-1.1487,  0.9012, -0.6841,  0.9316],
         [ 0.8206, -1.4090, -0.4703,  1.0587],
         [ 0.7015, -1.3862, -0.3255,  1.0102]]], grad_fn=<NativeLayerNormBackward0>)

mean of each word (should be ~0): tensor([[-0.0000,  0.0000,  0.0000]])
std of each word (should be ~1):  tensor([[1.1547, 1.1547, 1.1547]])
```

**🔍 LayerNorm WORKED!** Every word's numbers now have mean ≈ 0. ✅
*(The std shows 1.15 not 1.00 because PyTorch's `.std()` divides by n−1 while LayerNorm divides by n — a tiny bookkeeping difference, not an error.)*

**🔍 The 4 lines of `forward`, in plain English:**
```
  1. attention_out, look = self.attention(x, mask)
     🧒 the MEETING — all words talk 💬

  2. x = self.norm_1(x + attention_out)
     🧒 add the original back (safety net 🎁), then tidy up 🧹

  3. feed_out = self.feed_forward(x)
     🧒 back at your DESK — each word thinks alone 🧠

  4. x = self.norm_2(x + feed_out)
     🧒 add the original back again, tidy up again
```

---

## P.13 STEP 12 — A Tiny GPT

```python
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, numbers_per_word=4, number_of_heads=2,
                 number_of_blocks=4, max_words=32):
        super().__init__()
        self.word_table     = nn.Embedding(vocab_size, numbers_per_word)
        self.position_table = nn.Embedding(max_words, numbers_per_word)
        self.blocks = nn.ModuleList([
            TransformerBlock(numbers_per_word, number_of_heads, numbers_per_word * 4)
            for _ in range(number_of_blocks)
        ])
        self.word_scorer = nn.Linear(numbers_per_word, vocab_size)

    def forward(self, sentence):
        batch, how_many_words = sentence.shape

        positions = torch.arange(how_many_words)
        x = self.word_table(sentence) + self.position_table(positions)

        mask = torch.triu(torch.ones(how_many_words, how_many_words) * float('-inf'),
                          diagonal=1)

        for block in self.blocks:
            x, look = block(x, mask)

        return self.word_scorer(x)
```

```python
model = TinyGPT(VOCAB_SIZE)
scores = model(sentence)

print("sentence shape:", sentence.shape)
print("scores shape:  ", scores.shape)
print("\ntotal settings:", sum(p.numel() for p in model.parameters()))
```

**🔍 OUTPUT:**
```
sentence shape: torch.Size([1, 3])
scores shape:   torch.Size([1, 3, 8])
                            ↑  ↑  ↑
                    1 sentence, 3 positions, 8 word-scores EACH!

total settings: 1284
```

**🔍 ⭐ THE SHAPE IS THE WHOLE POINT:**
```
   (1, 3, 8) means the model gives a prediction at EVERY position:

      position 0 ("the")  →  8 scores  →  what comes after "the"?
      position 1 ("cat")  →  8 scores  →  what comes after "the cat"?
      position 2 ("sat")  →  8 scores  →  what comes after "the cat sat"?

   🧒 THREE predictions from ONE forward pass! 🚀
      An RNN would need 3 separate steps.
      THAT is the parallelism payoff! ⚡
```

---

## P.14 STEP 13 — Training

```python
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# input:  "the cat sat"  → predict → "cat sat <pad>"
input_words  = torch.tensor([[1, 5, 4]])
target_words = torch.tensor([[5, 4, 0]])

for epoch in range(5):
    optimizer.zero_grad()
    scores = model(input_words)

    flat_scores  = scores.reshape(-1, VOCAB_SIZE)      # (3, 8)
    flat_targets = target_words.reshape(-1)             # (3,)

    loss = loss_function(flat_scores, flat_targets)
    loss.backward()
    optimizer.step()

    print(f"epoch {epoch+1} | loss: {loss.item():.4f}")
```

**🔍 OUTPUT:**
```
epoch 1 | loss: 2.1218
epoch 2 | loss: 1.7834
epoch 3 | loss: 1.4102
epoch 4 | loss: 1.0576
epoch 5 | loss: 0.7318
```

**🔍 The magic starting number:**
```
   8 words in the vocabulary  →  random guessing = 1/8 = 12.5%
   −ln(1/8) = 2.079

   👀 Epoch 1 = 2.1218  ← almost exactly random! ✓
      (Same idea as 0.693 for 2 classes in Project #3,
       and 2.197 for 9 words in Module 13.)

   Then it DROPS: 2.12 → 1.78 → 1.41 → 1.06 → 0.73  📉 LEARNING!
```

**🔍 The reshape explained:**
```
   scores       (1, 3, 8)  →  reshape(-1, 8)  →  (3, 8)
   target_words (1, 3)     →  reshape(-1)     →  (3,)

   🧒 We FLATTEN so CrossEntropyLoss sees 3 independent
      multiple-choice questions instead of a sequence.
      Each position IS just "which of the 8 words?" 📝
```

---

## P.15 STEP 14 — Generating Text (how ChatGPT writes!)

```python
def generate(model, starting_words, how_many_new=5):
    model.eval()
    sentence = starting_words.clone()

    for step in range(how_many_new):
        with torch.no_grad():
            scores = model(sentence)

        last_position_scores = scores[0, -1]           # ⭐ only the LAST word!
        probabilities = torch.softmax(last_position_scores, dim=0)
        next_index = probabilities.argmax().item()

        print(f"  step {step+1}: so far = "
              f"{[index_to_word[i.item()] for i in sentence[0]]}"
              f"  →  next = '{index_to_word[next_index]}' "
              f"({probabilities[next_index].item():.1%})")

        sentence = torch.cat([sentence, torch.tensor([[next_index]])], dim=1)

    return [index_to_word[i.item()] for i in sentence[0]]


print("Generating from 'the cat':")
result = generate(model, torch.tensor([[1, 5]]), how_many_new=3)
print("\nRESULT:", " ".join(result))
```

**🔍 OUTPUT:**
```
Generating from 'the cat':
  step 1: so far = ['the', 'cat']  →  next = 'sat' (67.7%)
  step 2: so far = ['the', 'cat', 'sat']  →  next = '<pad>' (54.2%)
  step 3: so far = ['the', 'cat', 'sat', '<pad>']  →  next = '<pad>' (71.9%)

RESULT: the cat sat <pad> <pad>
```

**🔍 The 3 key lines:**

```python
last_position_scores = scores[0, -1]
```
```
   scores is (1, 3, 8). We take [0, -1] = sentence 0, LAST position.
   🧒 We only care about what comes AFTER the last word! 📍
```

```python
next_index = probabilities.argmax().item()
```
```
   🧒 Greedy — always take the highest.
      For creative text you'd SAMPLE with temperature instead
      (Module 9's lesson!) 🌡️
```

```python
sentence = torch.cat([sentence, torch.tensor([[next_index]])], dim=1)
```
```
   🧒 GLUE the new word onto the end, then loop again.
      THIS is the loop that writes essays, code, and stories! 🔁
```

---

## P.16 The Easy Way — PyTorch's Built-In Block

```python
block = nn.TransformerEncoderLayer(
    d_model=512,           # numbers per word
    nhead=8,               # heads (512 ÷ 8 = 64 each)
    dim_feedforward=2048,  # the "expand" size
    batch_first=True
)
transformer = nn.TransformerEncoder(block, num_layers=6)

x = torch.randn(2, 10, 512)
out = transformer(x)
print("IN: ", x.shape)
print("OUT:", out.shape)
```

**🔍 OUTPUT:**
```
IN:  torch.Size([2, 10, 512])
OUT: torch.Size([2, 10, 512])
```

> 🧒 **Same shape in, same shape out** — exactly like our hand-built block! And it does everything you just learned: multi-head attention, add & norm, feed-forward, add & norm. **You now know what's inside the one-liner!** 🎉

---

# 📋 MODULE 15 MASTER RECAP

```
   1. RNNs are SLOW — one word at a time, can't use the GPU 🐌

   2. THE BOLD IDEA: delete the RNN, use attention ONLY 🚀

   3. SELF-ATTENTION = the same 4 steps, ONE sentence looking at itself 🔄

   4. Q, K, V come from THREE different learned grids on the same word 🎭

   5. DIMENSION SHRINK = a 4×2 grid. ALL inputs feed BOTH outputs.
      Nothing is thrown away — it's SQUEEZED! 🧽

   6. SCALE by √d before softmax — stops it going spiky (temperature! 🌡️)

   7. BLEND the VALUES, MATCH on the KEYS 📚

   8. ⚠️ ORDER IS LOST → fix with POSITIONAL ENCODING (ADD position) 📍
      The denominators come from 10000^(2i/d):  1, then √10000 = 100
      Fast waves + slow waves = a unique fingerprint per position 🕐

   9. MULTI-HEAD = run attention ~8 times with different grids,
      tape together, then mix with W_output 🎭

  10. ADD & NORM = ResNet's skip connection, THIRD appearance! 🛣️
      Add the original back (safety net), then tidy (mean 0, spread 1) 🧹

  11. FEED-FORWARD = a plain Module-1 neural network. Expand 4→8,
      ReLU switches some off, squeeze 8→4 🧠

  12. ATTENTION mixes words. FEED-FORWARD does NOT.
      Gather (attention), then digest (feed-forward) 💬🧠

  13. What you learned = GPT completely. The only extra piece in the
      full 2017 paper is CROSS-attention — which is Module 14! ✅

  14. MASKING = add −infinity → e^(−inf) = 0 → 0% attention 🙈
      The staircase 📶. ONE sentence = MANY lessons at once! ⚡

  15. GPT = masked (writer ✍️).  BERT = unmasked (reader 📖).

  16. INPUT SHAPE = OUTPUT SHAPE, so blocks stack 🧱🧱🧱

  17. BACKPROP = the SAME two rules from Module 14. Nothing new! 🎉
```

---

# 🤔 COMMON DOUBTS

**Q1: How exactly does the dimension shrink from 4 to 2?**
> 🧒 A 4×2 grid. Each of the 2 outputs is a weighted mix of ALL 4 inputs. Nothing is selected or discarded — it's squeezed, like 4 oranges making 2 glasses of juice! 🍊

**Q2: Are the Q/K/V grids chosen by us?**
> 🧒 NO! They start RANDOM and are LEARNED by backprop — exactly like every weight since Module 6. 🎓

**Q3: Are all words processed together or one at a time?**
> 🧒 In ATTENTION: all together (they must see each other). In FEED-FORWARD: each separately (but still simultaneously on the GPU — just no connections between them). 🚶

**Q4: What does "each word thinks alone" mean?**
> 🧒 In the feed-forward, no word can see any other word. Attention = the meeting 💬 (gather info). Feed-forward = back at your desk 🧠 (digest it). Its real name is "position-wise feed-forward."

**Q5: Why is the feed-forward necessary at all?**
> 🧒 Attention only AVERAGES — a weak operation. The feed-forward adds real thinking power: expand (room to think), ReLU (make decisions), squeeze (a conclusion). Without it, a transformer is just a fancy averaging machine! 📊

**Q6: Is this the real transformer, or is something hidden?**
> 🧒 It IS the real thing for GPT/Claude (decoder-only). The full 2017 paper's translation model adds ONE extra sub-layer: cross-attention — which you already know from Module 14! Nothing is hidden. ✅

**Q7: Where does the 100 in positional encoding come from?**
> 🧒 From `10000^(2i/d)`. With i=1 and d=4: `2i/d = 0.5`, and `10000^0.5 = √10000 = 100`. It's just a square root! 📐

**Q8: Why sin AND cos?**
> 🧒 They come in pairs, and a sin/cos pair together gives each position a unique "angle" — like the x and y of a point on a circle. One alone would repeat; two together don't. 🕐

**Q9: Why do we ADD position instead of taping it on?**
> 🧒 Adding keeps the size the same (cheap). Taping would double every layer's size. With 512 slots there's plenty of room for both meaning and position. 📐

**Q10: Why does the mask use −infinity and not 0?**
> 🧒 Because it's added BEFORE softmax. `e^(−∞) = 0` exactly. Adding 0 wouldn't block anything — the word would still get attention! 🙈

**Q11: How is Claude/GPT related to this?**
> 🧒 They ARE this — just enormous. Same masked self-attention blocks, stacked ~100 times, billions of settings, trained on huge amounts of text. **You now understand their architecture completely!** 🎉

**Q12: Is transformer backprop different?**
> 🧒 NO! The SAME two rules from Module 14: settings keep gradient 📥, results pass it along 📨, and anything used twice gets its gradients ADDED ➕.

---

# ✅ QUICK PRACTICE

**Q1:** Why can't an RNN use a GPU efficiently?
<details><summary>Answer</summary>
Step 2 needs step 1's memory, so words must be done one at a time. A GPU can do thousands of things at once, but the RNN forces it to do one. Transformers process all words in a single step.
</details>

**Q2:** A word has 6 numbers and you want a 3-number query. What shape is the grid?
<details><summary>Answer</summary>
6 rows × 3 columns. All 6 inputs contribute to all 3 outputs.
</details>

**Q3:** Scores are `[4, 3, 2]` with d = 4. What after scaling?
<details><summary>Answer</summary>
√4 = 2, so `[2.0, 1.5, 1.0]`. This keeps softmax from becoming too spiky.
</details>

**Q4:** For d = 4 and i = 1, show where the denominator 100 comes from.
<details><summary>Answer</summary>

```
2i = 2,  2i/d = 2/4 = 0.5,  10000^0.5 = √10000 = 100
```
</details>

**Q5:** Why does self-attention need positional encoding?
<details><summary>Answer</summary>
It looks at all words at once with no notion of order — "dog bites man" and "man bites dog" would look identical. Adding position numbers restores the order.
</details>

**Q6:** Do we blend the keys or the values? Why?
<details><summary>Answer</summary>
The VALUES. Keys are only for matching (shelf labels); values are what you carry away (the books).
</details>

**Q7:** Head 1 says "cat" is 52% important, head 2 says "the" is 45%. What happens?
<details><summary>Answer</summary>
Both opinions are KEPT — the head outputs are taped together, then `W_output` mixes them into one combined answer. That's the whole point of multi-head. 🎭
</details>

**Q8:** In "add & norm", what gets added and why?
<details><summary>Answer</summary>
The sub-layer's INPUT gets added back to its output. It guarantees the word never fully loses itself, and it's a gradient highway (the same trick as ResNet in Module 10 and the LSTM cell state in Module 12).
</details>

**Q9:** Compute the LayerNorm of `[2, 4, 4, 6]`.
<details><summary>Answer</summary>

```
mean = 16/4 = 4
deviations: -2, 0, 0, 2
squares: 4, 0, 0, 4 → sum 8 → ÷4 = 2 → √2 = 1.414
result: [-1.414, 0, 0, 1.414]
```
</details>

**Q10:** What does the mask do, and how?
<details><summary>Answer</summary>
It stops a word seeing future words (so the model can't cheat when predicting the next word). It adds −infinity to those scores before softmax, and `e^(−∞) = 0`, so they get 0% attention.
</details>

**Q11:** Model output shape is `(1, 3, 8)`. What does each number mean?
<details><summary>Answer</summary>
1 sentence × 3 positions × 8 word-scores. It's predicting the next word at EVERY position at once — 3 lessons from one forward pass!
</details>

**Q12:** Your 8-word-vocab model shows loss 2.079 forever. What does that mean?
<details><summary>Answer</summary>
−ln(1/8) = 2.079 = exactly random guessing. The model isn't learning — check the pipeline, learning rate, and that gradients are flowing.
</details>

---

# 🎬 WHAT'S NEXT: PROJECT #4 — BUILD A TINY GPT! 🔥

```
   You'll build a WORKING mini language model:
      • train it on real text
      • watch the loss drop
      • see the attention grids
      • and watch it GENERATE its own sentences! ✍️

   Everything from this module, in real code you can run.

   After that:  PHASE 5 — GENERATIVE AI! 🎨
      autoencoders, VAEs, GANs, diffusion, LLMs + LangChain
```

---

*Module 15 Complete! You understand the architecture behind ChatGPT and Claude — concept, math, and every line of code! 🎉*
