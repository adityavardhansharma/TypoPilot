# TypoPilot — English and Hinglish Noisy-Text Correction Research Project

You are helping build **TypoPilot**, a research project for correcting heavily mistyped English and Hinglish text.

TypoPilot is intended to eventually become a system-wide desktop application for Windows and macOS. A user will be able to type messy text in almost any text field, press a keyboard shortcut, and have the text corrected and replaced within seconds.

However, **do not build the desktop application yet**.

The first and most important phase is to research, train, fine-tune, evaluate, and compare text-correction models. We should only start building the actual product after the model produces consistently useful corrections with acceptable latency.

---

# 1. Core problem

The model should correct heavily corrupted keyboard input such as:

Input:

```text
waht Is Th e weahter and hwo did the ai perfo rm in the taks giben to it
```

Expected output:

```text
What is the weather, and how did the AI perform in the task given to it?
```

It must also support Hinglish written in the Roman alphabet.

Input:

```text
mian aj clg nhi ja rha kyuki meri tabyat thk nhi hai
```

Possible corrected output:

```text
Main aaj college nahi ja raha kyunki meri tabiyat theek nahi hai.
```

Mixed English-Hinglish input:

```text
can u btao meeting kb hai aur mujhe kya preapre krna hai
```

Possible output:

```text
Can you batao meeting kab hai aur mujhe kya prepare karna hai?
```

The model should preserve the user’s original language choice. It must not automatically translate Hinglish into pure English or formal Hindi.

---

# 2. Initial project scope

The first phase must only include:

1. Dataset research and preparation.
2. Synthetic error generation.
3. Model selection.
4. Fine-tuning.
5. Automated and manual evaluation.
6. Latency measurement.
7. Experiment logging.
8. Comparison between models.
9. A simple local test interface or command-line script.
10. A final research report containing results, failures, risks, and recommendations.

Do not build:

* A Windows tray application.
* A macOS menu-bar application.
* Global keyboard shortcuts.
* Clipboard replacement.
* Accessibility integrations.
* Browser extensions.
* Mobile applications.
* User authentication.
* Cloud infrastructure.
* Production APIs.
* A polished graphical interface.

A very small local testing script or minimal interface is allowed, but only for testing the models.

---

# 3. Hardware constraints

The experiments will initially run on:

* ASUS ROG Zephyrus G14
* AMD Ryzen 9 4900HS
* NVIDIA GTX 1660 Ti
* 6GB VRAM
* 16GB system RAM

All model and training decisions must respect these limitations.

Prioritise:

* Small pretrained models.
* Efficient fine-tuning.
* FP16 mixed precision where stable.
* Gradient accumulation.
* Gradient checkpointing.
* Short sequence lengths.
* Small batch sizes.
* Memory-efficient data loading.
* Parameter-efficient fine-tuning when appropriate.
* ONNX or another optimised format for final inference testing.

Do not attempt to train a large language model from scratch.

Avoid models that cannot be trained or tested realistically on 6GB VRAM.

Cloud GPUs may be discussed as an optional later step, but the first working experiment should run locally wherever possible.

---

# 4. Research responsibility

Before implementing the full pipeline, independently research current datasets, models, papers, repositories, licences, and benchmarks relevant to this use case.

Do not blindly use only the datasets or models mentioned in this prompt.

Research whether newer or better resources are available for:

* English spelling correction.
* English grammatical error correction.
* Hinglish text normalisation.
* Hindi-English code-mixed text.
* Romanised Hindi correction.
* Keyboard typo correction.
* Word-boundary repair.
* Byte-level text correction.
* Character-level sequence modelling.
* Multilingual grammatical error correction.
* Synthetic noise generation.
* Meaning-preserving correction.
* Fast local inference.

For every dataset and model considered, record:

* Name.
* Source.
* Paper or repository.
* Languages.
* Dataset size or parameter count.
* Licence.
* Intended task.
* Strengths.
* Limitations.
* Suitability for TypoPilot.
* Whether it can legally be redistributed or used for training.
* Whether it contains personal, toxic, private, or low-quality data.
* Whether it is practical on the available hardware.

Prefer original papers, official repositories, Hugging Face model cards, dataset cards, and primary documentation.

---

# 5. Initial dataset candidates

Investigate these as starting candidates, but verify their current availability, licences, structure, and suitability.

## English datasets

### GitHub Typo Corpus

Potential use:

* Real typo-correction pairs.
* Real spelling edits.
* Some grammar and documentation corrections.

Risks:

* Technical and GitHub-specific language.
* Edits may contain code or identifier changes.
* Not every edit is a genuine typo correction.
* May require strong filtering.

### BEA-2019 / W&I + LOCNESS

Potential use:

* English grammatical error correction.
* Spelling, grammar, punctuation, and word-choice corrections.

Risks:

* Learner English differs from rapid keyboard typing.
* May contain longer formal writing rather than everyday messages.

### FCE and CoNLL grammatical-error datasets

Potential use:

* Additional English correction examples.
* Benchmarking grammar correction.

Risks:

* Limited domain.
* Licensing and access restrictions must be checked.
* Not primarily designed for severe keyboard corruption.

### C4 synthetic grammatical-error correction data

Potential use:

* Large-scale synthetic pretraining data.
* Grammar and correction patterns.

Risks:

* Very large.
* Mostly English.
* Synthetic distribution may not resemble real keyboard errors.
* Only use a manageable filtered subset if appropriate.

### Wikipedia corrective edits or WikEd-style corpora

Potential use:

* Real-world spelling and grammar edits.

Risks:

* Many edits may be stylistic, factual, formatting-related, or meaning-changing.
* Requires strong filtering.

## Hinglish datasets

### hinglishNorm or equivalent Hinglish normalisation datasets

Potential use:

* Noisy Romanised Hindi and Hinglish to normalised Hinglish pairs.
* Handling variations such as `nhi`, `nahi`, `nai`, `kyu`, and `kyun`.

Risks:

* Small dataset size.
* Normalisation rules may differ from TypoPilot’s desired behaviour.
* May over-normalise casual Hinglish.

### L3Cube-HingCorpus or newer Hinglish corpora

Potential use:

* Large amounts of naturally occurring Roman-script Hindi-English text.
* Clean or semi-clean source text for synthetic corruption.

Risks:

* Not necessarily correction pairs.
* Social-media text may already be noisy.
* Quality, duplicates, offensive content, and licence must be checked.

### Other code-mixed Hindi-English datasets

Search for datasets covering:

* Hinglish spelling normalisation.
* Romanised Hindi transliteration variants.
* Hindi-English code switching.
* Social-media normalisation.
* Code-mixed grammatical error correction.
* Hinglish sentiment or conversational corpora that could safely be repurposed.

Do not use sentiment labels as correction targets. Only use the underlying text if the licence permits it and if the text quality is useful.

---

# 6. Dataset policy

TypoPilot supports only:

* English.
* Hinglish written in the Roman alphabet.

Do not add unrelated languages.

The model should preserve:

* Natural Hinglish.
* Code-switching.
* User vocabulary.
* Names.
* Product names.
* Technical terminology.
* Casual tone where possible.

The model should correct:

* Character swaps.
* Missing characters.
* Duplicate characters.
* Nearby-key errors.
* Incorrect spacing.
* Split words.
* Merged words.
* Capitalisation.
* Basic punctuation.
* Obvious grammatical mistakes.
* Severe but recoverable typing errors.
* Common Romanised Hindi spelling mistakes.
* Inconsistent Hinglish abbreviations where correction is reasonably clear.

Examples:

```text
waht → what
teh → the
giben → given
perfo rm → perform
thetask → the task
mian → main
aajjj → aaj
kyki → kyunki
preapre → prepare
```

Avoid unnecessarily converting natural forms such as:

```text
nahi
nhi
nai
```

unless a clearly documented normalisation policy has been chosen.

Before training, define a precise policy for what counts as an error in Hinglish.

Possible policy:

> Correct clear keyboard mistakes, broken spacing, inconsistent capitalisation, severe abbreviations, and obvious grammatical problems while preserving the original code-mixed language, meaning, and conversational tone.

Document every normalisation decision.

---

# 7. Synthetic error-generation pipeline

Because existing datasets are unlikely to fully represent severe keyboard mistakes, build a configurable synthetic corruption pipeline.

Given a clean sentence, create one or more noisy versions.

Supported corruption types should include:

## Character-level errors

* Swap adjacent characters.

```text
what → waht
```

* Delete characters.

```text
weather → wether
```

* Duplicate characters.

```text
the → thhe
```

* Insert random characters.

```text
task → taksk
```

* Replace characters with nearby keyboard keys.

```text
given → giben
```

* Replace characters using common phonetic or Romanised Hindi confusions.

## Word-boundary errors

* Split a word.

```text
perform → perfo rm
```

* Merge adjacent words.

```text
the task → thetask
```

* Add multiple spaces.
* Remove necessary spaces.
* Insert spaces around punctuation.

## Capitalisation errors

```text
What is the AI model?
→ waht Is ThE ai Model
```

## Punctuation errors

* Remove punctuation.
* Duplicate punctuation.
* Replace punctuation.
* Add punctuation in the wrong location.

## Word-level errors

* Delete short words.
* Repeat words.
* Swap neighbouring words.
* Use common abbreviations.
* Corrupt helper verbs.
* Introduce common Hinglish shorthand.

## Multi-error corruption

Most examples should contain more than one error because the target problem involves heavily corrupted text.

However, control the severity.

Create categories such as:

* Light noise.
* Medium noise.
* Heavy noise.
* Extreme but recoverable noise.

Do not generate noise so severe that the original meaning cannot reasonably be recovered.

Use deterministic random seeds and log the corruption operations applied to each sentence.

---

# 8. Keyboard-layout awareness

Research and implement keyboard-aware corruption for relevant layouts.

At minimum:

* Standard QWERTY English keyboard.
* Romanised Hindi typed using QWERTY.

Nearby-key substitutions should use an actual keyboard adjacency map.

For example:

```text
g → f, h, t, y, v, b
```

Also investigate realistic mobile and laptop typing patterns, although the research priority is desktop typing.

Record whether keyboard-aware synthetic errors improve results compared with purely random character corruption.

---

# 9. Data cleaning and filtering

Create a robust cleaning pipeline.

Filter out or handle:

* Code.
* URLs.
* Email addresses.
* File paths.
* Markdown syntax.
* HTML.
* Secrets and API keys.
* Usernames.
* Hashtags where correction would be harmful.
* Extremely long sentences.
* Duplicate examples.
* Identical source and target pairs unless intentionally used.
* Meaning-changing edits.
* Factual edits.
* Style-only edits.
* Offensive or unsafe material where unnecessary.
* Personally identifiable information.
* Examples where the correction is uncertain.

Build scripts that produce clear statistics:

* Number of original examples.
* Number removed.
* Number retained.
* English examples.
* Hinglish examples.
* Mixed examples.
* Average input length.
* Average output length.
* Error-type distribution.
* Noise-severity distribution.
* Duplicate rate.
* Train/validation/test split sizes.

Prevent data leakage by deduplicating before splitting.

---

# 10. Model candidates

Research and compare several model families.

Initial candidates may include:

## ByT5-small

Why it may work:

* Byte-level processing.
* Strong fit for misspellings, broken words, unusual spacing, and Romanised Hindi.
* Does not depend as heavily on standard token boundaries.

Potential disadvantages:

* Longer byte sequences.
* Slower training and inference than subword models.
* May require careful sequence-length control.

## mT5-small

Why it may work:

* Multilingual pretraining.
* Supports English and Hindi-related patterns.
* Encoder-decoder architecture suitable for correction.

Potential disadvantages:

* Subword tokenisation may struggle with severe corruption.
* Approximately 300M parameters may be demanding on 6GB VRAM.
* Hinglish Romanisation may not be strongly represented.

## FLAN-T5-small

Why it may work:

* Small and practical.
* Fast to fine-tune.
* Strong English instruction-following baseline.

Potential disadvantages:

* Less naturally suited to Hinglish.
* Tokenisation may struggle with severe typos.
* May over-rewrite instead of minimally correcting.

## Character-level or byte-level custom models

Possible options:

* Small character-level transformer.
* Lightweight encoder-decoder.
* Character CNN plus transformer.
* Byte-level recurrent or transformer model.

Advantages:

* Potentially very fast.
* Specialised for the exact task.
* Can be trained with synthetic data.

Disadvantages:

* Training from scratch may need more data.
* May have weaker contextual and grammatical understanding.
* Requires additional architecture experimentation.

## Other compact models

Independently investigate newer compact models specifically designed for:

* Grammatical error correction.
* Spelling correction.
* Text normalisation.
* Byte-level generation.
* Character-level denoising.
* Multilingual sequence-to-sequence correction.

Do not select a model merely because it is popular. Select it based on quality, latency, memory requirements, licensing, and suitability.

---

# 11. Training strategy

Start with small experiments.

Suggested progression:

## Stage A: Pipeline validation

* Use 5,000–20,000 examples.
* Train for a small number of steps.
* Confirm that the model can learn simple corrections.
* Verify checkpoints, logging, evaluation, and inference.
* Do not run long experiments until the pipeline is stable.

## Stage B: English baseline

* Train or fine-tune on English correction data.
* Include real and synthetic examples.
* Evaluate by error category.

## Stage C: Hinglish baseline

* Train or fine-tune on Hinglish normalisation and synthetic Hinglish errors.
* Evaluate separately from English.

## Stage D: Combined English-Hinglish model

* Combine balanced English, Hinglish, and mixed-language data.
* Prevent English data from overwhelming Hinglish.
* Compare balanced sampling strategies.

## Stage E: Model comparison

Compare at least:

* One byte-level model.
* One multilingual subword model.
* One smaller English-oriented baseline where possible.

## Stage F: Optimised inference

For the best model:

* Test FP16 inference.
* Test quantisation where supported.
* Export to ONNX if practical.
* Measure CPU and GPU inference.
* Measure model-load time.
* Measure warm and cold latency.
* Measure memory usage.

---

# 12. Experiment tracking

Every training run must create a structured experiment folder.

Example:

```text
experiments/
  byt5-small-run-001/
    config.json
    dataset_summary.json
    train_metrics.json
    validation_metrics.json
    test_results.json
    latency_results.json
    sample_predictions.jsonl
    failures.jsonl
    training.log
    environment.txt
    model_card.md
```

Log:

* Experiment ID.
* Date and time.
* Git commit.
* Model name.
* Dataset versions.
* Dataset sizes.
* Sampling ratios.
* Synthetic corruption settings.
* Random seed.
* Batch size.
* Effective batch size.
* Learning rate.
* Optimiser.
* Scheduler.
* Number of epochs.
* Maximum input and output lengths.
* FP16 or FP32.
* Gradient accumulation.
* Gradient checkpointing.
* VRAM usage.
* RAM usage.
* Training duration.
* Validation duration.
* Inference speed.
* Checkpoint paths.
* Final scores.
* Manual evaluation score.
* Known problems.

Use TensorBoard, Weights & Biases, MLflow, CSV logs, or a lightweight local alternative. Prefer a setup that works without paid cloud services.

---

# 13. Evaluation requirements

Do not evaluate only using loss.

Use multiple evaluation methods.

## Automated metrics

Research suitable metrics such as:

* Character error rate.
* Word error rate.
* Exact-match accuracy.
* Sentence-level accuracy.
* Precision, recall, and F-score for edits.
* GLEU.
* ERRANT-based metrics for English where appropriate.
* Levenshtein distance.
* Normalised edit distance.
* ChrF.
* Meaning-similarity score.
* Over-correction rate.
* Under-correction rate.

Not every metric will work equally well for Hinglish. Explain the limitations.

## Error-category evaluation

Maintain separate test sets for:

* Character swaps.
* Character deletion.
* Character insertion.
* Nearby-key substitutions.
* Duplicate characters.
* Split words.
* Merged words.
* Capitalisation.
* Punctuation.
* English grammar.
* Hinglish spelling.
* Hinglish abbreviations.
* Mixed English-Hinglish sentences.
* Light noise.
* Medium noise.
* Heavy noise.
* Long paragraphs.
* Technical terms.
* Names and entities.

## Meaning preservation

The model must not alter the intended meaning.

Example:

```text
I don't think we should release it.
```

must never become:

```text
I think we should release it.
```

Track meaning-changing failures separately and treat them as severe errors.

## Minimal-edit behaviour

The system should change only what is necessary.

Input:

```text
bro kal meeting 4 bje hai
```

A suitable correction may be:

```text
Bro, kal meeting 4 baje hai.
```

An unsuitable correction would be:

```text
The meeting is scheduled for four o'clock tomorrow.
```

The second output may be grammatically correct, but it destroys the user’s natural language and tone.

---

# 14. Self-test suite

Create a manually curated test suite that remains unchanged across experiments.

It should include:

* Short sentences.
* Medium sentences.
* Long sentences.
* Multi-sentence paragraphs.
* English.
* Hinglish.
* Mixed English-Hinglish.
* Casual messages.
* Professional messages.
* Technical text.
* Names.
* Severe typo density.
* Broken word boundaries.
* Ambiguous inputs.
* Sentences that should remain mostly unchanged.

Include at least several hundred examples over time, but start with a smaller carefully designed set.

Example categories:

```text
tests/
  english_light.jsonl
  english_heavy.jsonl
  hinglish_light.jsonl
  hinglish_heavy.jsonl
  mixed_language.jsonl
  spacing_errors.jsonl
  meaning_preservation.jsonl
  no_change_needed.jsonl
  long_paragraphs.jsonl
  difficult_cases.jsonl
```

Each test example should contain:

```json
{
  "id": "hinglish-heavy-001",
  "input": "mian aj clg nhi ja rha kyki tabyat thk nhi",
  "reference": "Main aaj college nahi ja raha kyunki tabiyat theek nahi hai.",
  "language": "hinglish",
  "severity": "heavy",
  "error_types": [
    "character_swap",
    "abbreviation",
    "missing_word",
    "capitalisation"
  ]
}
```

---

# 15. Human-style success scoring

In addition to automated metrics, implement a simple manual review process.

For every tested output, assign one of these labels:

* **Perfect:** Correct output with meaning and tone fully preserved.
* **Good:** Minor stylistic issue, but clearly useful.
* **Acceptable:** Some errors remain, but output is better than input.
* **Bad:** Output is not meaningfully improved.
* **Harmful:** Meaning changed, important content removed, or wrong language used.

Calculate:

* Perfect percentage.
* Perfect + Good percentage.
* Acceptable-or-better percentage.
* Bad percentage.
* Harmful percentage.

The main practical metric should be:

> Out of all tested examples, how many corrections would a real user be happy to accept without editing?

Record this separately for:

* English.
* Hinglish.
* Mixed language.
* Short text.
* Long text.
* Light noise.
* Heavy noise.

---

# 16. Repeated self-testing

After every meaningful model checkpoint or completed run:

1. Run the fixed test suite.
2. Save all predictions.
3. Compare them with previous runs.
4. Count regressions.
5. Count improvements.
6. Record latency.
7. Record memory use.
8. Produce a concise summary.

Example summary:

```text
Model: ByT5-small run 004

Total test examples: 300

Perfect: 168
Good: 67
Acceptable: 31
Bad: 29
Harmful: 5

Perfect or good: 78.3%
Acceptable or better: 88.7%
Harmful: 1.7%

English perfect or good: 84.2%
Hinglish perfect or good: 73.4%
Mixed perfect or good: 69.8%

Median warm latency: 210 ms
P95 warm latency: 480 ms
Cold model load: 4.8 seconds
Peak GPU memory: 5.4GB
```

Do not claim success based only on a few impressive examples.

---

# 17. Latency testing

Speed is a core requirement.

The future application should feel nearly instant after the keyboard shortcut is pressed.

Measure:

* Model loading time.
* Tokenisation or byte-processing time.
* Inference time.
* Decoding time.
* Full correction pipeline time.
* CPU inference.
* GPU inference.
* Cold inference.
* Warm inference.
* Median latency.
* P90 latency.
* P95 latency.
* P99 latency.
* Short sentence latency.
* Long sentence latency.
* Paragraph latency.

Test inputs such as:

* 5–10 words.
* 20–30 words.
* 50–100 words.
* 150–300-word paragraphs.

Initial target goals:

* Short sentence: ideally under 500 ms after warm-up.
* Medium paragraph: ideally under 1 second.
* Long paragraph: preferably under 2 seconds.
* Model load can take longer, but the future application should keep the model loaded in memory.

These targets can be adjusted based on measured hardware performance, but latency must always be included in model comparisons.

A slightly less accurate model may be preferable if it is substantially faster and safer.

---

# 18. Long-paragraph testing

Do not evaluate only short examples.

Test long and realistic text containing several mistakes.

Example:

```text
hey i was cheking the modle results and i thnik the englsh cases are working fine but hinglsh me kuch sentnces ka meaning chnge ho rha hai and the inferrence time is also high wen the para is more than 100 words
```

The model should correct errors while preserving the original mixed-language structure.

For long text, test:

* Whether sentences are dropped.
* Whether content is repeated.
* Whether the model rewrites instead of correcting.
* Whether Hinglish is translated.
* Whether the ending is truncated.
* Whether latency increases sharply.
* Whether hallucinations appear.

Consider chunking long paragraphs by sentence or logical boundaries, but compare chunked and non-chunked approaches.

---

# 19. Baselines

Build simple baselines before claiming the fine-tuned model is effective.

Possible baselines:

* Hunspell or another spell checker.
* SymSpell.
* LanguageTool.
* A dictionary-based correction method.
* Character edit-distance correction.
* A pretrained model without fine-tuning.
* A general-purpose small language model prompted to correct text.
* Rule-based spacing and capitalisation correction.
* An English-only grammar tool.

Compare TypoPilot against them using the same test set.

This will show whether fine-tuning actually provides meaningful improvement.

---

# 20. Error analysis

After every important experiment, analyse failures manually.

Create categories such as:

* Failed to correct typo.
* Incorrect correction.
* Over-correction.
* Changed meaning.
* Translated Hinglish.
* Removed slang.
* Changed name.
* Changed technical term.
* Failed split-word repair.
* Failed merged-word repair.
* Hallucinated content.
* Repeated content.
* Truncated output.
* Incorrect punctuation.
* Incorrect capitalisation.
* Slow inference.
* Out-of-memory failure.

Save representative examples for each category.

Use error analysis to decide the next experiment instead of changing settings randomly.

---

# 21. Safety and privacy considerations

Even though this phase is model research, plan for the future product.

Potential risks include:

* Sensitive text being sent to a server.
* Passwords or private messages being processed.
* Model changing legal, financial, medical, or professional meaning.
* Names and technical terms being incorrectly modified.
* Hinglish being culturally or linguistically over-normalised.
* Training data containing private or offensive text.
* Harmful corrections that reverse negation.
* Silent replacement without a clear undo path.
* Model hallucination.
* Dataset licence violations.
* Bias toward formal English.
* Uneven performance between English and Hinglish.

The future product should preferably support fully local inference.

The correction engine should eventually detect and avoid:

* Password fields.
* URLs.
* Email addresses.
* Source code.
* Commands.
* API keys.
* File paths.
* Very short ambiguous selections.
* Content where correction confidence is too low.

For the model research phase, document these risks and create test cases for them.

---

# 22. Confidence and fallback behaviour

Research whether the model can estimate correction confidence.

The future product should not blindly replace text when the model is uncertain.

Possible strategies:

* Compare model probabilities.
* Use edit-distance thresholds.
* Detect unusually large rewrites.
* Reject outputs that differ too much from the input.
* Reject outputs with lower semantic similarity.
* Reject outputs containing new entities.
* Use a second validation model.
* Use rules to detect changed numbers, negations, dates, names, or URLs.
* Return the original text when confidence is low.

Create a post-processing safety layer and evaluate whether it reduces harmful corrections.

---

# 23. Future desktop application

Only after the model reaches acceptable quality and speed should development move to the product phase.

The intended future product is:

> A universal desktop text-correction layer for Windows and macOS.

Expected workflow:

1. User types messy text anywhere.
2. User selects the text or places the cursor in a sentence.
3. User presses a global keyboard shortcut.
4. TypoPilot captures the relevant text.
5. The local model corrects it.
6. The corrected text replaces the original.
7. The user can immediately undo the correction.

Possible correction modes:

* Fix selected text.
* Fix current sentence.
* Fix current paragraph.
* Spelling only.
* Grammar and punctuation.
* Preserve casual tone.
* Professional rewrite as an optional separate mode.

Potential implementation later:

## Windows

* Tray application.
* Global keyboard shortcut.
* Clipboard capture and replacement as the first fallback.
* Windows UI Automation for more direct text access.
* Clipboard restoration.
* Undo support.
* Password-field protection.

## macOS

* Menu-bar application.
* Global shortcut.
* Accessibility APIs.
* Clipboard fallback.
* Accessibility-permission onboarding.
* Secure-field protection.

## Local inference

* Model loaded in the background.
* ONNX Runtime or another optimised inference runtime.
* CPU and GPU options.
* Quantised model where quality remains acceptable.
* No internet required for correction.

Do not begin this phase until explicitly approved after model evaluation.

---

# 24. Success criteria before building the application

Do not recommend moving to desktop-app development until the model satisfies clearly defined thresholds.

Initial suggested thresholds:

* At least 85% of English examples rated acceptable or better.
* At least 80% of Hinglish examples rated acceptable or better.
* At least 80% of mixed-language examples rated acceptable or better.
* At least 70% rated perfect or good overall.
* Harmful correction rate below 1%.
* Very low negation, number, and named-entity corruption.
* Median warm latency below 500 ms for short sentences.
* P95 warm latency below 1 second for short and medium inputs.
* Long paragraphs corrected without truncation or hallucination.
* Model runs reliably within available memory.
* Clear improvement over rule-based and pretrained baselines.

These are starting targets and can be revised based on experimental evidence.

---

# 26. Reproducibility

The entire research project must be reproducible.

Include:

* Exact dependency versions.
* Environment setup instructions.
* CUDA version.
* PyTorch version.
* Dataset-download instructions.
* Dataset licences.
* Random seeds.
* Training configs.
* Model checkpoints.
* Evaluation scripts.
* Saved predictions.
* Hardware details.
* Known nondeterministic behaviour.

A new user should be able to reproduce a training run using a documented command.

Example:

```bash
python -m src.models.train \
  --config configs/training/byt5_small_combined.yaml
```

And evaluate it using:

```bash
python -m src.models.evaluate \
  --checkpoint experiments/byt5-small-run-004/checkpoint-best \
  --test-suite tests/curated/all.jsonl
```

---

# 27. Required research report

At the end of the initial phase, produce a detailed report containing:

1. Problem definition.
2. English and Hinglish correction policy.
3. Dataset research.
4. Dataset licences.
5. Data-cleaning methodology.
6. Synthetic-noise methodology.
7. Model candidates.
8. Hardware constraints.
9. Training configurations.
10. Evaluation metrics.
11. Baseline results.
12. Fine-tuned model results.
13. English results.
14. Hinglish results.
15. Mixed-language results.
16. Error-category results.
17. Long-paragraph results.
18. Latency results.
19. Memory usage.
20. Manual ratings.
21. Regression comparisons.
22. Failure examples.
23. Meaning-changing errors.
24. Risks and limitations.
25. Recommended final model.
26. Whether the model is ready for product development.
27. Next experiments if it is not ready.

Be honest about weak results.

Do not present a model as successful merely because a few selected examples look good.

---

# 28. First implementation milestone

Begin with the following milestone only:

## Milestone 1: Research and baseline prototype

1. Research available English and Hinglish datasets.
2. Verify licences and download methods.
3. Research suitable compact models.
4. Produce a dataset and model comparison document.
5. Select one English dataset and one Hinglish dataset.
6. Build the dataset loader and cleaner.
7. Build the synthetic-noise generator.
8. Prepare a small pilot dataset.
9. Fine-tune one small candidate model.
10. Build a command-line correction script.
11. Run the fixed self-test suite.
12. Save all predictions.
13. Calculate automated metrics.
14. Manually classify the outputs.
15. Measure inference latency.
16. Record training time and hardware usage.
17. Produce an experiment report.
18. Recommend the next model or dataset experiment.

Do not proceed to desktop integration.

---

# 29. Behaviour expected from the development agent

Work independently and make evidence-based decisions.

You are allowed and expected to:

* Research better datasets.
* Research newer model architectures.
* Challenge assumptions in this specification.
* Reject unsuitable datasets.
* Recommend a different compact model.
* Adjust training strategies based on memory limits.
* Create additional evaluation categories.
* Improve the synthetic corruption methodology.
* Identify licensing problems.
* Identify data-quality issues.
* Stop an experiment if it is clearly unproductive.
* Propose cloud training only when local hardware is insufficient.
* Document uncertainty.

Before making a major change, explain:

* What was discovered.
* Why the original approach may be insufficient.
* What change is recommended.
* What evidence supports the change.
* What trade-offs it introduces.

The main priority is not completing a checklist. The priority is finding the best realistic model for fast, safe English and Hinglish noisy-text correction.

---

# 30. Final objective

The final model should take heavily mistyped English, Hinglish, or mixed English-Hinglish text and return a minimally edited, meaning-preserving correction.

It should be:

* Fast.
* Small.
* Locally runnable.
* Accurate.
* Conservative.
* Robust to severe keyboard errors.
* Strong with broken and merged words.
* Strong with Romanised Hindi.
* Safe against meaning changes.
* Suitable for future integration into a Windows and macOS desktop application.

The product vision is:

> **TypoPilot — Type fast. TypoPilot fixes the rest.**

For now, focus entirely on proving that the correction model can work.
