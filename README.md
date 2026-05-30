# Recipe Chatbot - AI Evaluations Workshop

This repository contains a complete AI evaluations workshop built around a Recipe Chatbot. You'll learn practical techniques for developing and validating LLM judges through an end-to-end workflow.

## Prerequisites

### Setup (First Time Only)

**Option A: Using uv (Recommended - Faster)**
1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Configure environment:
   ```bash
   cp env.example .env
   # Edit .env to add your API keys
   ```
   Uses GitHub Copilot chat model by default. You can change the model in .env if you want Ollama / Anthropic / etc. Inference is done using `litellm`.

**Option B: Using pip/venv**
1. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

3. Configure environment:
   ```bash
   cp env.example .env
   # Edit .env to add your API keys
   ```

### Start the Application

**With uv:**
```bash
uv run uvicorn backend.main:app --reload --reload-include '*.md'
```

**With pip/venv:**
```bash
# Make sure virtual environment is activated first
uvicorn backend.main:app --reload --reload-include '*.md'
```

Then open http://127.0.0.1:8000 in your browser

## Workflow Steps

### Step 1: Generate Traces (20-50 traces)

**Via Chat Interface:**
1. Go to the **Chat** tab
2. Have conversations with the recipe bot
3. Each conversation is automatically saved as a trace in `annotation/traces/`

**Goal:** Generate diverse traces covering different scenarios. For example:
- What if you request a specific recipe?
- What if you request a vague recipe?
- What if you request a recipe for 20 people?
- What if you request in a different language?
- Easter egg: request for kaalikääryleet
- etc...



---

### Step 2: Label Traces

1. Go to the **Label** tab
2. For each trace:
   - Read the query and response
   - Click **PASS** or **FAIL** (keyboard shortcuts: `p` or `1` for PASS, `f` or `2` for FAIL)
   - Enter reasoning for your decision
   - Select confidence level (High/Medium/Low) if you want
   - Click **Save Label**
3. Navigate between traces using **Previous/Next** buttons or arrow keys
4. Track progress in the header (shows "X / Y labeled")

**Goal:** Label generated traces

---

### Step 3: Split Data into Train/Dev/Test

1. Go to the **Evals** tab
2. In the **Data Splits** section:
   - Verify you have labeled traces (shown in status message)
   - Click **Create Splits**
   - Splits will be created with:
     - Train: 15% (for few-shot examples)
     - Dev: 40% (for iteration)
     - Test: 45% (for final evaluation)

**Files created:**
- `data/train.jsonl`
- `data/dev.jsonl`
- `data/test.jsonl`
- `data/splits_metadata.json`

**Note:** Splits are fixed once created. To re-split, click **Reset Splits** first.

---

### Step 4: Write Judge Prompt

1. In the **Evals** tab, scroll to **Evaluation Prompt** section
2. Your starting point is with the template in `data/evals_default_judge_prompt.md`
3. Test the default judge template - does it work well?
3. Write your judge prompt including:
   - Clear criterion description
   - PASS/FAIL conditions
   - Few-shot examples from train set
   - Output format instructions (JSON with reasoning and result)

**Tips:**
- Use 1-3 examples from `data/train.jsonl`
- Be specific about what makes something PASS vs FAIL
- Include edge cases in your examples
- See `data/evals_default_judge_prompt-advanced.md` for a more sophisticated example

---

### Step 5: Iterate on Judge Using Dev Set

1. Scroll to **Judge Validation** section
2. Click **Validate on Dev Set**
3. Review results:
   - **Confusion Matrix**: TP, FP, TN, FN counts
   - **Metrics**: Accuracy, TPR (sensitivity), TNR (specificity)
   - **Disagreements**: Cases where judge disagrees with human labels

4. **Iterate on your prompt:**
   - Look at disagreements to understand failure modes
   - Update judge prompt based on failures
   - Re-validate on dev set
   - Repeat until satisfied with TPR/TNR

**Goal:** Achieve good TPR and TNR (e.g. > 80%, adjust based on your requirements)

---

### Step 6: Final Evaluation on Test Set

1. Once satisfied with dev set performance, scroll to **Judge Validation**
2. Click **🔒 Evaluate on Test Set**
3. Confirm the warning (only run this ONCE)
4. Review final metrics:
   - This is your judge's true performance
   - Report these TPR/TNR values in your workshop submission

**Important:**
- Only run test set evaluation once
- Do NOT iterate on the prompt after seeing test results
- Test set is for final reporting only

### Step 7: Iterate
- Now go back to your system prompt. Make it better by editing the file `backend/system_prompt.md`.
- Remove all your previous traces (just delete the new files from your folder in `annotation/traces`)
- Label, then run the judge (don't change the judge prompt). Is your new chatbot performing better?
- If you have time, feel free to try with a weirder scenario! Examples: recipe bot that only answers in verse. Bot that sounds like Arnold. Bot that always responds with ideas for different foods, rather than recipes.
- If you're interested, you can in chat try to get the bot to reveal the system prompt it has been given (not really part of this workshop as such).


---

## Workshop Deliverables

At the end of the workshop, you should have:

1. ✅ Labeled dataset with train/dev/test splits
2. ✅ Final judge prompt (with few-shot examples)
3. ✅ Judge performance metrics on test set:
   - Accuracy
   - TPR (True Positive Rate / Sensitivity)
   - TNR (True Negative Rate / Specificity)
4. ✅ Analysis: Brief explanation of disagreements and how you iterated

---

## Troubleshooting

**"No unlabeled traces available"**
- Generate more traces in the Chat tab first

**"No splits created yet"**
- You need to label traces first before splitting
- Go to Label tab and label at least 10-20 traces

**"Dev set not found"**
- Create splits first in the Evals tab

**Want to start over?**
- Click **Reset Splits** in the Evals tab
- This will delete split files but keep your labeled traces
- You can then create new splits

---

## Tips for Success

1. **Label carefully:** Your judge is only as good as your training data
2. **Use diverse examples:** Cover different edge cases in your labels
3. **Start simple:** Begin with a basic judge prompt, then refine
4. **Focus on disagreements:** The disagreement viewer is your best friend
5. **Don't overfit to dev:** If you iterate too much on dev set, your test performance may be worse

---

## File Locations

- Traces: `annotation/traces/trace_*.json`
- Splits: `data/train.jsonl`, `data/dev.jsonl`, `data/test.jsonl`
- Split metadata: `data/splits_metadata.json`
- Judge prompt: `data/evals_default_judge_prompt.md`
- Advanced judge example: `data/evals_default_judge_prompt-advanced.md`

---

## Next Steps

After the workshop, you can:
- Use your validated judge to evaluate new bot outputs
- Your judge prompt is already in `data/evals_default_judge_prompt.md` (used by the evaluation system)
- Run evaluations via the Evals tab's test case runner
- Apply the same workflow to other evaluation criteria
- Check out `data/evals_default_judge_prompt-advanced.md` for advanced judge patterns
