# Workshop Guide: End-to-End Judge Development Workflow

This guide walks through the complete workflow for developing and validating an LLM judge.

## Prerequisites

1. Start the application:
   ```bash
   uv run uvicorn backend.main:app --reload
   ```

2. Open http://127.0.0.1:8000 in your browser

## Workflow Steps

### Step 1: Generate Traces (50-100 traces)

**Via Chat Interface:**
1. Go to the **Chat** tab
2. Have conversations with the recipe bot
3. Each conversation is automatically saved as a trace in `annotation/traces/`

**Via Bulk Script:**
```bash
# Create a CSV file with queries
python scripts/bulk_test.py --csv data/sample_queries.csv
```

**Goal:** Generate 50-100 diverse traces covering different scenarios

---

### Step 2: Label Traces

1. Go to the **Label** tab
2. For each trace:
   - Read the query and response
   - Click **PASS** or **FAIL** (keyboard shortcuts: `p` or `1` for PASS, `f` or `2` for FAIL)
   - Enter reasoning for your decision
   - Select confidence level (High/Medium/Low)
   - Click **Save Label**
3. Navigate between traces using **Previous/Next** buttons or arrow keys
4. Track progress in the header (shows "X / Y labeled")

**Goal:** Label all generated traces

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
2. Optional: Start with the template in `data/judge_prompt_template.md`
3. Write your judge prompt including:
   - Clear criterion description
   - PASS/FAIL conditions
   - Few-shot examples from train set
   - Output format instructions (JSON with reasoning and result)

**Tips:**
- Use 1-3 examples from `data/train.jsonl`
- Be specific about what makes something PASS vs FAIL
- Include edge cases in your examples

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

**Goal:** Achieve TPR and TNR > 85% (adjust based on your requirements)

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
- Judge template: `data/judge_prompt_template.md`

---

## Next Steps

After the workshop, you can:
- Use your validated judge to evaluate new bot outputs
- Update `data/evals_default_judge_prompt.md` with your prompt
- Run evaluations via the Evals tab's test case runner
- Apply the same workflow to other evaluation criteria
