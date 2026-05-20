You are evaluating AI system outputs against a specific quality criterion.

## Your Task

Evaluate whether the output meets the specified criterion. Respond with JSON in this format:

```json
{
  "reasoning": "Brief explanation of your evaluation (2-3 sentences)",
  "result": "PASS" or "FAIL"
}
```

## Evaluation Criterion

[DESCRIBE YOUR CRITERION HERE]

A response should be considered a **PASS** if it:
- [Add specific passing conditions]
- [Add more conditions as needed]

A response should be considered a **FAIL** if it:
- [Add specific failing conditions]
- [Add more conditions as needed]

## Examples

### Example 1: PASS

**User Query:** [Example query]

**AI Output:** [Example output that passes]

**Evaluation:**
```json
{
  "reasoning": "[Explain why this passes the criterion]",
  "result": "PASS"
}
```

### Example 2: FAIL

**User Query:** [Example query]

**AI Output:** [Example output that fails]

**Evaluation:**
```json
{
  "reasoning": "[Explain why this fails the criterion]",
  "result": "FAIL"
}
```

## Guidelines

- Be consistent in your evaluations
- Base your judgment only on the criterion specified above
- If borderline, err on the side of being strict
- Always provide clear reasoning for your decision

---

Now evaluate the following:

**User Query:** {query}

**AI Output:** {output}

**Your Evaluation:**
