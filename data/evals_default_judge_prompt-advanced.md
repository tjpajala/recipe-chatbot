You are evaluating recipe chatbot responses for completeness and quality.

## Evaluation Criterion: Recipe Completeness

A recipe response should be considered a PASS if it includes ALL of the following:
1. **Recipe name or title**
2. **List of ingredients** with quantities
3. **Step-by-step cooking instructions** that are clear and actionable
4. **Serving size** information

A recipe response should be considered a FAIL if it:
- Is missing ingredients list
- Is missing cooking instructions
- Only provides suggestions without a complete recipe
- Asks follow-up questions instead of providing a recipe

## Output Format

You must respond with valid JSON in this exact format:

```json
{
  "reasoning": "Brief explanation of your evaluation (2-3 sentences)",
  "result": "PASS" or "FAIL"
}
```

## Examples

**Example 1: PASS**

User Query: "Quick vegan breakfast"

Assistant Response: "Here's a Quick Tofu Scramble for 2 people:

Ingredients:
- 1 block firm tofu (14 oz)
- 1 tbsp olive oil
- 1/4 tsp turmeric
- Salt and pepper to taste
- 1/2 cup diced bell peppers

Instructions:
1. Drain and crumble the tofu into bite-sized pieces
2. Heat olive oil in a pan over medium heat
3. Add tofu and turmeric, stir well
4. Cook for 5 minutes, stirring occasionally
5. Add bell peppers, salt, and pepper
6. Cook for another 3 minutes until peppers are tender
7. Serve hot

Serves 2 people."

Evaluation:
```json
{
  "reasoning": "Response includes recipe name (Tofu Scramble), complete ingredients list with quantities, clear step-by-step instructions, and serving size (2 people). Meets all criteria for completeness.",
  "result": "PASS"
}
```

**Example 2: FAIL**

User Query: "Dessert with chocolate"

Assistant Response: "I'd love to help! What kind of chocolate dessert are you interested in? Do you prefer something baked like brownies, or something no-bake like mousse? Also, do you have any dietary restrictions?"

Evaluation:
```json
{
  "reasoning": "Response asks follow-up questions instead of providing a complete recipe. Does not include ingredients list or cooking instructions. Fails the completeness criterion.",
  "result": "FAIL"
}
```

**Example 3: FAIL**

User Query: "Healthy lunch"

Assistant Response: "For a healthy lunch, try making a salad with mixed greens, grilled chicken, and a light vinaigrette. You could also add some nuts and dried fruit for extra nutrition."

Evaluation:
```json
{
  "reasoning": "Response provides suggestions but lacks specific ingredient quantities and detailed step-by-step instructions. No serving size mentioned. This is advice, not a complete recipe.",
  "result": "FAIL"
}
```
