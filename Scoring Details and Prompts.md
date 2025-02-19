# Scoring Translation across Datasets

```
- Self-explanation & Think-aloud
    - Paraphrase presence: 0 - Not present, 1 - Present, 2 - Good
    - Lexical change: BLANK - Not applicable, 0 - Not present, 1 - Present
    - Syntactic change: BLANK - Not applicable, 0 - Not present, 1 - Present
    - Misconception: BLANK - Not applicable, 0 - Not present, 1 - Present
    - Monitoring: BLANK - Not applicable, 0 - Not present, 1 - Present
    - Life event: BLANK - Not applicable, 0 - Not present, 1 - Present
    - Bridge presence: BLANK - Not applicable, 0 - Not present/Poor, 1 - Satisfactory, 2 - Good, 3 - Excellent
    - Bridge contribution: BLANK - Not applicable, 0 - Not present/Poor, 1 - Satisfactory, 2 - Good, 3 - Excellent
    - Elaboration presence: BLANK - Not applicable, 0 - Not present/Poor, 1 - Satisfactory, 2 - Good, 3 - Excellent
    - Overall: 0 - Poor, 1 - Satisfactory, 2 - Good, 3 - Excellent
- Summarization
    - Main Idea: 0 - Not applicable, 1 - Poor, 2 - Fair, 3 - Good, 4 - Excellent
    - Details: 0 - Not applicable, 1 - Poor, 2 - Fair, 3 - Good, 4 - Excellent
    - Cohesion: 0 - Not applicable, 1 - Poor, 2 - Fair, 3 - Good, 4 - Excellent
    - Objective language: 0 - Not applicable, 1 - Poor, 2 - Fair, 3 - Good, 4 - Excellent
    - Wording: 0 - Not applicable, 1 - Poor, 2 - Fair, 3 - Good, 4 - Excellent
    - Language beyond source text: 0 - Not applicable, 1 - Poor, 2 - Fair, 3 - Good, 4 - Excellent
    - Summary Length: 0 - Not applicable, 1 - Poor, 2 - Fair, 3 - Good, 4 - Excellent
- Paraphrasing
    - Garbage Content: 0 - Not present, 1 - Present, 2 - Too much
    - Frozen Expressions: 0 - Not present, 1 - Present, 2 - Too much
    - Syntactic Similarity: 0 - Not present, 1 - Present, 2 - Too much
    - Lexical Similarity: 0 - Not present, 1 - Present, 2 - Too much
    - Irrelevant: 0 - Relevant, 1 - Somewhat relevant, 2 - Irrelevant   
    - Elaboration: 0 - Not present/Poor, 1 - Good, 2 - Excellent 
    - Semantic Completeness: 0 - Not present/Poor, 1 - Good, 2 - Excellent 
    - Entailment: 0 - Not present/Poor, 1 - Good, 2 - Excellent 
    - Paraphrase Quality: 0 - Not present/Poor, 1 - Good, 2 - Excellent 
    - Writing Quality: 0 - Not present/Poor, 1 - Good, 2 - Excellent        
```

# Prompt Format
```
Rate the quality of the following performed task, based on the scoring rubric.

### Task description: {task_description}

{task_support_context}

### Execution: {student_response}

### Scoring rubric:
{detailed_scoring_rubric}

### Response:
{scores}
```

# Prompt Example
```
Rate the quality of the following performed task, based on the scoring rubric.

### Task description: Explain the meaning of the text, elaborating beyond your initial understanding of the text.

- Context: The heart is the hardest-working organ in the living body. Any disorder that terminates the body's blood supply is a threat to life. More people are killed every year in the U.S. by heart disease than by any other disease. A congenital disease is one with which a person is born. Most babies are born with perfect hearts, but something can go wrong for approximately one in 200 cases. Sometimes ...

- Phrase: Any disorder that terminates the body's blood supply is a threat to life.

### Execution: The text is explaining that any type of sickness that affects a person's heart puts this person in danger. This is because the heart is a crucial organ of the human body and supplies blood, so to threat the heart is to endanger the person's life as a whole.

### Scoring rubric:

- Paraphrase presence:

- - Not present: Paraphrase not present.

- - Present: Contains at least one full clause that overlaps; may include single words; contains some of (about half) of the main idea units from the target sentence.

- - Good: Contains most of (50\% or more) of the main idea units from the target sentence.

...

### Response:

- Paraphrase presence: Good

...

```
