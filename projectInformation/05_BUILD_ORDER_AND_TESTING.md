# Build Order & Testing Guide

Follow these steps exactly in order. Do not skip ahead.

---

## Phase 1: Environment Setup

1. Clone the repo and navigate into it
2. Create virtual environment: `python -m venv venv`
3. Activate it: `source venv/Scripts/activate` (Windows Git Bash)
4. Copy `.env.example` to `.env` and add your Gemini API key
5. Install dependencies: `pip install -r requirements.txt`
6. If conflicts appear about langchain-classic or langchain-openai: `pip uninstall langchain-classic langchain-openai -y`

---

## Phase 2: Create Source Documents

Create these three files in `data/docs/` using content from `04_SOURCE_DOCUMENTS.md`:
- `data/docs/company_overview.txt`
- `data/docs/hr_policy.txt`
- `data/docs/product_faq.txt`

---

## Phase 3: Build Files in This Order

Build files in this exact order (each depends on the previous):

1. `src/config.py`
2. `src/vectorstore/store.py`
3. `test_store.py`
4. Run test: `python test_store.py`
   - Expected output: "Loaded 3 documents, Split into 9 chunks, Vectorstore built..."
   - Expected output: 3 retrieved chunks about vacation days
   - If this works → proceed. If not → fix before continuing.
5. `src/agents/critic.py`
6. `src/pipeline/nodes.py`
7. `src/pipeline/graph.py`
8. `main.py`

---

## Phase 4: Run the Full Pipeline

```bash
python main.py
```

---

## Phase 5: Test Cases

Run these questions to verify the pipeline works correctly:

### Questions that should PASS on first try (answer is in the documents):
- "How many vacation days do NovaTech employees get?"
- "Who is the CEO of NovaTech?"
- "How much does DataPilot cost?"
- "What data sources does DataPilot integrate with?"
- "Does NovaTech offer health insurance?"
- "When are performance reviews conducted?"

### Questions that should trigger FAIL and retry (answer not in documents):
- "What is the stock price of NovaTech?" (not in docs)
- "Who is the VP of Marketing at NovaTech?" (not in docs)
- "What programming language is DataPilot built with?" (not in docs)

### Expected behavior on FAIL:
1. Critic rejects the answer
2. Query is reformulated
3. Retry retrieval with new query
4. If still fails → returns "I don't have enough information..."

---

## Verification Checklist

- [ ] `python test_store.py` runs without errors and retrieves relevant chunks
- [ ] `python main.py` starts without errors
- [ ] Questions about vacation days return "20 days"
- [ ] Questions about CEO return "Sarah Chen"
- [ ] Questions about DataPilot pricing return "$299/month"
- [ ] Questions about topics not in the documents return the graceful fallback message
- [ ] The terminal logs show [RETRIEVE], [GENERATE], [CRITIQUE] steps clearly
- [ ] On a hallucination, the log shows [CRITIQUE] Verdict: FAIL and a retry happens

---

## Common Issues and Fixes

### Issue: ModuleNotFoundError for any langchain package
Fix: Make sure venv is active (`source venv/Scripts/activate`) and run `pip install -r requirements.txt`

### Issue: GOOGLE_API_KEY error
Fix: Check that `.env` file exists in project root and contains `GOOGLE_API_KEY=your_actual_key`

### Issue: ChromaDB errors on second run
Fix: Delete the `data/chroma_db/` folder and let it rebuild: `rm -rf data/chroma_db`

### Issue: SentenceTransformer download hangs
Fix: Wait — it's downloading the ~80MB model for the first time. It caches after that.

### Issue: Gemini API quota exceeded
Fix: The free tier has rate limits. Wait 1 minute and try again, or reduce test frequency.
