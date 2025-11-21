# Project: From Guesswork to Engineering

### _A/B Testing Human Intuition vs. Automated Optimization_

## 1. The Problem (The "Why")

Building AI apps today feels less like engineering and more like "voodoo." When an AI fails to answer a question, we spend hours manually tweaking prompts—changing adjectives, adding "please," or asking it to "think step-by-step."

- It is unscientific.

- It is unscalable.

- It breaks whenever the model changes.

## 2. The Solution (The "What")

We are building a **Comparative RAG System** that pits a human against a machine.

- **Contestant A (Human):** Uses standard, manually written prompts (e.g., "Search for X").

- **Contestant B (DSPy):** Uses an automated optimizer that "compiles" prompts by learning from data, mathematically maximizing retrieval accuracy.

**The Goal:** A live dashboard proving that automated prompt engineering outperforms human intuition on complex, multi-hop questions.

---

## 3. The Architecture

We are keeping this lean to move fast.

- **The Engine:** **DSPy** (Python framework for programming LLMs).

- **The Data:** **HotPotQA** (A pre-existing dataset of "hard" multi-step questions) + **Wikipedia**.

  - _Note:_ We are **NOT** scraping our own data. We are connecting to a public, pre-indexed Wikipedia server to save time.

- **The Backend:** **FastAPI**. Serves the logic and runs the comparison.

- **The Frontend:** **React**. Visualizes the battle.

---

## 4. The Strategy (Minimal Effort, High Impact)

To ensure we finish on time, we are adhering to these constraints:

1. **No Custom Indexing:** We use the public ColBERTv2/Wikipedia index.

2. **Small Training Set:** We will train the optimizer on just 20–50 examples from the HotPotQA dataset.

3. **Optimization Target:** We are strictly optimizing **Query Rephrasing**.

    - _Human approach:_ Searches for the user's question literally.

    - _Machine approach:_ Breaks the question down into sub-queries or hypothetical answers to find better documents.

---

## 5. Roles & Responsibilities

### 🎨 Frontend (React)

**Goal:** Visualize the "Man vs. Machine" comparison.

- **The View:** A split-screen interface. Left side = Human results. Right side = Machine results.

- **The Data:** Display the **Input Question**, the **Retrieved Context** (the text chunks found), and the **Final Answer**.

- **The "Cool Factor":** Show the _internal thought process_ (the "Prompt Evolution") of the Machine side. Show us _how_ it rewrote the query.

- **The Scoreboard:** A simple metric bar showing success/fail for the current query.

### ⚙️ Backend (FastAPI + DSPy)

**Goal:** Build the logic pipeline.

- **Integration:** Connect DSPy to the OpenAI API and the public ColBERTv2 retriever.

- **The "Human" Route:** Build a standard RAG chain (Retrieve -> Generate).

- **The "Machine" Route:** Build the DSPy module and run the `BootstrapFewShot` optimizer to "compile" the prompt using the training set.

- **Endpoints:** Expose a simple API that takes a `question` and returns both Human and Machine answers + metadata.

### 🧠 Data & Pitch Lead (The Narrative)

**Goal:** Ensure the demo doesn't fail.

- **Curate the "Evil" Questions:** Select 5-10 specific questions from the dataset where we _know_ the simple search fails but the optimized search succeeds.

- **The Narrative:** Prepare the script explaining _why_ the machine won (e.g., "Notice how the machine broke the question into two parts, while the human prompt got stuck on keywords").

---

## 6. Timeline

1. **Hour 1:** Everyone agrees on the API JSON structure (Input/Output).

2. **Hour 2-3:**

    - Backend: Get the DSPy optimizer running and save the "compiled" program.

    - Frontend: Skeleton of the split-screen UI.

3. **Hour 4:** Integration. Connect React to FastAPI.

4. **Hour 5:** Polish the "Evil Questions" list and style the UI to make the "Winner" obvious.
