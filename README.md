# Mythborne Companions - CrewAI MVP

This project uses CrewAI to simulate a game development team working on a concept called "Mythborne Companions", a fictional collectible creature game with Pi Network integration.

## Project Overview
Mythborne Companions is a casual, free-to-play mobile experience combining collectible creature mechanics with social simulation elements, deeply integrated with ethical Pi Network crypto mechanics. Players assume the role of a Mythkeeper tasked with collecting, nurturing, and evolving captivating mythical companions (Mythbornes). Through engaging exploration, rewarding collection systems, crafting, battling, and customized interactions, players develop emotional bonds with their Mythbornes and progress through satisfying gameplay loops.

## Project Objectives

The agents collaborate to define game design elements, technical specifications, and other aspects based on assigned tasks.

## Setup

1.  **Environment Variables:**
    *   Create a file named `.env` in the project root.
    *   Add your OpenAI API key to the `.env` file:
        ```
        OPENAI_API_KEY='your_openai_api_key_here'
        ```

2.  **Dependencies:**
    *   Ensure you have Python 3 installed.
    *   Create and activate a virtual environment (recommended):
        ```bash
        python -m venv venv
        # Windows
        .\venv\Scripts\activate
        # macOS/Linux
        # source venv/bin/activate
        ```
    *   Install the required packages:
        ```bash
        pip install -r requirements.txt
        ```

3.  **Knowledge Base Index:**
    *   The agents use a local vector database (ChromaDB) to reference project documents stored in the `knowledge_base` directory.
    *   Create the initial index by running:
        ```bash
        python create_index.py
        ```
    *   **Important:** You must re-run `python create_index.py` *any time* you add, remove, or modify files in the `knowledge_base` directory to keep the index up-to-date.

## Usage

*   To run the CrewAI simulation, execute the main script:
    ```bash
    python main.py
    ```
*   Verbose output will be printed to the console, and a detailed log file (e.g., `crew_run_YYYYMMDD_HHMMSS.log`) will be saved in the project root.
*   The final output of the last task in the sequence will be printed at the end.

## Knowledge Base

The `knowledge_base` directory contains text files that serve as the project's documentation (GDD excerpts, specs, etc.). The agents use the RAG (Retrieval-Augmented Generation) tool (`Knowledge Base Search`) to query these documents for context during their tasks.
