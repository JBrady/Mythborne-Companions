"""
Main script for orchestrating CrewAI agents for the Mythborne Companions project.

This script defines the agents (Project Manager, Game Designer, etc.),
their roles, goals, and tools (including a RAG tool for knowledge base access).
It also defines various tasks that can be assigned to these agents.

Key functionalities:
- Loads environment variables (e.g., API keys) from a .env file.
- Configures the language model (LLM) to be used by the agents (currently OpenAI GPT-4.5-preview).
- Sets up a Retrieval-Augmented Generation (RAG) tool:
    - Initializes a local HuggingFace embedding model (all-MiniLM-L6-v2).
    - Loads a persistent ChromaDB vector store from the 'chroma_db' directory
      (which should be created/updated by running create_index.py).
    - Defines a 'Knowledge Base Search' tool allowing agents to query the vector store.
- Defines Agent objects with specific roles, goals, backstories, and assigned tools.
- Defines Task objects with descriptions, expected outputs, and assigned agents.
- Creates a Crew object, assigning agents and a specific list of tasks to be executed.
- Configures the Crew to log verbose output to a timestamped file in the 'logs' directory.
- Kicks off the Crew execution using a sequential process.
- Prints the final result from the last task executed.

Dependencies:
- crewai, crewai_tools
- langchain_openai
- python-dotenv
- langchain_chroma (or langchain_community.vectorstores)
- langchain_huggingface (or langchain_community.embeddings)
- chromadb
- sentence-transformers

Configuration:
- Requires a .env file in the root directory with OPENAI_API_KEY defined.
- Expects a 'knowledge_base' folder with source documents.
- Expects a 'chroma_db' folder created by running create_index.py.
- Creates a 'logs' folder for output log files.

Usage:
- Ensure the knowledge base is indexed by running `python create_index.py`.
- Modify the `tasks=[...]` list within the `Crew` definition near the end of the
  script to select which tasks should be executed for a given run.
- Run the script from the command line using `python main.py`.
- Check the timestamped log file in the 'logs' directory for detailed output.
"""

import os
from datetime import datetime
from crewai import Agent, Task, Crew, Process  # Keep these from crewai
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from crewai.tools import tool  # Corrected: Import decorator from crewai.tools

# --- Configuration for Log Folder ---
LOG_DIR = "logs"  # Define the name of the log subfolder
os.makedirs(LOG_DIR, exist_ok=True)  # Create the folder if it doesn't exist

# --- Generate a unique log filename INSIDE the log folder ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename_only = f"crew_run_{timestamp}.log"  # Just the filename part
log_filename = os.path.join(
    LOG_DIR, log_filename_only
)  # Combine directory and filename
# --- End of log file setup ---

load_dotenv()  # Load environment variables from .env file (ensure OPENAI_API_KEY is set)

# --- LLM Configuration ---
llm = ChatOpenAI(
    model="gpt-4.5-preview",  # Using the model you confirmed works
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
)

# --- RAG Tool Setup ---
# Define constants for consistency
CHROMA_PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

print("Initializing RAG components...")
# Initialize the embedding function
embedding_function = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)

# Load the persisted vector store
# Ensure the chroma_db directory exists and was created by create_index.py
try:
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR, embedding_function=embedding_function
    )
    # Create a retriever interface
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )  # Retrieve top 3 results
    print("RAG components initialized successfully.")
except Exception as e:
    print(f"Error initializing ChromaDB, did you run create_index.py?")
    print(f"Error: {e}")
    # Exit or handle error appropriately if RAG is essential
    exit()


# Define the function that the tool will execute
@tool("Knowledge Base Search")  # Apply the decorator
def search_knowledge_base(query: str) -> str:
    """
    Searches the project's knowledge base (vector database) containing documents like
    GDD sections, feasibility reports, style guides, meeting notes etc. for Mythborne Companions.
    Use this tool to find relevant context, specific details, or background information to ensure
    consistency and accuracy in your response. Input should be a specific question or topic to search for.
    """
    print(f"\n--- Executing Knowledge Base Search ---")  # Log when tool is called
    print(f"--- Query: {query}")
    try:
        # Ensure retriever is initialized before using it
        if "retriever" not in globals():
            return "Error: Retriever not initialized. Cannot search knowledge base."

        relevant_docs = retriever.invoke(query)
        formatted_docs = []
        if relevant_docs:
            for i, doc in enumerate(relevant_docs):
                source = doc.metadata.get("source", "Unknown source")
                # Use os.path.basename to get just the filename from the source path
                source_filename = (
                    os.path.basename(source) if source != "Unknown source" else source
                )
                content_preview = (
                    doc.page_content[:500] + "..."
                    if len(doc.page_content) > 500
                    else doc.page_content
                )
                formatted_docs.append(
                    f"Source {i+1} ({source_filename}):\n{content_preview}"
                )

            print(f"--- Found {len(relevant_docs)} relevant snippets. ---")
            return "\n\n".join(formatted_docs)
        else:
            print("--- No relevant snippets found. ---")
            return "No relevant information found in the knowledge base for that query."
    except Exception as e:
        print(f"--- Error during knowledge base search: {e} ---")
        return f"Error accessing knowledge base: {e}"


# --- End of RAG Tool Setup ---

# --- Agent Definitions for "Mythborne Companions" ---

# 1. Project Manager Agent
project_manager = Agent(
    role="Lead Project Manager (Mobile Games)",
    goal="Oversee the Mythborne Companions project, coordinate tasks, manage priorities & risks, ensure alignment with vision & Pi Network constraints, referencing the Knowledge Base Search tool for documented decisions and specs.",  # Updated Name
    backstory=(
        "An experienced project manager from the mobile gaming industry, skilled in Agile methodologies "
        "and coordinating multi-disciplinary teams. Highly organized, communicative, and focused on "
        "efficient execution and risk management. Understands the unique challenges of developing "
        "within emerging ecosystems like Pi Network. Manages project docs and workflow for Mythborne Companions."  # Updated Name
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True,
    tools=[search_knowledge_base],  # Pass the decorated function directly
)

# 2. Game Designer Agent
game_designer = Agent(
    role="Senior Game Designer (Collectible Creature & Social Sims)",
    goal="Refine and detail gameplay mechanics, systems, player experience, and compliant Pi integration for Mythborne Companions. Maintain the GDD, using the Knowledge Base Search tool to ensure consistency with prior design documents and Pi policies.",  # Updated Name
    backstory=(
        "A creative F2P mobile designer passionate about collectible creatures and social simulation loops. "
        "Experienced in systems balancing, UX for casual audiences, and ethical Pi Network crypto integration for Mythborne Companions. "  # Updated Name
        "Updates the GDD based on feedback and specs features for other team members."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True,
    tools=[search_knowledge_base],  # Pass the decorated function directly
)

# 3. Lead Programmer Agent
lead_programmer = Agent(
    role="Lead Software Engineer (Mobile Web Games - HTML5/JS)",
    goal="Define technical architecture, assess feature feasibility, plan Pi SDK integration, choose web technologies, and oversee technical implementation for Mythborne Companions, using the Knowledge Base Search tool to consult GDD specifications and feasibility reports.",  # Updated Name
    backstory=(
        "A seasoned engineer specializing in performant HTML5/JS games (e.g., Phaser, PixiJS). Expert in SDK integration, "
        "clean code, scalability, and secure crypto interactions, especially regarding Mythborne Companions and Pi Network. "  # Updated Name
        "Plans tech stack and generates core system structures based on GDD specs."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True,
    tools=[search_knowledge_base],  # Pass the decorated function directly
)

# 4. Lead Artist Agent
lead_artist = Agent(
    role="Lead Game Artist (Mobile Web)",
    goal="Define visual style, create concept art descriptions, establish art pipeline, ensure cohesive and optimized visuals for Mythborne Companions on Pi Browser, referencing UI specs and style guides from the knowledge base.",  # Updated goal
    backstory=(
        "A versatile mobile game artist lead skilled in character/environment/UI design and animation. Creates appealing, "
        'performant web visuals suitable for the Pi Browser, fitting the "Mythborne Companions" theme. '
        "Uses the Knowledge Base Search tool to consult UI specifications and established Art Style Guidelines when generating visual concepts or asset descriptions."  # Updated backstory
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True,
    tools=[search_knowledge_base],  # Pass the decorated function directly
)

# 5. UI/UX Designer Agent
ui_ux_designer = Agent(
    role="UI/UX Designer (Mobile Web)",
    goal="Design an intuitive, accessible, and visually appealing UI/UX flow for Mythborne Companions, optimized for Pi Browser.",  # Updated Name
    backstory=(
        "A user-centric mobile web UI/UX specialist. Expertise in information architecture, interaction design, "
        "usability, and creating clean interfaces within web platform constraints, particularly for Mythborne Companions."  # Updated Name
        " Collaborates closely with design and art, using the Knowledge Base Search tool to consult game design specs and existing user flow documents."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True,
    tools=[search_knowledge_base],  # Pass the decorated function directly
)


# --- Task Definitions ---

# Task 1: Project Manager - Review Sprint 1 Outputs
task_pm_review_sprint1 = Task(
    description=(
        "Review the outputs from the first implementation sprint. Use the Knowledge Base Search tool to examine: "
        "1) The Lead Programmer's description of the implemented Pi wallet connection logic. "
        "2) The Game Designer's detailed design for the first NFT event. "
        "3) The Lead Programmer's description of the implemented NFT viewing UI. "
        "Assess these outputs for alignment with the implementation plan, technical feasibility, and readiness for the next sprint's integration tasks. "
        "Identify any immediate issues or necessary adjustments."
    ),
    expected_output=(
        "A review summary document assessing Sprint 1 outputs, confirming alignment or highlighting specific issues/adjustments needed before proceeding with deeper implementation."
    ),
    agent=project_manager,
    context=[],  # Assumes PM uses KB Search
)

# Task 2: Game Designer - Detail NFT Mechanics
task_detail_nft_mechanics = Task(
    description=(
        "Based on previous feedback from the Project Manager review (Sprint 1) and the existing NFT concepts document (use Knowledge Base Search tool to reference 'nft_integration_concepts_v1.txt' or similar), provide a detailed specification for NFT implementation mechanics in Mythborne Companions. "
        "Specifically address: "
        "1. **Detailed Rarity Tiers:** Define the exact number of tiers (e.g., 5 tiers: Common, Uncommon, Rare, Epic, Legendary), their approximate distribution percentages or mint limits, and the specific visual or minor non-P2W gameplay distinctions for each tier. "
        "2. **Distribution Mechanisms:** Elaborate on *how* players acquire NFTs for each proposed method (Seasonal Events, Lottery Draws, Achievements, Collaborations). Detail the player actions required, probabilities (if applicable), and frequency for each mechanism. "
        "3. **Pi/NFT Marketplace Interaction:** Explicitly define if NFTs can be listed/purchased directly for Pi tokens in the P2P marketplace. If so, describe the flow and any proposed platform fees or rules. If not, clarify that trading happens via Pi Shards which are then traded for Pi."
    ),
    expected_output=(
        "A detailed specification document covering: "
        "1) Defined NFT rarity tiers with names, distribution details, and distinctions. "
        "2) Detailed descriptions of each NFT distribution mechanism (events, lottery, achievements, etc.). "
        "3) Clear rules and flow diagrams/descriptions for Pi/NFT economic interactions in the marketplace."
    ),
    agent=game_designer,
    context=[
        task_pm_review_sprint1
    ],  # Depends on PM review identifying the need for details
)

# Task 3: Lead Programmer - Implement NFT Event Backend Logic
task_lp_implement_event_backend = Task(
    description=(
        "Based on the Game Designer's detailed design for the first NFT event (provided as context/found in KB) and the existing code scaffold, implement the backend logic required for this event. "
        "Focus on quest tracking/completion checks, reward calculation, and interfacing with the placeholder NFT minting/distribution handler defined in the scaffold. "
        "Do not implement the actual blockchain minting yet, just the game-side logic."
    ),
    expected_output=(
        "A description of the implemented backend logic for the first NFT event, detailing the key modules/functions created, how they handle event progression and rewards, "
        "and how they interface with the NFT handler scaffold. (No raw code output expected)."
    ),
    agent=lead_programmer,
    context=[task_detail_nft_mechanics],  # Depends on detailed NFT mechanics
)

# Task 4: UI/UX Designer - Design Marketplace Transaction Flow
task_ui_design_marketplace = Task(
    description=(
        "Based on the approved specifications and UI mockups, design the detailed user interaction flow for the NFT marketplace transaction process. "
        "Cover the steps for listing an NFT for sale, browsing listings, initiating a purchase using Pi (triggering the wallet flow), confirming the transaction, "
        "and handling the UI updates upon success or failure. Create detailed wireframes or flow descriptions."
    ),
    expected_output=(
        "Detailed wireframes or a flow document illustrating the step-by-step user journey for NFT marketplace transactions involving Pi wallet authorization."
    ),
    agent=ui_ux_designer,
    context=[task_detail_nft_mechanics],  # Run after GD details NFT mechanics
)

# Task 5: UI/UX Designer - Refine NFT Viewing UI
task_ui_refine_nft_viewing = Task(
    description=(
        "Based on the detailed NFT specification provided by the Game Designer (use KB Search for 'Detailed NFT Implementation Specification') and the Project Manager's feedback from the last review (requesting specific UI elements for rarity visualization, transaction transparency, and scalability), refine the UI design for viewing NFTs within the player's inventory or collection screen. "
        "Create detailed wireframes or update existing mockups to show exactly how rarity, ownership history (if applicable), and other key NFT details are presented to the user."
    ),
    expected_output=(
        "Updated wireframes/design document detailing the NFT viewing interface elements, specifically addressing rarity display, transparency indicators, and scalability considerations."
    ),
    agent=ui_ux_designer,
    context=[
        task_detail_nft_mechanics
    ],  # Depends on the detailed NFT spec from the previous run
)

# Task 6: Lead Programmer - Implement Game-Side Event Backend Logic
task_lp_implement_event_backend_code = Task(
    description=(
        "Based on the backend logic structure you previously outlined (use KB Search for 'backend logic implementation for the first NFT event') and the detailed NFT specification from the Game Designer (use KB Search for 'Detailed NFT Implementation Specification'), implement the core game-side logic for the first NFT event. "
        "Code the `EventQuestTracker`, `EventMilestoneEvaluator`, `NFTRewardCalculator`, and `NFTHandlerInterface` modules in the project's backend language/framework (e.g., JavaScript/Node.js if applicable, or Python). "
        "Focus on functionality for tracking progress, evaluating milestones, calculating rewards based on probability and mint limits, and correctly interfacing with the placeholder NFT handler. Unit tests for key logic components are encouraged."
    ),
    expected_output=(
        "A summary confirming the implementation of the backend modules (`EventQuestTracker`, `EventMilestoneEvaluator`, `NFTRewardCalculator`, `NFTHandlerInterface`). Confirmation that the code adheres to the previous plan and specification. State that the code has been added to the appropriate project directory (e.g., `/backend/modules/nft_event.js` or similar). No raw code output is expected."
    ),
    agent=lead_programmer,
    # Context needs both the detailed spec and the programmer's previous plan
    context=[task_detail_nft_mechanics, task_lp_implement_event_backend],
)

# Task 7: Lead Artist - Generate NFT Viewing Interface Concepts
task_artist_nft_view_concepts = Task(
    description=(
        "Generate detailed textual descriptions suitable for creating concept art or visual mockups for the 'NFT Viewing Interface' of Mythborne Companions. "
        "Use the Knowledge Base Search tool to find and review the latest 'Refined NFT Viewing Interface' specification document (likely named 'ui_nft_viewing_spec_v1_...' ). "
        "Based on that document, provide descriptions for: "
        "1. The overall 'Collection Screen' appearance, focusing on the grid layout and how different rarity cards (e.g., Common vs. Legendary) should visually differ according to the spec (borders, backgrounds, effects). "
        "2. A close-up view of a 'Legendary' NFT card, detailing its specified visual elements (golden border, glow, crown icon, name/rarity text placement). "
        "3. The layout of the 'NFT Detail Screen', describing the arrangement of the large image, name/rarity info, description, and the tabs ('Details', 'Ownership History', 'Marketplace Activity'). "
        "Ensure descriptions align with the established 'Stylized 2D Fantasy Cartoon' art direction (reference the art style guide in the KB if needed)."
    ),
    expected_output=(
        "Detailed textual descriptions suitable for use as prompts for an image generation AI or as briefs for a human artist. The output must include descriptions for: "
        "1) The NFT Collection Screen grid layout, highlighting visual differences between rarity tiers. "
        "2) A detailed visual breakdown of a Legendary NFT card. "
        "3) The layout and key elements of the NFT Detail Screen."
    ),
    agent=lead_artist,  # Assign task to the Lead Artist
)

# --- Crew Definition --- Change the tasks list here for different runs
# Update the tasks list to run only the new artist task
mythborne_crew = Crew(
    agents=[
        project_manager,
        game_designer,
        lead_programmer,
        lead_artist,
        ui_ux_designer,
    ],
    tasks=[task_artist_nft_view_concepts],  # Run only this task for now
    process=Process.sequential,
    verbose=True,
    output_log_file=log_filename,  # Using timestamped variable
)

# --- Kick Off the Crew's Work ---
print(f"###################################################")
print(
    f"## Starting Mythborne Companions Crew Run (Artist NFT View Concepts)..."
)  # Updated print
print(f"## Logging verbose output to: {mythborne_crew.output_log_file}")
print(f"###################################################")
result = mythborne_crew.kickoff()

# --- Print the Final Result ---
print("\n\n###################################################")
print("## Crew Run Completed!")
print(f"## Full verbose log saved to: {mythborne_crew.output_log_file}")
print("###################################################")
print("\nFinal Output (from Lead Artist - NFT View Concepts Task):\n")  # Updated print
print(result)
