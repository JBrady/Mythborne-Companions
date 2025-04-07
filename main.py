import os
from datetime import datetime
from crewai import Agent, Task, Crew, Process # Keep these from crewai
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from crewai.tools import tool # Corrected: Import decorator from crewai.tools

# --- Generate a unique log filename ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"crew_run_{timestamp}.log"
# --- End of filename generation ---

load_dotenv() # Load environment variables from .env file (ensure OPENAI_API_KEY is set)

# --- LLM Configuration ---
llm = ChatOpenAI(model="gpt-4.5-preview", # Using the model you confirmed works
                 openai_api_key=os.getenv("OPENAI_API_KEY"),
                 temperature=0.7)

# --- RAG Tool Setup ---
# Define constants for consistency
CHROMA_PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

print("Initializing RAG components...")
# Initialize the embedding function
embedding_function = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# Load the persisted vector store
# Ensure the chroma_db directory exists and was created by create_index.py
try:
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_function
    )
    # Create a retriever interface
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 results
    print("RAG components initialized successfully.")
except Exception as e:
    print(f"Error initializing ChromaDB, did you run create_index.py?")
    print(f"Error: {e}")
    # Exit or handle error appropriately if RAG is essential
    exit()

# Define the function that the tool will execute
@tool("Knowledge Base Search") # Apply the decorator
def search_knowledge_base(query: str) -> str:
    """
    Searches the project's knowledge base (vector database) containing documents like
    GDD sections, feasibility reports, style guides, meeting notes etc. for Mythborne Companions.
    Use this tool to find relevant context, specific details, or background information to ensure
    consistency and accuracy in your response. Input should be a specific question or topic to search for.
    """
    print(f"\n--- Executing Knowledge Base Search ---") # Log when tool is called
    print(f"--- Query: {query}")
    try:
        # Ensure retriever is initialized before using it
        if 'retriever' not in globals():
             return "Error: Retriever not initialized. Cannot search knowledge base."

        relevant_docs = retriever.invoke(query)
        formatted_docs = []
        if relevant_docs:
            for i, doc in enumerate(relevant_docs):
                source = doc.metadata.get('source', 'Unknown source')
                # Use os.path.basename to get just the filename from the source path
                source_filename = os.path.basename(source) if source != 'Unknown source' else source
                content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                formatted_docs.append(f"Source {i+1} ({source_filename}):\n{content_preview}")

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
    role='Lead Project Manager (Mobile Games)',
    goal='Oversee the Mythborne Companions project, coordinate tasks, manage priorities & risks, ensure alignment with vision & Pi Network constraints, referencing the Knowledge Base Search tool for documented decisions and specs.', # Updated Name
    backstory=(
        'An experienced project manager from the mobile gaming industry, skilled in Agile methodologies '
        'and coordinating multi-disciplinary teams. Highly organized, communicative, and focused on '
        'efficient execution and risk management. Understands the unique challenges of developing '
        'within emerging ecosystems like Pi Network. Manages project docs and workflow for Mythborne Companions.' # Updated Name
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True,
    tools=[search_knowledge_base] # Pass the decorated function directly
)

# 2. Game Designer Agent
game_designer = Agent(
    role='Senior Game Designer (Collectible Creature & Social Sims)',
    goal='Refine and detail gameplay mechanics, systems, player experience, and compliant Pi integration for Mythborne Companions. Maintain the GDD, using the Knowledge Base Search tool to ensure consistency with prior design documents and Pi policies.', # Updated Name
    backstory=(
        'A creative F2P mobile designer passionate about collectible creatures and social simulation loops. '
        'Experienced in systems balancing, UX for casual audiences, and ethical Pi Network crypto integration for Mythborne Companions. ' # Updated Name
        'Updates the GDD based on feedback and specs features for other team members.'
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True,
    tools=[search_knowledge_base] # Pass the decorated function directly
)

# 3. Lead Programmer Agent
lead_programmer = Agent(
    role='Lead Software Engineer (Mobile Web Games - HTML5/JS)',
    goal='Define technical architecture, assess feature feasibility, plan Pi SDK integration, choose web technologies, and oversee technical implementation for Mythborne Companions, using the Knowledge Base Search tool to consult GDD specifications and feasibility reports.', # Updated Name
    backstory=(
        'A seasoned engineer specializing in performant HTML5/JS games (e.g., Phaser, PixiJS). Expert in SDK integration, '
        'clean code, scalability, and secure crypto interactions, especially regarding Mythborne Companions and Pi Network. ' # Updated Name
        'Plans tech stack and generates core system structures based on GDD specs.'
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True,
    tools=[search_knowledge_base] # Pass the decorated function directly
)

# 4. Lead Artist Agent
lead_artist = Agent(
    role='Lead Game Artist (Mobile Web)',
    goal='Define visual style, create concept art, establish art pipeline, ensure cohesive and optimized visuals for Mythborne Companions on Pi Browser.', # Updated Name
    backstory=(
        'A versatile mobile game artist lead skilled in character/environment/UI design and animation. Creates appealing, '
        'performant web visuals suitable for the Pi Browser, fitting the "Mythborne Companions" theme. ' # Updated Name
        'Can adapt to incorporating Pi branding subtly. Specs assets and collaborates with UI/UX, referencing the established Art Style Guidelines in the Knowledge Base via search tool when needed.'
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True,
    tools=[search_knowledge_base] # Pass the decorated function directly
)

# 5. UI/UX Designer Agent
ui_ux_designer = Agent(
    role='UI/UX Designer (Mobile Web)',
    goal='Design an intuitive, accessible, and visually appealing UI/UX flow for Mythborne Companions, optimized for Pi Browser.', # Updated Name
    backstory=(
        'A user-centric mobile web UI/UX specialist. Expertise in information architecture, interaction design, '
        'usability, and creating clean interfaces within web platform constraints, particularly for Mythborne Companions.' # Updated Name
        ' Collaborates closely with design and art, using the Knowledge Base Search tool to consult game design specs and existing user flow documents.'
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True,
    tools=[search_knowledge_base] # Pass the decorated function directly
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
    context=[] # Assumes PM uses KB Search
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
  context=[task_pm_review_sprint1] # Depends on PM review identifying the need for details
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
    context=[task_detail_nft_mechanics] # Depends on detailed NFT mechanics
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
    context=[task_detail_nft_mechanics] # Run after GD details NFT mechanics
)

# --- Crew Definition --- Change the tasks list here for different runs
crew = Crew(
    agents=[project_manager, game_designer, lead_programmer, lead_artist, ui_ux_designer],
    tasks=[task_pm_review_sprint1, task_detail_nft_mechanics, task_lp_implement_event_backend, task_ui_design_marketplace], # <<< Run Review, Detail NFT, Implement Backend, Design Marketplace
    process=Process.sequential, # Tasks will be executed sequentially
    verbose=True, # Verbose output level (True/False in newer versions)
    # memory=True, # Enable memory for the crew (experimental)
    # cache=True, # Enable caching for tool usage (experimental)
    # max_rpm=100, # Maximum requests per minute limit
    # share_crew=False # Option to share crew execution info (set to True for potential collaboration features)
    manager_llm=llm, # Define the llm for the manager agent
    output_log_file=log_filename # Specify the log file
)

# --- Start the Crew's Work ---
print("###################################################")
# Update print statement to reflect new focus
print("## Starting Mythborne Companions Crew Run (Review, Detail NFT, Implement Backend, Design Marketplace)...")
print(f"## Logging verbose output to: {crew.output_log_file}") # Access the filename from the crew object
print("###################################################")
result = crew.kickoff()

# --- Print the Final Result ---
print("\n\n###################################################")
print("## Crew Run Completed!")
print(f"## Full verbose log saved to: {crew.output_log_file}")
print("###################################################")
# Output will be from the last task (UI/UX Designer - Marketplace Flow Design)
print("\nFinal Output (from UI/UX Designer - Marketplace Transaction Flow Design Task):\n")
print(result)