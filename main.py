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


# --- Define Tasks Based on Project Manager's Plan ---

# Task 1 (PM Output Item 1)
task_define_core_loop = Task(
    description=(
        'Based on the initial concept for Mythborne Companions, clearly define the core gameplay loop. '
        'Detail the primary player objectives, core actions (e.g., exploring, collecting, crafting, battling/mini-games), '
        'progression mechanics (e.g., leveling up companions, unlocking features), and how players interact '
        'meaningfully with their Mythborne companions.'
    ),
    expected_output=(
        'A document section detailing the core gameplay loop for Mythborne Companions, covering player objectives, '
        'core actions, progression systems, and key companion interactions.'
    ),
    agent=game_designer # Assigned by PM
)

# Task 2 (PM Output Item 2)
task_define_pi_integration = Task(
    description=(
        'Specify the exact Pi token integrations for Mythborne Companions. Detail potential ways players might earn Pi '
        '(if allowed by Pi Network policies, specify this constraint) and clear ways they can spend Pi within the game '
        '(e.g., for cosmetic items, speeding up timers, special crafting recipes, companion slots). '
        'Crucially, ensure these mechanics strictly align with current Pi Network technical guidelines and platform policies.'
    ),
    expected_output=(
        'A specification document detailing proposed Pi earning (if applicable) and spending mechanics. '
        'Include example costs/rewards in Pi. Explicitly state how alignment with Pi Network policies will be maintained.'
    ),
    agent=game_designer # Assigned by PM
)

# Task 3 (PM Output Item 3)
task_check_feasibility = Task(
    description=(
        'Evaluate the technical feasibility of integrating the proposed Pi Network functionalities '
        '(wallet interaction, transaction handling for Pi spending/earning actions defined in the previous task) '
        'into Mythborne Companions using the Pi SDK/APIs. Consider potential blockchain constraints (e.g., transaction times, fees), '
        'API limitations, security best practices, and the impact on the likely mobile web technology stack (HTML5/JS). '
        'Outline potential challenges and recommended technical approaches or solutions.'
    ),
    expected_output=(
        'A technical feasibility report covering Pi SDK/API integration for the defined mechanics. '
        'Highlight potential challenges (latency, security, policy compliance), constraints, '
        'and recommended technical solutions or necessary workarounds.'
    ),
    agent=lead_programmer # Assigned by PM
)

# Task 4 (PM Output Item 4)
task_define_art_style = Task(
    description=(
        'Establish initial visual art style guidelines for Mythborne Companions. Provide text descriptions '
        'and optionally list key visual references or inspirations (e.g., specific games, art styles) that define the '
        'desired look and feel for the Mythborne companions (creatures), game environments, and UI elements. '
        'The style should be appealing, consistent with the "Mythborne" theme, suitable for the Pi Network community, '
        'and technically feasible for performant mobile web rendering.'
    ),
    expected_output=(
        'A brief document outlining the proposed art style direction (e.g., "Stylized 2D Cartoon", "Semi-Realistic Fantasy"). '
        'Include descriptions of the desired look for companions, environments, and UI. List 2-3 visual reference points if possible.'
    ),
    agent=lead_artist # Assigned by PM
)

# Task 5 (PM Output Item 5)
task_map_core_ux_flows = Task(
    description=(
        'Map out the initial high-level UI/UX user flows for 2-3 core features of Mythborne Companions. '
        'Focus on: (1) Companion Management (viewing stats, feeding/interacting, initiating evolution if applicable) '
        'and (2) a key flow involving Pi token transactions (e.g., purchasing an item from a shop with Pi). '
        'Describe the steps the user takes and the screens they interact with, prioritizing intuitive navigation '
        'and a smooth experience on mobile web.'
    ),
    expected_output=(
        'A document describing the step-by-step user flow for Companion Management and a Pi Transaction screen. '
        'Use text descriptions for each step and screen involved. Focus on clarity and ease of use.'
    ),
    agent=ui_ux_designer # Assigned by PM
)

# --- Define New Tasks Based on User Feedback ---

# Task to Re-evaluate Pi Earning Mechanics
task_revise_pi_earning = Task(
    description=(
        "Review the previously generated Pi Token Integration Specification for Mythborne Companions. "
        "User feedback indicates a strong desire for players to feel they are EARNING Pi value more directly "
        "through gameplay, beyond just P2P trading of Pi Shards. "
        "Investigate and propose creative, engaging game mechanics or systems that are STRICTLY COMPLIANT "
        "with current Pi Network policies but provide players with a tangible sense of earning Pi-related value "
        "directly from their in-game actions (e.g., high-value achievements rewarding items tradable for Pi, "
        "more visible Pi Shard accumulation linked to effort, special non-transferable Pi-linked rewards?). "
        "If direct Pi earning remains impossible due to policy, clearly state this and focus on maximizing the *feeling* "
        "of earning valuable, Pi-exchangeable assets through gameplay. Update the Pi Integration Specification accordingly."
    ),
    expected_output=(
        "An updated Pi Integration Specification document for Mythborne Companions. Detail revised or new mechanics "
        "that allow players to earn significant Pi-related value directly through gameplay actions while strictly adhering "
        "to Pi Network policies. Clearly explain the compliance rationale and how these mechanics create a sense of direct earning."
    ),
    agent=game_designer # Game Designer to handle mechanics and policy alignment
)

# Task to Explore NFT Integration
task_explore_nfts = Task(
    description=(
        "Explore potential concepts for integrating NFTs (Non-Fungible Tokens) into Mythborne Companions. "
        "Brainstorm 2-3 specific, creative use cases (e.g., unique/limited edition Mythborne companions as NFTs, "
        "special cosmetic items/skins, player-owned decorative land plots within their habitat). "
        "For each use case, briefly describe how it might function, how players could acquire/trade them (potentially using Pi), "
        "and how it could enhance gameplay or collection aspects. Also include brief considerations on technical needs "
        "and potential compatibility/policy alignment with the Pi Network blockchain and ecosystem."
    ),
    expected_output=(
        "A brainstorming document outlining 2-3 distinct NFT integration concepts for Mythborne Companions. "
        "Each concept should include: Use Case (what is the NFT?), Acquisition/Trading (how players get/trade it, possibly with Pi?), "
        "Gameplay Impact (how does it enhance the game?), and brief Technical/Pi Network Considerations."
    ),
    agent=game_designer # Game Designer to brainstorm concepts initially
    # context=[task_revise_pi_earning] # Make this task depend on the Pi revision if needed, or run sequentially
)

# --- Create the Crew ---
# Update the tasks list to include the new tasks
mythborne_crew = Crew(
    agents=[project_manager, game_designer, lead_programmer, lead_artist, ui_ux_designer],
    tasks=[
        task_revise_pi_earning, # Focus on this task first
        task_explore_nfts       # Then explore NFTs
        # You could add back other tasks here if needed, or run them separately
    ],
    process=Process.sequential, # Run these two tasks sequentially
    verbose=True,
    # Use the timestamped log file name generated earlier
    output_log_file=log_filename
)

# --- Kick Off the Crew's Work ---
print("###################################################")
# Update print statement to reflect new focus
print("## Starting Mythborne Companions Crew Run (Revising Pi Earning & Exploring NFTs)...")
print(f"## Logging verbose output to: {mythborne_crew.output_log_file}") # Access the filename from the crew object
print("###################################################")
result = mythborne_crew.kickoff()

# --- Print the Final Result ---
print("\n\n###################################################")
print("## Crew Run Completed!")
print(f"## Full verbose log saved to: {mythborne_crew.output_log_file}")
print("###################################################")
# Output will be from the last task (NFT Exploration)
print("\nFinal Output (from Game Designer - NFT Exploration Task):\n")
print(result)