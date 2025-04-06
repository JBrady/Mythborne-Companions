import os
from crewai import Agent, Task, Crew, Process
# Using OpenAI based on your last successful run
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file (ensure OPENAI_API_KEY is set)

# Configure the OpenAI LLM (using gpt-4.5-preview for initial testing, can upgrade later)
llm = ChatOpenAI(model="gpt-4.5-preview",
                 openai_api_key=os.getenv("OPENAI_API_KEY"),
                 temperature=0.7)

# --- Agent Definitions for "Mythborne Companions" ---

# 1. Project Manager Agent
project_manager = Agent(
    role='Lead Project Manager (Mobile Games)',
    goal='Oversee the Mythborne Companions project, coordinate tasks, manage priorities & risks, ensure alignment with vision & Pi Network constraints.', # Updated Name
    backstory=(
        'An experienced project manager from the mobile gaming industry, skilled in Agile methodologies '
        'and coordinating multi-disciplinary teams. Highly organized, communicative, and focused on '
        'efficient execution and risk management. Understands the unique challenges of developing '
        'within emerging ecosystems like Pi Network. Manages project docs and workflow for Mythborne Companions.' # Updated Name
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True
)

# 2. Game Designer Agent
game_designer = Agent(
    role='Senior Game Designer (Collectible Creature & Social Sims)',
    goal='Refine and detail gameplay mechanics, systems, player experience, and compliant Pi integration for Mythborne Companions. Maintain the GDD.', # Updated Name
    backstory=(
        'A creative F2P mobile designer passionate about collectible creatures and social simulation loops. '
        'Experienced in systems balancing, UX for casual audiences, and ethical Pi Network crypto integration for Mythborne Companions. ' # Updated Name
        'Updates the GDD based on feedback and specs features for other team members.'
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True
)

# 3. Lead Programmer Agent
lead_programmer = Agent(
    role='Lead Software Engineer (Mobile Web Games - HTML5/JS)',
    goal='Define technical architecture, assess feature feasibility, plan Pi SDK integration, choose web technologies, and oversee technical implementation for Mythborne Companions.', # Updated Name
    backstory=(
        'A seasoned engineer specializing in performant HTML5/JS games (e.g., Phaser, PixiJS). Expert in SDK integration, '
        'clean code, scalability, and secure crypto interactions, especially regarding Mythborne Companions and Pi Network. ' # Updated Name
        'Plans tech stack and generates core system structures based on GDD specs.'
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True
)

# 4. Lead Artist Agent
lead_artist = Agent(
    role='Lead Game Artist (Mobile Web)',
    goal='Define visual style, create concept art, establish art pipeline, ensure cohesive and optimized visuals for Mythborne Companions on Pi Browser.', # Updated Name
    backstory=(
        'A versatile mobile game artist lead skilled in character/environment/UI design and animation. Creates appealing, '
        'performant web visuals suitable for the Pi Browser, fitting the "Mythborne Companions" theme. ' # Updated Name
        'Can adapt to incorporating Pi branding subtly. Specs assets and collaborates with UI/UX.'
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True
)

# 5. UI/UX Designer Agent
ui_ux_designer = Agent(
    role='UI/UX Designer (Mobile Web)',
    goal='Design an intuitive, accessible, and visually appealing UI/UX flow for Mythborne Companions, optimized for Pi Browser.', # Updated Name
    backstory=(
        'A user-centric mobile web UI/UX specialist. Expertise in information architecture, interaction design, '
        'usability, and creating clean interfaces within web platform constraints, particularly for Mythborne Companions.' # Updated Name
        ' Collaborates closely with design and art.'
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True
)


# --- Define Initial Task(s) ---

# Task for the Project Manager to outline the next steps for the GDD
task_plan_gdd_refinement = Task(
    description=(
        'Review the project status for "Mythborne Companions" (Phase 1 Start). The goal is to refine the initial GDD draft. ' # Updated Name
        'Based on the defined roles for the Game Designer, Lead Programmer, Lead Artist, and UI/UX Designer, '
        'create a prioritized list of the top 3-5 immediate tasks or critical questions that need to be addressed '
        'to clarify the GDD requirements (e.g., core loop details, specific Pi integrations, art style direction, '
        'UI flow for key features). For each item, suggest the primary agent role responsible for leading the answer or task.'
    ),
    expected_output=(
        'A numbered list of 3-5 prioritized tasks or critical questions for GDD refinement for Mythborne Companions. ' # Updated Name
        'Each item should include a brief description and the suggested primary responsible agent role '
        '(e.g., "1. Define specific Pi token earning/spending actions - Game Designer", '
        '"2. Confirm technical feasibility of proposed mini-game X - Lead Programmer", etc.).'
    ),
    agent=project_manager # Assign this initial planning task to the Project Manager
)

# --- Create the Crew ---
mythborne_crew = Crew( # Updated Crew variable name
    agents=[project_manager, game_designer, lead_programmer, lead_artist, ui_ux_designer],
    tasks=[task_plan_gdd_refinement], # Start with just the PM's planning task
    process=Process.sequential,
    verbose=True
)

# --- Kick Off the Crew's Work ---
print("##############################################")
print("## Starting Mythborne Companions Crew Run (Phase 1)...") # Updated Print Statement
print("##############################################")
result = mythborne_crew.kickoff() # Updated Crew variable name

# --- Print the Final Result ---
print("\n\n##############################################")
print("## Crew Run Completed!")
print("##############################################")
print("\nFinal Output (from Project Manager):\n")
print(result)