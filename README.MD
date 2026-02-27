My Python Monorepo

    This repository contains multiple independent simulation projects that share a core set of utilities and modules located in the common/ folder.



Project Structure
.
├── common/                     # Shared modules used across all projects
├── simulation/
│   ├── EmergencyDepartment/    # Simulation for emergency department
│   └── PortTerminal/           # Simulation for port terminal
├── notebooks/                  # Shared Jupyter Notebooks for research
├── pyproject.toml              # The "Brain" that links everything together
└── .gitignore                  # Keeps the repo clean

Getting Started (Setup)

To ensure that the projects can see the common folder, you must install the repository in editable mode.

1. Create and Activate Virtual Environment
PowerShell:
    # Create the environment
    python -m venv .venv

# Activate (Windows PowerShell)
    .\.venv\Scripts\Activate.ps1

2. Install in Editable Mode

This step "links" the common and projects folders so they can be imported from anywhere.

PowerShell:
    pip install -e .



How to Use
Importing Shared Modules

In any script within this repo, you can import shared tools simply by using:
Python:

import common
# or
from common import shared_function



Running Notebooks

    1. Open the .ipynb file in VS Code.

    2. Select the Kernel: Ensure it is set to ('.venv': venv).

    3. If prompted, install ipykernel via the pop-up or run pip install ipykernel.



Project Details

Project,Description,Status
Common,"Shared constants, function, processes.",Core
Emergency Deparment, simulation script for capacity analysis,completed
Port terminal, simulation script for capacity analysis,Planned