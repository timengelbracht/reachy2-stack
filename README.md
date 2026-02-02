# Reachy2 Stack 🤖

A modular research and demo stack for the Reachy 2 robot, designed for reproducible development using Docker and VS Code devcontainers.

---

## Quick Start (Devcontainer – Recommended)

### 1. Clone the repository (with submodules)

git clone --recursive https://github.com/cvg-robotics/reachy2-stack.git  
cd reachy2-stack  

If you already cloned without `--recursive`:

git submodule update --init --recursive

----

### 2. Open in VS Code Devcontainer

code .

Then reopen the workspace in the devcontainer:
- Press `F1`
- Select **Dev Containers: Reopen in Container**

---

### 3. Install Python dependencies

Inside the devcontainer terminal:

pip install -e .

---

## Optional Dependency Groups

vision-models: vision / foundation models  
pip install -e .[vision-models]
---

## Third-Party Code

Third-party research code is included via git submodules in:

third_party/hloc

Do not install these via pip.

