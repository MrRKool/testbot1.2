#!/bin/bash

# Kleuren voor output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Functie om commando's te checken
command_exists() {
    command -v "$1" &> /dev/null
}

echo -e "${GREEN}Setting up trading bot environment...${NC}"

# 1. Check of Homebrew is geïnstalleerd
if ! command_exists brew; then
    echo -e "${GREEN}Homebrew wordt geïnstalleerd...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo -e "${GREEN}Homebrew installatie voltooid.${NC}"
fi

# 2. Installeer libomp (voor XGBoost)
if ! brew list libomp &>/dev/null; then
    echo -e "${GREEN}libomp wordt geïnstalleerd via Homebrew...${NC}"
    brew install libomp
else
    echo -e "${GREEN}libomp is al geïnstalleerd.${NC}"
fi

# 3. Check of Python 3 is geïnstalleerd
if ! command_exists python3; then
    echo -e "${RED}Python 3 is niet geïnstalleerd. Installeer Python 3 eerst.${NC}"
    exit 1
fi

# 4. Check of virtualenv is geïnstalleerd
if ! command_exists virtualenv; then
    echo -e "${GREEN}virtualenv wordt geïnstalleerd...${NC}"
    pip3 install virtualenv
fi

# 5. Maak virtual environment aan als die nog niet bestaat
if [ ! -d "venv" ]; then
    echo -e "${GREEN}Virtuele omgeving wordt aangemaakt...${NC}"
    virtualenv venv
fi

# 6. Activeer virtual environment
echo -e "${GREEN}Virtuele omgeving wordt geactiveerd...${NC}"
source venv/bin/activate

# 7. Upgrade pip
echo -e "${GREEN}pip wordt geüpdatet...${NC}"
pip install --upgrade pip

# 8. Installeer Python dependencies
echo -e "${GREEN}Python dependencies worden geïnstalleerd...${NC}"
pip install --force-reinstall -r requirements.txt

# 9. Controleer dependencies
echo -e "${GREEN}Dependencies worden gecontroleerd...${NC}"
python3 -c "from utils.dependency_checker import check_dependencies; check_dependencies()"

echo -e "${GREEN}Setup compleet! Je kunt nu de bot starten met:${NC}"
echo -e "${GREEN}source venv/bin/activate${NC}"
echo -e "${GREEN}python main.py${NC}" 