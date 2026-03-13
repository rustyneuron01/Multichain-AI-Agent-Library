#!/bin/bash

PM2_PROCESS_NAME=$1
VENV_PATH=".venv"

# Ensure we're in the project root directory
cd "$(dirname "$0")"

# Function to activate virtual environment
activate_venv() {
    if [ -f "$VENV_PATH/bin/activate" ]; then
        source "$VENV_PATH/bin/activate"
    else
        echo "Virtual environment not found at $VENV_PATH"
        exit 1
    fi
}

# Activate virtual environment initially
activate_venv

while true; do
    sleep 300

    VERSION=$(git rev-parse HEAD)
    
    # Pull latest changes
    git pull --rebase --autostash

    NEW_VERSION=$(git rev-parse HEAD)

    if [ $VERSION != $NEW_VERSION ]; then
        echo "Code updated, reinstalling dependencies..."
        
        # Reactivate venv to ensure we're in the right environment
        activate_venv
        
        # Install dependencies using uv
        uv pip install -e ".[validator]"
        
        # Restart the PM2 process
        pm2 restart $PM2_PROCESS_NAME
        
        echo "Update completed"
    fi
done
