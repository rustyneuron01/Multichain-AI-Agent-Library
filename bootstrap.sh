#!/bin/bash

# install_prerequisites.sh
set -e  # Exit on any error

echo "Installing system prerequisites for Score Vision..."

# Function to run command with or without sudo based on user permissions
run_cmd() {
    if [ "$(id -u)" -eq 0 ]; then
        # If root, run without sudo
        $@
    else
        # If not root, run with sudo
        sudo $@
    fi
}

# Update package list
run_cmd apt-get update

# Set timezone to UTC
run_cmd ln -fs /usr/share/zoneinfo/UTC /etc/localtime
run_cmd apt-get install -y tzdata
run_cmd dpkg-reconfigure -f noninteractive tzdata

# Add deadsnakes PPA for Python 3.10
run_cmd apt-get install -y software-properties-common
run_cmd add-apt-repository -y ppa:deadsnakes/ppa
run_cmd apt-get update

# Install Python 3.10 and related tools
run_cmd apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    libncurses5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    nano

# Install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash

# Load nvm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Install latest LTS version of Node.js
nvm install --lts

# Install PM2
npm install -g pm2

# Verify installations
echo "Checking installed versions:"
node -v     # Node.js version
npm -v      # npm version
pm2 -v      # PM2 version

# Setup PM2 startup (without sudo if we're root)
if [ "$(id -u)" -eq 0 ]; then
    env PATH=$PATH:/root/.nvm/versions/node/$(node -v)/bin pm2 startup
else
    sudo env PATH=$PATH:/home/$USER/.nvm/versions/node/$(node -v)/bin pm2 startup
fi

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH for current session
export PATH="$HOME/.local/bin:$PATH"

# Add uv to PATH permanently (for both root and non-root users)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> $HOME/.bashrc
source $HOME/.local/bin/env

# Source the updated PATH
source $HOME/.bashrc

echo "----------------------------------------"
echo "Prerequisites installation complete!"
echo "PM2 is now installed. You can manage processes using 'pm2' command."
echo "You can now proceed with the installation steps from README.md"
echo "----------------------------------------"
