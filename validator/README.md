# Score Vision (SN44) - Validator

This is the validator component for Subnet 44 (Soccer Video Analysis). For full subnet documentation, please see the [main README](../README.md).

## System Requirements

Please see [REQUIREMENTS.md](REQUIREMENTS.md) for detailed system requirements.

## Setup Instructions

1. **Bootstrap System Dependencies**

```bash
# Clone repository
git clone https://github.com/score-technologies/score-vision.git
cd score-vision
chmod +x bootstrap.sh
./bootstrap.sh
```

2. Setup Bittensor Wallet:

```bash
# Create hotkey directory
mkdir -p ~/.bittensor/wallets/[walletname]/hotkeys/

# If copying from local machine:
scp ~/.bittensor/wallets/[walletname]/hotkeys/[hotkeyname] [user]@[SERVERIP]:~/.bittensor/wallets/[walletname]/hotkeys/[hotkeyname]
scp ~/.bittensor/wallets/[walletname]/coldkeypub.txt [user]@[SERVERIP]:~/.bittensor/wallets/[walletname]/coldkeypub.txt
```

## Installation

1. Create and activate virtual environment:

```bash
uv venv
source .venv/bin/activate  # On Unix-like systems

```

2. Install dependencies:

```bash
uv pip install -e ".[validator]"
```

3. Setup environment:

```bash
cp validator/.env.example validator/.env
# Edit .env with your configuration
```

## Running the Validator

### Production Deployment (PM2)

Start the auto-updater and validator:

```bash
# Start the auto-updater (handles code updates and dependency management)
pm2 start validator_auto_update.sh --name validator-updater --interpreter bash -- sn44-validator

# Start the validator
cd validator
pm2 start \
  --name "sn44-validator" \
  --interpreter "../.venv/bin/python" \
  "main.py"
```

### Development Mode

```bash
cd validator
python main.py
```

## Operational Overview

The validator operates several key processes to manage the subnet:

### 1. Challenge Management

- Generates and distributes video analysis challenges
- Validates miner stake and registration
- Manages challenge timeouts and retries
- Implements rate limiting and load balancing
- Tracks challenge distribution metrics

### 2. Response Evaluation

- Processes miner responses using GPT-4
- Validates detection accuracy and completeness
- Calculates performance scores
- Manages evaluation queues
- Implements scoring algorithms

### 3. Weight Setting

- Processes stored responses in batches
- Uses random frames from the challenges
- Uses GPT-4o to validate frame annotations:
  - Player detection accuracy
  - Goalkeeper identification
  - Referee detection
  - Ball tracking
- Calculates frame-level scores
- Stores evaluation results

### 4. Weight Setting Loop (Every 21 minutes)

- Aggregates recent evaluation scores
- Calculates miner performance metrics:
  - Evaluation accuracy (60%)
  - Availability (30%)
  - Response speed (10%)
- Updates miner weights on-chain
- Manages reward distribution

## Configuration Reference

Key environment variables in `.env`:

```bash
# Subnet Configuration (Testnet 261)
NETUID=261
SUBTENSOR_NETWORK=local
SUBTENSOR_ADDRESS=wss://test.finney.opentensor.ai:443
# PROD wss://entrypoint-finney.opentensor.ai:443
# TEST wss://test.finney.opentensor.ai:443

# Validator Configuration
HOTKEY_NAME=default
WALLET_NAME=validator
MIN_STAKE_THRESHOLD=2

# Port and Host
VALIDATOR_PORT=8000
VALIDATOR_HOST=0.0.0.0

# OpenAI Configuration
OPENAI_API_KEY=your_openai_key

# Logging
LOG_LEVEL=DEBUG
```

## Troubleshooting

### Common Issues

1. **OpenAI API Issues**

   - Verify API key is valid and has GPT-4o access
   - Check API rate limits and quotas
   - Monitor API response times

2. **Network Connectivity**

   - Ensure ports are properly exposed
   - Check firewall settings
   - Verify subtensor network connectivity

3. **Database Issues**
   - Check disk space
   - Verify write permissions
   - Monitor database growth

For advanced configuration options and architecture details, see the [main README](../README.md).
