# **Multichain AI Agent Library with Voice Support 🤖🎙️**

## ✨ **Features**

* 🛠️ **Full-featured Discord, Twitter, and Telegram connectors** for seamless social media integration
* 🔗 **Chat with Several ChatGPT Models** (Llama, Grok, OpenAI, Anthropic, Gemini, etc.)
* 👥 **Multi-agent and room support** to create dynamic environments for collaboration
* 💾 **Retrievable memory and document store** for consistent interaction history
* 💹 **Trade Tokens and Make Rewards** on Bitcoin, Solana, Ethereum & SUI Networks
* 🎙️ **Voice-enabled Interaction**: Engage with AI agents using voice commands on supported platforms.

## 🎯 **Use Cases**

* 🤖 **Voice-activated Chatbots** for customer service, personal assistants, and more
* 📈 **Business Process Automation** for task management and workflow handling
* 🧠 **Crypto Trading** with AI-driven strategies across Bitcoin, Solana, Ethereum, and SUI networks
* 🎤 **Voice-enabled Crypto Interactions**: Execute trades, check balances, and more using voice commands

## 🚀 **Quick Start**

### **Prerequisites**

Make sure you have the following installed:

* [Python 2.7+](https://www.python.org/downloads/)
* [Node.js 23+](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)
* [pnpm](https://pnpm.io/installation)
* [Web Speech API or Third-Party Voice Libraries](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API)

> **Note for Windows Users:** [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install-manual) is required for certain functionalities.

### **Use the Starter (Recommended)**

Clone the repository and get started quickly:

```bash
git clone https://github.com/toptrendev0829/Multchain-AI-Agent-Library.git
cd Crypto-AI-Agent
cp .env.example .env
pnpm i && pnpm build && pnpm start
```

### **Manually Start AI Agent (For Advanced Users)**

1. **Checkout the Latest Release**

```bash
git clone https://github.com/toptrendev0829/Multchain-AI-Agent-Library.git
```

2. **Edit the .env file**

Copy `.env.example` to `.env` and fill in the appropriate values.

```bash
cp .env.example .env
```

3. **Start Crypto AI**

```bash
pnpm i
pnpm build
pnpm start
```

If you encounter issues, you can clean and rebuild the project:

```bash
pnpm clean
pnpm build
pnpm start
```

### **Interact via Browser or Voice**

Once the project is running, you can interact with it either through a browser or with voice commands:

```bash
pnpm start:client
```

Open the browser at the given URL to chat with your agent, or use your voice-enabled device for hands-free interaction.

---

### **Automatically Start AI Agent**

You can also set up and run the AI Agent automatically using the provided start script:

```bash
sh scripts/start.sh
```

### **Modify Character**

1. **Load Custom Characters**

   To load a custom character, use:

   ```bash
   pnpm start --characters="path/to/your/character.json"
   ```

2. **Voice Support Integration**

   To enable voice interaction, ensure your character JSON includes voice-enabled commands:

   ```json
   "voice": true,
   "clients": ["twitter", "discord", "telegram"]
   ```

---

### **Crypto Trading Setup**

#### Supported Networks:

* **SUI**
* **Solana**
* **Ethereum (EVM)**
* **Bitcoin**

#### Features:

* **AI-driven Trading Strategies**: Automatically trade and earn rewards across multiple blockchain networks.
* **Secure Wallet Integration**: Connect your wallets to perform transactions directly via the AI agent.
* **Voice-enabled Crypto Interactions**: Perform crypto trading and wallet transactions with simple voice commands.

> **Note**: Ensure that you have set up the appropriate API keys for trading and connected your wallet before starting.

---

#### Additional Requirements

If you encounter errors during setup, install the necessary dependencies such as **Sharp**:

```bash
pnpm install --include=optional sharp
```

Additionally, integrate voice support by configuring your environment with the Web Speech API or your preferred voice library to handle speech recognition and responses.
