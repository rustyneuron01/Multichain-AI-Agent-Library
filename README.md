# **Trading Support AGENT 🤖**

## ✨ **Features**

-   🛠️ **Full-featured Discord, Twitter, and Telegram connectors** for social media integration
-   🔗 **Chat with Several ChatGPT Models** (Llama, Grok, OpenAI, Anthropic, Gemini, etc.)
-   👥 **Multi-agent and room support** to create dynamic environments
-   💾 **Retrievable memory and document store** for consistent interaction history
-   💹 **Trade Tokens and Make Rewards** on Bitcoin, Solana & Ethereum Networks

## 🎯 **Use Cases**

-   🤖 **Chatbots** for customer service, personal assistants, and more
-   📈 **Business Process Handling** including automation and task management
-   🧠 **Crypto Trading** with AI-driven strategies on Bitcoin, Solana, and Ethereum networks

## 🚀 **Quick Start**

### **Prerequisites**

Make sure you have the following installed:

-   [Python 2.7+](https://www.python.org/downloads/)
-   [Node.js 23+](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)
-   [pnpm](https://pnpm.io/installation)

> **Note for Windows Users:** [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install-manual) is required.

### **Use the Starter (Recommended)**

Clone the repository and get started quickly:

```bash
git clone https://github.com/FCBtc1116/Trading-Support-Agent.git
cd Crypto-AI-Agent
cp .env.example .env
pnpm i && pnpm build && pnpm start
```

### **Manually Start AI Agent (For Advanced Users)**

1. **Checkout the Latest Release**

```bash
git clone https://github.com/toptrendev0829/Trading-Support-Agent.git
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

### **Interact via Browser**

Once project is running, you can interact with it via a browser:

```bash
pnpm start:client
```

Open the browser at the given URL to chat with your agent.

---

### **Automatically Start AI Agent**

You can also set up and run AI Agent automatically using the provided start script:

```bash
sh scripts/start.sh
```

### **Modify Character**

1. **Load Custom Characters**

    To load a custom character, use:

    ```bash
    pnpm start --characters="path/to/your/character.json"
    ```

2. **Connect with X (Twitter)**

    To integrate with **X (formerly Twitter)**, modify the character JSON:

    ```json
    "clients": ["twitter"]
    ```

---

### **Crypto Trading Setup**

#### Supported Networks:

-   **SUI**
-   **Solana**
-   **Ethereum**

#### Features:

-   **AI-driven Trading Strategies**: Automatically trade and earn rewards on supported blockchain networks.
-   **Secure Wallet Integration**: Connect your wallets to perform transactions directly via the AI agent.

> **Note**: Ensure that you have set up the appropriate API keys for trading and connected your wallet before starting.

---

#### Additional Requirements

If you face errors during setup, install the necessary dependencies such as **Sharp**:

```bash
pnpm install --include=optional sharp
```
