# 🤖🎙️ Multichain AI Agent Library with Voice Support

A powerful, extensible AI agent framework designed for **multichain crypto interactions**, **voice-enabled interfaces**, and **cross-platform communication**. Seamlessly connect to **Twitter, Discord, Telegram**, and leverage LLMs like **OpenAI, Gemini, LLaMA, Grok**, and more.

---

## ✨ Features

* 🛠️ **Full-featured Connectors** – Easily integrate with Discord, Twitter, and Telegram
* 🔗 **LLM Flexibility** – Supports OpenAI, Gemini, LLaMA, Anthropic, Grok, and more
* 👥 **Multi-Agent & Room Support** – Build collaborative AI environments with distinct personalities
* 💾 **Memory & Document Store** – Persistent context and retrievable interaction history
* 💹 **Multichain Crypto Trading** – Execute trades and earn rewards on Bitcoin, Solana, Ethereum, and SUI
* 🎙️ **Voice Command Support** – Interact hands-free using browser-based or external voice tools

---

## 🎯 Use Cases

* 🤖 **Voice-Activated AI Agents** – Personal assistants, customer support bots, or crypto traders
* 📈 **Business Process Automation** – Handle workflows, reminders, and task routing via AI agents
* 🧠 **Crypto Automation** – AI-driven portfolio strategies on Solana, Ethereum, SUI, and Bitcoin
* 🎤 **Voice-Controlled Trading** – Perform actions like sending tokens or checking balances with speech

---

## 🚀 Quick Start

### 📦 Prerequisites

* [Node.js 23+](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)
* [Python 2.7+](https://www.python.org/downloads/)
* [pnpm](https://pnpm.io/installation)
* Voice support via [Web Speech API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API) or other compatible libraries

> 🪟 **Windows Users**: Please install [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install-manual) for full functionality.

---

### 🔧 Installation (Recommended)

```bash
git clone https://github.com/rustyneuron01/Multichain-AI-Agent-Library.git
cd Multichain-AI-Agent-Library
cp .env.example .env
pnpm install
pnpm build
pnpm start
```

---

### ⚙️ Advanced Manual Setup

```bash
git clone https://github.com/rustyneuron01/Multichain-AI-Agent-Library.git
cd Multichain-AI-Agent-Library
cp .env.example .env  # Configure API keys, tokens, and agent settings
pnpm install
pnpm build
pnpm start
```

> Optional clean build:

```bash
pnpm clean
pnpm build
pnpm start
```

---

## 🧠 Agent Interaction

### Via Browser or Voice

```bash
pnpm start:client
```

* Open your browser and navigate to the provided local URL.
* Interact through chat or supported voice commands.

---

### 📜 Auto Start Script

For simplified deployment:

```bash
sh scripts/start.sh
```

---

## 🧙 Custom Agent Configuration

### 1. Load Custom Characters

```bash
pnpm start --characters="path/to/your/character.json"
```

### 2. Enable Voice Support in Character File

```json
{
  "voice": true,
  "clients": ["twitter", "discord", "telegram"]
}
```

---

## 💹 Crypto Trading Integration

### 🔗 Supported Chains

* **Solana**
* **Ethereum (EVM)**
* **SUI**
* **Bitcoin**

### ⚙️ Trading Features

* AI-powered trade execution
* Portfolio monitoring and real-time price feedback
* Voice-enabled token transfers and trading commands

> Ensure your `.env` contains all necessary **API keys** and **wallet credentials**.

---

### 🧰 Additional Setup Notes

Some modules (e.g., image processing or voice support) may require native dependencies:

```bash
pnpm install --include=optional sharp
```

For full voice support, configure your environment using the **Web Speech API**, or install preferred voice recognition libraries.

---

## 🖼️ Sample UI

You can preview the agent interface via browser. Add assets like:


![A streaming conversation between the user and an AI agent](/public/image/sample.png)

---

## 📫 Contact

Maintained by [**RustyNeuron**](https://github.com/rustyneuron01)
Twitter: [@rustyneuron\_01](https://x.com/rustyneuron_01)
