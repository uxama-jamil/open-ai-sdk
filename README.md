## Project Overview

This is a Python project demonstrating various AI agent patterns using the `openai-agents` framework. The project showcases different agent architectures including simple agents, agent routing/handoffs, agents-as-tools, and various agent interaction patterns.

## Development Commands

### Running the Project

- Install dependencies: `uv sync` (uses uv for package management)
- Run simple agent: `uv run llm` (mapped to `testprj.simpleAgent:runAgent`)
- Run tool calling example: `uv run tool` (mapped to `testprj.toolCalling:testAsync`)

### Project Scripts (from pyproject.toml)

- `testprj = "testprj:main"` - Main entry point
- `llm = "testprj.simpleAgent:runAgent"` - Simple agent demo
- `tool = "testprj.toolCalling:testAsync"` - Async tool calling demo

### Running Individual Examples

Examples can be run directly with: `python -m src.testprj.examples.agent_patterns.{filename}`

- Agent routing: `python -m src.testprj.examples.agent_patterns.agent_routing`
- Agents as tools: `python -m src.testprj.examples.agent_patterns.agents_as_tools`

## Architecture Overview

### Core Framework

The project uses the `openai-agents` framework with these key components:

- `Agent`: Core agent class with instructions, model, tools, and handoffs
- `Runner`: Executes agents synchronously (`run_sync`) or asynchronously (`run`)
- `AsyncOpenAI`: Client for external LLM providers (Gemini, Ollama)
- `OpenAIChatCompletionsModel`: Model wrapper for OpenAI-compatible APIs

### Directory Structure

- `src/testprj/practices/`: Core implementation examples and patterns
- `src/testprj/examples/agent_patterns/`: Advanced agent pattern demonstrations
- `src/testprj/dynamic-tool-call/`: Dynamic tool calling examples
- `src/testprj/under_the_hood/`: Lower-level implementation details

### Key Agent Patterns

#### 1. Simple Agents (`practices/simpleAgent.py`)

Basic agent setup with external providers (Gemini, Ollama). Shows fundamental agent creation and execution.

#### 2. Tool Calling (`practices/toolCalling.py`)

Demonstrates function tools with `@function_tool` decorator. Includes weather API integration and math operations.

#### 3. Agent Handoffs (`practices/handOff.py`)

Shows agent-to-agent handoffs using the `handoffs` parameter. Tutor agents specialized by subject.

#### 4. Agents as Tools (`examples/agent_patterns/agents_as_tools.py`)

Uses `.as_tool()` method to convert agents into tools for orchestrator agents. Translation example with multiple language agents.

#### 5. Agent Routing (`examples/agent_patterns/agent_routing.py`)

Multi-language routing system with triage agent that hands off to specialized language agents.

### LLM Provider Configuration

The project supports multiple providers:

- **Gemini**: Uses `https://generativelanguage.googleapis.com/v1beta/openai/` endpoint
- **Ollama**: Local deployment at `http://localhost:11434/v1`
- **Configuration**: API keys loaded from `.env` file using `python-dotenv`

### Common Patterns

- All examples disable tracing with `set_tracing_disabled(True)`
- Agents use `handoff_description` for tool conversion context
- Custom hooks available via `AgentHooks` class for lifecycle events
- Streaming responses supported via `Runner.run_streamed()`

## Dependencies

- `openai-agents>=0.0.17`: Core agent framework
- `agentops>=0.4.16`: Agent operation tracking
- `chainlit>=2.5.5`: UI framework for chat interfaces
- `python-dotenv>=1.1.0`: Environment variable management

## Environment Variables

Create a `.env` file with:

```
GEMINI_API_KEY=your_gemini_key_here
```

## Testing and Development

- No specific test framework configured - examples are run directly
- AgentOps integration available for tracking agent behavior
- Chainlit support for building chat UIs around agents
