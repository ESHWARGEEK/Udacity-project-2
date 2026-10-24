"""
Customer Support AI Agent — Starter Code
==========================================
Your task is to complete this file by implementing all sections marked
with # TODO comments.

Reference the step-by-step solution files and INSTRUCTIONS.md for guidance.
Do NOT copy the solution directly — work through each section yourself.

Run locally:
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-NEW-781", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy
"""

# ── Imports ───────────────────────────────────────────────────────────────────
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
import logging
import uuid
from typing import Dict, List
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CSAI_Agent")

# ── TODO 1 — App Initialisation ───────────────────────────────────────────────
app = BedrockAgentCoreApp()
os.environ["BYPASS_TOOL_CONSENT"] = "true"

# ── TODO 2 — Configuration ────────────────────────────────────────────────────
GATEWAY_URL = "https://customersupportgateway-uywtgjtdn0.gateway.bedrock-agentcore.us-west-1.amazonaws.com/mcp"
KB_ID       = "8WXRUI76CW"
REGION      = "us-east-1"  # Note: Gateway is in us-west-1. Ensure memory/KB are actually in us-east-1.
MEMORY_ID   = "CustomerSupportMemory-jC8zw0Ho7z"

SYSTEM_PROMPT = """
You are a helpful and polite customer support AI agent. 
Use your available tools to assist the customer with their queries, 
including tracking orders, calculating loyalty discounts, 
and searching the knowledge base. Always verify information from the system.

If retrieved customer memory or context is provided, treat it as trusted conversation history. 
Use it naturally in your answers and NEVER claim you cannot remember previous preferences or lack prior context when this information is available.
Keep responses concise and customer-support oriented.
"""

# ── TODO 3 — Model and Clients ────────────────────────────────────────────────
model_id = "global.amazon.nova-2-lite-v1:0"
model = BedrockModel(model_id=model_id)
memory_client = MemoryClient(region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


# ── TODO 4 — Memory Helpers ─────────────────────────────────
def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict[str, List[str]]:
    """Capture ALL namespaces per strategy, not just the first one."""
    strategies = mem_client.get_memory_strategies(memory_id)
    namespaces = {}
    for strategy in strategies:
        if strategy.get("namespaces"):
            namespaces[strategy["type"]] = strategy["namespaces"]
    return namespaces

def load_memory_for_customer(mem_client: MemoryClient, memory_id: str, customer_id: str, user_prompt: str) -> List[str]:
    """Retrieve facts and preferences for the stable customer identity across all namespaces."""
    namespaces_dict = get_namespaces(mem_client, memory_id)
    collected_memories = []
    unique_texts = set()

    for strategy_type, ns_list in namespaces_dict.items():
        for ns_template in ns_list:
            formatted_namespace = ns_template.format(actorId=customer_id)
            try:
                memories = mem_client.retrieve_memories(
                    memory_id=memory_id,
                    namespace=formatted_namespace,
                    query=user_prompt,
                    top_k=5,
                )
                
                iterable_memories = memories if isinstance(memories, list) else []
                if isinstance(memories, dict) and "memories" in memories:
                    iterable_memories = memories["memories"]

                for mem in iterable_memories:
                    text = mem.get("text", "").strip() if isinstance(mem, dict) else str(mem).strip()
                    if text and text not in unique_texts:
                        unique_texts.add(text)
                        collected_memories.append(f"- [{strategy_type}] {text}")
            except Exception as e:
                logger.warning(f"Failed to retrieve memory for namespace {formatted_namespace}: {e}")
                
    return collected_memories

def format_memory_for_prompt(retrieved_memories: List[str]) -> str:
    """Convert retrieved memory into a short, model-friendly system note."""
    if not retrieved_memories:
        return ""
        
    return (
        "[SYSTEM NOTE: RETRIEVED CUSTOMER MEMORY]\n"
        "The following details are from previous interactions with this customer:\n"
        + "\n".join(retrieved_memories) + "\n\n"
        "Use this context when answering. Do not say you lack memory or prior context."
    )

def build_effective_prompt(user_prompt: str, memory_text: str) -> str:
    """Create the exact text the model should see by prepending memory context if it exists."""
    if memory_text:
        return f"{memory_text}\n\nCurrent customer request:\n{user_prompt}"
    return user_prompt

def make_strands_user_message(final_prompt_text: str) -> List[Dict]:
    """Wrap the final string in the structured block-list format Strands expects."""
    return [{"role": "user", "content": [{"text": final_prompt_text}]}]

def extract_assistant_text(response) -> str:
    """Robustly normalize agent output into one clean string, strictly ignoring non-text blocks to avoid memory pollution."""
    if not response:
        return ""
        
    content = getattr(response, "content", response)
    
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
                # Explicitly ignore tool_use / tool_result dicts without text
            elif hasattr(block, "text"):
                parts.append(str(getattr(block, "text")))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts).strip()
        
    return ""

def save_clean_interaction(mem_client: MemoryClient, memory_id: str, customer_id: str, session_id: str, user_prompt: str, assistant_text: str):
    """Persist the real conversation (clean user prompt + clean assistant response)."""
    if not assistant_text:
        logger.warning("Assistant text is empty or extraction failed. Skipping memory save.")
        return

    # Backup guard filter for stateless hallucinations
    bad_phrases = [
        "don't have access", "don't remember", "cannot remember", 
        "lack prior", "no memory", "persistent memory", "previous conversation",
        "unable to retain"
    ]
    if any(bp in assistant_text.lower() for bp in bad_phrases):
        logger.warning("Assistant hallucinated statelessness. Refusing to poison memory.")
        return

    logger.info(f"Saving to memory -> USER: '{user_prompt[:50]}...' | ASSISTANT: '{assistant_text[:50]}...'")
    try:
        mem_client.create_event(
            memory_id=memory_id,
            actor_id=customer_id,
            session_id=session_id,
            messages=[
                (user_prompt, "USER"),
                (assistant_text, "ASSISTANT"),
            ],
        )
    except Exception as e:
        logger.error(f"Failed to save memory event: {e}")


# ── TODO 6 — Knowledge Base Tool ─────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    if not KB_ID:
        return "Knowledge base not configured."

    try:
        response = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query}
        )

        results = response.get("retrievalResults", [])
        if not results:
            return "No relevant information found in the knowledge base."

        chunks = [
            result["content"]["text"]
            for result in results
            if "content" in result and "text" in result["content"]
        ]

        if not chunks:
            return "No relevant information found in the knowledge base."

        return "\n---\n".join(chunks)
    except Exception as e:
        logger.error(f"Knowledge Base retrieval failed: {e}")
        return f"Error accessing knowledge base: {e}"


# ── TODO 7 — Loyalty Discount Tool (Code Interpreter) ────────────────────────
@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    code = f"""
import json
import math

loyalty_points = {loyalty_points}
tier = "{tier}"
order_total = {order_total}
product_category = "{product_category}"

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

max_points_discount = order_total * 0.50
max_usable_points = int(max_points_discount * 100)

usable_points = min(loyalty_points, max_usable_points)
points_redeemed = (usable_points // 500) * 500
points_discount = points_redeemed / 100.0

subtotal_after_points = max(0.0, order_total - points_discount)
tier_rate = tier_rates.get(tier, 0.0)
tier_discount = round(subtotal_after_points * tier_rate, 2)

final_total = round(subtotal_after_points - tier_discount, 2)
total_savings = round(points_discount + tier_discount, 2)

earn_rate = earn_rates.get(product_category.lower(), 1)
points_earned = math.floor(final_total * earn_rate)
remaining_points = loyalty_points - points_redeemed + points_earned

result = {{
    "order_total": order_total,
    "points_redeemed": points_redeemed,
    "points_discount": points_discount,
    "tier": tier,
    "tier_discount": tier_discount,
    "final_total": final_total,
    "total_savings": total_savings,
    "points_earned": points_earned,
    "remaining_points": remaining_points
}}

print(json.dumps(result))
"""
    try:
        session = code_session(REGION)
        response = session.invoke(
            "executeCode",
            {
                "code": code,
                "language": "python",
                "clearContext": True,
            },
        )
        for event in response:
            if "result" in event:
                return event["result"]
            if "output" in event:
                return event["output"]
        return json.dumps({"raw_response": str(response)})

    except Exception as e:
        logger.error(f"Code interpreter execution failed: {e}")
        return json.dumps({
            "error": "Execution failed. Could not determine exact loyalty discount safely.",
            "details": str(e)
        })


# ── TODO 8 — Agent Entrypoint ─────────────────────────────────────────────────
@app.entrypoint
async def invoke(payload, context=None):
    try:
        # 1. Parse Inputs (Fail fast if customer_id is missing)
        user_input = payload.get("prompt", "")
        actor_id = payload.get("customer_id")
        session_id = payload.get("session_id") or str(uuid.uuid4())

        if not user_input:
            return "No prompt provided."
        if not actor_id:
            logger.error("customer_id is required but missing from payload.")
            return "Error: customer_id is required for personalized support."

        # 2. Retrieve Memory
        retrieved_memories = load_memory_for_customer(memory_client, MEMORY_ID, actor_id, user_input)
        
        # 3. Assemble Final Prompt
        memory_text = format_memory_for_prompt(retrieved_memories)
        final_prompt_text = build_effective_prompt(user_input, memory_text)
        
        user_message = make_strands_user_message(final_prompt_text)
        logger.info(f"Final assembled user_message for agent: {json.dumps(user_message)}")

        # 4. Setup Agent Tools
        agent_core_browser = AgentCoreBrowser(region=REGION)
        tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
            agent_core_browser.browser,
        ]

        # 5. Run the Agent (Graceful Gateway Fallback)
        agent_invoked = False
        if GATEWAY_URL:
            logger.info("Connecting to MCP Gateway to fetch tools...")
            try:
                client = MCPClient(lambda: streamable_http_client(url=GATEWAY_URL))
                with client:
                    gateway_tools = client.list_tools_sync()
                    tools.extend(gateway_tools)
                    logger.info(f"Successfully loaded {len(gateway_tools)} gateway tools.")

                    agent = Agent(
                        model=model,
                        tools=tools,
                        system_prompt=SYSTEM_PROMPT,
                    )
                    logger.info("Invoking agent with Gateway tools...")
                    response = agent(user_message)
                    agent_invoked = True
            except Exception as e:
                logger.error(f"Gateway connection or execution failed: {e}. Falling back to standard tools.")
        
        # Fallback path if gateway is disabled or failed
        if not agent_invoked:
            logger.info("Invoking agent without Gateway tools (fallback mode)...")
            agent = Agent(
                model=model,
                tools=tools,
                system_prompt=SYSTEM_PROMPT,
            )
            response = agent(user_message)

        # 6. Extract Clean Interaction Text
        assistant_text = extract_assistant_text(response)
        logger.info(f"Extracted assistant text: {assistant_text[:100]}...")

        # 7. Save Clean Interaction (We only save the original user_input)
        save_clean_interaction(memory_client, MEMORY_ID, actor_id, session_id, user_input, assistant_text)

        return assistant_text or str(response)

    except Exception as e:
        logger.error(f"Error executing agent invocation: {e}", exc_info=True)
        return json.dumps({
            "error": "Failed to process invocation",
            "details": str(e)
        })
        
# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()