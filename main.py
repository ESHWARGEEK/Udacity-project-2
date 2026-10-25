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
from strands.hooks import HookProvider, HookRegistry, MessageAddedEvent, AfterInvocationEvent
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

# ── App Initialisation ───────────────────────────────────────────────
app = BedrockAgentCoreApp()
os.environ["BYPASS_TOOL_CONSENT"] = "true"

# ── Configuration ────────────────────────────────────────────────────
GATEWAY_URL = "https://customersupportgateway-uywtgjtdn0.gateway.bedrock-agentcore.us-west-1.amazonaws.com/mcp"
KB_ID       = "8WXRUI76CW"
REGION      = "us-east-1"
MEMORY_ID   = "CustomerSupportMemory-jC8zw0Ho7z"

SYSTEM_PROMPT = """
You are a helpful and polite customer support AI agent. 
Use your available tools to assist the customer with their queries, 
including tracking orders, calculating loyalty discounts, 
and searching the knowledge base. Always verify information from the system.

Before calling refund-processor___initiate_refund, look up the order with order-tracker___get_order and pass the order total as amount. Never report a $0 refund as approved.

If retrieved customer memory or context is provided, treat it as trusted conversation history. 
Use it naturally in your answers and NEVER claim you cannot remember previous preferences or lack prior context when this information is available.
Keep responses concise and customer-support oriented.
"""

# ── Model and Clients (Lazy Loaded) ──────────────────────────────────
model_id = "global.amazon.nova-2-lite-v1:0"
model = BedrockModel(model_id=model_id)

_memory_client = None
_bedrock_runtime = None

def get_memory_client():
    global _memory_client
    if _memory_client is None:
        _memory_client = MemoryClient(region_name=REGION)
    return _memory_client

def get_bedrock_runtime():
    global _bedrock_runtime
    if _bedrock_runtime is None:
        _bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)
    return _bedrock_runtime

# ── Memory Hooks ─────────────────────────────────
def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict[str, List[str]]:
    """Capture namespaces per strategy, favoring namespaceTemplates fallback."""
    try:
        strategies = mem_client.get_memory_strategies(memory_id)
        namespaces = {}
        for strategy in strategies:
            ns_list = strategy.get("namespaceTemplates") or strategy.get("namespaces")
            if ns_list:
                namespaces[strategy["type"]] = ns_list
        return namespaces
    except Exception as e:
        logger.warning(f"Could not fetch memory strategies: {e}")
        return {}

class MemoryHook(HookProvider):
    def __init__(self, actor_id: str, session_id: str, mem_client: MemoryClient, memory_id: str):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = mem_client
        self.memory_id = memory_id
        self.original_user_prompt = ""
        self.namespaces = get_namespaces(mem_client, memory_id)

    def register_hooks(self, registry: HookRegistry):
        try:
            registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
            registry.add_callback(AfterInvocationEvent, self.save_support_interaction)
        except Exception as e:
            logger.error(f"Failed to register hooks: {e}")

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve memories and prepend to system prompt when a user message is added."""
        try:
            message = getattr(event, 'message', {})
            if not isinstance(message, dict):
                return

            role = message.get("role")
            if role != "user":
                return
                
            content = message.get("content", [])
            
            # Extract plain text from the content block
            if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict) and "text" in content[0]:
                self.original_user_prompt = content[0]["text"]
            elif isinstance(content, str):
                self.original_user_prompt = content
            else:
                return

            collected_memories = []
            unique_texts = set()

            for strategy_type, ns_list in self.namespaces.items():
                for ns_template in ns_list:
                    formatted_namespace = ns_template.format(actorId=self.actor_id)
                    try:
                        memories = self.memory_client.retrieve_memories(
                            memory_id=self.memory_id,
                            namespace=formatted_namespace,
                            query=self.original_user_prompt,
                            top_k=5,
                        )
                        
                        iterable_memories = memories if isinstance(memories, list) else []
                        if isinstance(memories, dict) and "memories" in memories:
                            iterable_memories = memories["memories"]

                        for mem in iterable_memories:
                            if isinstance(mem, dict) and "content" in mem and "text" in mem["content"]:
                                text = mem["content"]["text"].strip()
                            else:
                                continue
                                
                            if text and text not in unique_texts:
                                unique_texts.add(text)
                                collected_memories.append(f"- [{strategy_type}] {text}")
                    except Exception as e:
                        logger.warning(f"Failed to retrieve memory for namespace {formatted_namespace}: {e}")

            if collected_memories:
                memory_text = (
                    "[SYSTEM NOTE: RETRIEVED CUSTOMER MEMORY]\n"
                    "The following details are from previous interactions with this customer:\n"
                    + "\n".join(collected_memories) + "\n\n"
                    "Use this context when answering. Do not say you lack memory or prior context.\n\n"
                )
                
                # Prepend retrieved memories back into the prompt
                if isinstance(content, list) and isinstance(content[0], dict) and "text" in content[0]:
                    content[0]["text"] = memory_text + "Current customer request:\n" + self.original_user_prompt
                elif isinstance(content, str):
                    message["content"] = memory_text + "Current customer request:\n" + self.original_user_prompt

        except Exception as e:
            logger.error(f"Error in retrieve_customer_context hook: {e}")

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the conversation to memory using the clean original user prompt."""
        try:
            # Safely navigate: event -> result -> message -> content
            result = getattr(event, 'result', None)
            if not result:
                return
                
            message = getattr(result, 'message', {})
            if not isinstance(message, dict):
                return
                
            content = message.get("content", [])
            assistant_text = ""
            
            if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict) and "text" in content[0]:
                assistant_text = content[0]["text"]
            elif isinstance(content, str):
                assistant_text = content

            if not assistant_text or not self.original_user_prompt:
                return

            bad_phrases = [
                "don't have access", "don't remember", "cannot remember", 
                "lack prior", "no memory", "persistent memory", "previous conversation"
            ]
            if any(bp in assistant_text.lower() for bp in bad_phrases):
                logger.info("Skipping memory save due to stateless hallucination.")
                return

            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[
                    (self.original_user_prompt, "USER"),
                    (assistant_text, "ASSISTANT"),
                ],
            )
            logger.info(f"Saved memory event for actor={self.actor_id} session={self.session_id}")
        except Exception as e:
            logger.error(f"Failed to save memory event in hook: {e}")


# ── Knowledge Base Tool ─────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    if not KB_ID:
        return "Knowledge base not configured."

    try:
        client = get_bedrock_runtime()
        response = client.retrieve(
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

        return "\n---\n".join(chunks) if chunks else "No relevant information found."
    except Exception as e:
        logger.error(f"Knowledge Base retrieval failed: {e}")
        return f"Error accessing knowledge base: {e}"


# ── Loyalty Discount Tool (Code Interpreter) ────────────────────────
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
tier = {json.dumps(tier)}
order_total = {order_total}
product_category = {json.dumps(product_category)}

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
    "tier_discount_pct": round(tier_rate * 100),
    "final_total": final_total,
    "total_savings": total_savings,
    "points_earned": points_earned,
    "remaining_points": remaining_points
}}

print(json.dumps(result))
"""
    try:
        with code_session(REGION) as ci:
            response = ci.invoke(
                "executeCode",
                {
                    "code": code,
                    "language": "python",
                    "clearContext": True,
                },
            )
            for event in response.get("stream", []):
                result = event.get("result", {})
                if result.get("isError"):
                    raise RuntimeError("Code Interpreter error")
                stdout = result.get("structuredContent", {}).get("stdout", "")
                if stdout:
                    return json.dumps(json.loads(stdout))
            return json.dumps({"error": "No output returned from code interpreter"})

    except Exception as e:
        logger.error(f"Code interpreter execution failed: {e}")
        tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
        tier_rate = tier_rates.get(tier, 0.0)
        tier_discount = round(order_total * tier_rate, 2)
        final_total = round(order_total - tier_discount, 2)
        
        fallback_result = {
            "order_total": order_total,
            "points_redeemed": 0,
            "points_discount": 0.0,
            "tier": tier,
            "tier_discount": tier_discount,
            "tier_discount_pct": round(tier_rate * 100),
            "final_total": final_total,
            "total_savings": tier_discount,
            "points_earned": 0, 
            "remaining_points": loyalty_points
        }
        return json.dumps(fallback_result)


# ── Agent Entrypoint ─────────────────────────────────────────────────
@app.entrypoint
async def invoke(payload, context=None):
    try:
        user_input = payload.get("prompt", "")
        actor_id = payload.get("customer_id")
        session_id = payload.get("session_id") or str(uuid.uuid4())

        if not user_input:
            return "No prompt provided."
        if not actor_id:
            return "Error: customer_id is required for personalized support."

        user_message = [{"role": "user", "content": [{"text": user_input}]}]

        # Inject MemoryHook configuration using lazy loaded memory client
        memory_hook = MemoryHook(actor_id, session_id, get_memory_client(), MEMORY_ID)
        
        agent_core_browser = AgentCoreBrowser(region=REGION)
        tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
            agent_core_browser.browser,
        ]

        agent_invoked = False
        response = None
        if GATEWAY_URL:
            try:
                client = MCPClient(lambda: streamable_http_client(url=GATEWAY_URL))
                with client:
                    gateway_tools = client.list_tools_sync()
                    tools.extend(gateway_tools)
                    
                    agent = Agent(
                        model=model,
                        tools=tools,
                        system_prompt=SYSTEM_PROMPT,
                        hooks=[memory_hook]
                    )
                    response = agent(user_message)
                    agent_invoked = True
            except Exception as e:
                logger.error(f"Gateway failed: {e}. Falling back to standard tools.")
        
        if not agent_invoked:
            agent = Agent(
                model=model,
                tools=tools,
                system_prompt=SYSTEM_PROMPT,
                hooks=[memory_hook]
            )
            response = agent(user_message)

        # Extract textual response
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
                elif hasattr(block, "text"):
                    parts.append(str(getattr(block, "text")))
                elif isinstance(block, str):
                    parts.append(block)
            return " ".join(parts).strip()

        return str(response)

    except Exception as e:
        return json.dumps({
            "error": "Failed to process invocation",
            "details": str(e)
        })
        
# ── CLI entry point ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()

    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)

if __name__ == "__main__":
    app.run()