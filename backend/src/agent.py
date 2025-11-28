import logging
import json
import os
from datetime import datetime
from typing import Annotated, Optional, List, Dict, Any
from dataclasses import dataclass, asdict, field

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

# 🔌 PLUGINS (Kept for agent environment setup)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# 💾 1. CATALOG & RECIPES SETUP
# ======================================================

CATALOG_FILE = "catalog.json"
ORDER_FOLDER = "orders"

# Simple Recipe Mapping for 'Intelligent Bundling'
# Format: Dish Name -> List of (Item Name, Quantity)
RECIPES = {
    "peanut butter sandwich": [
        ("Whole Wheat Bread", 1),
        ("Peanut Butter", 1),
    ],
    "pasta for two": [
        ("Spaghetti Pasta", 1),
        ("Tomato Sauce", 1),
    ],
    "basic breakfast": [
        ("Eggs (dozen)", 1),
        ("Milk (gallon)", 1),
        ("Bacon (pack)", 1),
    ],
}


@dataclass
class CatalogItem:
    """Schema for an item in the catalog."""
    name: str
    category: str
    price: float
    units: str
    tags: List[str] = field(default_factory=list)

def load_catalog() -> Dict[str, CatalogItem]:
    """Loads the catalog from JSON and indexes by name."""
    path = os.path.join(os.path.dirname(__file__), CATALOG_FILE)
    if not os.path.exists(path):
        # Create a sample catalog if it doesn't exist
        sample_catalog = [
            {"name": "Whole Wheat Bread", "category": "Groceries", "price": 4.50, "units": "loaf", "tags": ["vegan"]},
            {"name": "Eggs (dozen)", "category": "Groceries", "price": 5.25, "units": "dozen", "tags": ["protein"]},
            {"name": "Milk (gallon)", "category": "Groceries", "price": 3.00, "units": "gallon", "tags": ["dairy"]},
            {"name": "Peanut Butter", "category": "Groceries", "price": 6.80, "units": "jar", "tags": ["protein"]},
            {"name": "Spaghetti Pasta", "category": "Groceries", "price": 1.50, "units": "pack", "tags": ["carb"]},
            {"name": "Tomato Sauce", "category": "Groceries", "price": 2.20, "units": "jar", "tags": ["sauce"]},
            {"name": "Cheese Pizza (large)", "category": "Prepared Food", "price": 15.00, "units": "pizza", "tags": ["ready-to-eat"]},
            {"name": "Bag of Chips (large)", "category": "Snacks", "price": 4.00, "units": "bag", "tags": ["salt"]},
            {"name": "Bacon (pack)", "category": "Groceries", "price": 8.00, "units": "pack", "tags": ["meat"]},
            {"name": "Cereal (box)", "category": "Groceries", "price": 5.50, "units": "box", "tags": ["breakfast"]},
        ]
        with open(path, "w", encoding='utf-8') as f:
            json.dump(sample_catalog, f, indent=4)
        print(f"✅ Catalog seeded at {CATALOG_FILE}")

    with open(path, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    # Index by lowercased name for easy lookup
    catalog = {item["name"].lower(): CatalogItem(**item) for item in data}
    return catalog

# Initialize Catalog on load
CATALOG = load_catalog()

# ======================================================
# 🧠 2. STATE MANAGEMENT (Cart)
# ======================================================

@dataclass
class CartItem:
    """An item currently in the user's cart."""
    name: str
    quantity: int
    price: float  # Price per unit
    notes: str = ""

@dataclass
class OrderingState:
    """Holds the current state of the ordering session."""
    customer_name: Optional[str] = None
    customer_address: Optional[str] = None
    cart: List[CartItem] = field(default_factory=list)

    def get_cart_summary(self) -> str:
        """Returns a formatted summary of the cart."""
        if not self.cart:
            return "Your cart is currently empty."
        
        summary = ["Current Cart:"]
        total = 0.0
        for item in self.cart:
            line_total = item.quantity * item.price
            total += line_total
            summary.append(f"- {item.quantity} x {item.name} (${item.price:.2f} each) -> ${line_total:.2f}")
        
        summary.append(f"TOTAL: ${total:.2f}")
        return "\n".join(summary)
    
    def calculate_total(self) -> float:
        """Calculates the current total price of the cart."""
        return sum(item.quantity * item.price for item in self.cart)

# ======================================================
# 🛠️ 3. ORDERING AGENT TOOLS
# ======================================================

@function_tool
async def add_to_cart(
    ctx: RunContext[OrderingState],
    item_or_recipe_name: Annotated[str, Field(description="The exact name of the product (e.g., 'Milk') or the recipe (e.g., 'ingredients for pasta for two').")],
    quantity: Annotated[int, Field(description="The number of units to add (e.g., 2, 3, 1). Defaults to 1 if not specified.")] = 1,
    notes: Annotated[Optional[str], Field(description="Any specific customer requests, e.g., 'whole wheat', 'almond milk'.")] = None,
) -> str:
    """
    🛒 Adds a specific item OR all ingredients for a simple recipe to the cart. 
    Use this for all requests: single items like 'bread' or recipes like 'ingredients for a sandwich'.
    """
    state = ctx.userdata
    
    # 1. Check for Recipe Match (Intelligent Bundling)
    recipe_key = item_or_recipe_name.lower().replace("ingredients for ", "").strip()
    if recipe_key in RECIPES:
        items_added = []
        for item_name, default_qty in RECIPES[recipe_key]:
            # Scale quantity based on user request (e.g., "pasta for two people" might scale ingredients)
            final_qty = default_qty * quantity 

            if item_name.lower() in CATALOG:
                cat_item = CATALOG[item_name.lower()]
                
                # Check if item is already in the cart
                existing_item = next((i for i in state.cart if i.name.lower() == item_name.lower()), None)

                if existing_item:
                    existing_item.quantity += final_qty
                else:
                    new_item = CartItem(
                        name=cat_item.name,
                        quantity=final_qty,
                        price=cat_item.price,
                        notes=notes if notes else "",
                    )
                    state.cart.append(new_item)
                items_added.append(f"{final_qty} x {cat_item.name}")
        
        return f"SUCCESS: Added ingredients for '{recipe_key}' to the cart: {', '.join(items_added)}."

    # 2. Handle Single Item
    item_key = item_or_recipe_name.lower()
    if item_key not in CATALOG:
        # Try a fuzzy search on the catalog
        match = next((i for k, i in CATALOG.items() if item_key in k or item_key in " ".join(i.tags).lower()), None)
        if match:
            item_key = match.name.lower()
        else:
            return f"ERROR: I could not find '{item_or_recipe_name}' in the catalog. Please try a different item or check the spelling."

    cat_item = CATALOG[item_key]
    
    # Check if item is already in the cart
    existing_item = next((i for i in state.cart if i.name.lower() == item_key), None)

    if existing_item:
        existing_item.quantity += quantity
        return f"SUCCESS: Increased quantity of {cat_item.name} to {existing_item.quantity}."
    else:
        new_item = CartItem(
            name=cat_item.name,
            quantity=quantity,
            price=cat_item.price,
            notes=notes if notes else "",
        )
        state.cart.append(new_item)
        return f"SUCCESS: Added {quantity} x {cat_item.name} to the cart."

@function_tool
async def list_cart_contents(ctx: RunContext[OrderingState]) -> str:
    """
    📝 Displays all items and their quantities currently in the cart.
    Call this when the user asks, 'What is in my cart?'
    """
    return ctx.userdata.get_cart_summary()

@function_tool
async def place_order(
    ctx: RunContext[OrderingState],
    customer_name: Annotated[str, Field(description="The customer's name for the order.")],
    customer_address: Annotated[str, Field(description="The customer's address for delivery.")]
) -> str:
    """
    💾 Finalizes the order, calculates the total, and saves the order to a JSON file. 
    Call this when the user says they are done, e.g., 'That's all' or 'Place my order.'
    """
    state = ctx.userdata
    if not state.cart:
        return "ERROR: The cart is empty. Please add items before placing an order."

    # 1. Prepare Order Data
    order_id = datetime.now().strftime("%Y%m%d%H%M%S")
    order_total = state.calculate_total()
    
    order_items = [
        {
            "item_name": item.name,
            "quantity": item.quantity,
            "unit_price": item.price,
            "line_total": round(item.quantity * item.price, 2),
            "notes": item.notes,
        }
        for item in state.cart
    ]

    order_object = {
        "order_id": order_id,
        "timestamp": datetime.now().isoformat(),
        "customer_info": {
            "name": customer_name,
            "address": customer_address,
        },
        "items": order_items,
        "order_total": round(order_total, 2)
    }

    # 2. Save to JSON file
    os.makedirs(ORDER_FOLDER, exist_ok=True)
    filename = os.path.join(ORDER_FOLDER, f"order_{order_id}.json")
    
    try:
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(order_object, f, indent=4)
        
        print(f"✅ ORDER PLACED: Saved to {filename}")

        # 3. Clear cart after successful placement
        state.cart = [] 
        state.customer_name = customer_name
        state.customer_address = customer_address
        
        return (f"SUCCESS: Your order (ID: {order_id}) has been placed. The total is ${order_total:.2f}. "
                f"I've saved the details to a JSON file. Thank you for shopping with us! Goodbye.")
    
    except Exception as e:
        return f"ERROR: Failed to save the order file. Details: {str(e)}"

# ======================================================
# 🤖 4. AGENT DEFINITION
# ======================================================

class OrderingAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=f"""
            You are 'Nick', the friendly Food & Grocery Ordering Assistant for 'Daily Pantry'.
            Your primary goal is to efficiently take the user's order and finalize it.

            🛒 **ORDERING PROTOCOL (FOLLOW STRICTLY):**
            
            1. **GREETING & INSTRUCTION:**
                - Greet the user warmly.
                - State clearly: "I can help you order groceries, snacks, and prepared meals. You can tell me individual items or even things like 'I need ingredients for a peanut butter sandwich'."

            2. **ADDING ITEMS:**
                - Use the `add_to_cart` tool for every item requested, including quantity and any notes (like 'gluten-free').
                - **CRITICAL:** If the user asks for "ingredients for X", use the full phrase as the `item_or_recipe_name` in the `add_to_cart` tool.
                - After a tool call returns SUCCESS, verbally confirm the item(s) added and the current item count.

            3. **CART MANAGEMENT:**
                - When the user asks "What's in my cart?", use the `list_cart_contents` tool.
                - The LLM's response should paraphrase the summary returned by the tool.

            4. **FINALIZATION & CHECKOUT:**
                - When the user says they are done (e.g., "That's all," "Checkout," "Place my order"):
                    a. Politely ask for their **Name** and **Delivery Address**.
                    b. Once you have both pieces of information, use the `place_order` tool with the name and address.
                - The final verbal message must be the confirmation returned by the `place_order` tool.
            
            **Available Catalog Categories:** Groceries, Snacks, Prepared Food.
            **TONE:** Friendly, efficient, and helpful. Always confirm changes verbally.
            """,
            tools=[add_to_cart, list_cart_contents, place_order],
        )

# ======================================================
# 🎬 ENTRYPOINT (Remains mostly the same, only class name changes)
# ======================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    print("\n" + "📦" * 25)
    print("🚀 STARTING GROCERY ORDERING SESSION")
    
    # 1. Initialize State
    userdata = OrderingState()

    # 2. Setup Agent
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"), 
        tts=murf.TTS(
            voice="en-US-marcus", 
            style="Conversational",        
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )
    
    # 3. Start
    await session.start(
        agent=OrderingAgent(), # <-- Changed from FraudAgent
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))