import logging
import json
import os
import asyncio
from datetime import datetime
from typing import Annotated, Literal
from dataclasses import dataclass, field

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
    metrics,
    MetricsCollectedEvent,
    RunContext,
    function_tool,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# ORDER MANAGEMENT SYSTEM
# ======================================================
@dataclass
class OrderState:
    """Coffee shop order state with validation"""
    drinkType: str | None = None
    size: str | None = None
    milk: str | None = None
    extras: list[str] = field(default_factory=list)
    name: str | None = None
    
    def is_complete(self) -> bool:
        """Check if all required fields are filled"""
        return all([
            self.drinkType is not None,
            self.size is not None,
            self.milk is not None,
            self.extras is not None,
            self.name is not None
        ])
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "drinkType": self.drinkType,
            "size": self.size,
            "milk": self.milk,
            "extras": self.extras,
            "name": self.name
        }
    
    def get_summary(self) -> str:
        """Get friendly order summary"""
        if not self.is_complete():
            return "Order in progress..."
        
        extras_text = f" with {', '.join(self.extras)}" if self.extras else ""
        return f"{self.size.upper()} {self.drinkType.title()} with {self.milk.title()} milk{extras_text} for {self.name}"

@dataclass
class Userdata:
    """User session data"""
    order: OrderState
    session_start: datetime = field(default_factory=datetime.now)

# ======================================================
# BARISTA AGENT FUNCTION TOOLS
# ======================================================

@function_tool
async def set_drink_type(
    ctx: RunContext[Userdata],
    drink: Annotated[
        Literal["latte", "cappuccino", "americano", "espresso", "mocha", "coffee", "cold brew", "matcha"],
        Field(description="The type of coffee drink the customer wants"),
    ],
) -> str:
    """Set the drink type. Call when customer specifies which coffee they want."""
    ctx.userdata.order.drinkType = drink
    return f"Excellent choice! One {drink} coming up!"

@function_tool
async def set_size(
    ctx: RunContext[Userdata],
    size: Annotated[
        Literal["small", "medium", "large", "extra large"],
        Field(description="The size of the drink"),
    ],
) -> str:
    """Set the size. Call when customer specifies drink size."""
    ctx.userdata.order.size = size
    return f"{size.title()} size - perfect for your {ctx.userdata.order.drinkType}!"

@function_tool
async def set_milk(
    ctx: RunContext[Userdata],
    milk: Annotated[
        Literal["whole", "skim", "almond", "oat", "soy", "coconut", "none"],
        Field(description="The type of milk for the drink"),
    ],
) -> str:
    """Set milk preference. Call when customer specifies milk type."""
    ctx.userdata.order.milk = milk
    
    if milk == "none":
        return "Got it! Black coffee - strong and simple!"
    return f"{milk.title()} milk - great choice!"

@function_tool
async def set_extras(
    ctx: RunContext[Userdata],
    extras: Annotated[
        list[Literal["sugar", "whipped cream", "caramel", "extra shot", "vanilla", "cinnamon", "honey"]] | None,
        Field(description="List of extras, or empty/None for no extras"),
    ] = None,
) -> str:
    """Set extras. Call when customer specifies add-ons or says no extras."""
    ctx.userdata.order.extras = extras if extras else []
    
    if ctx.userdata.order.extras:
        return f"Added {', '.join(ctx.userdata.order.extras)} - making it special!"
    return "No extras - keeping it classic and delicious!"

@function_tool
async def set_name(
    ctx: RunContext[Userdata],
    name: Annotated[str, Field(description="Customer's name for the order")],
) -> str:
    """Set customer name. Call when customer provides their name."""
    ctx.userdata.order.name = name.strip().title()
    return f"Wonderful, {ctx.userdata.order.name}! Almost ready to complete your order!"

@function_tool
async def complete_order(ctx: RunContext[Userdata]) -> str:
    """Finalize and save order to JSON. ONLY call when ALL fields are filled."""
    order = ctx.userdata.order
    
    if not order.is_complete():
        missing = []
        if not order.drinkType: missing.append("drink type")
        if not order.size: missing.append("size")
        if not order.milk: missing.append("milk")
        if order.extras is None: missing.append("extras")
        if not order.name: missing.append("name")
        
        return f"Almost there! Just need: {', '.join(missing)}"
    
    try:
        save_order_to_json(order)
        extras_text = f" with {', '.join(order.extras)}" if order.extras else ""
        
        return f"""🎉 PERFECT! Your {order.size} {order.drinkType} with {order.milk} milk{extras_text} is confirmed, {order.name}! 
We're preparing your drink now - it'll be ready in 3-5 minutes!
Thanks for using our AI Barista!"""
        
    except Exception as e:
        return "Order recorded but there was a small issue. Don't worry, we'll make your drink right away!"

@function_tool
async def get_order_status(ctx: RunContext[Userdata]) -> str:
    """Get current order status. Call when customer asks about their order."""
    order = ctx.userdata.order
    if order.is_complete():
        return f"Your order is complete! {order.get_summary()}"
    
    progress = order.get_summary()
    return f"Order in progress: {progress}"

class BaristaAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
            You are a FRIENDLY and PROFESSIONAL barista at a modern café.
            
            MISSION: Take coffee orders by systematically collecting:
            Drink Type: latte, cappuccino, americano, espresso, mocha, coffee, cold brew, matcha
            Size: small, medium, large, extra large
            Milk: whole, skim, almond, oat, soy, coconut, none
            Extras: sugar, whipped cream, caramel, extra shot, vanilla, cinnamon, honey, or none
            Customer Name: for the order
            
            PROCESS:
            1. Greet warmly and ask for drink type
            2. Ask for size preference 
            3. Ask for milk choice
            4. Ask about extras
            5. Get customer name
            6. Confirm and complete order
            
            STYLE:
            - Be warm, enthusiastic, and professional
            - Use emojis to make it friendly
            - Ask one question at a time
            - Confirm choices as you go
            - Celebrate when order is complete
            
            Use the function tools to record each piece of information.
            """,
            tools=[
                set_drink_type,
                set_size,
                set_milk,
                set_extras,
                set_name,
                complete_order,
                get_order_status,
            ],
        )

def create_empty_order():
    """Create a fresh order state"""
    return OrderState()

# ======================================================
# ORDER STORAGE & PERSISTENCE
# ======================================================
def get_orders_folder():
    """Get the orders directory path"""
    base_dir = os.path.dirname(__file__)
    backend_dir = os.path.abspath(os.path.join(base_dir, ".."))
    folder = os.path.join(backend_dir, "orders")
    os.makedirs(folder, exist_ok=True)
    return folder

def save_order_to_json(order: OrderState) -> str:
    """Save order to JSON file"""
    folder = get_orders_folder()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"order_{timestamp}.json"
    path = os.path.join(folder, filename)

    try:
        order_data = order.to_dict()
        order_data["timestamp"] = datetime.now().isoformat()
        order_data["session_id"] = f"session_{timestamp}"
        
        with open(path, "w", encoding='utf-8') as f:
            json.dump(order_data, f, indent=4, ensure_ascii=False)
        
        return path
        
    except Exception as e:
        raise e

# ======================================================
# SYSTEM VALIDATION & TESTING
# ======================================================
def test_order_saving():
    """Test function to verify order saving works"""
    test_order = OrderState()
    test_order.drinkType = "latte"
    test_order.size = "medium"
    test_order.milk = "oat"
    test_order.extras = ["extra shot", "vanilla"]
    test_order.name = "TestCustomer"
    
    try:
        save_order_to_json(test_order)
        return True
    except Exception:
        return False

# ======================================================
# SYSTEM INITIALIZATION & PREWARMING
# ======================================================
def prewarm(proc: JobProcess):
    """Preload VAD model for better performance"""
    proc.userdata["vad"] = silero.VAD.load()

# ======================================================
# AGENT SESSION MANAGEMENT
# ======================================================
async def entrypoint(ctx: JobContext):
    """Main agent entrypoint - handles customer sessions"""
    ctx.log_context_fields = {"room": ctx.room.name}

    test_order_saving()

    userdata = Userdata(order=create_empty_order())
    
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )

    usage_collector = metrics.UsageCollector()
    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        usage_collector.collect(ev.metrics)

    await session.start(
        agent=BaristaAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

# ======================================================
# APPLICATION BOOTSTRAP & LAUNCH
# ======================================================
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
