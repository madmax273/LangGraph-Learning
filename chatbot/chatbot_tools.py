from langchain_core.tools import tool
from dotenv import load_dotenv
import logging
from datetime import datetime

load_dotenv()
logger = logging.getLogger(__name__)

### TOOLs ###

@tool
async def get_current_time() -> str:
    """Get the current time"""
    try:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.error(f"Error getting current time: {e}")
        return "Unable to fetch current time."

@tool
async def calculate(first_number: float, second_number: float, operator: str) -> float | str:
    """Perform a mathematical operation.
    supports: add, subtract, multiply, divide"""
    try:    
        if operator == "add":
            return first_number + second_number
        elif operator == "subtract":
            return first_number - second_number
        elif operator == "multiply":
            return first_number * second_number
        elif operator == "divide":
            if second_number == 0:
                return "Cannot divide by zero"
            return first_number / second_number
        else:
            return f"Invalid operator '{operator}'. Supported operators: add, subtract, multiply, divide."
    except Exception as e:
        logger.error(f"Error in calculation tool: {e}")
        return f"Error: {str(e)}"

tools = [get_current_time, calculate]
