from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

### TOOLs ###

@tool
def get_current_time():
    """Get the current time"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate(first_number:float, second_number:float, operator:str):
    """Perform a mathematical operation
    supports: add, subtract, multiply, divide"""
    try:    
        if operator == "add":
            return first_number + second_number
        elif operator == "subtract":
            return first_number - second_number
        elif operator == "multiply":
            return first_number * second_number
        elif operator == "divide":
            return first_number / second_number
        else:
            return "Invalid operator"
    except ZeroDivisionError:
        return "Cannot divide by zero"
    except Exception as e:
        return f"Error: {str(e)}"


tools = [get_current_time, calculate]


