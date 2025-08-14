#!/usr/bin/env python3
"""Test script for the tool belt tools."""

import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from tools import check_ip, roll_dice, get_tool_belt

def test_roll_dice():
    """Test the dice rolling tool."""
    print("Testing dice rolling tool...")
    
    # Test basic dice roll
    result = roll_dice("1d6")
    print(f"1d6: {result}")
    
    # Test multiple dice
    result = roll_dice("2d20k1")
    print(f"2d20k1: {result}")
    
    # Test multiple rolls
    result = roll_dice("1d10", 3)
    print(f"1d10 (3 rolls): {result}")
    
    # Test invalid notation
    result = roll_dice("invalid")
    print(f"Invalid notation: {result}")
    
    print()

def test_check_ip():
    """Test the IP checking tool."""
    print("Testing IP checking tool...")
    
    # Test with a public IP (Google DNS)
    result = check_ip("8.8.8.8")
    print(f"8.8.8.8: {result}")
    
    # Test with localhost
    result = check_ip("127.0.0.1")
    print(f"127.0.0.1: {result}")
    
    print()

def test_tool_belt():
    """Test that the tool belt can be created."""
    print("Testing tool belt creation...")
    
    try:
        tools = get_tool_belt()
        print(f"Successfully created tool belt with {len(tools)} tools:")
        for i, tool in enumerate(tools):
            tool_name = getattr(tool, '__name__', str(type(tool).__name__))
            print(f"  {i+1}. {tool_name}")
    except Exception as e:
        print(f"Error creating tool belt: {e}")
    
    print()

if __name__ == "__main__":
    print("Testing LangGraph Platform Tools\n")
    
    test_roll_dice()
    test_check_ip()
    test_tool_belt()
    
    print("Tool testing complete!")
