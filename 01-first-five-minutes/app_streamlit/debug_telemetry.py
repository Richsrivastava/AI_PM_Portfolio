#!/usr/bin/env python3
"""
Quick script to create some test telemetry data
Run this to create sample data for the dashboard
"""

import sys
import os

# Add current directory to path to import telemetry
sys.path.append(os.path.dirname(__file__))

try:
    from telemetry import log, ensure, get_stats
    
    print("📊 Creating test telemetry data...")
    
    # Ensure database exists
    ensure()
    
    # Create some test data
    test_data = [
        ("test-session-1", "A", 250, 15, 45, "direct_chat_planning"),
        ("test-session-2", "B", 180, 12, 38, "wizard_holiday_planning"),
        ("test-session-3", "C", 320, 20, 52, "sample_itinerary_planning"),
        ("test-session-4", "A", 200, 18, 41, "direct_chat_planning"),
        ("test-session-5", "B", 150, 10, 35, "wizard_holiday_planning"),
    ]
    
    for session_id, variant, latency, tokens_in, tokens_out, task in test_data:
        log(session_id, variant, latency, tokens_in, tokens_out, task)
        print(f"  ✅ Logged: {variant} - {task} - {latency}ms")
    
    # Show stats
    stats = get_stats()
    if stats:
        print(f"\n📈 Database Stats:")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Average latency: {stats['avg_latency']:.1f}ms")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Total cost: ${stats['total_cost']:.4f}")
    
    print(f"\n🎉 Test data created! Database location:")
    from telemetry import DB_PATH
    print(f"  {DB_PATH}")
    
except ImportError as e:
    print(f"❌ Could not import telemetry module: {e}")
    print("Make sure you're running this from the same directory as telemetry.py")
except Exception as e:
    print(f"❌ Error creating test data: {e}")