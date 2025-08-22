
"""
Entry point for utils module commands
Allows running: python -m src.utils.test_motion --camera 0
"""
import sys
import argparse
from pathlib import Path

def main():
    """Main entry point for utils module commands"""
    
    if len(sys.argv) < 2:
        print("Available utils commands:")
        print("  test_motion    - Test motion detection capabilities")
        print("  benchmark      - Benchmark system performance")  
        print("  export_character - Export character animations")
        print("\nUsage: python -m src.utils.<command> [args...]")
        return 1
    
    command = sys.argv[1]
    
    # Remove the command from sys.argv so submodules can parse their args
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    try:
        if command == "test_motion":
            from .test_motion import main as test_motion_main
            return test_motion_main()
        elif command == "benchmark":
            from .benchmark import main as benchmark_main
            return benchmark_main()
        elif command == "export_character":
            from .export_character import main as export_main
            return export_main()
        else:
            print(f"Unknown command: {command}")
            return 1
            
    except ImportError as e:
        print(f"Failed to import command '{command}': {e}")
        return 1
    except Exception as e:
        print(f"Command '{command}' failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
