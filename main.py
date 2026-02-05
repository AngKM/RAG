import sys
from code.pipeline import run as run_agent

def main():
    print("Starting RAG agent...")
    print("Press Ctrl+C to stop.")
    try:
        run_agent()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
