""" Real-time demo of Video Mimic AI """
import sys 
from pathlib import Path
# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.pipeline.realtime_pipeline import RealtimePipeline

def main():
    assets_path = Path(__file__).parent.parent / "assets"

    print("🎥 Video Mimic AI - Real-time Demo")
    print("Press SPACE to pause/resume, ESC to quit")
    try:
        pipeline = RealtimePipeline()
        pipeline.run()
    except KeyboardInterrupt:
        print("\n⏹️  Demo stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()