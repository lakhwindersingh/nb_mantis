""" Batch processing demo """
import sys
import argparse
from pathlib import Path
# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.pipeline.batch_pipeline import BatchPipeline
def main():
    parser = argparse.ArgumentParser(description="Batch processing demo")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("--output", help="Output video file")
    args = parser.parse_args()

    print(f"🎬 Processing video: {args.input}")

    try:
        pipeline = BatchPipeline()
        pipeline.process_video(args.input, args.output)
        print("✅ Batch processing completed!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()