import argparse
import sys
from pathlib import Path
import logging
# Add src to path
sys.path.append(str(Path(__file__).parent))
from src.pipeline.realtime_pipeline import RealtimePipeline
from src.pipeline.batch_pipeline import BatchPipeline
from src.utils.logging_config import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Video Mimic AI Agent")
    parser.add_argument("--mode", choices=["realtime", "batch"], default="realtime", help="Processing mode")
    parser.add_argument("--input", type=str, help="Input video file (for batch mode)")
    parser.add_argument("--output", type=str, help="Output video file")
    parser.add_argument("--config", type=str, help="Custom configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    try:
        if args.mode == "realtime":
            logger.info("Starting realtime processing...")
            pipeline = RealtimePipeline(config_path=args.config)
            pipeline.run()
        else:
            if not args.input:
                logger.error("Input file required for batch mode")
            return 1

        logger.info(f"Starting batch processing: {args.input}")
        pipeline = BatchPipeline(config_path=args.config)
        pipeline.process_video(args.input, args.output)

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return 1

    logger.info("Processing completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())