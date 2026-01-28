"""Allow running as: python -m evaluation"""
import asyncio

from evaluation import main

if __name__ == "__main__":
    asyncio.run(main())
