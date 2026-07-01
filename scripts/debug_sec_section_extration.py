import asyncio

from app.services.tools.sec_filings_tool import get_latest_10k


async def main() -> None:
    for ticker in ["AAPL", "MSFT", "NVDA"]:
        filing = await get_latest_10k(ticker)

        print(f"\n=== {ticker} ===")
        if filing is None:
            print("No filing returned")
            continue

        print("error:", filing.error)
        print("success:", filing.success)
        print("sections:", list(filing.sections.keys()))

        for name, section in filing.sections.items():
            text = section.content
            print(f"\n[{name}]")
            print("chars:", len(text))
            print("words:", section.word_count)
            print("preview:", text[:300].replace("\n", " "))


if __name__ == "__main__":
    asyncio.run(main())
