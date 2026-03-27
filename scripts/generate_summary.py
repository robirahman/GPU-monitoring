"""Generate a markdown summary report from the paper database."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.storage.paper_store import PaperStore
from src.analysis.summarizer import generate_summary


def main():
    store = PaperStore()
    output_path = project_root / "data" / "summary.md"

    papers = store.get_all()
    if not papers:
        print("No papers in database. Run search first.")
        return

    report = generate_summary(store, output_path=output_path)

    print(f"Generated summary for {len(papers)} papers")
    print(f"Report saved to: {output_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
