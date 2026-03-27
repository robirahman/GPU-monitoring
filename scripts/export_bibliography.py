"""Export the paper database to BibTeX format."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.storage.paper_store import PaperStore
from src.storage.bibtex_export import export_to_bibtex


def main():
    store = PaperStore()
    output_path = project_root / "data" / "bibliography.bib"

    papers = store.get_all()
    if not papers:
        print("No papers in database. Run search first.")
        return

    bibtex = export_to_bibtex(store, output_path=output_path)

    print(f"Exported {len(papers)} papers to {output_path}")
    print(f"BibTeX file size: {len(bibtex)} characters")


if __name__ == "__main__":
    main()
