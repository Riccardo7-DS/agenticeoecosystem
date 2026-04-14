import re
import unittest
from agenticeoecosystem.servers.EOdashboard.eodashboard_server import search_narratives


def parse_narratives(output_str):
    entries = output_str.split("\n- ")
    parsed = []

    for entry in entries[1:]:
        lines = entry.strip().split("\n")
        title = lines[0].strip()

        theme_match = re.search(r"Theme:\s*(.*)", entry)
        theme = theme_match.group(1).strip() if theme_match else ""

        parsed.append({
            "title": title,
            "theme": theme
        })

    return parsed


class TestNarratives(unittest.TestCase):

    def check_query(self, query):
        result = search_narratives(query)
        parsed = parse_narratives(result)

        try:
            self.assertGreater(len(parsed), 0, "No results returned")

            for item in parsed:
                title = item["title"].lower()
                theme = item["theme"].lower()
                q = query.lower()

                self.assertTrue(
                    (q in title) or (q in theme),
                    f"Query '{query}' not found in title or theme:\nTitle: {title}\nTheme: {theme}"
                )

            print(f"✅ Test passed for query: '{query}'")

        except AssertionError as e:
            print(f"❌ Test FAILED for query: '{query}'")
            print(e)
            raise  # re-raise so unittest still marks it as failed

    # ===== SEPARATE TESTS =====

    def test_atmosphere(self):
        self.check_query("atmosphere")

    def test_pollution(self):
        self.check_query("pollution")

    def test_economy(self):
        self.check_query("economy")

    def test_climate(self):
        self.check_query("climate")


if __name__ == "__main__":
    unittest.main(verbosity=2)