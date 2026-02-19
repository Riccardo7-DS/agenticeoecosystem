"""
Test script for PhIdeas MCP Server
Creates mock data and tests all tools without requiring the actual dataset
"""

import shutil
import sys
import tempfile
from pathlib import Path
import pandas as pd


# Mock the ollama module since we're testing structure, not actual LLM calls
class MockOllama:
    @staticmethod
    def generate(model, prompt, options):
        """Mock LLM responses for testing"""
        if "Extract only the names" in prompt:
            return {"response": "John Doe, Jane Smith"}
        elif "Summarize" in prompt:
            return {"response": "This is a test summary of the idea."}
        elif "commercial, non-profit" in prompt:
            return {"response": "commercial"}
        elif "in-kind, financial" in prompt:
            return {"response": "financial"}
        elif "funded/ not funded/ in-kind" in prompt:
            return {"response": "funded||0.85"}
        elif "Propose at most" in prompt:
            return {
                "response": '["Earth Observation", "Data Processing", "Climate Science", "Other"]'
            }
        elif "Assign up to" in prompt:
            return {
                "response": '{"0": {"Earth Observation": 0.7, "Climate Science": 0.3}}'
            }
        else:
            return {"response": "test response"}


sys.modules["ollama"] = MockOllama()

# Now import the server functions
from agenticeoecosystem.servers.phideas.phideas_server import (
    categorize_cofunding_type,
    categorize_commercial_status,
    categorize_funding_binary,
    categorize_ideas_weighted,
    categorize_ideas_with_imposed_categories,
    clean_expert_column,
    get_dataframe_info,
    list_available_columns,
    summarize_description_column,
)


def create_mock_dataframe():
    """Create a mock PhIdeas dataframe with realistic columns"""
    data = {
        "ID": [1, 2, 3],
        "Title": [
            "AI for Earth Observation",
            "Climate Data Processing",
            "Satellite Image Analysis",
        ],
        "Description": [
            "This is a comprehensive project about using artificial intelligence and machine learning to process earth observation data from satellites for climate monitoring purposes.",
            "A detailed initiative focused on processing large-scale climate data using advanced computational methods and distributed systems.",
            "An innovative approach to analyzing satellite imagery using deep learning and computer vision techniques.",
        ],
        "ESA experts": [
            "I would like to collaborate with Dr. John Smith from ESA and also Dr. Jane Doe if possible",
            "Please connect me with Prof. Alice Johnson",
            "No specific expert requested",
        ],
        "Other non-ESA experts": [
            "We want to work with Bob Wilson from MIT and Sarah Chen from Stanford",
            "",
            "External expert: Dr. Michael Brown",
        ],
        "Non-Staff - company/institution of origin": [
            "TechCorp Solutions Inc.",
            "University of Copenhagen",
            "SpaceData Analytics Ltd.",
        ],
        "Leverage & Co-funding.1": [
            "We can provide 50,000 EUR in funding for this project",
            "Computing resources and data storage will be provided in-kind",
            "No external funding available",
        ],
    }
    return pd.DataFrame(data)


def test_server_tools():
    """Test all PhIdeas server tools"""

    print("=" * 60)
    print("PhIdeas MCP Server Test Suite")
    print("=" * 60)

    # Create temporary directory for test files
    test_dir = Path(tempfile.mkdtemp(prefix="phideas_test_"))
    print(f"\n✓ Created temporary test directory: {test_dir}")

    try:
        # Create mock data
        df = create_mock_dataframe()
        test_file = test_dir / "test_data.csv"
        df.to_csv(test_file, index=False)
        print(f"✓ Created mock dataset with {len(df)} rows")

        # Test 1: Get dataframe info
        print("\n" + "-" * 60)
        print("TEST 1: Get Dataframe Info")
        print("-" * 60)
        result = get_dataframe_info(str(test_file))
        print(result)
        assert "Shape: 3 rows" in result
        print("✓ Test passed")

        # Test 2: List available columns
        print("\n" + "-" * 60)
        print("TEST 2: List Available Columns")
        print("-" * 60)
        result = list_available_columns(str(test_file))
        print(result)
        assert "ESA experts" in result
        assert "Description" in result
        print("✓ Test passed")

        # Test 3: Clean expert column
        print("\n" + "-" * 60)
        print("TEST 3: Clean Expert Column")
        print("-" * 60)
        output_file = test_dir / "step1_cleaned.csv"
        result = clean_expert_column(
            str(test_file), column_name="ESA experts", output_filepath=str(output_file)
        )
        print(result)
        assert "Successfully cleaned" in result
        assert output_file.exists()
        print("✓ Test passed - Output file created")

        # Test 4: Summarize description column
        print("\n" + "-" * 60)
        print("TEST 4: Summarize Description Column")
        print("-" * 60)
        output_file = test_dir / "step2_summarized.csv"
        result = summarize_description_column(
            str(test_file),
            source_column="Description",
            target_column="Description Summary",
            output_filepath=str(output_file),
        )
        print(result)
        assert "Successfully summarized" in result
        assert output_file.exists()
        print("✓ Test passed - Output file created")

        # Test 5: Categorize commercial status
        print("\n" + "-" * 60)
        print("TEST 5: Categorize Commercial Status")
        print("-" * 60)
        output_file = test_dir / "step3_commercial.csv"
        result = categorize_commercial_status(
            str(test_file), output_filepath=str(output_file)
        )
        print(result)
        assert "Successfully categorized" in result
        assert output_file.exists()
        print("✓ Test passed - Output file created")

        # Test 6: Categorize cofunding type
        print("\n" + "-" * 60)
        print("TEST 6: Categorize Cofunding Type")
        print("-" * 60)
        output_file = test_dir / "step4_cofunding.csv"
        result = categorize_cofunding_type(
            str(test_file), output_filepath=str(output_file)
        )
        print(result)
        assert "Successfully categorized" in result
        assert output_file.exists()
        print("✓ Test passed - Output file created")

        # Test 7: Categorize funding binary (requires Description Summary)
        print("\n" + "-" * 60)
        print("TEST 7: Categorize Funding Binary")
        print("-" * 60)
        # First add Description Summary column
        df_with_summary = df.copy()
        df_with_summary["Description Summary"] = [
            "AI for climate monitoring",
            "Climate data processing system",
            "Satellite image analysis tool",
        ]
        temp_file = test_dir / "temp_with_summary.csv"
        df_with_summary.to_csv(temp_file, index=False)

        output_file = test_dir / "step5_funding_binary.csv"
        result = categorize_funding_binary(
            str(temp_file), output_filepath=str(output_file)
        )
        print(result)
        assert "Successfully categorized" in result
        assert output_file.exists()
        print("✓ Test passed - Output file created")

        # Test 8: Categorize ideas weighted
        print("\n" + "-" * 60)
        print("TEST 8: Categorize Ideas Weighted (Auto-generate)")
        print("-" * 60)
        output_file = test_dir / "step6_weighted.csv"
        result = categorize_ideas_weighted(
            str(temp_file),
            idea_column="Description Summary",
            max_categories=5,
            max_per_idea=2,
            output_filepath=str(output_file),
        )
        print(result)
        assert "Successfully categorized" in result
        assert output_file.exists()
        print("✓ Test passed - Output file created")

        # Test 9: Categorize with imposed categories
        print("\n" + "-" * 60)
        print("TEST 9: Categorize Ideas with Imposed Categories")
        print("-" * 60)
        output_file = test_dir / "step7_imposed.csv"
        result = categorize_ideas_with_imposed_categories(
            str(temp_file),
            idea_column="Description Summary",
            imposed_categories="AI & Machine Learning, Data Processing, Remote Sensing, Other",
            max_per_idea=2,
            output_filepath=str(output_file),
        )
        print(result)
        assert "Successfully categorized" in result
        assert output_file.exists()
        print("✓ Test passed - Output file created")

        # Test 10: Error handling - missing column
        print("\n" + "-" * 60)
        print("TEST 10: Error Handling - Missing Column")
        print("-" * 60)
        result = clean_expert_column(str(test_file), column_name="NonExistentColumn")
        print(result)
        assert "Error" in result
        assert "not found" in result
        print("✓ Test passed - Error handled correctly")

        # Test 11: Error handling - invalid file
        print("\n" + "-" * 60)
        print("TEST 11: Error Handling - Invalid File")
        print("-" * 60)
        result = get_dataframe_info("/nonexistent/path/file.csv")
        print(result)
        assert "Error" in result
        print("✓ Test passed - Error handled correctly")

        # Test 12: Test Excel format
        print("\n" + "-" * 60)
        print("TEST 12: Excel File Format Support")
        print("-" * 60)
        excel_file = test_dir / "test_data.xlsx"
        df.to_excel(excel_file, index=False)
        result = get_dataframe_info(str(excel_file))
        print(result)
        assert "Shape: 3 rows" in result
        print("✓ Test passed - Excel format supported")

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print(f"\nGenerated files in: {test_dir}")
        print("\nTo inspect test outputs:")
        print(f"  ls -la {test_dir}")
        print(f"  cat {test_dir}/step1_cleaned.csv")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback

        traceback.print_exc()

    finally:
        # Ask user if they want to keep test files
        print(f"\n\nTest files location: {test_dir}")
        keep = input("Keep test files? (y/n): ").lower().strip()
        if keep != "y":
            shutil.rmtree(test_dir)
            print("✓ Cleaned up test files")
        else:
            print(f"✓ Test files preserved at: {test_dir}")




def test_server_tools():
    """Test all PhIdeas server tools"""

    print("=" * 60)
    print("PhIdeas MCP Server Test Suite")
    print("=" * 60)

    # Create temporary directory for test files
    test_dir = Path(tempfile.mkdtemp(prefix="phideas_test_"))
    print(f"\n✓ Created temporary test directory: {test_dir}")

    try:
        # Create mock data
        df = create_mock_dataframe()
        test_file = test_dir / "test_data.csv"
        df.to_csv(test_file, index=False)
        print(f"✓ Created mock dataset with {len(df)} rows")

        # Test 1: Get dataframe info
        print("\n" + "-" * 60)
        print("TEST 1: Get Dataframe Info")
        print("-" * 60)
        result = get_dataframe_info(str(test_file))
        print(result)
        assert "Shape: 3 rows" in result
        print("✓ Test passed")

        # Test 2: List available columns
        print("\n" + "-" * 60)
        print("TEST 2: List Available Columns")
        print("-" * 60)
        result = list_available_columns(str(test_file))
        print(result)
        assert "ESA experts" in result
        assert "Description" in result
        print("✓ Test passed")

        # Test 3: Clean expert column
        print("\n" + "-" * 60)
        print("TEST 3: Clean Expert Column")
        print("-" * 60)
        output_file = test_dir / "step1_cleaned.csv"
        result = clean_expert_column(
            str(test_file), column_name="ESA experts", output_filepath=str(output_file)
        )
        print(result)
        assert "Successfully cleaned" in result
        assert output_file.exists()
        print("✓ Test passed - Output file created")

        # Test 4: Summarize description column
        print("\n" + "-" * 60)
        print("TEST 4: Summarize Description Column")
        print("-" * 60)
        output_file = test_dir / "step2_summarized.csv"
        result = summarize_description_column(
            str(test_file),
            source_column="Description",
            target_column="Description Summary",
            output_filepath=str(output_file),
        )
        print(result)
        assert "Successfully summarized" in result
        assert output_file.exists()
        print("✓ Test passed - Output file created")

        # Test 5: Categorize commercial status
        print("\n" + "-" * 60)
        print("TEST 5: Categorize Commercial Status")
        print("-" * 60)
        output_file = test_dir / "step3_commercial.csv"
        result = categorize_commercial_status(
            str(test_file), output_filepath=str(output_file)
        )
        print(result)
        assert "Successfully categorized" in result
        assert output_file.exists()
        print("✓ Test passed - Output file created")

        # Test 6: Categorize cofunding type
        print("\n" + "-" * 60)
        print("TEST 6: Categorize Cofunding Type")
        print("-" * 60)
        output_file = test_dir / "step4_cofunding.csv"
        result = categorize_cofunding_type(
            str(test_file), output_filepath=str(output_file)
        )
        print(result)
        assert "Successfully categorized" in result
        assert output_file.exists()
        print("✓ Test passed - Output file created")

        # Test 7: Categorize funding binary (requires Description Summary)
        print("\n" + "-" * 60)
        print("TEST 7: Categorize Funding Binary")
        print("-" * 60)
        # First add Description Summary column
        df_with_summary = df.copy()
        df_with_summary["Description Summary"] = [
            "AI for climate monitoring",
            "Climate data processing system",
            "Satellite image analysis tool",
        ]
        temp_file = test_dir / "temp_with_summary.csv"
        df_with_summary.to_csv(temp_file, index=False)

        output_file = test_dir / "step5_funding_binary.csv"
        result = categorize_funding_binary(
            str(temp_file), output_filepath=str(output_file)
        )
        print(result)
        assert "Successfully categorized" in result
        assert output_file.exists()
        print("✓ Test passed - Output file created")

        # Test 8: Categorize ideas weighted
        print("\n" + "-" * 60)
        print("TEST 8: Categorize Ideas Weighted (Auto-generate)")
        print("-" * 60)
        output_file = test_dir / "step6_weighted.csv"
        result = categorize_ideas_weighted(
            str(temp_file),
            idea_column="Description Summary",
            max_categories=5,
            max_per_idea=2,
            output_filepath=str(output_file),
        )
        print(result)
        assert "Successfully categorized" in result
        assert output_file.exists()
        print("✓ Test passed - Output file created")

        # Test 9: Categorize with imposed categories
        print("\n" + "-" * 60)
        print("TEST 9: Categorize Ideas with Imposed Categories")
        print("-" * 60)
        output_file = test_dir / "step7_imposed.csv"
        result = categorize_ideas_with_imposed_categories(
            str(temp_file),
            idea_column="Description Summary",
            imposed_categories="AI & Machine Learning, Data Processing, Remote Sensing, Other",
            max_per_idea=2,
            output_filepath=str(output_file),
        )
        print(result)
        assert "Successfully categorized" in result
        assert output_file.exists()
        print("✓ Test passed - Output file created")

        # Test 10: Error handling - missing column
        print("\n" + "-" * 60)
        print("TEST 10: Error Handling - Missing Column")
        print("-" * 60)
        result = clean_expert_column(str(test_file), column_name="NonExistentColumn")
        print(result)
        assert "Error" in result
        assert "not found" in result
        print("✓ Test passed - Error handled correctly")

        # Test 11: Error handling - invalid file
        print("\n" + "-" * 60)
        print("TEST 11: Error Handling - Invalid File")
        print("-" * 60)
        result = get_dataframe_info("/nonexistent/path/file.csv")
        print(result)
        assert "Error" in result
        print("✓ Test passed - Error handled correctly")

        # Test 12: Test Excel format
        print("\n" + "-" * 60)
        print("TEST 12: Excel File Format Support")
        print("-" * 60)
        excel_file = test_dir / "test_data.xlsx"
        df.to_excel(excel_file, index=False)
        result = get_dataframe_info(str(excel_file))
        print(result)
        assert "Shape: 3 rows" in result
        print("✓ Test passed - Excel format supported")

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print(f"\nGenerated files in: {test_dir}")
        print("\nTo inspect test outputs:")
        print(f"  ls -la {test_dir}")
        print(f"  cat {test_dir}/step1_cleaned.csv")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback

        traceback.print_exc()

    finally:
        # Ask user if they want to keep test files
        print(f"\n\nTest files location: {test_dir}")
        keep = input("Keep test files? (y/n): ").lower().strip()
        if keep != "y":
            shutil.rmtree(test_dir)
            print("✓ Cleaned up test files")
        else:
            print(f"✓ Test files preserved at: {test_dir}")



if __name__ == "__main__":
    test_server_tools()
