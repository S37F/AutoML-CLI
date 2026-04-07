"""
Final Portfolio Test - Automated Testing
"""
import subprocess
import sys

def run_test(name, command, input_text=""):
    """Run a test and display results"""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    
    try:
        if input_text:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                input=input_text,
                timeout=120
            )
        else:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120
            )
        
        print(result.stdout)
        if result.returncode == 0:
            print(f"\n✅ {name} - PASSED")
            return True
        else:
            print(f"\n❌ {name} - FAILED")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"\n❌ {name} - ERROR: {str(e)}")
        return False


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║             AUTOML CLI - PORTFOLIO TESTING SUITE               ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Regression with full features
    total_tests += 1
    if run_test(
        "Regression (House Prices) - Full Features",
        "python automl_cli.py data/house_prices.csv --auto --all",
        "n\n"
    ):
        tests_passed += 1
    
    # Test 2: Classification with basic features
    total_tests += 1
    if run_test(
        "Classification (Iris) - Basic Features",
        "python automl_cli.py data/iris_classification.csv --auto",
        "n\n"
    ):
        tests_passed += 1
    
    # Test 3: Fast mode
    total_tests += 1
    if run_test(
        "Fast Mode Test",
        "python automl_cli.py data/house_prices.csv --auto --quick",
        "n\n"
    ):
        tests_passed += 1
    
    # Results
    print(f"\n\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    print(f"Success Rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print("\n✅ ALL TESTS PASSED - Portfolio Ready! 🚀")
        sys.exit(0)
    else:
        print(f"\n⚠️ {total_tests - tests_passed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
