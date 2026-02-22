import json
import os
import subprocess
import sys


def verify_fixture(minizinc_file, json_fixture):
    # Run minizinc to get the valid solution
    # Note: parsing minizinc output is tricky, better if minizinc outputs json, but standard minizinc output is plain text
    # command: minizinc --solver Gecode planning_validation.mzn

    # Check if minizinc is installed
    try:
        subprocess.run(["minizinc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError:
        print("Error: minizinc executable not found in PATH.")
        sys.exit(1)

    result = subprocess.run(["minizinc", minizinc_file], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running minizinc: {result.stderr}")
        sys.exit(1)

    output = result.stdout
    print("Minizinc output:")
    print(output)

    # Parse expected sequence from minizinc output
    # Output format defined in .mzn:
    # "Action sequence:
    #   1: action_transfer_flag(0, 1)
    #   ..."

    minizinc_actions = []
    for line in output.splitlines():
        if ": action_transfer_flag(" in line:
            # extract params
            # format: "  1: action_transfer_flag(0, 1)"
            parts = line.split("action_transfer_flag(")[1].strip(")").split(",")
            from_flag = int(parts[0].strip())
            to_flag = int(parts[1].strip())
            minizinc_actions.append(["action_transfer_flag", from_flag, to_flag])

    # Load JSON fixture
    with open(json_fixture, "r") as f:
        fixture_data = json.load(f)

    # Compare
    print("\nComparing...")
    print(f"Fixture:  {fixture_data}")
    print(f"Minizinc: {minizinc_actions}")

    if fixture_data == minizinc_actions:
        print("\nSUCCESS: JSON fixture matches Minizinc validation model.")
        sys.exit(0)
    else:
        print("\nFAILURE: JSON fixture does NOT match Minizinc validation model.")
        sys.exit(1)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    minizinc_file = os.path.join(base_dir, "planning_validation.mzn")
    # Fixture is in ../fixtures/ipyhop_sample_test_1.json
    fixture_file = os.path.join(base_dir, "../fixtures/ipyhop_sample_test_1.json")

    verify_fixture(minizinc_file, fixture_file)
