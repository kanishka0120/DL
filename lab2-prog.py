import numpy as np

# Vectorized unit step function for array inputs
def unit_step(x):
    return np.where(x >= 0, 1, 0)

# Generalized perceptron model
def perceptron_model(x, w, b):
    v = np.dot(w, x) + b
    return unit_step(v)

# Logic Gate Definitions
def NOT_logic(x):
    return perceptron_model(x, w=-1, b=0.5)

def AND_logic(x):
    return perceptron_model(x, w=np.array([1, 1]), b=-1.5)

def OR_logic(x):
    return perceptron_model(x, w=np.array([1, 1]), b=-0.5)

def NAND_logic(x):
    return NOT_logic(AND_logic(x))

def XOR_logic(x):
    # XOR implemented using AND, OR, and NOT gates
    y1 = AND_logic(x)
    y2 = OR_logic(x)
    y3 = NOT_logic(y1)
    final_x = np.array([y2, y3])
    return AND_logic(final_x)

# General testing function for logic gates
def test_gate(gate_function, gate_name, test_cases):
    print(f"\nTesting {gate_name} gate:")
    for test in test_cases:
        result = gate_function(test)
        print(f"{gate_name} gate({test[0]},{test[1]}) = {result}")

# Define test cases
test_cases = [np.array([0, 1]), np.array([1, 1]), np.array([0, 0]), np.array([1, 0])]

# Testing all gates
test_gate(XOR_logic, "XOR", test_cases)
test_gate(AND_logic, "AND", test_cases)
test_gate(OR_logic, "OR", test_cases)
test_gate(NAND_logic, "NAND", test_cases)

# Testing NOT gate separately (single input)
print("\nTesting NOT gate:")
for x in [0, 1]:
    print(f"NOT gate({x}) = {NOT_logic(np.array(x))}")