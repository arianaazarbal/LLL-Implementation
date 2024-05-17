
from LLL import grahamSchmidt, LLLNaive, LLLOptimized

###############################################################################
#MANUALLY CREATED TEST CASES
######################################################################################

test_basis_2 = np.array([[201, 37], [1648, 297]])
reduced_test_basis_2 = np.array([[1, 32], [40, 1]])

test_basis_31 = np.asarray([[1, 1, 1], [-1, 0, 2], [3, 5, 6]])
reduced_test_basis_31 = np.array([[0, 1, 0], [1, 0, 1], [-1, 0, 2]])

test_basis_32 = np.asarray([[15, 23, 11], [46, 15, 3], [32, 1, 1]])
reduced_test_basis_32 = np.asarray([[1, 9, 9], [13, 5, -7], [6, -9, 15]])

test_basis_6 = np.asarray([[19, 2, 32, 46, 3, 33],
                           [15, 42, 11, 0, 3, 24],
                           [43, 15, 0, 24, 4, 16],
                           [20, 44, 44, 0, 18, 15],
                           [0, 48, 35, 16, 31, 31],
                           [48, 33, 32, 9, 1, 29]])

reduced_test_basis_6 = np.asarray(
    [[7, -12,  -8,   4,  19,   9],
     [-20,   4,  -9,  16,  13,  16],
     [  5,   2,  33,   0,  15,  -9],
     [ -6,  -7, -20, -21,   8, -12],
     [-10, -24,  21, -15,  -6, -11],
     [  7,   4,  -9, -11,   1,  31]])

def checkAccuracyManual(lll_func):
    assert(np.isclose(lll_func(test_basis_2), reduced_test_basis_2).all())
    assert(np.isclose(lll_func(test_basis_31), reduced_test_basis_31).all())
    assert(np.isclose(lll_func(test_basis_32),reduced_test_basis_32).all())
    assert(np.isclose(lll_func(test_basis_6), reduced_test_basis_6).all())
    return True

###############################################################################
#AUTOMATED TESTING FUNCTIONS
###############################################################################


def compareEfficiencyPrint(func_1_name, func_2_name, speedup_only=False, **kwargs):
    t1, t2 = compareEfficiency(**kwargs)
    if not speedup_only:
        print("Lattice Dimension:", kwargs.get('dimension', 10), ", number of tests:", kwargs.get('num_test_cases', 100))
        print(func_1_name, ": ", t1, "s")
        print(func_2_name, ": ", t2, "s")
        print("Speedup: ", t1/t2, "x\n")
    else: 
        print("Lattice Dimension:", kwargs.get('dimension', 10), "Speedup: ", t1/t2, "x\n")
    
def compareEfficiency(lll_func_1, lll_func_2, num_test_cases=100, dimension=10, int_range=(-100, 100)):
    test_bases = []
    for i in range(num_test_cases):
        random_basis = np.random.randint(size=(dimension, dimension), low=int_range[0], high=int_range[1], dtype=int)
        if np.linalg.det(random_basis == 0): 
            i -= 1
        else:
            test_bases += [random_basis]

    start_1 = time.time()
    for basis in test_bases:
        lll_func_1(basis)
    end_1 = time.time()

    start_2 = time.time()
    for basis in test_bases:
        lll_func_2(basis)
    end_2 = time.time()

    return (end_1 - start_1, end_2 - start_2)

def plotTimeData(lll_func, dimension_bits=[2, 3, 4, 5, 6, 7, 8, 9],
                 int_range=(-100, 100)):
    dims = []
    bit_dims = []
    times = []
    for d_b in dimension_bits:
        bit_dims += [d_b]
        dims += [2 ** d_b]
        random_basis = np.random.randint(size=(2 ** d_b, 2 ** d_b), low=-100, high=100, dtype=int)
        while np.linalg.det(random_basis == 0): 
            random_basis = np.random.randint(size=(2 ** d_b, 2 ** d_b), low=-100, high=100, dtype=int)
        
        start = time.time()
        lll_func(random_basis)
        end = time.time()
        times += [end - start]
        
    # Plot the data points and the best fit line
    plt.scatter(dims, times)
    
    # Add labels and legend
    plt.xlabel('Lattice Dimension')
    plt.ylabel('Time (s)')
    plt.title('LLL Runtime vs. Dimension')
    plt.legend()
    
    # Show the plot
    plt.show()
    degrees = np.arange(len(bit_dims))
    min_degree = np.argmin(np.array([np.polyval(np.polyfit(bit_dims, times, deg), bit_dims).std() for deg in degrees])) + 1
    
    # Calculate the coefficients of the best fit polynomial
    coefficients = np.polyfit(bit_dims, times, min_degree - 1)
    
    # Create a line with the best fit polynomial
    polynomial = np.poly1d(coefficients)
    x_line = np.linspace(min(bit_dims), max(bit_dims), 100)
    y_line = polynomial(x_line)
    
    # Plot the data points and the best fit curve
    plt.scatter(bit_dims, times)
    #plt.plot(x_line, y_line, 'r', label=f'Best Fit Curve: {polynomial}')
    
    # Add labels and legend
    plt.xlabel('Bits of Dimension')
    plt.ylabel('Time (s)')
    plt.title('LLL Runtime vs. Bits of Dimension')
    plt.legend()
    
    # Show the plot
    plt.show()
    
                                                       
def testingOracle(lll_func, num_test_cases=100, dimension_range=(2, 40), int_range=(-100, 100)):
    assert(checkAccuracyManual(lll_func))
    for i in range(num_test_cases):
        n = np.random.randint(low=dimension_range[0], high=dimension_range[1])
        random_basis = np.random.randint(size=(n, n), low=int_range[0], high=int_range[1], dtype=int)
        if np.linalg.det(random_basis == 0): 
            i -= 1
        else:
            if not verifyOutput(random_basis, lll_func(random_basis)):
                return random_basis, lll_func(random_basis)
    return True
        
    
def verifyOutput(basis, lll_basis):
    same_lattice = np.isclose(abs(np.linalg.det(basis)), abs(np.linalg.det(lll_basis)))
    if not same_lattice: 
        print("Not the initial lattice")
        return False
    lovasz = checkLovaszCondition(lll_basis)
    size = checkSizeCondition(lll_basis)
    if not (lovasz and size):
        print("Not LLL reduced")
        return False
    return True

def checkLovaszCondition(lll_basis):
    gs = grahamSchmidt(lll_basis)
    n = lll_basis.shape[0]
    
    for i in range(1, n):   
        left_hand_side = dot(gs[i], gs[i])
        right_hand_side = ((3 / 4) - np.square(get_u(lll_basis[i], gs[i - 1]))) * dot(gs[i - 1], gs[i - 1])
        lovasz_condition = (left_hand_side >= right_hand_side)
        if not lovasz_condition:
            return False
    return True

def checkSizeCondition(lll_basis):
    n = lll_basis.shape[0]
    gs = grahamSchmidt(lll_basis)
    for i in range(1, n):
        for j in range(i):
            if get_u(lll_basis[i], gs[j]) > 0.5:
                return False
    return True
