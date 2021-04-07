
def TrapezoidRule(integrand, h):
    """ Perform integration using the trapezoid rule.
    """
    return h * (0.5 * (integrand[0] + integrand[-1]) + \
                sum([integrand[int(k * h)] for k in range(1, (integrand.size -
                    1)/h)]))

def RombergStep(R_im, R_iminus, m): 
    return R_im + 1 / (np.pow(4, m) - 1) * (R_im - R_iminus)

def RombergIntegral(integrand, h, delta): 
    """ Compute integral using Romberg integration.

        === params ===
        integrand: array of values to integrate over
        h: int timestep
    """
    error = inf
    bottom_out = False

    roberg_matrix = np.zeros((10,10))
    roberg_matrix[0,0] = self.TrapezoidRule(integrand, h)  # R1_1

    for i in range(1, 9):
        # calculate the next trapezoid rule estimate
        romberg_matrix[i,0] = self.TrapezoidRule(integrand, np.pow(h, 2*i)) 
        # fill the interior of the matrix
        for j in range(1, i+1):
            romberg_matrix[i, j] = self.RombergStep(romberg_matrix[i, j-1],
                                                     romber_matrix[i-1, j-1],
                                                     j-1)

        # calculate error
        error = np.abs(romberg_matrix[i, j] - romberg_matrix[i, j-1])
        if error < delta:
            return romberg_matrix[i, i]

    # default return last integral calculated
    print("error condition not reached")
    print("returning last calculated integral")
    return romberg_matrix[9, 9]

def REN(pct_array):
    """ Calculate the Renyi Entropy
    """
    pass

