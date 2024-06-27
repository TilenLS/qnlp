def sum_odd(arr):
    """
    Compute the maximum value of the sum of the given 
    array `arr` with odd number of elements first mutliplied
    by -1.
    """
    arr_neg = [a for a in arr if a <= 0]
    arr_abs = [abs(a) for a in arr]

    s = sum(arr_abs)

    if len(arr_neg) % 2 == 0:
        return s - 2 * min(arr_abs)
    else:
        return s