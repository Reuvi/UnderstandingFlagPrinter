

'''

Polynomial Addition:

For Polynomial add we take two lists of coefficients,
then we 0 pad them in case one list is a higher degree than the other
and then we just do a list comprehension adding all the individuals together 
from the ziped iterable we made.
'''
def poly_add(a: list[int], b: list[int]):
    return [x + y for x, y in zip(
        a + [0]*(len(b) - len(a)),
        b + [0]*(len(a) - len(b))
    )]


'''

Polynomial Multiplication:

Polynomial A has degree A
Poly B has degree B
Theirfore the new Poly has degree Da + Db
Which would be Da + Db + a Coefficients (The term of x^0)
len(a) = Da + 1 and len(b) = Db + 1 theirfore len(a) + len(b) - 1 gives us
the new multiplied size. Coefficients are 0 by default

'''

def poly_mul(a: list[int], b: list[int]) -> list[int]:
    res = [0] * (len(a) + len(b) - 1)
    for i in range(len(a)):
        for j in range(len(b)):
            res[i + j] += a[i] * b[j] #Simple Algo for multiplying polynomials
    return res


"""
Build the polynomial coefficiants using lagrange interpolation from a set
of points given.

Polynomials are represented in code as Lists [a0, a1, a2, .... aN]
--> a0 + a1x + a2x^2 + aNx^n

"""

def langrange_poly(X: list[int], Y: list[int]) -> list[int]:
    n = len(X)
    P = [0]

    for j in range(n):
        
        #We need to construct one of the Sub Polynomials for point J

        subPoly = [1]
        denom = 1
        for m in range(n):
            if m != j:
                subPoly = poly_mul(subPoly, [-X[m], 1]) # subpoly * (x - X[m])
                denom *= (X[j] - X[m])
            
        subPoly = [c * Y[j] / denom for c in subPoly]
        P = poly_add(P, subPoly)

    return P


#Testing the Algo

X = []
Y = []

with open("testPoints.txt", "r") as myFile:
    
    x = ""
    for line in myFile.readlines():
        point = line.split(",")
        X.append(int(point[0]))
        Y.append(int(point[1]))



coeffs = langrange_poly(X, Y)
eq = "F(x) = "
for i in range(len(coeffs)):
    eq += str(coeffs[i]) + "x^" + str(i) + " "

eq + " \n"
print(eq)


